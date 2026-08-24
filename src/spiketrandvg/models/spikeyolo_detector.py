"""SpikeYOLO for event-frame object detection, and the backbone the grounding model reuses.

The topology follows SpikeYOLO's own `snn_yolov8.yaml` (its Gen1 event-detection variant),
rebuilt here through the fork's module constructors rather than the ultralytics YAML
parser -- that parser only runs inside the fork's package, whose `__init__` pulls the whole
YOLO/SAM stack and writes to `~/.config`. Every block is SpikeYOLO's; nothing is edited.

    stem   MS_DownSampling  2 -> C0, k7 s4          /4
    b1     MS_AllConvBlock  x d0                    /4     <- tap s4
    ds2    MS_DownSampling  C0 -> C1, k3 s2         /8
    b2     MS_AllConvBlock  x d1                    /8     <- tap s8
    ds3    MS_DownSampling  C1 -> C2                /16
    b3     MS_ConvBlock     x d2                    /16    <- tap s16
    ds4    MS_DownSampling  C2 -> C3                /32
    b4     MS_ConvBlock     x d3                    /32
    sppf   SpikeSPPF                                /32    <- tap s32
    neck   PAN, top-down then bottom-up                    -> P3 /8, P4 /16, P5 /32
    head   SpikeDetect (anchor-free, DFL reg_max=16)

Why this exists
---------------
Training the grounding model end to end from an ImageNet-initialised classifier backbone
failed to localise: the finished run reached 0.22 mIoU, and a caption-blind control scored
within 0.001 of it, i.e. the model was predicting a salient box from events alone and
never learned where the *referred* object was. Localisation is learnable from events with
no language involved, so it is learned here first, on the same frames, as plain detection.
`DetectionBackbone` then hands the grounding model a vision tower that already knows where
objects are, leaving cross-attention with only the selection problem.

`MS_GetT` from the YAML is deliberately absent: it exists to replicate a single frame
across T, and our input already carries a real T axis of event bins.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiketrandvg.datasets.events_voxel_cube import T_STEPS
from spiketrandvg.utils import forks

__all__ = ["DetectionBackbone", "SpikeYOLODetector", "TAP_STRIDES"]

TAP_STRIDES = {"s4": 4, "s8": 8, "s16": 16, "s32": 32}


def _up2(x: torch.Tensor) -> torch.Tensor:
    T, B, C, H, W = x.shape
    return F.interpolate(x.flatten(0, 1), scale_factor=2.0, mode="nearest").reshape(
        T, B, C, H * 2, W * 2
    )


def _cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Channel-concat two (T,B,C,H,W) maps, trimming to the smaller spatial size."""
    if a.shape[-2:] != b.shape[-2:]:
        h = min(a.shape[-2], b.shape[-2])
        w = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :h, :w], b[..., :h, :w]
    return torch.cat((a, b), dim=2)


class DetectionBackbone(nn.Module):
    """SpikeYOLO backbone over event cubes, tapped at strides 4/8/16/32.

    forward(cube) : (T, B, 2, H, W) -> {"s4","s8","s16","s32"}: (T, B, C, H/s, W/s)

    Outputs are BatchNorm outputs (analog membranes), matching what `MS_DownSampling` and
    the conv blocks emit internally -- every consumer opens with its own neuron, which is
    SpikeYOLO's own convention throughout.
    """

    def __init__(
        self,
        in_channels: int = 2,
        width: float = 0.5,
        depths: tuple[int, int, int, int] = (1, 2, 3, 1),
        base: tuple[int, int, int, int] = (128, 256, 512, 1024),
    ):
        super().__init__()
        sy = forks.load_spikeyolo()
        c = [max(16, int(round(b * width))) for b in base]
        self.channels = {"s4": c[0], "s8": c[1], "s16": c[2], "s32": c[3]}

        self.stem = sy.MS_DownSampling(in_channels, c[0], kernel_size=7, stride=4,
                                       padding=2, first_layer=True)
        self.b1 = nn.Sequential(*[sy.MS_AllConvBlock(c[0], 4.0, 7) for _ in range(depths[0])])
        self.ds2 = sy.MS_DownSampling(c[0], c[1], kernel_size=3, stride=2,
                                      padding=1, first_layer=False)
        self.b2 = nn.Sequential(*[sy.MS_AllConvBlock(c[1], 4.0, 7) for _ in range(depths[1])])
        self.ds3 = sy.MS_DownSampling(c[1], c[2], kernel_size=3, stride=2,
                                      padding=1, first_layer=False)
        self.b3 = nn.Sequential(*[sy.MS_ConvBlock(c[2], 3.0, 7) for _ in range(depths[2])])
        self.ds4 = sy.MS_DownSampling(c[2], c[3], kernel_size=3, stride=2,
                                      padding=1, first_layer=False)
        self.b4 = nn.Sequential(*[sy.MS_ConvBlock(c[3], 2.0, 7) for _ in range(depths[3])])
        self.sppf = sy.SpikeSPPF(c[3], c[3], 5)

    def forward(self, cube: torch.Tensor) -> dict[str, torch.Tensor]:
        if cube.dim() != 5:
            raise ValueError(f"expected (T, B, C, H, W), got {tuple(cube.shape)}")
        if cube.shape[-1] % 32 or cube.shape[-2] % 32:
            raise ValueError(f"H and W must be divisible by 32, got {tuple(cube.shape[-2:])}")
        s4 = self.b1(self.stem(cube))
        s8 = self.b2(self.ds2(s4))
        s16 = self.b3(self.ds3(s8))
        s32 = self.sppf(self.b4(self.ds4(s16)))
        return {"s4": s4, "s8": s8, "s16": s16, "s32": s32}


class SpikeYOLODetector(nn.Module):
    """Backbone + PAN neck + SpikeDetect, for the Talk2Event frame-detection pretraining.

    forward(cube)
        train : list of (B, 4*reg_max + nc, H_s, W_s), one per level  -- feed to the loss
        eval  : (pred, raw); pred (B, 4 + nc, A) with xywh in PIXELS then class scores

    `self.head.stride` is set here from the actual pyramid strides. Ultralytics normally
    infers it with a dummy forward inside its model builder, which is never run.
    """

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 2,
        width: float = 0.5,
        depths: tuple[int, int, int, int] = (1, 2, 3, 1),
        T: int = T_STEPS,
    ):
        super().__init__()
        sy = forks.load_spikeyolo()
        self.T = T
        self.num_classes = num_classes
        self.backbone = DetectionBackbone(in_channels, width, depths)
        c = self.backbone.channels
        c8, c16, c32 = c["s8"], c["s16"], c["s32"]

        # top-down
        self.lat32 = sy.MS_StandardConv(c32, c16, k=1, s=1)
        self.td16 = sy.MS_ConvBlock(c16, 3.0, 7)
        self.red16 = sy.MS_StandardConv(2 * c16, c8, k=1, s=1)
        self.td8 = sy.MS_AllConvBlock(c8, 4.0, 7)
        self.red8 = sy.MS_StandardConv(2 * c8, c8, k=1, s=1)
        self.out8 = sy.MS_AllConvBlock(c8, 4.0, 7)
        # bottom-up
        self.down8 = sy.MS_StandardConv(c8, c8, k=3, s=2)
        self.out16 = sy.MS_ConvBlock(2 * c8, 3.0, 7)
        self.down16 = sy.MS_StandardConv(2 * c8, c16, k=3, s=2)
        self.out32 = sy.MS_ConvBlock(c16 + c16, 1.0, 7)

        ch = (c8, 2 * c8, 2 * c16)
        self.head = sy.SpikeDetect(nc=num_classes, ch=ch)
        self.strides = (8, 16, 32)
        self.head.stride = torch.tensor(self.strides, dtype=torch.float32)
        self.head.bias_init()

    @property
    def reg_max(self) -> int:
        return self.head.reg_max

    def neck(self, f: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        t32 = self.lat32(f["s32"])                       # C16 @ /32
        p16 = self.red16(_cat(self.td16(_up2(t32)), f["s16"]))     # C8 @ /16
        p8 = self.out8(self.red8(_cat(self.td8(_up2(p16)), f["s8"])))   # C8 @ /8
        p16b = self.out16(_cat(self.down8(p8), p16))               # 2*C8 @ /16
        p32b = self.out32(_cat(self.down16(p16b), t32))            # 2*C16 @ /32
        return [p8, p16b, p32b]

    def forward(self, cube: torch.Tensor):
        if cube.shape[0] != self.T:
            raise ValueError(f"cube has T={cube.shape[0]}, model built for T={self.T}")
        feats = self.neck(self.backbone(cube))
        return self.head(list(feats))       # SpikeDetect mutates the list it is given
