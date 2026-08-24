"""E-3DSNN sparse-3D encoder over Talk2Event event volumes.

Wraps the OpenPCDet backbones from the E-3DSNN fork (arXiv 2412.07360) so an event
stream can be fed to them as what it natively is: a SPARSE 3D VOLUME. Nothing in
`repositories/E-3DSNN` is edited; the classes are loaded through
`spiketrandvg.utils.forks.load_e3dsnn`.

The mapping: events are already spike-voxel-coded
------------------------------------------------
A Talk2Event `.npz` holds (20, 480, 640) uint8 event COUNTS = 2 polarities x 10 time
bins (polarity-major, `ch = p*10 + t`). E-3DSNN's Spike Voxel Coding step exists to
turn an unordered point cloud into exactly this kind of sparse occupancy grid, so for
events that step is already done by the sensor. What remains is choosing which axis of
the network's 3D grid each event dimension occupies:

    network grid x  <-  event pixel x     (0..639)
    network grid y  <-  event pixel y     (0..479)
    network grid z  <-  event TIME bin    (0..9, placed at z = Z_STRIDE * t)

This assignment is forced, not free. The detection stack collapses z into channels
(`dense().view(N, C*D, H, W)`), so whatever occupies grid x/y is what indexes the
output feature map -- and the output map has to be the image plane for a 2D box to
come out of it. Time therefore lands on the axis the pretrained weights learned as
HEIGHT. That is the domain gap of this transfer, and it is the reason the numbers in
the notebook are reported per activation variant rather than assumed.

Z_SIZE is 41, not 10
--------------------
`VoxelBackBone8x` halves z three times (conv2, conv3, conv4) and then applies
`conv_out` with kernel (3,1,1) stride (2,1,1) padding 0. Reaching the D=2 that makes
the collapsed map 128*2 = 256 channels -- the width `BaseBEVBackbone` was trained for
-- requires z_in in {41, 42}: 41 -> 21 -> 11 -> 5 -> 2. Ten time bins are therefore
placed at z = 4t (0, 4, ..., 36) inside a 41-deep grid rather than at z = t. The bins
stay distinct until the third downsampling, the active-site count is unchanged (no
replication), and the pretrained conv_out / 2D trunk load unmodified.

Features per active voxel
-------------------------
KITTI's `MeanVFE` hands the backbone the mean raw point feature of each voxel, i.e.
(x, y, z) in METRES over the KITTI range plus reflectance in [0, 1]. The pretrained
convs and, more importantly, the frozen BatchNorm running statistics expect that
scale. Two feature modes are provided:

    "kitti_like" (default) -- each active voxel gets its grid coordinate mapped
        linearly onto KITTI's ranges (x -> [0, 70.4], y -> [-40, 40], z -> [-3, 1])
        plus count/10 as the reflectance analogue. Input statistics then resemble
        what the checkpoint was trained on, at the cost of discarding polarity.
    "event_counts" -- [pos/10, neg/10, total/10, t/T]. Keeps polarity, matches no
        pretrained statistic; the honest choice when the encoder is fine-tuned.

Use `recalibrate_bn` to re-estimate the BatchNorm running statistics on event data
(conv weights untouched) -- the cheapest available correction for the domain shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from spiketrandvg.utils import forks

__all__ = [
    "EVENT_H",
    "EVENT_W",
    "T_FILE",
    "Z_SIZE",
    "Z_STRIDE",
    "BEV_STRIDE",
    "BEV_CHANNELS",
    "E3DSNNEventEncoder",
    "LoadReport",
    "events_to_spike_voxels",
    "load_pretrained_weights",
    "recalibrate_bn",
]

EVENT_H, EVENT_W = 480, 640
T_FILE = 10          # time bins stored in the npz
POLARITIES = 2
Z_STRIDE = 4         # grid z of time bin t is Z_STRIDE * t
Z_SIZE = 41          # see module docstring: the only depth giving D=2 after conv_out
BEV_STRIDE = 8       # spatial stride of the feature map the backbones return
BEV_CHANNELS = 256

# KITTI point-cloud range the checkpoint was trained on, [x_min, y_min, z_min, x_max, y_max, z_max]
KITTI_RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)


class _Cfg(dict):
    """Attribute-access dict standing in for pcdet's EasyDict model configs."""

    __getattr__ = dict.get

    def get(self, key, default=None):
        return dict.get(self, key, default)


def events_to_spike_voxels(
    events: torch.Tensor,
    feature_mode: str = "kitti_like",
    count_scale: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 20, H, W) event counts -> (features (N, 4), coords (N, 4)) for spconv.

    `coords` columns are [batch_idx, z, y, x] as pcdet's backbones expect, with
    z = Z_STRIDE * time_bin. A voxel is active when either polarity fired in it, so
    the two polarity planes are merged into one active site carrying both counts.
    """
    if events.dim() == 3:
        events = events[None]
    b, c, h, w = events.shape
    if c != POLARITIES * T_FILE:
        raise ValueError(f"expected {POLARITIES * T_FILE} event channels, got {c}")
    ev = events.float().view(b, POLARITIES, T_FILE, h, w)   # polarity-major, ch = p*10 + t
    neg, pos = ev[:, 0], ev[:, 1]
    total = pos + neg

    bi, ti, yi, xi = torch.nonzero(total > 0, as_tuple=True)
    coords = torch.stack([bi, ti * Z_STRIDE, yi, xi], dim=1).int()

    if feature_mode == "kitti_like":
        x_min, y_min, z_min, x_max, y_max, z_max = KITTI_RANGE
        feats = torch.stack(
            [
                x_min + xi.float() / (w - 1) * (x_max - x_min),
                y_min + yi.float() / (h - 1) * (y_max - y_min),
                z_min + ti.float() / (T_FILE - 1) * (z_max - z_min),
                total[bi, ti, yi, xi] / count_scale,
            ],
            dim=1,
        )
    elif feature_mode == "event_counts":
        feats = torch.stack(
            [
                pos[bi, ti, yi, xi] / count_scale,
                neg[bi, ti, yi, xi] / count_scale,
                total[bi, ti, yi, xi] / count_scale,
                ti.float() / (T_FILE - 1),
            ],
            dim=1,
        )
    else:
        raise ValueError(f"unknown feature_mode {feature_mode!r}")
    return feats, coords


@dataclass
class LoadReport:
    """Outcome of pushing a checkpoint into the encoder."""

    loaded: int
    permuted: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"loaded {self.loaded} tensors ({self.permuted} relaid out for spconv 2.x), "
            f"{len(self.missing)} missing, {len(self.unexpected)} unexpected"
        )


class E3DSNNEventEncoder(nn.Module):
    """E-3DSNN sparse 3D backbone + BEV trunk over an event volume.

    forward(events) -> dict:
        "bev"      (B, 256, H/8, W/8)  feature map aligned with the event frame
        "x_conv1"  .. "x_conv4"        the sparse multi-scale tensors, strides 1/2/4/8
        "voxels"   number of active input sites (diagnostics)

    `activation="multispike"` swaps every ReLU for E-3DSNN's multi-bit spike neuron
    (integers in {0..4}), turning the loaded ANN weights into a spiking forward pass;
    `"relu"` keeps the network the checkpoint was actually trained as.
    """

    def __init__(
        self,
        activation: str = "multispike",
        feature_mode: str = "kitti_like",
        input_channels: int = 4,
        height: int = EVENT_H,
        width: int = EVENT_W,
    ) -> None:
        super().__init__()
        if activation not in ("multispike", "relu"):
            raise ValueError(f"activation must be 'multispike' or 'relu', got {activation!r}")
        fork = forks.load_e3dsnn()
        self.activation = activation
        self.feature_mode = feature_mode
        self.height, self.width = height, width

        # pcdet takes grid_size as (x, y, z) and does grid_size[::-1] + [1, 0, 0]
        # elementwise on a numpy array, so z is passed one short of Z_SIZE.
        grid_size = np.array([width, height, Z_SIZE - 1])
        self.backbone_3d = fork.VoxelBackBone8x(
            model_cfg=_Cfg(), input_channels=input_channels, grid_size=grid_size
        )
        self.backbone_2d = fork.BaseBEVBackbone(
            model_cfg=_Cfg(
                LAYER_NUMS=[5, 5],
                LAYER_STRIDES=[1, 2],
                NUM_FILTERS=[64, 128],
                UPSAMPLE_STRIDES=[1, 2],
                NUM_UPSAMPLE_FILTERS=[128, 128],
            ),
            input_channels=BEV_CHANNELS,
        )
        self.out_channels = self.backbone_2d.num_bev_features
        if activation == "multispike":
            self.n_spike_swaps = _swap_relu(self, fork.Multispike)
        else:
            self.n_spike_swaps = 0

    @property
    def sparse_shape(self) -> list[int]:
        return [int(v) for v in self.backbone_3d.sparse_shape]

    def forward(self, events: torch.Tensor) -> dict[str, object]:
        feats, coords = events_to_spike_voxels(events, self.feature_mode)
        return self.forward_voxels(feats, coords, batch_size=len(events))

    def forward_voxels(
        self, feats: torch.Tensor, coords: torch.Tensor, batch_size: int
    ) -> dict[str, object]:
        """Same as `forward` for pre-built voxels (lets a DataLoader worker do the encoding)."""
        # The fork's `VoxelBackBone8x.forward` is deliberately NOT called: it ends with
        # five unconditional debug prints (spconv_backbone.py:180-184) that would fire
        # on every training step. Its submodules are driven directly instead, which is
        # the same computation and also exposes the intermediate scales.
        import spconv.pytorch as spconv  # noqa: PLC0415

        b3 = self.backbone_3d
        x = spconv.SparseConvTensor(
            features=feats, indices=coords.int(), spatial_shape=b3.sparse_shape,
            batch_size=batch_size,
        )
        x = b3.conv_input(x)
        x1 = b3.conv1(x)
        x2 = b3.conv2(x1)
        x3 = b3.conv3(x2)
        x4 = b3.conv4(x3)
        enc = b3.conv_out(x4)
        dense = enc.dense()                              # (B, 128, 2, H/8, W/8)
        n, c, d, hh, ww = dense.shape
        bev = self.backbone_2d({"spatial_features": dense.view(n, c * d, hh, ww)})
        return {
            "bev": bev["spatial_features_2d"],
            "voxels": int(coords.shape[0]),
            "x_conv1": x1, "x_conv2": x2, "x_conv3": x3, "x_conv4": x4,
        }


def _swap_relu(module: nn.Module, spike_cls: type[nn.Module]) -> int:
    """Replace every nn.ReLU in the tree with a fresh `spike_cls`; returns the count."""
    swapped = 0
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, spike_cls())
            swapped += 1
        else:
            swapped += _swap_relu(child, spike_cls)
    return swapped


def load_pretrained_weights(
    encoder: E3DSNNEventEncoder,
    checkpoint: str | Path,
    strict: bool = True,
) -> dict[str, LoadReport]:
    """Load `kitti.pth` (`Xuerui123/E-3DSNN`) into the encoder's two backbones.

    The checkpoint was saved by an spconv 1.x-style codebase, whose conv weights are
    laid out (k1, k2, k3, c_in, c_out); spconv 2.x wants (c_out, k1, k2, k3, c_in).
    Mismatching 5D tensors are permuted exactly as pcdet's own
    `Detector3DTemplate.load_params_from_file` does it.

    `strict=True` raises unless every checkpoint tensor for a backbone is consumed and
    no parameter is left at its initialisation -- the failure this guards against is a
    silent partial load that looks like a working transfer.
    """
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = state.get("model_state", state)
    reports: dict[str, LoadReport] = {}
    for prefix, module in (("backbone_3d", encoder.backbone_3d),
                           ("backbone_2d", encoder.backbone_2d)):
        target = module.state_dict()
        sub, permuted = {}, 0
        for key, val in state.items():
            if not key.startswith(prefix + "."):
                continue
            name = key[len(prefix) + 1:]
            if name in target and target[name].shape != val.shape and val.dim() == 5:
                val = val.permute(4, 0, 1, 2, 3).contiguous()
                permuted += 1
            sub[name] = val
        result = module.load_state_dict(sub, strict=False)
        report = LoadReport(
            loaded=len(sub) - len(result.unexpected_keys),
            permuted=permuted,
            missing=tuple(result.missing_keys),
            unexpected=tuple(result.unexpected_keys),
        )
        if strict and (report.missing or report.unexpected):
            raise RuntimeError(f"{prefix}: partial load -- {report}")
        reports[prefix] = report
    return reports


@torch.no_grad()
def recalibrate_bn(
    encoder: E3DSNNEventEncoder,
    batches,
    momentum: float | None = None,
    device: str = "cuda",
) -> int:
    """Re-estimate BatchNorm running statistics on event data; conv weights untouched.

    The loaded statistics describe KITTI LiDAR voxels. Event voxels have a different
    occupancy and a different feature distribution, so every normalisation in the
    network is biased before this runs. `batches` is any iterable of
    (features, coords, batch_size) tuples -- one pass is enough with momentum=None,
    which switches BatchNorm to a cumulative average over the batches seen.

    Returns the number of BatchNorm layers that were updated.
    """
    bns = [m for m in encoder.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = momentum
    encoder.train()
    for feats, coords, bsz in batches:
        encoder.forward_voxels(feats.to(device), coords.to(device), bsz)
    encoder.eval()
    return len(bns)
