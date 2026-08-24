"""SpikeTransDVG: end-to-end spiking visual grounding on Talk2Event.

    events  (T, B, 2, H, W)          caption (B, L) token ids + mask
        |                                     |
    EventVisionEncoder                 SpikeLMTextEncoder
    (Meta-SpikeFormer, SDTv2)          (SpikeLM BERT, roberta-base init)
        |  {s4, s8, s16b} membranes           |  (B, L, d) tokens
        +---------------> CrossModalFusion <--+
                          (CMSF spiking cross-attention, per scale)
                                  |
                          SpikingPAN neck
                          (SpikeYOLO MS_StandardConv / MS_ConvBlock)
                                  |
                          SingleBoxHead
                          (spiking MLP, SpikeYOLO conv blocks)
                                  |
                        ONE box (B, 4) cxcywh

Every stage is spiking and every stage comes from a frozen fork; this module only wires
them and supplies the neck.

Design decisions that are not arbitrary
---------------------------------------
**One box, by construction.** Grounding returns exactly one region per expression, so the
head regresses four numbers directly. There is no anchor grid, no per-location score, no
label assignment and no argmax -- the model cannot emit a second box, and the loss is a
plain regression on the one it does emit. SpikeYOLO's `SpikeDetect` and its DFL are
therefore not used; SpikeYOLO still supplies every spiking conv block in the neck and the
head's front end.

**Pyramid at strides 4/8/16, not the usual 8/16/32.** Talk2Event boxes are small -- the
median target is ~62x66 px at 480x640 and the smallest is under 10 px -- so a stride-32
level would be 15x20 cells carrying nothing the stride-16 level does not already resolve,
while a stride-4 level is what gives sub-10 px objects any cells at all. The strides are a
constructor argument; nothing hard-codes them.

**Prediction is text-conditioned, not text-filtered.** The caption enters before the neck,
so the head never sees an unconditioned feature map. With a single-box head this is not
optional: there is no scoring stage downstream that could select among candidates, so the
caption must shape the features themselves.

Timing and state
----------------
The vision encoder's T (default `T_STEPS = 5`) flows through fusion, neck and head. The
text encoder's own SpikeLM T is independent (4) and is averaged away inside SpikeLM, so a
caption yields one static representation reused at every vision timestep -- correct, the
expression does not change over the event window. The head averages its four outputs over
T, so the prediction is per-frame, not per-timestep.

Every LIF holds membrane state across calls; `forward` resets all three stages first.
"""

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron as sj_neuron

from spiketrandvg.datasets.events_voxel_cube import T_STEPS
from spiketrandvg.models.event_encoder import TAPS, EventVisionEncoder
from spiketrandvg.models.fusion import CrossModalFusion
from spiketrandvg.models.text_encoder import MAX_TEXT_LEN, SpikeLMTextEncoder
from spiketrandvg.utils import forks

__all__ = ["SpikingPAN", "SingleBoxHead", "SpikeTransDVG", "build_spiketrandvg",
           "cxcywh_to_xyxy", "DEFAULT_TAPS"]


def cxcywh_to_xyxy(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(N, 4) normalised cxcywh -> (N, 4) xyxy in pixels."""
    cx, cy, w, h = box.unbind(-1)
    scale = box.new_tensor([width, height, width, height])
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1) * scale

# stride 4 / 8 / 16, finest first -- see the module docstring for why not 8/16/32
DEFAULT_TAPS: tuple[str, ...] = ("s4", "s8", "s16b")


def _up2(x: torch.Tensor) -> torch.Tensor:
    """Nearest 2x upsample on a (T, B, C, H, W) spiking tensor."""
    T, B, C, H, W = x.shape
    y = F.interpolate(x.flatten(0, 1), scale_factor=2.0, mode="nearest")
    return y.reshape(T, B, C, H * 2, W * 2)


class SpikingPAN(nn.Module):
    """Spiking PAN over N scales, built from SpikeYOLO's own blocks.

    Top-down then bottom-up, all at a single width, so the head sees `ch=(d,)*N`:

        P_n  --(1x1)-->  T_n  --up--> + T_{n-1} ... --> N_0  --down--> + ... --> N_n

    `MS_ConvBlock` is the refine unit at every node and `MS_StandardConv` does the 1x1
    reductions and the stride-2 downsamples, so every conv in the neck is preceded by an
    integer-LIF and is therefore spike-driven.

    Inputs may be spike-valued (fused scales) or analog membranes (unfused ones); each
    block opens with its own neuron, so the mix is handled uniformly.
    """

    def __init__(self, d_model: int = 256, num_scales: int = 3, mlp_ratio: int = 4):
        super().__init__()
        sy = forks.load_spikeyolo()
        self.n = num_scales
        self.d_model = d_model

        # top-down: fuse P_{i+1} (coarser, upsampled) into P_i
        self.td_reduce = nn.ModuleList(
            sy.MS_StandardConv(2 * d_model, d_model, k=1, s=1) for _ in range(num_scales - 1)
        )
        self.td_refine = nn.ModuleList(
            sy.MS_ConvBlock(d_model, mlp_ratio=mlp_ratio) for _ in range(num_scales - 1)
        )
        # bottom-up: fuse N_{i-1} (finer, strided down) into N_i
        self.bu_down = nn.ModuleList(
            sy.MS_StandardConv(d_model, d_model, k=3, s=2) for _ in range(num_scales - 1)
        )
        self.bu_reduce = nn.ModuleList(
            sy.MS_StandardConv(2 * d_model, d_model, k=1, s=1) for _ in range(num_scales - 1)
        )
        self.bu_refine = nn.ModuleList(
            sy.MS_ConvBlock(d_model, mlp_ratio=mlp_ratio) for _ in range(num_scales - 1)
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """[finest ... coarsest] (T,B,d,H,W) -> same shapes, language- and scale-mixed."""
        if len(feats) != self.n:
            raise ValueError(f"expected {self.n} scales, got {len(feats)}")

        # top-down
        td = [None] * self.n
        td[-1] = feats[-1]
        for i in range(self.n - 2, -1, -1):
            up = _up2(td[i + 1])
            if up.shape[-2:] != feats[i].shape[-2:]:  # odd sizes: trim to the finer map
                up = up[..., : feats[i].shape[-2], : feats[i].shape[-1]]
            x = self.td_reduce[i](torch.cat((feats[i], up), dim=2))
            td[i] = self.td_refine[i](x)

        # bottom-up
        out = [td[0]]
        for i in range(1, self.n):
            down = self.bu_down[i - 1](out[-1])
            if down.shape[-2:] != td[i].shape[-2:]:
                down = down[..., : td[i].shape[-2], : td[i].shape[-1]]
            x = self.bu_reduce[i - 1](torch.cat((td[i], down), dim=2))
            out.append(self.bu_refine[i - 1](x))
        return out


class SingleBoxHead(nn.Module):
    """Spiking MLP that regresses ONE box: (T,B,d,H,W) per scale -> (B, 4).

    Grounding returns exactly one region per expression, so this head asserts that
    structurally -- there is no anchor grid, no per-location score, no assignment and no
    argmax anywhere. The output is four numbers.

    Pipeline, per scale: a stride-2 spiking conv, an adaptive average pool to a common
    `grid`, and a BatchNorm; the three grids are concatenated over channels, mixed by a
    1x1 spiking conv, flattened, and pushed through a two-layer MLP. `sigmoid` puts the
    result in normalised cxcywh, which is also how `Talk2EventDataset` stores its boxes,
    so nothing has to convert units.

    Why pool to a grid rather than globally
    ---------------------------------------
    A global average pool destroys the spatial information a box regressor needs -- every
    location would contribute identically and the MLP could only ever emit the dataset's
    mean box. A 6x8 grid keeps a coarse spatial layout at a size the MLP can afford
    (256*48 = 12288 inputs), and the neck has already mixed fine detail into it.

    Why the normalisers here are GroupNorm and LayerNorm, never BatchNorm
    ---------------------------------------------------------------------
    A normaliser is needed at all because average-pooling spikes yields values well under
    1 and SpikeYOLO's integer LIF fires only above 0.5, so the pooled maps would arrive
    dead at the next neuron -- the same failure `CrossModalFusion`'s `tap_norm` prevents.

    But it must not be BatchNorm. This model trains at batch size 1 (a trainable backbone
    at 480x640 leaves no room for more), so a BatchNorm's "batch" is the T=5 timesteps of
    ONE sample, and it subtracts that sample's own per-channel mean. For a convolutional
    BatchNorm deep in a backbone that is survivable -- spatial structure carries the
    signal. For a regressor it is fatal: the per-sample feature level IS the answer, and
    normalising it away leaves every sample looking alike, while eval swaps in pooled
    running statistics and scrambles whatever mapping did get learned.

    Measured, fitting 8 samples on frozen features with `BatchNorm1d` before the MLP:
    training loss fell 3.25 -> 0.54 while eval mean IoU fell to 0.000, and the final
    predictions were uncorrelated with their targets (gt cx 0.255 -> pred 0.68, gt 0.824
    -> pred 0.30). GroupNorm and LayerNorm normalise within a sample instead, so they are
    identical in train and eval and independent of batch size.

    The MLP is spike-driven: an integer LIF precedes each Linear, so both matmuls consume
    spikes in {0,1,2,3,4}. Regression precision does not suffer -- each output is a sum
    over 12288 and 512 terms respectively, so the achievable resolution is far finer than
    any single neuron's.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_scales: int = 3,
        grid: tuple[int, int] = (6, 8),
        hidden: int = 512,
        norm_groups: int = 32,
        d_text: int = 256,
        film: bool = True,
    ):
        super().__init__()
        sy = forks.load_spikeyolo()
        self.grid = grid
        self.reduce = nn.ModuleList(
            sy.MS_StandardConv(d_model, d_model, k=3, s=2) for _ in range(num_scales)
        )
        self.pool_norm = nn.ModuleList(
            nn.GroupNorm(norm_groups, d_model) for _ in range(num_scales)
        )
        self.mix = sy.MS_StandardConv(d_model * num_scales, d_model, k=1, s=1)

        # FiLM: the caption scales and shifts the pooled feature map before the MLP reads
        # it. Measured need, not decoration -- with the head seeing vision only, feeding a
        # deliberately WRONG caption changed the trained model's mIoU by 0.001, i.e. the
        # language branch was contributing nothing and the model was predicting a salient
        # box from events alone. Cross-attention alone did not survive the head's pooling;
        # this gives the caption a direct, unavoidable path to the four output numbers.
        self.use_film = film
        if film:
            self.film = nn.Sequential(
                nn.Linear(d_text, d_model), nn.GELU(), nn.Linear(d_model, 2 * d_model)
            )
            # small, NOT zero: a zero final weight would make gamma/beta identity but also
            # send exactly zero gradient back into the text encoder -- the same trap that
            # cost the box head its backbone gradient (see fc2 below).
            nn.init.normal_(self.film[-1].weight, std=1e-2)
            nn.init.zeros_(self.film[-1].bias)

        self.lif1 = sy.mem_update()
        self.fc1 = nn.Linear(d_model * grid[0] * grid[1], hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.lif2 = sy.mem_update()
        self.fc2 = nn.Linear(hidden, 4)
        # Start every prediction near the centre of the frame, so the first steps do not
        # have to climb out of a saturated corner of the sigmoid. A SMALL weight, not a
        # zero one: with `fc2.weight = 0` the gradient reaching this layer's input is
        # `grad_out @ W = 0`, so the vision backbone, fusion and neck all receive exactly
        # zero gradient -- measured, and it silently costs the first optimiser step.
        # 1e-3 over 512 spiking inputs lands the pre-sigmoid within ~0.02 of the origin.
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self, feats: list[torch.Tensor], sentence: torch.Tensor | None = None
    ) -> torch.Tensor:
        """[finest ... coarsest] (T,B,d,H,W) + (B, d_text) -> (B, 4) normalised cxcywh."""
        pooled = []
        for f, red, bn in zip(feats, self.reduce, self.pool_norm):
            x = red(f)
            T, B, C, H, W = x.shape
            x = F.adaptive_avg_pool2d(x.flatten(0, 1), self.grid)
            pooled.append(bn(x).reshape(T, B, C, *self.grid))

        x = self.mix(torch.cat(pooled, dim=2))              # (T, B, d, Gh, Gw)
        T, B = x.shape[:2]
        if self.use_film:
            if sentence is None:
                raise ValueError("head was built with film=True but got no sentence vector")
            gamma, beta = self.film(sentence).chunk(2, dim=-1)          # (B, d) each
            x = x * (1.0 + gamma[None, :, :, None, None]) + beta[None, :, :, None, None]
        x = self.lif1(x).reshape(T, B, -1)                  # spikes
        x = self.norm1(self.fc1(x.flatten(0, 1))).reshape(T, B, -1)
        x = self.lif2(x)
        x = self.fc2(x.flatten(0, 1)).reshape(T, B, 4)
        return x.mean(0).sigmoid()                          # (B, 4)


class SpikeTransDVG(nn.Module):
    """The full model. See the module docstring for the wiring diagram.

    forward(cube, input_ids, attention_mask)
        cube            (T, B, 2, H, W)   event voxel cube, H and W divisible by 16
        input_ids       (B, L)            roberta-base ids, L <= MAX_TEXT_LEN
        attention_mask  (B, L)
        -> (B, 4) normalised cxcywh in (0, 1) -- ONE box, train and eval alike

    Use `predict` for xyxy pixels. There is no separate training output: the head emits
    the same four numbers in both modes.
    """

    def __init__(
        self,
        d_model: int = 256,
        taps: tuple[str, ...] = DEFAULT_TAPS,
        T: int = T_STEPS,
        # vision
        vision_ckpt: str | None = None,
        freeze_vision: bool = False,
        vision_trainable_from: str | None = None,
        # text
        text_donor: str | None = "roberta-base",
        text_layers: int = 12,
        freeze_text: bool = False,
        # fusion
        fusion_depth: int = 1,
        fusion_heads: int = 8,
        fusion_mlp_ratio: float = 2.0,
        fuse_scales: tuple[str, ...] | None = None,
        attn_bn_gain: float = 3.0,
        # neck
        neck_mlp_ratio: int = 4,
        # head
        head_grid: tuple[int, int] = (6, 8),
        head_hidden: int = 512,
        head_film: bool = True,
    ):
        """
        Args:
            head_grid: spatial grid the head pools every scale down to before the MLP.
            head_hidden: width of the MLP's single hidden layer.
            head_film: condition the head on the sentence vector (FiLM). Turning this
                off reproduces the caption-blind model measured at 0.001 mIoU of
                caption sensitivity; keep it on.
            taps: encoder taps, FINEST FIRST. Strides come from `TAPS`.
            vision_ckpt: Meta-SpikeFormer ImageNet checkpoint, or None for random init.
            text_donor: donor for the text encoder, or None to leave it random.
            fuse_scales: which taps get cross-attention; default all of them.
            attn_bn_gain: see `CrossModalFusion._scale_attention_bn`. Leave at 3.0
                unless you have measured a better value; 1.0 makes the model text-blind.
            freeze_vision: run the backbone under no_grad (large memory saving).
        """
        super().__init__()
        bad = set(taps) - set(TAPS)
        if bad:
            raise ValueError(f"unknown taps {sorted(bad)}; choose from {sorted(TAPS)}")
        strides = [TAPS[t][1] for t in taps]
        if strides != sorted(strides):
            raise ValueError(f"taps must be finest-first; got strides {strides}")

        self.taps = tuple(taps)
        self.strides = tuple(strides)
        self.T = T
        self.d_model = d_model

        self.vision = EventVisionEncoder(
            in_channels=2,
            taps=self.taps,
            ckpt_path=vision_ckpt,
            freeze=freeze_vision,
            trainable_from=vision_trainable_from,
        )
        self.text = SpikeLMTextEncoder(
            d_model=d_model, num_hidden_layers=text_layers, freeze=freeze_text
        )
        if text_donor is not None:
            from spiketrandvg.models.text_encoder import load_pretrained_weights

            self.text_report = load_pretrained_weights(self.text, donor=text_donor)

        self.fusion = CrossModalFusion(
            in_channels=self.vision.out_channels,
            d_model=d_model,
            num_heads=fusion_heads,
            depth=fusion_depth,
            mlp_ratio=fusion_mlp_ratio,
            fuse=fuse_scales,
            attn_bn_gain=attn_bn_gain,
            T=T,
        )
        self.neck = SpikingPAN(d_model=d_model, num_scales=len(taps), mlp_ratio=neck_mlp_ratio)

        self.head = SingleBoxHead(
            d_model=d_model, num_scales=len(taps), grid=head_grid, hidden=head_hidden,
            d_text=d_model, film=head_film,
        )

    # -- state -----------------------------------------------------------------
    def reset(self) -> None:
        """Zero every membrane in every stage. `forward` does this automatically.

        Reset is done per-neuron rather than through `sj_functional.reset_net(self)`,
        which would call this very method and recurse. SpikeYOLO's `mem_update` is
        stateless (its membrane lives inside one forward), so only spikingjelly neurons
        need touching; the vision encoder gets its own documented reset.
        """
        self.vision.reset()
        for m in self.modules():
            if isinstance(m, sj_neuron.MultiStepLIFNode):
                m.reset()

    @torch.no_grad()
    def calibrate_bn(self, batches, momentum: float | None = None) -> int:
        """Populate every BatchNorm's running statistics from real data.

        Required before the first eval of a freshly built model, and not optional: an
        untrained BatchNorm has running_mean 0 / running_var 1, which makes it the
        identity, which leaves the tapped membranes (std ~0.02 off a random backbone) and
        CMSF's q/k/v pre-activations far below their neurons' thresholds. The eval path
        then returns an all-zero pyramid and a caption-independent prediction -- silently,
        with no error anywhere.

        This also settles SpikeYOLO's `BNAndPadLayer`, whose padding value is computed
        from `running_mean`/`running_var` while the same BatchNorm is updating them, so
        its train-mode output legitimately drifts between identical calls until the
        statistics converge.

        Args:
            batches: iterable of `(cube, input_ids, attention_mask)` already on-device.
                Around 40 batches gets running stats to ~99% of their converged value at
                the default momentum of 0.1.
            momentum: temporarily override BatchNorm momentum; None keeps each layer's own.
        Returns:
            number of batches consumed.
        """
        was_training = self.training
        self.train()
        saved = {}
        if momentum is not None:
            for name, m in self.named_modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    saved[name] = m.momentum
                    m.momentum = momentum
        n = 0
        try:
            for cube, input_ids, attention_mask in batches:
                self(cube, input_ids, attention_mask)
                n += 1
        finally:
            if momentum is not None:
                for name, m in self.named_modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm) and name in saved:
                        m.momentum = saved[name]
            self.train(was_training)
        return n

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        cube: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        if cube.dim() != 5:
            raise ValueError(f"expected (T, B, 2, H, W) cube, got {tuple(cube.shape)}")
        if cube.shape[0] != self.T:
            raise ValueError(f"cube has T={cube.shape[0]}, model built for T={self.T}")
        if input_ids.shape[1] > MAX_TEXT_LEN:
            raise ValueError(
                f"caption length {input_ids.shape[1]} exceeds MAX_TEXT_LEN {MAX_TEXT_LEN}"
            )
        if cube.shape[1] != input_ids.shape[0]:
            raise ValueError(
                f"batch mismatch: {cube.shape[1]} event frames vs {input_ids.shape[0]} captions"
            )

        self.reset()
        maps = self.vision(cube)                                  # {tap: (T,B,C,H,W)}
        tokens, sentence = self.text(input_ids, attention_mask)   # (B,L,d), (B,d)
        fused = self.fusion(maps, tokens, attention_mask)
        feats = self.neck([fused[t] for t in self.taps])
        return self.head(feats, sentence)            # (B, 4) normalised cxcywh

    @torch.no_grad()
    def predict(
        self,
        cube: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The one box, as (B, 4) xyxy in pixels of the input frame."""
        was_training = self.training
        self.eval()
        try:
            box = self(cube, input_ids, attention_mask)
        finally:
            self.train(was_training)
        H, W = cube.shape[-2:]
        return cxcywh_to_xyxy(box, H, W)


def build_spiketrandvg(**kwargs) -> SpikeTransDVG:
    return SpikeTransDVG(**kwargs)
