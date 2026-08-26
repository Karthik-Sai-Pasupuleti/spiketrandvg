"""Loader for the frozen reference repositories under ``repositories/``.

Those repos are never edited. They all define generically-named top-level modules
(``models.py``, ``datasets/``, ``utils/``), so each is loaded via importlib under a
UNIQUE alias rather than by putting its directory on ``sys.path`` -- otherwise the
names collide with each other and with this package.

Some fork files import modules at top level that the code paths we use never touch
(``torchinfo``, ``einops``, SpikeYOLO's ``ultralytics`` package). Those are satisfied
with stub modules; a stub is only installed when the real package is unimportable.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch
from spikingjelly.clock_driven import neuron as sj_neuron

# This file sits at <ws>/spiketrandvg/src/spiketrandvg/forks.py, so the
# workspace root (the parent of this repo, holding repositories/) is 5 levels up.
WS = Path(os.environ.get("T2E_WS", Path(__file__).resolve().parents[3]))
REPOS = WS / "repositories"

FORKS = {
    "sdt2": REPOS / "Spike-Driven-Transformer-V2",
    "spikeyolo": REPOS / "SpikeYOLO",
    "talk2event": REPOS / "Talk2Event",
    "sfod": REPOS / "SFOD",
    "e3dsnn": REPOS / "E-3DSNN",
    "cmsf": REPOS / "CMSF",
    "spiliformer": REPOS / "SpiLiFormer",
}


def _stub(name: str, **attrs) -> None:
    """Register a placeholder for `name`, unless it is genuinely importable."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def _load_file(alias: str, path: Path) -> types.ModuleType:
    """Execute a single .py file as a module under `alias`."""
    if alias in sys.modules:
        return sys.modules[alias]
    if not path.is_file():
        raise FileNotFoundError(f"fork file missing: {path}")
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod  # registered before exec so self-imports resolve
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def load_metaspikformer() -> types.ModuleType:
    """Meta-SpikeFormer (Spike-Driven Transformer V2, ICLR 2024) model definitions.

    Loaded as `sdt2_models`. Facts this relies on, verified against the source:
      * `metaspikformer_8_512(**kwargs)` builds the 55M variant with
        embed_dim=[128, 256, 512, 640]; `in_channels` and `num_classes` are
        forwarded to the constructor.
      * `forward_features(x)` takes (T, B, C, H, W) and returns
        (T, B, 640, H/16, W/16); H and W need only be divisible by 16.
      * `forward()` would replicate ONE frame across T steps, so we never call it.
      * `torchinfo` (models.py:3) and `einops.layers.torch.Rearrange` (models.py:13)
        are imported and never used.
    """
    _stub("torchinfo")
    _stub("einops")
    _stub("einops.layers")
    _stub("einops.layers.torch", Rearrange=object)
    return _load_file("sdt2_models", FORKS["sdt2"] / "classification" / "models.py")


@lru_cache(maxsize=1)
def load_ilif() -> tuple[type, type]:
    """SpikeYOLO's integer-LIF neuron: returns (mem_update, MultiSpike4).

    `mem_update.forward` takes (T, B, ...) and emits integer spikes in {0..4}; it is
    parameter-free and holds no state across calls. Its file imports
    `ultralytics.utils.tal`, and importing that for real would execute the fork's
    `ultralytics/__init__.py` (which builds YOLO/SAM machinery and writes
    ~/.config/Ultralytics/settings.yaml), so those names are stubbed.
    """
    _stub("torchinfo")
    _stub("einops")
    _stub("einops.layers")
    _stub("einops.layers.torch", Rearrange=object)
    _stub("ultralytics")
    _stub("ultralytics.utils")
    _stub("ultralytics.utils.tal", TORCH_1_10=True, dist2bbox=object, make_anchors=object)
    mod = _load_file(
        "spikeyolo_modules",
        FORKS["spikeyolo"] / "ultralytics" / "nn" / "modules" / "yolo_spikformer.py",
    )
    return mod.mem_update, mod.MultiSpike4


def fork_status() -> dict[str, str]:
    """Report each fork's presence -- cheap handshake for sanity checks."""
    return {k: ("ok" if p.is_dir() else "MISSING") for k, p in FORKS.items()}


@lru_cache(maxsize=1)
def load_e3dsnn() -> types.SimpleNamespace:
    """E-3DSNN's OpenPCDet backbones (ICLR-submission code, arXiv 2412.07360).

    Returns a namespace with:
      * `VoxelBackBone8x`          -- the classic (conv, norm, act) sparse 3D backbone
      * `BaseBEVBackbone`          -- the classic 2D trunk over the collapsed BEV map
      * `VoxelBackBone8x_3dv_snn`  -- E-3DSNN's spike-first variant (act, conv, norm)
      * `BaseBEVBackbone_spike`    -- its spiking 2D trunk
      * `Multispike`               -- the multi-bit spike neuron, floor(clamp(x, 0, 4) + 0.5)
      * `AnchorGenerator`, `ResidualCoder` -- what `AnchorHeadSingle` needs to turn the
        checkpoint's 1x1 head convs into boxes, without pcdet's compiled NMS ops

    Why the classic classes are here too
    ------------------------------------
    The only released detection checkpoint (`kitti.pth` on HuggingFace
    `Xuerui123/E-3DSNN`) has the CLASSIC parameter layout, not the spiking one:
    its blocks put the conv at index 0 and the norm at index 1
    (`backbone_3d.conv1.0.0.weight` + `conv1.0.1.*`), and its 2D trunk puts the
    conv at index 1 after the ZeroPad2d (`backbone_2d.blocks.0.1.weight`).
    E-3DSNN's own blocks are spike-FIRST in every commit of this fork
    (`Multispike(), conv, norm` -> conv at index 1, norm at 2), which shifts every
    index by one and additionally expects three convs in `conv_input` where the
    checkpoint has one. Loaded into the classic classes the checkpoint matches
    exactly -- 72/72 tensors for `backbone_3d`, 84/84 for `backbone_2d`, nothing
    missing, nothing unexpected -- so that is how the weights are consumed here,
    with the ReLUs swapped for `Multispike` when a spiking forward is wanted.

    How the fork is loaded
    ----------------------
    The backbone files use package-relative imports (`from ...utils.spconv_utils
    import replace_feature, spconv`), so a single-file `_load_file` cannot resolve
    them. A synthetic package tree (`e3dsnn_fork.models.backbones_3d`, plus a
    stub `e3dsnn_fork.utils.spconv_utils` supplying `spconv` and
    `replace_feature`) is registered in `sys.modules` first, and the fork files
    are then executed inside it. Nothing in `repositories/E-3DSNN` is edited, and
    pcdet's own `__init__` chain -- which reaches for compiled CUDA ops
    (`iou3d_nms_cuda`, `roiaware_pool3d`) that are not built here -- is never run.

    Requires `spconv` (spconv-cu126 2.3.8 is verified working on sm_120).
    """
    import spconv.pytorch as spconv_pt  # noqa: PLC0415  (optional heavy dependency)

    pkg = "e3dsnn_fork"
    for name in (pkg, f"{pkg}.models", f"{pkg}.models.backbones_3d",
                 f"{pkg}.models.backbones_2d", f"{pkg}.utils"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # marks it a package so submodules can be registered
            sys.modules[name] = mod
    su_name = f"{pkg}.utils.spconv_utils"
    if su_name not in sys.modules:
        su = types.ModuleType(su_name)
        su.spconv = spconv_pt
        su.replace_feature = lambda out, new_features: out.replace_feature(new_features)
        sys.modules[su_name] = su
        sys.modules[f"{pkg}.utils"].spconv_utils = su

    det = FORKS["e3dsnn"] / "det" / "pcdet" / "models"
    classic3d = _load_file(f"{pkg}.models.backbones_3d.spconv_backbone",
                           det / "backbones_3d" / "spconv_backbone.py")
    spike3d = _load_file(f"{pkg}.models.backbones_3d.spconv_backbone_spike",
                         det / "backbones_3d" / "spconv_backbone_spike.py")
    bev = _load_file(f"{pkg}.models.backbones_2d.base_bev_backbone",
                     det / "backbones_2d" / "base_bev_backbone.py")
    # torch-only files, so these load standalone (no relative imports to satisfy)
    anchors = _load_file("e3dsnn_anchor_generator",
                         det / "dense_heads" / "target_assigner" / "anchor_generator.py")
    coder = _load_file("e3dsnn_box_coder", FORKS["e3dsnn"] / "det" / "pcdet" / "utils"
                       / "box_coder_utils.py")
    return types.SimpleNamespace(
        AnchorGenerator=anchors.AnchorGenerator,
        ResidualCoder=coder.ResidualCoder,
        VoxelBackBone8x=classic3d.VoxelBackBone8x,
        VoxelBackBone8x_3dv_snn=spike3d.VoxelBackBone8x_3dv_snn,
        BaseBEVBackbone=bev.BaseBEVBackbone,
        BaseBEVBackbone_spike=bev.BaseBEVBackbone_spike,
        Multispike=spike3d.Multispike,
    )


@contextlib.contextmanager
def allow_cupy_construction():
    """Let a spikingjelly neuron be CONSTRUCTED with backend='cupy' on a box without it.

    `MultiStepLIFNode.__init__` asserts cupy is importable when the cupy backend is
    requested, and both CMSF and Meta-SpikeFormer hard-code that backend at every call
    site. cupy is only touched at FORWARD time, so a sentinel gets construction through;
    callers must then flip every neuron to the 'torch' backend (`use_torch_backend`).
    """
    if getattr(sj_neuron, "cupy", None) is not None:
        yield
        return
    sj_neuron.cupy = types.SimpleNamespace()
    try:
        yield
    finally:
        sj_neuron.cupy = None


def use_torch_backend(module) -> int:
    """Switch every spikingjelly multi-step neuron in `module` to the torch backend.

    `backend` is read at forward time, so this is safe to call any time after
    construction. Returns how many neurons were switched.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, sj_neuron.MultiStepLIFNode):
            m.backend = "torch"
            n += 1
    return n


@lru_cache(maxsize=1)
def load_spikeyolo() -> types.SimpleNamespace:
    """SpikeYOLO's spiking modules, including the detection head (arXiv 2407.20708).

    Returns a namespace with:
      * `mem_update`, `MultiSpike4`  -- the integer-LIF neuron, spikes in {0,1,2,3,4}
      * `SpikeConv`, `SpikeConvWithoutBN`, `MS_StandardConv`, `SpikeSPPF`,
        `MS_ConvBlock`, `MS_AllConvBlock`, `MS_DownSampling`  -- spiking (T,B,C,H,W) ops
      * `SpikeDetect`, `SpikeDFL`     -- the anchor-free decoupled head with DFL
      * `make_anchors`, `dist2bbox`, `bbox2dist` -- the real ultralytics helpers
      * `TaskAlignedAssigner`, `bbox_iou` -- the real YOLOv8 label assigner

    Why `make_anchors`/`dist2bbox` need special handling
    ----------------------------------------------------
    `yolo_spikformer.py` does `from ultralytics.utils.tal import make_anchors, dist2bbox`
    at module scope, and importing that package for real executes the fork's
    `ultralytics/__init__.py` -- which builds the YOLO/SAM machinery and writes
    `~/.config/Ultralytics/settings.yaml`. So the import is satisfied with stubs, and the
    two genuine functions are then loaded out of `tal.py` in isolation and rebound in the
    fork module's globals. `tal.py` itself only needs `check_version` (from `.checks`,
    the whole reason the package import is unaffordable) and `bbox_iou` (from `.metrics`,
    which pulls matplotlib and `ultralytics.utils.LOGGER`). `bbox_iou` is used solely by
    `TaskAlignedAssigner`, which we do not use, so it is stubbed; `make_anchors` and
    `dist2bbox` are pure torch and self-contained.

    Nothing on disk is edited -- the rebinding happens on the in-memory module object.

    Gradient note: `MultiSpike4`'s surrogate passes gradient only where the membrane is
    in (0, 4). Anything driven far outside that window trains at exactly zero gradient.
    """
    _stub("torchinfo")
    _stub("einops")
    _stub("einops.layers")
    _stub("einops.layers.torch", Rearrange=object)
    _stub("ultralytics")
    _stub("ultralytics.utils")
    _stub("ultralytics.utils.tal", TORCH_1_10=True, dist2bbox=object, make_anchors=object)

    mod = _load_file(
        "spikeyolo_modules",
        FORKS["spikeyolo"] / "ultralytics" / "nn" / "modules" / "yolo_spikformer.py",
    )

    # real make_anchors / dist2bbox, loaded in a synthetic package so tal.py's two
    # relative imports resolve without dragging in the ultralytics package proper
    pkg = "spikeyolo_tal"
    if f"{pkg}.tal" not in sys.modules:
        base = types.ModuleType(pkg)
        base.__path__ = []
        sys.modules[pkg] = base
        checks = types.ModuleType(f"{pkg}.checks")
        checks.check_version = lambda *a, **k: True
        sys.modules[f"{pkg}.checks"] = checks
        metrics = types.ModuleType(f"{pkg}.metrics")
        # `TaskAlignedAssigner` needs the real `bbox_iou`, but importing metrics.py drags
        # in matplotlib and `ultralytics.utils.LOGGER`. The function body uses nothing but
        # `torch` and `math` (verified against the source), so it is extracted by text and
        # executed in an isolated namespace -- no fork file is edited, no package init runs.
        src = (FORKS["spikeyolo"] / "ultralytics" / "utils" / "metrics.py").read_text()
        start = src.index("def bbox_iou(")
        end = src.index("\ndef ", start + 1)
        ns: dict = {"torch": torch, "math": math}
        exec(compile(src[start:end], "<spikeyolo bbox_iou>", "exec"), ns)
        metrics.bbox_iou = ns["bbox_iou"]
        sys.modules[f"{pkg}.metrics"] = metrics
        _load_file(
            f"{pkg}.tal",
            FORKS["spikeyolo"] / "ultralytics" / "utils" / "tal.py",
        )
    tal = sys.modules[f"{pkg}.tal"]

    mod.make_anchors = tal.make_anchors      # rebind the stubs SpikeDetect closes over
    mod.dist2bbox = tal.dist2bbox

    return types.SimpleNamespace(
        mem_update=mod.mem_update,
        MultiSpike4=mod.MultiSpike4,
        SpikeConv=mod.SpikeConv,
        SpikeConvWithoutBN=mod.SpikeConvWithoutBN,
        MS_StandardConv=mod.MS_StandardConv,
        MS_DownSampling=mod.MS_DownSampling,
        MS_ConvBlock=mod.MS_ConvBlock,
        MS_AllConvBlock=mod.MS_AllConvBlock,
        SpikeSPPF=mod.SpikeSPPF,
        SpikeDetect=mod.SpikeDetect,
        SpikeDFL=mod.SpikeDFL,
        make_anchors=tal.make_anchors,
        dist2bbox=tal.dist2bbox,
        bbox2dist=tal.bbox2dist,
        TaskAlignedAssigner=tal.TaskAlignedAssigner,
        bbox_iou=sys.modules["spikeyolo_tal.metrics"].bbox_iou,
    )


@lru_cache(maxsize=1)
def load_cmsf() -> types.SimpleNamespace:
    """CMSF's spiking cross-modal attention (Multimodal SNN for Image-Text Retrieval).

    Returns a namespace with:
      * `SpikingCrossAttention` -- linear spiking cross-attention, (T,B,L,D) x (T,B,L',D)
      * `SCA_Block`             -- that attention plus CMSF's `Spiking_GFNN` gated MLP
      * `SpikingQKAttention`, `QK_Block` -- the cheaper QK variant CMSF ships as default
      * `Spiking_GFNN`, `RepeatTextEncoder`, `Dynamic_Threshold_LIFNode`

    Cost note, and why this scales to dense feature maps: `SpikingCrossAttention`
    computes `k^T @ v` first (D x D) and then `q @ (k^T v)`, so it is O(L*D^2), LINEAR in
    the number of queries rather than quadratic. A 60x80 feature map flattened to 4800
    query tokens is therefore affordable; a softmax attention over the same map would not
    be.

    Loading: `CrossEncoder.py` does `from lib.CPG import ...` and
    `from lib.positional_embedding import ...`, so CMSF's `lib/` is registered under the
    generic name `lib` for the duration of the import and removed afterwards, keeping
    that name out of the process.

    CUDA note: every neuron is constructed with `backend='cupy'`. Construction is allowed
    through with `allow_cupy_construction()` and the neurons are switched to the torch
    backend, so this runs on CPU and on GPUs without a matching cupy build.

    Quirk worth knowing: `SCA_Block` builds `norm1`/`norm2` LayerNorms and never applies
    them -- its forward is a bare `x + attn(x, y, y)` then `x + mlp(x)`. Left as-is;
    the parameters are inert.
    """
    root = FORKS["cmsf"]
    if not root.is_dir():
        raise FileNotFoundError(f"CMSF fork missing: {root}")

    had_lib = "lib" in sys.modules
    if not had_lib:
        pkg = types.ModuleType("lib")
        pkg.__path__ = [str(root / "lib")]
        sys.modules["lib"] = pkg
    try:
        with allow_cupy_construction():
            mod = _load_file("cmsf_cross_encoder", root / "lib" / "CrossEncoder.py")
    finally:
        if not had_lib:
            sys.modules.pop("lib", None)
            for name in [k for k in sys.modules if k.startswith("lib.")]:
                sys.modules.pop(name, None)

    return types.SimpleNamespace(
        SpikingCrossAttention=mod.SpikingCrossAttention,
        SCA_Block=mod.SCA_Block,
        SpikingQKAttention=mod.SpikingQKAttention,
        QK_Block=mod.QK_Block,
        Spiking_GFNN=mod.Spiking_GFNN,
        RepeatTextEncoder=mod.RepeatTextEncoder,
        Dynamic_Threshold_LIFNode=mod.Dynamic_Threshold_LIFNode,
    )


@lru_cache(maxsize=1)
def load_spiliformer():
    """SpiLiFormer's spiking transformer (ICCV 2025), the RGB branch of the hybrid model.

    Returns the `Spike_Lateral_Transformer` class and the `SpiLiFormer_10_768` factory.

    Architecture, verified against the source: a convolutional stem (`PatchEmbedInit`, two
    stride-2 maxpools -> /4) then three stages at 192 / 384 / 768 channels, each preceded
    by a patch-embedding downsample, so `forward_features` returns a (T, B, 768, H/16,
    W/16) SPATIAL map -- not a pooled classification vector. At 480x640 that is a 30x40
    grid, the same resolution as the event backbone's s16 tap.

    Two things the fork does that matter to a caller:

    * `forward()` replicates one frame across T (`x.unsqueeze(0).repeat(T,...)`), which is
      right for a static RGB image but means T is the model's own, independent of the event
      stream's.
    * The lateral-inhibition feedback path is a TWO-pass design: the first pass returns
      `(x, tmp)` plus a feedback tensor, and a second pass consumes it. `forward_features`
      with `second_forward=None` gives the feedforward features on their own, which is what
      a feature extractor wants.

    Loading: `spiliformer.py` does `from util.factory import Decoder`, so the ImageNet
    directory is put on `sys.path` for the duration of the import and removed afterwards,
    keeping the generic name `util` out of the process (the same trick used elsewhere here).
    """
    # `einops.layers.torch.Rearrange` is imported at module scope and never used
    # (verified by grep over spiliformer.py and util/factory.py), so a stub suffices.
    _stub("torchinfo")
    _stub("einops")
    _stub("einops.layers")
    _stub("einops.layers.torch", Rearrange=object)
    d = FORKS["spiliformer"] / "imagetnet_1k"
    if not d.is_dir():
        raise FileNotFoundError(f"SpiLiFormer fork missing: {d}")
    added = str(d)
    had = added in sys.path
    if not had:
        sys.path.insert(0, added)
    try:
        mod = _load_file("spiliformer_model", d / "spiliformer.py")
    finally:
        if not had and added in sys.path:
            sys.path.remove(added)
        for name in [k for k in list(sys.modules) if k == "util" or k.startswith("util.")]:
            sys.modules.pop(name, None)
    _fix_spiliformer_square_assumption(mod)
    return types.SimpleNamespace(
        Spike_Lateral_Transformer=mod.Spike_Lateral_Transformer,
        SpiLiFormer_10_768=mod.SpiLiFormer_10_768,
    )


def _fix_spiliformer_square_assumption(mod) -> None:
    """Correct two transposed reshapes that only work on a SQUARE feature grid.

    `MLP.forward` (spiliformer.py:43) and `FB_LiDiff_Attention.forward` (:157) both end in
    `.reshape(T, B, C, W, H)` -- width and height the wrong way round. On ImageNet's 14x14
    grid H == W so the error is invisible and the released weights are unaffected. On a
    480x640 frame the grid is 30x40 and the block returns a (..., 40, 30) tensor, which
    then fails to add to its own (..., 30, 40) residual. `FF_LiDiff_Attention` (:94) has it
    right, so this is a slip in two of three places rather than a convention.

    Patched by wrapping `forward` on the in-memory classes: if the input grid was
    non-square and the output comes back transposed, transpose it back. On square inputs
    the wrapper is a no-op, so ImageNet behaviour is bit-identical.
    """
    for cls in (mod.MLP, mod.FB_LiDiff_Attention):
        if getattr(cls, "_hw_patched", False):
            continue
        original = cls.forward

        def forward(self, x, *args, _orig=original, **kwargs):
            h, w = x.shape[-2:]
            out = _orig(self, x, *args, **kwargs)
            if h != w and out.shape[-2:] == (w, h):
                out = out.transpose(-1, -2).contiguous()
            return out

        cls.forward = forward
        cls._hw_patched = True
