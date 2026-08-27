"""Shared infrastructure: frozen-fork loading, and checkpoint diagnostics.

**Fork loading.** Every model component comes from a reference repository under
`../repositories/` that is never edited. `load_*` executes individual files under unique
aliases and stubs the imports the used code paths never touch, so importing one model
definition does not drag in a whole framework's `__init__`.

**Diagnostics.** `firing_rates`, `positional_ratio`, `attention_perplexity` and
`mode_gap` inspect a trained checkpoint for the failure modes this project has actually
hit -- a dead spiking layer silently passing its residual through, a positional embedding
thresholded away before it can inform attention, an attention map that is uniform and
therefore carries no location, and a train/eval discrepancy that invalidates every other
number. Run them with:

    uv run python -m spiketrandvg.utils --run runs/<name>

The model is imported lazily inside those functions: `model.py` imports THIS module for
the fork loaders, so a module-level import here would be circular.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from spikingjelly.clock_driven import neuron as sj_neuron
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import os
import sys
import torch
import types



# ====================================================================================
# from forks.py
# ====================================================================================

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


# ====================================================================================
# from diagnose.py
# ====================================================================================

def build_from_args(run_args: dict, device: str):
    """Reconstruct the exact architecture a run was trained with.

    `rgb_ckpt=None` deliberately: the checkpoint's own state_dict overwrites every
    backbone weight immediately after construction, so loading SpiLiFormer's ImageNet
    checkpoint first would be wasted work.
    """
    from spiketrandvg.model import RefCOCOGrounding      # lazy: avoids a cycle
    size = tuple(run_args["size"])
    return RefCOCOGrounding(
        rgb_ckpt=None, text_model=run_args.get("text_model", "roberta-base"),
        img_size=size, T=run_args.get("T", 4), rgb_T=run_args.get("rgb_T", 1),
        depth=run_args.get("depth", 2), attn_type=run_args.get("attn_type", "spatial_softmax"),
        freeze_rgb=False, freeze_text=not run_args.get("train_text", False),
        head_type=run_args.get("head_type", "pooled_mlp"),
        attn_map=run_args.get("attn_map", "last"),
        pos_std=run_args.get("pos_std", 0.02),
        text_unfreeze_last=(0 if run_args.get("train_text") else
                            run_args.get("text_unfreeze_last", 0)),
    ).to(device)


@torch.no_grad()
def firing_rates(model, rgb, ids, mask) -> dict[str, float]:
    """Mean fraction of nonzero activations per spiking layer, one forward pass.

    Hooked by class name (`"LIF" in type(m).__name__` or `"mem_update"`) rather than by
    isinstance against imported classes, so this covers CMSF's `Dynamic_Threshold_LIFNode`,
    spikingjelly's `MultiStepLIFNode`, and SpikeYOLO's `mem_update` without needing to know
    their exact module paths.
    """
    rates: dict[str, float] = {}
    handles = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            rates[name] = (t != 0).float().mean().item()
        return hook

    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "LIF" in cls or cls == "mem_update":
            handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        model(rgb, ids, mask)
    finally:
        for h in handles:
            h.remove()
    return rates


@torch.no_grad()
def positional_ratio(model: RefCOCOGrounding) -> float | None:
    """RMS(vision.pos) / RMS(pre-position lateral output from the last forward call)."""
    lat = model.vision.last_lateral_rms
    if lat is None or lat == 0:
        return None
    pos_rms = model.vision.pos.detach().float().pow(2).mean().sqrt().item()
    return pos_rms / lat


@torch.no_grad()
def attention_perplexity(model, rgb, ids, mask) -> dict[str, float]:
    """exp(entropy) of each real token's attention distribution, per block.

    Uses `return_diagnostics=True` rather than re-deriving attention outside the model,
    so this always reflects exactly what the model itself computed -- no duplicated
    forward logic to drift out of sync.
    """
    _, diag = model(rgb, ids, mask, return_diagnostics=True)
    out = {}
    m = mask.to(torch.float32).unsqueeze(-1)                    # (B,L,1)
    for i, attn in enumerate(diag["attn_maps"]):
        a = attn.float().mean(dim=(0, 2))                       # (B,L,N)
        a = a.clamp_min(1e-12)
        entropy = -(a * a.log()).sum(-1)                        # (B,L)
        perplexity = entropy.exp()
        real = m.squeeze(-1).bool()
        out[f"block{i}"] = perplexity[real].mean().item()
    return out


@torch.no_grad()
def mode_gap(model, loader, device, amp_ctx) -> dict[str, float]:
    """mIoU on the same samples, train() vs eval(). See the module docstring."""
    def run(training: bool) -> float:
        model.train(training)
        ious = []
        for rgb, ids, mask, gt, _c, _m in loader:
            rgb, ids, mask, gt = (t.to(device) for t in (rgb, ids, mask, gt))
            with amp_ctx():
                pred = model(rgb, ids, mask)
            ious.append(box_iou(cxcywh_to_xyxy_norm(pred.float()),
                                cxcywh_to_xyxy_norm(gt)).diagonal().cpu())
        return torch.cat(ious).mean().item()

    train_mode = run(True)
    model.eval()
    eval_mode = run(False)
    return {"train_mode_mIoU": train_mode, "eval_mode_mIoU": eval_mode,
            "gap": eval_mode - train_mode}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/of100_pooled")
    ap.add_argument("--ckpt", default="best.pth", help="filename inside --run")
    ap.add_argument("--n-batches", type=int, default=4,
                    help="batches to average firing rates / perplexity over")
    ap.add_argument("--mode-gap-samples", type=int, default=100,
                    help="samples for the train/eval mode-gap check")
    args = ap.parse_args()

    from spiketrandvg.dataloader import RefCOCO, make_collate   # lazy: avoids a cycle
    from spiketrandvg.textencoder import build_tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run)
    run_args = json.loads((run_dir / "args.json").read_text())
    blob = torch.load(run_dir / args.ckpt, map_location=device, weights_only=False)

    model = build_from_args(run_args, device)
    # strict=False: a checkpoint from before head_norm0 (item 2) existed is missing that
    # key. Report it rather than silently leaving head_norm0 at its random init, which
    # would make this run's mode-gap and firing-rate numbers for head_lif1 incomparable
    # to a fresh run without saying so.
    missing = model.load_state_dict(blob["model"], strict=False)
    print(f"loaded {run_dir / args.ckpt} (epoch {blob.get('epoch')}, "
          f"head_type={run_args.get('head_type', 'pooled_mlp')})")
    if missing.missing_keys or missing.unexpected_keys:
        print(f"  NOTE: missing {missing.missing_keys}, unexpected {missing.unexpected_keys}"
              f" -- checkpoint predates a code change; those params are at random init")
    print()

    tokenizer = build_tokenizer()
    size = tuple(run_args["size"])
    ds = RefCOCO(run_args["dataset"], "train", size=size, augment=None,
                limit=run_args.get("limit_train"))
    dl = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True,
                    collate_fn=make_collate(tokenizer))
    amp_ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
               if device == "cuda" else __import__("contextlib").nullcontext)

    # --- 1 & 3: firing rates + attention perplexity, averaged over n_batches ----------
    model.eval()
    rate_sums: dict[str, float] = defaultdict(float)
    perp_sums: dict[str, float] = defaultdict(float)
    n = 0
    it = iter(dl)
    for _ in range(args.n_batches):
        try:
            rgb, ids, mask, gt, _c, _m = next(it)
        except StopIteration:
            it = iter(dl)
            rgb, ids, mask, gt, _c, _m = next(it)
        rgb, ids, mask = (t.to(device) for t in (rgb, ids, mask))
        with amp_ctx():
            rates = firing_rates(model, rgb, ids, mask)
            perps = attention_perplexity(model, rgb, ids, mask)
        for k, v in rates.items():
            rate_sums[k] += v
        for k, v in perps.items():
            perp_sums[k] += v
        n += 1

    print(f"=== firing rate per spiking layer (n={n} batches) ===")
    print("    (< 1% = likely dead, silently identity-passing its residual branch)")
    print("    (> 90% = likely saturated)")
    for k in sorted(rate_sums):
        v = rate_sums[k] / n
        flag = "  <-- DEAD" if v < 0.01 else "  <-- SATURATED" if v > 0.90 else ""
        print(f"  {k:45s} {v*100:6.2f}%{flag}")

    print(f"\n=== positional RMS ratio (RMS(pos) / RMS(pre-pos lateral output)) ===")
    ratio = positional_ratio(model)
    if ratio is None:
        print("  unavailable (no forward call captured a lateral RMS)")
    else:
        flag = "  <-- position may be lost before the LIF threshold" if ratio < 0.05 else ""
        print(f"  {ratio:.4f}{flag}")

    print(f"\n=== attention perplexity per block (n={n} batches, ~{model.vision.pos.shape[-2]}"
          f"x{model.vision.pos.shape[-1]} = "
          f"{model.vision.pos.shape[-2]*model.vision.pos.shape[-1]} keys) ===")
    print("    (near the key count = map is ~uniform, carries no location)")
    print("    (near 1 = map has collapsed onto a single position)")
    for k in sorted(perp_sums):
        print(f"  {k:15s} {perp_sums[k]/n:8.2f}")

    # --- 4: train mode vs eval mode, on the exact training samples --------------------
    print(f"\n=== train/eval mode gap (mIoU on {min(args.mode_gap_samples, len(ds))} "
          f"training samples) ===")
    from torch.utils.data import Subset
    n_gap = min(args.mode_gap_samples, len(ds))
    gap_idxs = list(range(0, len(ds), max(1, len(ds) // n_gap)))[:n_gap]
    gap_dl = DataLoader(Subset(ds, gap_idxs), batch_size=8, shuffle=False,
                        collate_fn=make_collate(tokenizer))
    g = mode_gap(model, gap_dl, device, amp_ctx)
    flag = "  <-- SUSPECT: fix the eval path before trusting other numbers" \
        if abs(g["gap"]) > 0.10 else ""
    print(f"  train() mIoU {g['train_mode_mIoU']:.4f}  eval() mIoU {g['eval_mode_mIoU']:.4f}"
          f"  gap {g['gap']:+.4f}{flag}")

    if run_args.get("head_type") == "attn_softargmax":
        scale = model.attn_logit_scale.exp().item()
        print(f"\nattn_logit_scale.exp() = {scale:.4f}  "
              f"(collapsing toward 0 flattens the map toward uniform, which pins "
              f"every centre to (0.5, 0.5) -- the same output a fresh sigmoid head "
              f"gives at init, so this failure looks like 'no progress', not a bug)")


if __name__ == "__main__":
    main()


@lru_cache(maxsize=1)
def load_spiliformer_dvs() -> types.ModuleType:
    """SpiLiFormer's CIFAR10-DVS variant -- the event-native member of the family.

    Why this one is interesting for events: `in_channels=2` is its DESIGN, not an
    adaptation. The ImageNet variants take 3 RGB channels and have to have their stem
    averaged onto event polarity; this one was built for DVS data from the start, and the
    authors trained it on CIFAR10-DVS and N-Caltech101.

    Two things to know before using it, both verified here:

    * **No pretrained weights are published.** The repo's README links ImageNet
      checkpoints only (T1/T4 224, T4 288, T4 384) -- nothing for CIFAR10-DVS or
      N-Caltech101. This backbone therefore starts from random init.
    * **It is small.** 1.7M parameters against Meta-SpikeFormer's ~55M, because it was
      sized for 128x128 ten-class classification.

    Two source defects are patched on the in-memory module, never in the fork:

    * `img_size_h/w` default to 128 and the `SpiLiFormer()` factory does not forward them,
      so the model builds with a 128x128 assumption and raises a broadcast error on any
      other input. Pass the real size to `Spike_Lateral_Transformer` instead.
    * `FB_LiDiff_Attention.forward` (model.py:173) ends in `.reshape(T, B, C, W, H)` with
      W and H transposed. Invisible on square DVS frames, fatal at 480x640. The same bug
      exists at lines 43 and 157 of the ImageNet variant, patched the same way.
    """
    _stub("timm")
    mod = _load_file("spili_dvs", FORKS["spiliformer"] / "cifar10dvs" / "model.py")
    _fix_spiliformer_square_assumption(mod)
    return mod
