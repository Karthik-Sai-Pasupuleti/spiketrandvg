"""Caption -> (B, L, d_model) token embeddings. A stock ANN encoder, deliberately.

    forward(input_ids, attention_mask) -> (B, L, d_model)

This is the one component in the model that is genuinely pretrained for its job, and it
is frozen by default. The projection to `d_model` always trains -- it has no pretrained
counterpart, so freezing it would leave a random map between roberta and the fusion.

Why this is an ANN and not a spiking encoder
--------------------------------------------
It used to be SpikeLM: 124.3M parameters of roberta-base transplanted by name into a
spiking BERT (197 of 199 tensors matched). SpikeLM ships no weights, so nothing about it
was ever spike-pretrained -- the language knowledge was ANN knowledge wearing a spiking
forward pass. Measured with it frozen over 85 epochs, the grounding model was
caption-blind: mean caption delta +0.0009 against a mIoU of 0.21, i.e. a deliberately
wrong caption cost 0.4% of the prediction. The same architecture with encoders unfrozen
reached +0.051 in two epochs.

So the model is a hybrid on purpose: spiking vision, ANN language. Swapping a
known-good text encoder in removes one confound, so that any remaining failure is
attributable to the fusion rather than to a transplanted encoder that was never trained
in the regime it runs in.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

__all__ = ["MAX_TEXT_LEN", "TOKENIZER_NAME", "TextEncoder", "build_tokenizer",
           "ATTRIBUTES", "AttributeQueryTagger", "SpikingTextEncoder"]

# Talk2Event annotates every caption with phrases under exactly these four headings.
# They are the dataset's own labels, not a taxonomy invented here -- see
# `talk2event.py`, which locates each phrase back in the caption to build token-level
# supervision for the tagger below.
ATTRIBUTES = ("appearance", "status", "relation_viewer", "relation_others")

TOKENIZER_NAME = "roberta-base"

# RefCOCO captions are short: median 5 tokens on refcoco, 10 on refcocog, longest 49
# across all three datasets. This is a truncation guard, not the padding length --
# `dataset.make_collate` pads to the longest caption in each batch instead.
MAX_TEXT_LEN = 80


def build_tokenizer():
    """The one tokenizer for this project. Must match whatever built the annotations."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


class TextEncoder(nn.Module):
    """HuggingFace encoder + linear projection to the fusion width.

    Args:
        model_name: any HF encoder.
        d_model: fusion width the grounding model expects.
        freeze: hold the encoder fixed (the projection always trains).
        unfreeze_last: unfreeze only the last N transformer layers (0 = respect `freeze`
            exactly). Requires `model.encoder.layer` -- true of RobertaModel and most HF
            encoder classes; raises otherwise rather than silently leaving everything
            frozen. Untested hypothesis: on Talk2Event, unfreezing the VISION side alone
            moved caption delta from +0.0009 to +0.051, but the text side has never been
            tested this way.
    """

    def __init__(self, model_name: str = TOKENIZER_NAME, d_model: int = 256,
                 freeze: bool = True, unfreeze_last: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, d_model)
        self.d_model = d_model
        self.unfreeze_last = unfreeze_last
        # frozen: forward() runs the WHOLE encoder under no_grad, so per-parameter
        # requires_grad is irrelevant. partial: forward() runs under enable_grad and only
        # the last N layers accumulate gradient -- the rest sit at requires_grad=False and
        # get none, despite the relaxed context.
        self.frozen = freeze and unfreeze_last == 0
        self.partial = freeze and unfreeze_last > 0
        if freeze:
            self.encoder.requires_grad_(False)
            if unfreeze_last > 0:
                if not hasattr(self.encoder, "encoder") or not hasattr(
                        self.encoder.encoder, "layer"):
                    raise ValueError(
                        f"{model_name} has no `.encoder.layer` -- unfreeze_last needs "
                        "a standard HF encoder stack (RobertaModel, BertModel, ...)")
                layers = self.encoder.encoder.layer
                if unfreeze_last > len(layers):
                    raise ValueError(f"unfreeze_last={unfreeze_last} exceeds the "
                                     f"{len(layers)} layers {model_name} has")
                for layer in layers[-unfreeze_last:]:
                    layer.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            # eval() also disables the encoder's dropout, which otherwise makes the text
            # features nondeterministic and any train-mode comparison meaningless
            self.encoder.eval()
        elif self.partial:
            # dropout stays active in the unfrozen tail, like every other trainable
            # module -- only the frozen prefix's PARAMETERS are held fixed
            self.encoder.train(mode)
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > MAX_TEXT_LEN:
            raise ValueError(f"caption length {input_ids.shape[1]} exceeds {MAX_TEXT_LEN}")
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
        return self.proj(hidden)                       # (B, L, d_model)


class AttributeQueryTagger(nn.Module):
    """Caption -> four attribute sub-queries, via a learned per-token span tagger.

        forward(tokens, attention_mask) -> (queries (B,4,d), logits (B,L,5))

    Instead of pooling the caption into ONE vector, every token is tagged with which of
    the four Talk2Event attributes it belongs to (or none), and tokens are pooled per
    attribute into `q_appearance`, `q_status`, `q_viewer`, `q_others`.

    Why four and not one
    --------------------
    The four attributes ask for different evidence from the scene. "A dark-coloured car"
    is an appearance question; "driving" is a motion question; "on the right side of the
    road" is a viewer-relative geometry question. A single pooled vector cannot tell the
    encoder which of those it is being asked, so it cannot condition on it -- and
    conditioning is the point (see `ThresholdModulator`).

    It is also a clean single-variable ablation against EventRefer, which recovers the
    same four groups by fuzzy string-matching the raw caption at data-loading time. Here
    they are *learned*, supervised by spans that the dataset already provides, so the
    tagger can generalise to phrasings the string matcher misses.

    Soft pooling, not hard
    ----------------------
    Pooling uses the softmax posterior rather than an argmax, so the whole path stays
    differentiable and a token that is genuinely ambiguous ("parked" is both status and
    appearance-ish) contributes to both queries in proportion. `logits` is returned so
    the trainer can add the auxiliary cross-entropy against the dataset's spans; without
    that loss the tagger is free to learn any partition it likes, which is a valid
    ablation but no longer "the four attributes".
    """

    def __init__(self, d_model: int = 256, n_attr: int = len(ATTRIBUTES)):
        super().__init__()
        self.n_attr = n_attr
        # n_attr + 1: the extra class is "belongs to no attribute", which most tokens
        # (articles, prepositions, the object noun itself) legitimately are.
        self.tag = nn.Linear(d_model, n_attr + 1)
        # Learned fallback for an attribute with no tokens assigned to it -- common,
        # since not every caption mentions all four. Zero would push a meaningless
        # all-zeros vector into cross-attention; a learned token says "not asked".
        self.null_query = nn.Parameter(torch.zeros(n_attr, d_model))
        nn.init.trunc_normal_(self.null_query, std=0.02)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        """tokens (B,L,d) from TextEncoder; attention_mask (B,L)."""
        if tokens.dim() != 3:
            raise ValueError(f"expected (B,L,d) tokens, got {tuple(tokens.shape)}")
        logits = self.tag(tokens)                                  # (B,L,n_attr+1)
        mask = attention_mask.to(tokens.dtype).unsqueeze(-1)       # (B,L,1)

        # padded positions must not win any attribute's pooling weight
        p = logits.softmax(-1)[..., :self.n_attr] * mask           # (B,L,n_attr)
        w = p.transpose(1, 2)                                      # (B,n_attr,L)
        denom = w.sum(-1, keepdim=True)                            # (B,n_attr,1)
        pooled = torch.bmm(w, tokens) / denom.clamp(min=1e-6)      # (B,n_attr,d)

        # where an attribute drew essentially no mass, substitute the learned null query
        empty = (denom < 1e-4).to(tokens.dtype)                    # (B,n_attr,1)
        queries = pooled * (1 - empty) + self.null_query[None] * empty
        return queries, logits


class SpikingTextEncoder(nn.Module):
    """SpikeLM's spiking BERT + projection. Drop-in for `TextEncoder`.

        forward(input_ids, attention_mask) -> (B, L, d_model)

    Why this exists
    ---------------
    The rest of the model is spike-driven but the language half is a conventional ANN --
    roberta-base is 124.8M of the 183.9M parameters, so by parameter count the system is
    only ~30% spiking. That is the weakest point in any "spiking grounding" claim. This
    swaps the one remaining ANN for a spiking encoder, taking the model to ~98% spiking.

    What SpikeLM actually is, and the caveat that governs every result
    ------------------------------------------------------------------
    SpikeLM is a quantisation-aware-training FRAMEWORK for converting an ANN language
    model into an SNN. The repository ships the recipe and the architecture; it ships
    **no trained weights**. So `pretrained_name` here transplants roberta/BERT tensors by
    NAME into the spiking forward pass -- 197 of 199 matched last time this was tried.
    Those weights were trained for continuous activations and are being run through
    binary ones. Nothing about this encoder is spike-pretrained, and a frozen transplant
    should be expected to underperform its ANN source rather than match it.

    Prior result, and why it is not decisive
    -----------------------------------------
    A frozen SpikeLM encoder previously measured caption delta **+0.0009 over 85 epochs**
    -- caption-blind. That run is confounded: the architecture it sat in was caption-blind
    with roberta too (+0.0009 frozen, +0.051 unfrozen), because the attention map was
    uniform and position could not reach the head at all. The encoder was never the
    isolated variable. Re-running it on the current architecture -- which grounds at
    caption delta +0.302 on held-out test -- is the first measurement that actually
    isolates the text encoder.

    CUDA-only: `SpikeLinear.forward` hard-codes `.cuda()` in the fork.
    """

    def __init__(self, pretrained_name: str = TOKENIZER_NAME, d_model: int = 256,
                 freeze: bool = False, T: int = 4, input_bits: int = 1,
                 weight_bits: int = 1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        from spiketrandvg import utils as forks

        sl = forks.load_spikelm()
        cfg = AutoConfig.from_pretrained(pretrained_name)
        # the five extra attributes the spiking code reads off the config
        cfg.weight_bits, cfg.input_bits = weight_bits, input_bits
        cfg.quantize_act, cfg.clip_val, cfg.T = True, 2.5, T
        self.encoder = sl.BertModel(cfg, add_pooling_layer=False)

        # transplant the ANN weights by name -- see the caveat above
        src = AutoModel.from_pretrained(pretrained_name).state_dict()
        msg = self.encoder.load_state_dict(src, strict=False)
        matched = len(src) - len(msg.unexpected_keys)
        print(f"[textencoder] SpikeLM BERT from {pretrained_name}: {matched}/{len(src)} "
              f"tensors transplanted (missing {len(msg.missing_keys)}, "
              f"unexpected {len(msg.unexpected_keys)}) -- NOT spike-pretrained")
        self.transplant_report = {"matched": matched, "of": len(src),
                                  "missing": len(msg.missing_keys)}

        self.proj = nn.Linear(cfg.hidden_size, d_model)
        self.d_model, self.T = d_model, T
        # `frozen` here means the same as in TextEncoder: no gradient through the encoder.
        # Default is FALSE, unlike TextEncoder -- a transplant that was never trained in
        # this regime has no pretraining worth protecting, and freezing it is what the
        # previous +0.0009 result did.
        self.frozen = freeze
        self.partial = False
        if freeze:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > MAX_TEXT_LEN:
            raise ValueError(f"caption length {input_ids.shape[1]} exceeds {MAX_TEXT_LEN}")
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            # BertEncoder owns its own time axis and averages it away internally, so
            # last_hidden_state is (B, L, hidden) exactly as the ANN encoder returns
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
        return self.proj(hidden)
