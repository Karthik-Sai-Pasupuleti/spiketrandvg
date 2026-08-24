"""Spiking text encoder: SpikeLM's spiking BERT behind a projection.

Wraps the frozen SpikeLM fork. Nothing in it is edited -- the model is built through
its own constructor with a config we assemble.

Tokenizer rule (the important constraint)
-----------------------------------------
`Talk2EventDataset` tokenizes captions with **roberta-base** and builds
`target["positive_map"]` against those exact token positions. Any text encoder in this
model must therefore share that vocabulary, or the soft-token alignment supervision
points at the wrong words.

SpikeLM is BERT-*architecture*, but `vocab_size` and `pad_token_id` are plain config
arguments feeding an ordinary `nn.Embedding`, so the architecture is instantiated with
roberta-base's vocabulary (50265 tokens, pad id 1) rather than BERT's 30522. Use
`build_tokenizer()` here so the two can never drift apart.

Time semantics
--------------
SpikeLM owns its own spiking time axis: `BertEncoder` repeats the embeddings to
(T, B, L, D) and averages over T before returning (spike_bert.py:479 and :506). Its T
is therefore INDEPENDENT of the vision encoder's T_STEPS, and the output is a single
(B, L, hidden) tensor. That is the right semantics here: a referring expression is
constant over the event sequence, so one text representation is reused across every
vision timestep rather than being recomputed per step.

CUDA-only: `SpikeLinear.forward` hard-codes `.cuda()` (spiking.py:116-117), so this
module cannot run on CPU without editing the fork.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertConfig, RobertaTokenizerFast

from spiketrandvg.utils import forks

# roberta-base, matching Talk2EventDataset's tokenizer exactly
TOKENIZER_NAME = "roberta-base"
ROBERTA_VOCAB_SIZE = 50265
ROBERTA_PAD_TOKEN_ID = 1

# SpikeLM's own spiking timesteps (run_pretrain.py sets 4); unrelated to vision T_STEPS
SPIKELM_T = 4

# Padding length for captions. Measured over the eval split: median 38 tokens, p95 53,
# max 72, and the highest non-zero positive_map index seen is 63. A 64-token window
# therefore truncates ~0.2% of captions and sits exactly on the boundary of the
# supervision, so 80 is used to leave headroom. Note positive_map is only 256 wide, so
# nothing beyond token 255 could ever be supervised anyway.
MAX_TEXT_LEN = 80


def build_tokenizer() -> RobertaTokenizerFast:
    """The one tokenizer for this project: the same one the dataset uses."""
    return RobertaTokenizerFast.from_pretrained(TOKENIZER_NAME)


def make_spikelm_config(
    vocab_size: int = ROBERTA_VOCAB_SIZE,
    pad_token_id: int = ROBERTA_PAD_TOKEN_ID,
    num_hidden_layers: int = 12,
    hidden_size: int = 768,
    T: int = SPIKELM_T,
) -> BertConfig:
    """Stock BertConfig plus the attributes SpikeLM's spiking layers read.

    The five that are load-bearing (missing any raises AttributeError deep inside
    SpikeLinear.__init__ or BertSelfAttention.__init__) are weight_bits,
    quantize_act, clip_val, input_bits and T. The rest are set to the values the
    fork's own pretraining script uses, for fidelity.
    """
    cfg = BertConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
    )
    cfg.weight_bits = 32          # weights stay full precision; only activations spike
    cfg.input_bits = 2            # ternary {-1, 0, +1} elastic bi-spiking
    cfg.clip_init_val = 2.5
    cfg.weight_layerwise = True
    cfg.input_layerwise = True
    cfg.hidden_act = "relu"
    cfg.quantize_act = True
    cfg.clip_val = 1.0
    cfg.T = T
    return cfg


class SpikeLMTextEncoder(nn.Module):
    """Referring expression -> token features and a sentence vector."""

    def __init__(
        self,
        d_model: int = 256,
        num_hidden_layers: int = 12,
        T: int = SPIKELM_T,
        freeze: bool = False,
    ):
        super().__init__()
        spikelm = forks.load_spikelm()
        self.config = make_spikelm_config(num_hidden_layers=num_hidden_layers, T=T)
        self.encoder = spikelm.BertModel(self.config, add_pooling_layer=False)
        # declared-analog projection to the fusion width
        self.proj = nn.Linear(self.config.hidden_size, d_model)
        self.d_model = d_model
        self.frozen = freeze
        if freeze:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L) ids and mask -> (tokens (B, L, d_model), sentence (B, d_model)).

        The sentence vector is a MASKED mean, so padding never dilutes it.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"expected (B, L) input_ids, got {tuple(input_ids.shape)}")
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tokens = self.proj(out.last_hidden_state)              # (B, L, d_model)
        m = attention_mask.unsqueeze(-1).to(tokens.dtype)
        sentence = (tokens * m).sum(1) / m.sum(1).clamp(min=1.0)
        return tokens, sentence


def build_text_encoder(d_model: int = 256, **kwargs) -> SpikeLMTextEncoder:
    return SpikeLMTextEncoder(d_model=d_model, **kwargs)


def load_pretrained_weights(
    encoder: "SpikeLMTextEncoder",
    donor: str = TOKENIZER_NAME,
    verbose: bool = True,
) -> dict[str, int]:
    """Initialise SpikeLM's BertModel from a pretrained ANN checkpoint.

    SpikeLM ships NO weights of its own -- the repo contains no checkpoint files, its
    `base_spike/` holds only args.json and `bert-base-uncased/` is empty, and the paper
    expects you to pretrain. Left random, the text encoder starts with no linguistic
    knowledge and must learn language from the grounding task alone.

    Because SpikeLM is BERT-architecture with standard parameter names, a real
    checkpoint transfers almost completely: 195 of roberta-base's tensors match by
    name AND shape (the whole 12-layer transformer plus word embeddings). Two need
    conversion, handled below. SpikeLM's own spiking parameters (the per-timestep
    act_clip_val / clip_key / clip_value ParameterLists and weight_clip_val buffers)
    have no counterpart and keep their initialisation -- they are calibrated lazily
    from the first forward pass anyway.

    The donor MUST share the tokenizer used for positive_map, hence roberta-base by
    default; its word embeddings are only meaningful for that vocabulary.

    Returns a report dict of tensor counts.
    """
    from transformers import AutoModel

    target = encoder.encoder
    tgt_sd = target.state_dict()
    donor_sd = AutoModel.from_pretrained(donor).state_dict()
    # roberta-base prefixes everything with 'roberta.'; strip so names line up
    donor_sd = {
        (k.split(".", 1)[1] if k.startswith(("roberta.", "bert.")) else k): v
        for k, v in donor_sd.items()
    }

    new_sd, converted = {}, []
    for key, tgt in tgt_sd.items():
        src = donor_sd.get(key)
        if src is None:
            continue
        if src.shape == tgt.shape:
            new_sd[key] = src
        elif key.endswith("position_embeddings.weight"):
            # RoBERTa reserves ids 0 and 1 (pad offset), so rows 2..N are the real
            # positions; take exactly as many as the BERT-style config expects.
            n = tgt.shape[0]
            if src.shape[0] >= n + 2:
                new_sd[key] = src[2 : 2 + n].clone()
                converted.append(f"{key}: {tuple(src.shape)} -> {tuple(tgt.shape)} (dropped ids 0,1)")
        elif key.endswith("token_type_embeddings.weight"):
            # RoBERTa has a single segment; replicate it across BERT's segments
            new_sd[key] = src[:1].expand_as(tgt).clone()
            converted.append(f"{key}: {tuple(src.shape)} -> {tuple(tgt.shape)} (segment replicated)")

    msg = target.load_state_dict(new_sd, strict=False)
    spiking_kept = len(msg.missing_keys)
    report = {
        "donor_tensors": len(donor_sd),
        "loaded": len(new_sd),
        "converted": len(converted),
        "spiking_params_left_at_init": spiking_kept,
    }
    if verbose:
        print(f"[text_encoder] initialised from '{donor}': "
              f"loaded {report['loaded']} tensors "
              f"({report['converted']} converted), "
              f"{spiking_kept} SpikeLM-specific params kept at init")
        for line in converted:
            print(f"    {line}")
    return report
