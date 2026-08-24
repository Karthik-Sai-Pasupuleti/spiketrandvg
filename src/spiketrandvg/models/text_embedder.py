"""Caption -> (B, L, d_model) token embeddings. The slot SpikeLM used to occupy.

`SpikeGroundingV2` no longer owns a text encoder; it consumes token embeddings and leaves
their production to the caller. This is the default provider: a stock HuggingFace encoder
plus a linear projection to the fusion width.

Why SpikeLM is gone
-------------------
It was 124.3M parameters of roberta-base transplanted by name into SpikeLM's spiking BERT
(197 of 199 tensors matched). SpikeLM ships no weights, so nothing about it was ever
spike-pretrained -- the language knowledge was ANN knowledge wearing a spiking forward
pass. Measured with it frozen, over 85 epochs, the grounding model was caption-blind:
mean caption delta +0.0009 against a mIoU of 0.21, i.e. feeding a deliberately wrong
caption cost 0.4% of the prediction. The same architecture with encoders unfrozen reached
+0.051 in two epochs, so the encoders -- not the fusion or the head -- were the blocker.

Nothing here claims the replacement will ground either. It removes one confound: a
conventional, genuinely-pretrained text encoder whose representations are known-good, so
that any remaining failure is attributable to the fusion rather than to a transplanted
encoder that was never trained in the regime it runs in.

Contract
--------
    forward(input_ids, attention_mask) -> (B, L, d_model)

`L` must match the mask, and `d_model` must equal the grounding model's. The tokenizer
must be the one `Talk2EventDataset` uses to build `positive_map` -- roberta-base -- or the
soft-token supervision points at the wrong words. `build_tokenizer()` enforces that.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

__all__ = ["MAX_TEXT_LEN", "TOKENIZER_NAME", "TextEmbedder", "build_tokenizer"]

# roberta-base, matching Talk2EventDataset's own tokenizer exactly
TOKENIZER_NAME = "roberta-base"

# Measured over the eval split: median 38 tokens, p95 53, max 72, and the highest non-zero
# positive_map index is 63. 80 leaves headroom without truncating supervision.
MAX_TEXT_LEN = 80


def build_tokenizer():
    """The one tokenizer for this project: the same one the dataset uses."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


class TextEmbedder(nn.Module):
    """HuggingFace encoder + linear projection to the fusion width.

    Args:
        model_name: any HF encoder; must share roberta-base's vocabulary if the dataset's
            `positive_map` is used downstream.
        d_model: fusion width the grounding model expects.
        freeze: hold the encoder fixed (the projection always trains -- it has no
            pretrained counterpart, so freezing it would leave a random map in place).
    """

    def __init__(self, model_name: str = TOKENIZER_NAME, d_model: int = 256,
                 freeze: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, d_model)
        self.d_model = d_model
        self.frozen = freeze
        if freeze:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            # eval() also disables the encoder's dropout, which otherwise makes the text
            # features nondeterministic and any train-mode comparison meaningless
            self.encoder.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > MAX_TEXT_LEN:
            raise ValueError(f"caption length {input_ids.shape[1]} exceeds {MAX_TEXT_LEN}")
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
        return self.proj(hidden)                       # (B, L, d_model)
