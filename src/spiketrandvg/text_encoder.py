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

__all__ = ["MAX_TEXT_LEN", "TOKENIZER_NAME", "TextEncoder", "build_tokenizer"]

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
