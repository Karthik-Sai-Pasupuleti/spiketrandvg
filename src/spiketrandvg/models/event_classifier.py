"""Classification head over the event vision encoder.

Used to sanity-check the encoder on a labelled event benchmark (CIFAR10-DVS): if the
encoder cannot learn to classify event cubes, it will not support grounding either.

The readout follows Meta-SpikeFormer's own classifier path exactly (models.py:551-556):

    features -> spatial mean over H,W -> LIF -> Linear -> mean over T

so the only float operation on the spike path is the final mean over timesteps, and
the Linear consumes neuron output (an accumulation, not a MAC).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional as sj_functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from spiketrandvg.models.event_encoder import TAPS, EventVisionEncoder


class EventClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 2,
        tap: str = "s16b",
        ckpt_path: str | None = None,
        freeze_backbone: bool = False,
        trainable_from: str | None = None,
    ):
        super().__init__()
        self.tap = tap
        self.encoder = EventVisionEncoder(
            in_channels=in_channels,
            taps=(tap,),
            ckpt_path=ckpt_path,
            freeze=freeze_backbone,
            trainable_from=trainable_from,
        )
        dim = TAPS[tap][2]
        # backend='torch' because cupy is not installed; see event_encoder for why
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.head = nn.Linear(dim, num_classes)

    def reset(self) -> None:
        """Clear membrane state everywhere, including this head's own neuron.

        The encoder resets its backbone inside its forward, but `self.lif` lives
        outside it. Left unreset, its retained `self.v` still belongs to the previous
        iteration's autograd graph, and the next backward fails with "Trying to
        backward through the graph a second time" -- the state, not the graph, is the
        real culprit.
        """
        sj_functional.reset_net(self.lif)
        self.encoder.reset()

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """(T, B, C, H, W) -> logits (B, num_classes)."""
        self.reset()
        feat = self.encoder(cube)[self.tap]      # (T, B, C, h, w) membrane
        pooled = feat.flatten(3).mean(3)         # (T, B, C) spatial average
        spikes = self.lif(pooled)                # (T, B, C) spikes
        return self.head(spikes).mean(0)         # average logits over timesteps
