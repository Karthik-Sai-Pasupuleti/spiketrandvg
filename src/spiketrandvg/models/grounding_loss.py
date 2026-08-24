"""Loss for SpikeTransDVG's single-box head: one prediction, one target.

The head emits four numbers, so the loss is a plain box regression -- no anchors, no
label assignment, no distribution focal loss, no classification term. This is the DETR /
TransVG single-box objective:

    L = l1_weight * L1(pred, target)  +  ciou_weight * (1 - CIoU(pred, target))

Both terms are needed and they fail in opposite directions. L1 on normalised cxcywh has a
stable gradient everywhere, including when the boxes do not overlap at all -- which is the
entire early-training regime, where IoU is 0 and its gradient carries no direction. CIoU
is what actually matches the metric, and unlike plain IoU it keeps pulling on centre
distance and aspect ratio once the boxes do overlap. DETR's 5:2 weighting is the default
here for the same reason it is there: L1 on numbers in (0, 1) is small compared to an IoU
term that starts near 1.

Everything is in **normalised cxcywh**, which is how `Talk2EventDataset` stores boxes and
what the head emits, so no unit conversion happens anywhere in the training loop. CIoU is
computed on xyxy because that is what torchvision wants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, complete_box_iou_loss

__all__ = ["SingleBoxLoss", "cxcywh_to_xyxy_norm"]


def cxcywh_to_xyxy_norm(box: torch.Tensor) -> torch.Tensor:
    """(N, 4) normalised cxcywh -> (N, 4) normalised xyxy."""
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class SingleBoxLoss(nn.Module):
    """forward(pred, target) -> (total, parts), both (B, 4) normalised cxcywh."""

    def __init__(self, l1_weight: float = 5.0, ciou_weight: float = 2.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ciou_weight = ciou_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pred.shape != target.shape or pred.shape[-1] != 4:
            raise ValueError(
                f"expected matching (B, 4) boxes, got {tuple(pred.shape)} and "
                f"{tuple(target.shape)}"
            )
        l1 = F.l1_loss(pred, target)

        p_xyxy = cxcywh_to_xyxy_norm(pred)
        t_xyxy = cxcywh_to_xyxy_norm(target)
        ciou = complete_box_iou_loss(p_xyxy, t_xyxy, reduction="mean")

        with torch.no_grad():
            iou = box_iou(p_xyxy, t_xyxy).diagonal().mean()

        total = self.l1_weight * l1 + self.ciou_weight * ciou
        return total, {"l1": l1.detach(), "ciou": ciou.detach(), "iou": iou}
