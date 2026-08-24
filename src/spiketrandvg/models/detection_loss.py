"""YOLOv8 detection loss (BCE + CIoU + DFL) with the real task-aligned assigner.

This is ultralytics' `v8DetectionLoss` reassembled around the components loaded out of the
SpikeYOLO fork -- `TaskAlignedAssigner`, `bbox_iou`, `make_anchors`, `dist2bbox`,
`bbox2dist` are all the originals, so the assignment rule and the IoU definition are the
ones the architecture was designed against. Only the plumbing is local, because the loss
class itself lives inside `ultralytics.utils.loss`, whose import chain runs the package
`__init__`.

Units, which is where this kind of loss usually goes wrong:

* `make_anchors` returns centres in FEATURE-GRID cells plus a per-anchor stride, so
  `anchor_points * stride_tensor` is pixels.
* The DFL distribution predicts distances in CELLS, bounded by `reg_max` (16 => at most
  15 cells per side).
* Assignment and the CIoU term run in PIXELS; the DFL term runs in cells. Targets are
  divided by the stride exactly once, right before the box losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiketrandvg.utils import forks

__all__ = ["DetectionLoss"]


class DetectionLoss(nn.Module):
    """forward(feats, gt_boxes, gt_labels, mask) -> (total, {"box","cls","dfl"}).

    feats      list of (B, 4*reg_max + nc, H_s, W_s), the head's training output
    gt_boxes   (B, n_max, 4) xyxy in pixels, zero-padded
    gt_labels  (B, n_max) int64, zero-padded
    mask       (B, n_max, 1) 1 where the box is real
    """

    def __init__(
        self,
        model,
        topk: int = 10,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
    ):
        super().__init__()
        sy = forks.load_spikeyolo()
        self.nc = model.head.nc
        self.no = model.head.no
        self.reg_max = model.head.reg_max
        self.strides = tuple(model.strides)
        self.box_weight, self.cls_weight, self.dfl_weight = box_weight, cls_weight, dfl_weight

        self._make_anchors = sy.make_anchors
        self._dist2bbox = sy.dist2bbox
        self._bbox2dist = sy.bbox2dist
        self._bbox_iou = sy.bbox_iou
        self.assigner = sy.TaskAlignedAssigner(
            topk=topk, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float32))

    def _decode(self, pred_dist: torch.Tensor) -> torch.Tensor:
        """(B, A, 4*reg_max) logits -> (B, A, 4) expected ltrb in cells."""
        b, a, _ = pred_dist.shape
        return pred_dist.view(b, a, 4, self.reg_max).softmax(3).matmul(
            self.proj.to(pred_dist.dtype)
        )

    def _df_loss(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred_dist (N*4, reg_max) logits, target (N, 4) in cells -> (N, 1)."""
        tl = target.long()
        tr = (tl + 1).clamp(max=self.reg_max - 1)
        wl = tr.to(target.dtype) - target
        wr = 1.0 - wl
        ce_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        ce_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (ce_l * wl + ce_r * wr).mean(-1, keepdim=True)

    def forward(self, feats, gt_boxes, gt_labels, mask):
        dtype = feats[0].dtype
        device = feats[0].device
        B = feats[0].shape[0]

        flat = torch.cat([f.view(B, self.no, -1) for f in feats], dim=2)      # (B, no, A)
        pred_dist, pred_scores = flat.split((4 * self.reg_max, self.nc), dim=1)
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()                   # (B, A, 4*rm)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()               # (B, A, nc)

        stride_t = torch.tensor(self.strides, device=device, dtype=dtype)
        anchor_points, stride_tensor = self._make_anchors(feats, stride_t, 0.5)

        # boxes in CELLS, then in pixels for assignment
        pred_ltrb = self._decode(pred_dist)
        pred_bboxes = self._dist2bbox(pred_ltrb, anchor_points, xywh=False, dim=-1)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).to(gt_boxes.dtype),
            anchor_points * stride_tensor,
            gt_labels.unsqueeze(-1),
            gt_boxes,
            mask,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)

        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes = target_bboxes / stride_tensor          # pixels -> cells
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = self._bbox_iou(
                pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
            )
            loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum
            target_ltrb = self._bbox2dist(anchor_points, target_bboxes, self.reg_max)
            dfl = self._df_loss(
                pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]
            )
            loss_dfl = (dfl * weight).sum() / target_scores_sum
        else:
            loss_box = pred_bboxes.sum() * 0
            loss_dfl = pred_dist.sum() * 0

        total = (self.box_weight * loss_box + self.cls_weight * loss_cls
                 + self.dfl_weight * loss_dfl)
        return total, {
            "box": loss_box.detach(),
            "cls": loss_cls.detach(),
            "dfl": loss_dfl.detach(),
            "fg": fg_mask.sum().detach(),
        }
