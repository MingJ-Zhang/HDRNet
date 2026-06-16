# Copyright (c) 2026.
# HDRNet decode head for MMSegmentation 1.x / MMEngine.
# Modules:
#   DRCA: Disaster Relation-aware Context Aggregation
#   DSBR: Damage-sensitive Structural Boundary Refinement
#   HDAD: Hierarchical Damage-aware Decoding Head

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


class DilatedDWConv(nn.Module):
    def __init__(self, channels: int, dilation: int, norm_cfg=None, act_cfg=None):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DRCA(nn.Module):
    """Disaster Relation-aware Context Aggregation.

    This is a lightweight replacement for ASPP/PPM at the highest-level feature.
    It combines multi-scale branches using coarse disaster semantic gates and an
    optional learnable co-occurrence relation projection.
    """

    def __init__(self,
                 channels: int,
                 num_coarse_classes: int = 5,
                 dilations: Sequence[int] = (1, 6, 12),
                 use_relation: bool = True):
        super().__init__()
        self.num_branches = 2 + len(dilations)  # point + dilated branches + global
        self.use_relation = use_relation

        self.point = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.ModuleList([
            DilatedDWConv(channels, d) for d in dilations
        ])
        self.global_fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
        )
        self.coarse_cls = nn.Sequential(
            nn.Linear(channels, max(64, channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(64, channels // 4), num_coarse_classes),
        )
        self.gate = nn.Linear(num_coarse_classes, self.num_branches)

        if use_relation:
            self.relation_proj = nn.Sequential(
                nn.Linear(num_coarse_classes, channels),
                nn.Sigmoid(),
            )

        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        fp = self.point(x)
        branches = [fp]
        branches += [branch(fp) for branch in self.dilated]

        pooled = F.adaptive_avg_pool2d(fp, 1).flatten(1)
        coarse_logits = self.coarse_cls(pooled)
        coarse_prob = F.softmax(coarse_logits, dim=1)

        fg = self.global_fc(pooled).view(b, c, 1, 1).expand(-1, -1, h, w)
        branches.append(fg)

        weights = F.softmax(self.gate(coarse_prob), dim=1)
        fused = 0.0
        for i, feat in enumerate(branches):
            fused = fused + weights[:, i].view(b, 1, 1, 1) * feat

        if self.use_relation:
            rel = self.relation_proj(coarse_prob).view(b, c, 1, 1)
            fused = fused + fused * rel

        return self.proj(fused), coarse_logits


class DirectionalStructureConv(nn.Module):
    """Learnable directional structural response initialized by line kernels."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels * 4, 3, padding=1,
                                   groups=channels, bias=False)
        kernels = torch.tensor([
            [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
            [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
            [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
            [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
        ], dtype=torch.float32)
        with torch.no_grad():
            weight = torch.zeros(channels * 4, 1, 3, 3)
            for ch in range(channels):
                for k in range(4):
                    weight[ch * 4 + k, 0] = kernels[k]
            self.depthwise.weight.copy_(weight)
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.depthwise(x), inplace=True))


class DSBR(nn.Module):
    """Damage-sensitive Structural Boundary Refinement."""

    def __init__(self, low_channels: int, dec_channels: int):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, dec_channels, 1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )
        self.struct = DirectionalStructureConv(dec_channels)
        self.boundary_head = nn.Conv2d(dec_channels, 1, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(dec_channels * 2 + 1, dec_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_channels, dec_channels, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(dec_channels, dec_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, dec_feat: torch.Tensor, low_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low = self.low_proj(low_feat)
        low = resize(low, size=dec_feat.shape[2:], mode='bilinear', align_corners=False)
        struct = self.struct(low)
        boundary_logit = self.boundary_head(struct)
        boundary_prob = torch.sigmoid(boundary_logit)
        gate = self.attn(torch.cat([dec_feat, struct, boundary_prob], dim=1))
        refined = dec_feat + dec_feat * gate + struct * gate
        return self.out_proj(refined), boundary_logit


def _make_target_from_mapping(seg_gt: torch.Tensor,
                              mapping: Optional[Sequence[int]],
                              ignore_index: int) -> Optional[torch.Tensor]:
    if mapping is None:
        return None
    target = torch.full_like(seg_gt, ignore_index)
    for src_id, dst_id in enumerate(mapping):
        if dst_id is None or dst_id < 0:
            continue
        target[seg_gt == src_id] = int(dst_id)
    target[seg_gt == ignore_index] = ignore_index
    return target


def _boundary_from_seg(seg_gt: torch.Tensor,
                       ignore_index: int,
                       damage_ids: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Generate binary boundary labels from segmentation masks.

    Args:
        seg_gt: (B,H,W), long.
        damage_ids: if set, only boundaries related to these semantic ids are used.
    Returns:
        boundary: (B,1,H,W), float.
    """
    valid = (seg_gt != ignore_index)
    if damage_ids:
        mask = torch.zeros_like(seg_gt, dtype=torch.bool)
        for cid in damage_ids:
            mask |= (seg_gt == int(cid))
        work = mask.long()
    else:
        work = seg_gt.clone()
        work[~valid] = 0

    x = work.unsqueeze(1).float()
    dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    boundary = (dilated != eroded).float()
    boundary = boundary * valid.unsqueeze(1).float()
    return boundary


@MODELS.register_module()
class HDRNetHead(BaseDecodeHead):
    """SegFormer-style decode head with DRCA + DSBR + HDAD.

    During inference ``forward`` returns final semantic logits only, so it remains
    compatible with MMSegmentation's EncoderDecoder sliding-window inference.
    During training ``loss`` computes auxiliary hierarchical and boundary losses.
    """

    def __init__(self,
                 feature_strides=(4, 8, 16, 32),
                 num_coarse_classes: int = 5,
                 num_object_classes: int = 8,
                 num_damage_classes: int = 4,
                 object_class_mapping: Optional[Sequence[int]] = None,
                 damage_class_mapping: Optional[Sequence[int]] = None,
                 coarse_class_mapping: Optional[Sequence[int]] = None,
                 damage_ordinal: bool = True,
                 damage_semantic_ids: Optional[Sequence[int]] = None,
                 loss_weights: Optional[Dict[str, float]] = None,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(self.in_channels) == len(feature_strides)
        self.feature_strides = feature_strides
        self.num_coarse_classes = num_coarse_classes
        self.num_object_classes = num_object_classes
        self.num_damage_classes = num_damage_classes
        self.object_class_mapping = object_class_mapping
        self.damage_class_mapping = damage_class_mapping
        self.coarse_class_mapping = coarse_class_mapping
        self.damage_ordinal = damage_ordinal and num_damage_classes > 2
        self.damage_semantic_ids = damage_semantic_ids or []
        self.loss_weights = dict(
            seg=1.0, coarse=0.1, obj=0.4, damage=0.6, hier=0.3, boundary=0.4)
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        self.proj_convs = nn.ModuleList([
            ConvModule(ch, self.channels, 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            for ch in self.in_channels
        ])
        self.drca = DRCA(self.channels, num_coarse_classes=num_coarse_classes)
        self.fpn_fuse = ConvModule(
            self.channels * len(self.in_channels), self.channels, 1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.dsbr = DSBR(low_channels=self.in_channels[0], dec_channels=self.channels)

        self.object_head = nn.Conv2d(self.channels, num_object_classes, 1)
        dmg_out = num_damage_classes - 1 if self.damage_ordinal else num_damage_classes
        self.damage_head = nn.Conv2d(self.channels, dmg_out, 1)
        # Auxiliary logits can help the final semantic branch, but directly
        # concatenating unbounded logits often over-constrains the decoder.
        # Start with a small, learnable contribution and let training increase it
        # only when the hierarchy is useful for the final mask.
        self.object_fuse_logit = nn.Parameter(torch.tensor(-2.1972))
        self.damage_fuse_logit = nn.Parameter(torch.tensor(-2.1972))
        self.final_fuse = ConvModule(
            self.channels + num_object_classes + dmg_out, self.channels, 3,
            padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, 1)

    def _forward_all(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self._transform_inputs(inputs)
        out_size = feats[0].shape[2:]
        projected = []
        for i, feat in enumerate(feats):
            x = self.proj_convs[i](feat)
            if i == len(feats) - 1:
                x, coarse_logits = self.drca(x)
            x = resize(x, size=out_size, mode='bilinear', align_corners=self.align_corners)
            projected.append(x)
        dec = self.fpn_fuse(torch.cat(projected, dim=1))
        dec, boundary_logit = self.dsbr(dec, feats[0])
        object_logits = self.object_head(dec)
        damage_logits = self.damage_head(dec)
        object_for_fuse = torch.tanh(object_logits) * torch.sigmoid(self.object_fuse_logit)
        damage_for_fuse = torch.tanh(damage_logits) * torch.sigmoid(self.damage_fuse_logit)
        fused = self.final_fuse(torch.cat([dec, object_for_fuse, damage_for_fuse], dim=1))
        seg_logits = self.cls_seg(fused)
        return dict(
            seg_logits=seg_logits,
            coarse_logits=coarse_logits,
            object_logits=object_logits,
            damage_logits=damage_logits,
            boundary_logit=boundary_logit,
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return self._forward_all(inputs)['seg_logits']

    def _stack_gt(self, batch_data_samples) -> torch.Tensor:
        seg_gt = [sample.gt_sem_seg.data.squeeze(0) for sample in batch_data_samples]
        return torch.stack(seg_gt, dim=0).long()

    def loss(self, inputs: Tuple[torch.Tensor], batch_data_samples, train_cfg) -> Dict[str, torch.Tensor]:
        outputs = self._forward_all(inputs)
        seg_gt = self._stack_gt(batch_data_samples).to(outputs['seg_logits'].device)
        seg_logits = resize(outputs['seg_logits'], size=seg_gt.shape[1:],
                            mode='bilinear', align_corners=self.align_corners)

        losses: Dict[str, torch.Tensor] = {}
        losses['loss_seg'] = self.loss_weights['seg'] * F.cross_entropy(
            seg_logits, seg_gt, ignore_index=self.ignore_index)

        # Object-level branch.
        obj_gt = _make_target_from_mapping(seg_gt, self.object_class_mapping, self.ignore_index)
        if obj_gt is not None:
            obj_logits = resize(outputs['object_logits'], size=seg_gt.shape[1:],
                                mode='bilinear', align_corners=self.align_corners)
            losses['loss_obj'] = self.loss_weights['obj'] * F.cross_entropy(
                obj_logits, obj_gt, ignore_index=self.ignore_index)

        # Damage/status branch.
        dmg_gt = _make_target_from_mapping(seg_gt, self.damage_class_mapping, self.ignore_index)
        if dmg_gt is not None:
            dmg_logits = resize(outputs['damage_logits'], size=seg_gt.shape[1:],
                                mode='bilinear', align_corners=self.align_corners)
            valid = (dmg_gt != self.ignore_index)
            if valid.any():
                if self.damage_ordinal:
                    # damage logits: (B,K-1,H,W), target_k = 1[y >= k].
                    targets = []
                    for k in range(1, self.num_damage_classes):
                        targets.append((dmg_gt >= k).float())
                    ordinal_target = torch.stack(targets, dim=1)
                    valid_f = valid.unsqueeze(1).float()
                    loss_ord = F.binary_cross_entropy_with_logits(
                        dmg_logits, ordinal_target, reduction='none')
                    losses['loss_damage'] = self.loss_weights['damage'] * (loss_ord * valid_f).sum() / valid_f.sum().clamp_min(1.0)
                else:
                    losses['loss_damage'] = self.loss_weights['damage'] * F.cross_entropy(
                        dmg_logits, dmg_gt, ignore_index=self.ignore_index)

        # Coarse branch: generate image-level label from an explicit semantic to
        # coarse mapping when available; fall back to object-level majority.
        coarse_gt = _make_target_from_mapping(
            seg_gt, self.coarse_class_mapping, self.ignore_index)
        if coarse_gt is None:
            coarse_gt = obj_gt
        if coarse_gt is not None:
            coarse_targets = []
            for b in range(coarse_gt.shape[0]):
                vals = coarse_gt[b][coarse_gt[b] != self.ignore_index]
                if vals.numel() == 0:
                    coarse_targets.append(torch.tensor(0, device=coarse_gt.device))
                else:
                    coarse_vals = vals.clamp(min=0, max=self.num_coarse_classes - 1)
                    hist = torch.bincount(coarse_vals, minlength=self.num_coarse_classes)
                    coarse_targets.append(hist.argmax())
            coarse_targets = torch.stack(coarse_targets).long()
            losses['loss_coarse'] = self.loss_weights['coarse'] * F.cross_entropy(
                outputs['coarse_logits'], coarse_targets)

        # Boundary branch.
        boundary_gt = _boundary_from_seg(seg_gt, self.ignore_index, self.damage_semantic_ids)
        boundary_logit = resize(outputs['boundary_logit'], size=seg_gt.shape[1:],
                                mode='bilinear', align_corners=self.align_corners)
        bce = F.binary_cross_entropy_with_logits(boundary_logit, boundary_gt, reduction='mean')
        prob = torch.sigmoid(boundary_logit)
        inter = (prob * boundary_gt).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + boundary_gt.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
        losses['loss_boundary'] = self.loss_weights['boundary'] * (bce + dice)

        # Pixel accuracy for convenient logs.
        with torch.no_grad():
            pred = seg_logits.argmax(dim=1)
            valid = seg_gt != self.ignore_index
            acc = (pred[valid] == seg_gt[valid]).float().mean() if valid.any() else seg_logits.new_tensor(0.)
        losses['acc_seg'] = acc
        return losses
