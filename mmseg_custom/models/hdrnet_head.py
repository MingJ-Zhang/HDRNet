# Copyright (c) 2026.
# HDRNet decode head for MMSegmentation 1.x / MMEngine.
# This implementation is aligned with the Methodology document:
#   DRCA: Disaster Relation-aware Context Aggregation with an explicit
#         class co-occurrence prior and relation attention.
#   DSBR: Damage-sensitive Structural Boundary Refinement with F1/F2 low-level
#         structural cues and boundary-attention injection.
#   HDAD: Hierarchical Damage-aware Decoding Head with object/status branches,
#         independent feature heads, and hierarchical KL consistency loss.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


class DilatedDWConv(nn.Module):
    """Depthwise separable dilated convolution branch."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CategoryRelationAttention(nn.Module):
    """Relation attention over coarse disaster-category tokens.

    Given an image-level coarse disaster distribution p_c and a category
    co-occurrence prior R, this module first constructs category tokens, performs
    relation-biased category self-attention, and then injects the resulting
    relation-enhanced tokens back into the spatial feature map through cross
    attention.

    This realizes the paper formula in an implementation-friendly form:
        A = Softmax(QK^T / sqrt(d) + beta R)
        F_hat = F + CrossAttn(F, RelationAttn(T, R))
    """

    def __init__(self,
                 channels: int,
                 num_coarse_classes: int,
                 cooccurrence_matrix: Optional[Sequence[Sequence[float]]] = None,
                 num_heads: int = 4,
                 learnable_beta: bool = True):
        super().__init__()
        assert channels % num_heads == 0, 'channels must be divisible by num_heads'
        self.channels = channels
        self.num_coarse_classes = num_coarse_classes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.category_embed = nn.Parameter(torch.randn(num_coarse_classes, channels) * 0.02)
        self.token_q = nn.Linear(channels, channels)
        self.token_k = nn.Linear(channels, channels)
        self.token_v = nn.Linear(channels, channels)
        self.token_out = nn.Linear(channels, channels)

        self.pixel_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.pixel_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.token_k_cross = nn.Linear(channels, channels)
        self.token_v_cross = nn.Linear(channels, channels)

        if cooccurrence_matrix is None:
            prior = torch.eye(num_coarse_classes, dtype=torch.float32)
        else:
            prior = torch.tensor(cooccurrence_matrix, dtype=torch.float32)
            assert prior.shape == (num_coarse_classes, num_coarse_classes), \
                f'cooccurrence_matrix must be {num_coarse_classes}x{num_coarse_classes}'
        # Row-normalize to make it a stable prior bias.
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer('relation_prior', prior)

        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_buffer('beta', torch.tensor(1.0, dtype=torch.float32))

        self.norm = nn.BatchNorm2d(channels)

    def _reshape_heads_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,K,C) -> (B,heads,K,head_dim)
        b, k, c = x.shape
        return x.view(b, k, self.num_heads, self.head_dim).transpose(1, 2)

    def _reshape_heads_pixels(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,heads,HW,head_dim)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, feat: torch.Tensor, coarse_prob: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        # Category tokens weighted by predicted coarse disaster distribution.
        tokens = self.category_embed.unsqueeze(0) * coarse_prob.unsqueeze(-1)  # (B,K,C)

        q = self._reshape_heads_tokens(self.token_q(tokens))
        k = self._reshape_heads_tokens(self.token_k(tokens))
        v = self._reshape_heads_tokens(self.token_v(tokens))
        relation_bias = self.beta * self.relation_prior.view(1, 1, self.num_coarse_classes, self.num_coarse_classes)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale + relation_bias
        attn = F.softmax(attn, dim=-1)
        rel_tokens = torch.matmul(attn, v)  # (B,heads,K,head_dim)
        rel_tokens = rel_tokens.transpose(1, 2).contiguous().view(b, self.num_coarse_classes, c)
        rel_tokens = self.token_out(rel_tokens)

        q_pix = self._reshape_heads_pixels(self.pixel_q(feat))
        k_tok = self._reshape_heads_tokens(self.token_k_cross(rel_tokens))
        v_tok = self._reshape_heads_tokens(self.token_v_cross(rel_tokens))
        cross = torch.matmul(q_pix, k_tok.transpose(-2, -1)) * self.scale
        cross = F.softmax(cross, dim=-1)
        ctx = torch.matmul(cross, v_tok)  # (B,heads,HW,head_dim)
        ctx = ctx.transpose(1, 2).contiguous().view(b, h * w, c).transpose(1, 2).view(b, c, h, w)
        ctx = self.pixel_out(ctx)
        return self.norm(feat + ctx)


class DRCA(nn.Module):
    """Disaster Relation-aware Context Aggregation.

    Compared with standard ASPP/PPM, DRCA has two explicit mechanisms:
    1) coarse disaster-semantic gated multi-scale aggregation;
    2) class co-occurrence prior guided relation attention.
    """

    def __init__(self,
                 channels: int,
                 num_coarse_classes: int = 5,
                 dilations: Sequence[int] = (1, 6, 12),
                 cooccurrence_matrix: Optional[Sequence[Sequence[float]]] = None,
                 relation_heads: int = 4,
                 use_relation: bool = True):
        super().__init__()
        self.num_branches = 2 + len(dilations)  # point + dilated branches + global
        self.use_relation = use_relation

        self.point = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.ModuleList([DilatedDWConv(channels, d) for d in dilations])
        self.global_fc = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True))
        self.coarse_cls = nn.Sequential(
            nn.Linear(channels, max(64, channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(64, channels // 4), num_coarse_classes),
        )
        self.gate = nn.Linear(num_coarse_classes, self.num_branches)
        self.relation_attn = CategoryRelationAttention(
            channels=channels,
            num_coarse_classes=num_coarse_classes,
            cooccurrence_matrix=cooccurrence_matrix,
            num_heads=relation_heads) if use_relation else None
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        fp = self.point(x)
        branches = [fp] + [branch(fp) for branch in self.dilated]

        pooled = F.adaptive_avg_pool2d(fp, 1).flatten(1)
        coarse_logits = self.coarse_cls(pooled)
        coarse_prob = F.softmax(coarse_logits, dim=1)

        fg = self.global_fc(pooled).view(b, c, 1, 1).expand(-1, -1, h, w)
        branches.append(fg)

        weights = F.softmax(self.gate(coarse_prob), dim=1)
        fused = 0.0
        for i, feat in enumerate(branches):
            fused = fused + weights[:, i].view(b, 1, 1, 1) * feat

        if self.relation_attn is not None:
            fused = self.relation_attn(fused, coarse_prob)
        return self.proj(fused), coarse_logits, coarse_prob


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
    """Damage-sensitive Structural Boundary Refinement.

    The module uses two low-level encoder features F1 and F2. It predicts an
    auxiliary boundary map and injects boundary-aware attention into decoder
    features according to: F_r = F_d + F_d * M_b.
    """

    def __init__(self, low1_channels: int, low2_channels: int, dec_channels: int):
        super().__init__()
        self.low1_proj = nn.Sequential(
            nn.Conv2d(low1_channels, dec_channels, 1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )
        self.low2_proj = nn.Sequential(
            nn.Conv2d(low2_channels, dec_channels, 1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )
        self.low_fuse = nn.Sequential(
            nn.Conv2d(dec_channels * 2, dec_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )
        self.struct = DirectionalStructureConv(dec_channels)
        self.boundary_head = nn.Conv2d(dec_channels, 1, 1)
        self.boundary_gate = nn.Sequential(
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

    def forward(self,
                dec_feat: torch.Tensor,
                low1_feat: torch.Tensor,
                low2_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = dec_feat.shape[2:]
        low1 = resize(self.low1_proj(low1_feat), size=size, mode='bilinear', align_corners=False)
        low2 = resize(self.low2_proj(low2_feat), size=size, mode='bilinear', align_corners=False)
        low = self.low_fuse(torch.cat([low1, low2], dim=1))
        struct_feat = self.struct(low)
        boundary_logit = self.boundary_head(struct_feat)
        boundary_prob = torch.sigmoid(boundary_logit)
        gate = self.boundary_gate(torch.cat([dec_feat, struct_feat, boundary_prob], dim=1))
        refined = dec_feat + dec_feat * gate
        return self.out_proj(refined), boundary_logit, gate


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


def _make_soft_distribution_from_mapping(seg_gt: torch.Tensor,
                                         mapping: Optional[Sequence[int]],
                                         num_targets: int,
                                         ignore_index: int) -> Optional[torch.Tensor]:
    """Image-level coarse distribution from pixel labels."""
    mapped = _make_target_from_mapping(seg_gt, mapping, ignore_index)
    if mapped is None:
        return None
    dists = []
    for b in range(mapped.shape[0]):
        vals = mapped[b][mapped[b] != ignore_index]
        if vals.numel() == 0:
            hist = torch.zeros(num_targets, device=mapped.device, dtype=torch.float32)
            hist[0] = 1.0
        else:
            hist = torch.bincount(vals.clamp(0, num_targets - 1), minlength=num_targets).float()
            hist = hist / hist.sum().clamp_min(1.0)
        dists.append(hist)
    return torch.stack(dists, dim=0)


def _boundary_from_seg(seg_gt: torch.Tensor,
                       ignore_index: int,
                       damage_ids: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Generate binary boundary labels from segmentation masks.

    If damage_ids is provided, only boundaries of disaster-sensitive semantic
    categories are used, which matches DSBR's damage-sensitive supervision.
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


def _ordinal_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 ordinal logits into K class probabilities."""
    cum = torch.sigmoid(logits)
    ones = torch.ones_like(cum[:, :1])
    zeros = torch.zeros_like(cum[:, :1])
    cum = torch.cat([ones, cum, zeros], dim=1)
    probs = cum[:, :-1] - cum[:, 1:]
    # Numerical safety: independent sigmoid classifiers may violate monotonicity
    # early in training. Clamp and re-normalize for probability losses/logging.
    probs = probs.clamp_min(1e-6)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _project_semantic_prob(seg_prob: torch.Tensor,
                           mapping: Optional[Sequence[int]],
                           num_targets: int) -> Optional[torch.Tensor]:
    if mapping is None:
        return None
    b, c, h, w = seg_prob.shape
    out = seg_prob.new_zeros((b, num_targets, h, w))
    for src_id, dst_id in enumerate(mapping):
        if dst_id is None or dst_id < 0 or src_id >= c:
            continue
        out[:, int(dst_id)] = out[:, int(dst_id)] + seg_prob[:, src_id]
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _masked_kl(p: torch.Tensor, q: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """D_KL(p || q) averaged over valid pixels."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl = (p * (p.log() - q.log())).sum(dim=1)
    return (kl * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


@MODELS.register_module()
class HDRNetHead(BaseDecodeHead):
    """SegFormer-style decode head with DRCA + DSBR + HDAD.

    During inference ``forward`` returns final semantic logits only, so it stays
    compatible with MMSegmentation EncoderDecoder sliding-window inference.
    During training ``loss`` computes segmentation, coarse context, object,
    damage/status, hierarchical consistency, and boundary losses.
    """

    def __init__(self,
                 feature_strides=(4, 8, 16, 32),
                 num_coarse_classes: int = 5,
                 num_object_classes: int = 8,
                 num_damage_classes: int = 4,
                 coarse_class_mapping: Optional[Sequence[int]] = None,
                 object_class_mapping: Optional[Sequence[int]] = None,
                 damage_class_mapping: Optional[Sequence[int]] = None,
                 class_cooccurrence_matrix: Optional[Sequence[Sequence[float]]] = None,
                 damage_ordinal: bool = True,
                 damage_semantic_ids: Optional[Sequence[int]] = None,
                 loss_weights: Optional[Dict[str, float]] = None,
                 relation_heads: int = 4,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(self.in_channels) == len(feature_strides)
        assert len(self.in_channels) >= 2, 'DSBR needs at least F1 and F2 features.'
        self.feature_strides = feature_strides
        self.num_coarse_classes = num_coarse_classes
        self.num_object_classes = num_object_classes
        self.num_damage_classes = num_damage_classes
        self.coarse_class_mapping = coarse_class_mapping
        self.object_class_mapping = object_class_mapping
        self.damage_class_mapping = damage_class_mapping
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
        self.drca = DRCA(
            self.channels,
            num_coarse_classes=num_coarse_classes,
            cooccurrence_matrix=class_cooccurrence_matrix,
            relation_heads=relation_heads,
            use_relation=True)
        self.fpn_fuse = ConvModule(
            self.channels * len(self.in_channels), self.channels, 1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.dsbr = DSBR(
            low1_channels=self.in_channels[0],
            low2_channels=self.in_channels[1],
            dec_channels=self.channels)

        self.object_feat_head = ConvModule(
            self.channels, self.channels, 3, padding=1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.object_head = nn.Conv2d(self.channels, num_object_classes, 1)

        self.damage_feat_head = ConvModule(
            self.channels, self.channels, 3, padding=1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        dmg_out = num_damage_classes - 1 if self.damage_ordinal else num_damage_classes
        self.damage_head = nn.Conv2d(self.channels, dmg_out, 1)

        self.final_fuse = ConvModule(
            self.channels * 3, self.channels, 3, padding=1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, 1)

    def _forward_all(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self._transform_inputs(inputs)
        out_size = feats[0].shape[2:]
        projected = []
        coarse_logits = None
        coarse_prob = None
        for i, feat in enumerate(feats):
            x = self.proj_convs[i](feat)
            if i == len(feats) - 1:
                x, coarse_logits, coarse_prob = self.drca(x)
            x = resize(x, size=out_size, mode='bilinear', align_corners=self.align_corners)
            projected.append(x)
        dec = self.fpn_fuse(torch.cat(projected, dim=1))
        refined, boundary_logit, boundary_gate = self.dsbr(dec, feats[0], feats[1])

        object_feat = self.object_feat_head(refined)
        object_logits = self.object_head(object_feat)

        damage_feat = self.damage_feat_head(refined)
        damage_logits = self.damage_head(damage_feat)

        fused = self.final_fuse(torch.cat([refined, object_feat, damage_feat], dim=1))
        seg_logits = self.cls_seg(fused)
        return dict(
            seg_logits=seg_logits,
            coarse_logits=coarse_logits,
            coarse_prob=coarse_prob,
            object_logits=object_logits,
            damage_logits=damage_logits,
            boundary_logit=boundary_logit,
            boundary_gate=boundary_gate,
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
        valid = seg_gt != self.ignore_index
        losses['loss_seg'] = self.loss_weights['seg'] * F.cross_entropy(
            seg_logits, seg_gt, ignore_index=self.ignore_index)

        # DRCA auxiliary coarse disaster-distribution supervision.
        coarse_dist = _make_soft_distribution_from_mapping(
            seg_gt, self.coarse_class_mapping, self.num_coarse_classes, self.ignore_index)
        if coarse_dist is not None and outputs['coarse_logits'] is not None:
            log_pc = F.log_softmax(outputs['coarse_logits'], dim=1)
            losses['loss_coarse'] = self.loss_weights['coarse'] * (-(coarse_dist * log_pc).sum(dim=1).mean())

        # HDAD object-level branch.
        obj_gt = _make_target_from_mapping(seg_gt, self.object_class_mapping, self.ignore_index)
        obj_logits = resize(outputs['object_logits'], size=seg_gt.shape[1:],
                            mode='bilinear', align_corners=self.align_corners)
        if obj_gt is not None:
            losses['loss_obj'] = self.loss_weights['obj'] * F.cross_entropy(
                obj_logits, obj_gt, ignore_index=self.ignore_index)

        # HDAD damage/status branch.
        dmg_gt = _make_target_from_mapping(seg_gt, self.damage_class_mapping, self.ignore_index)
        dmg_logits = resize(outputs['damage_logits'], size=seg_gt.shape[1:],
                            mode='bilinear', align_corners=self.align_corners)
        if dmg_gt is not None:
            dmg_valid = (dmg_gt != self.ignore_index)
            if dmg_valid.any():
                if self.damage_ordinal:
                    targets = []
                    for k in range(1, self.num_damage_classes):
                        targets.append((dmg_gt >= k).float())
                    ordinal_target = torch.stack(targets, dim=1)
                    valid_f = dmg_valid.unsqueeze(1).float()
                    loss_ord = F.binary_cross_entropy_with_logits(
                        dmg_logits, ordinal_target, reduction='none')
                    losses['loss_damage'] = self.loss_weights['damage'] * (loss_ord * valid_f).sum() / valid_f.sum().clamp_min(1.0)
                else:
                    losses['loss_damage'] = self.loss_weights['damage'] * F.cross_entropy(
                        dmg_logits, dmg_gt, ignore_index=self.ignore_index)

        # HDAD hierarchical consistency loss: object/status branch probabilities
        # should be consistent with final semantic probabilities projected by
        # semantic-to-object and semantic-to-damage mappings.
        if self.loss_weights.get('hier', 0.0) > 0:
            seg_prob = F.softmax(seg_logits, dim=1)
            hier_terms = []
            obj_from_seg = _project_semantic_prob(seg_prob, self.object_class_mapping, self.num_object_classes)
            if obj_from_seg is not None:
                obj_prob = F.softmax(obj_logits, dim=1)
                hier_terms.append(_masked_kl(obj_prob, obj_from_seg, valid))
            dmg_from_seg = _project_semantic_prob(seg_prob, self.damage_class_mapping, self.num_damage_classes)
            if dmg_from_seg is not None:
                if self.damage_ordinal:
                    dmg_prob = _ordinal_probs_from_logits(dmg_logits)
                else:
                    dmg_prob = F.softmax(dmg_logits, dim=1)
                # Use pixels with valid damage labels if mapping ignores some classes.
                dmg_mask = valid if dmg_gt is None else (dmg_gt != self.ignore_index)
                hier_terms.append(_masked_kl(dmg_prob, dmg_from_seg, dmg_mask))
            if hier_terms:
                losses['loss_hier'] = self.loss_weights['hier'] * sum(hier_terms) / len(hier_terms)

        # DSBR auxiliary damage-sensitive boundary supervision.
        boundary_gt = _boundary_from_seg(seg_gt, self.ignore_index, self.damage_semantic_ids)
        boundary_logit = resize(outputs['boundary_logit'], size=seg_gt.shape[1:],
                                mode='bilinear', align_corners=self.align_corners)
        bce = F.binary_cross_entropy_with_logits(boundary_logit, boundary_gt, reduction='mean')
        prob = torch.sigmoid(boundary_logit)
        inter = (prob * boundary_gt).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + boundary_gt.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
        losses['loss_boundary'] = self.loss_weights['boundary'] * (bce + dice)

        with torch.no_grad():
            pred = seg_logits.argmax(dim=1)
            acc = (pred[valid] == seg_gt[valid]).float().mean() if valid.any() else seg_logits.new_tensor(0.)
        losses['acc_seg'] = acc
        return losses
