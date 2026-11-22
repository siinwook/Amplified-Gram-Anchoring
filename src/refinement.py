import torch
from torch import nn
import torch.nn.functional as F

from dinov2.hub.backbones import dinov2_vits14
from dinov2.layers import DINOHead
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss

from .config import (
    DINO_OUT_DIM, IBOT_OUT_DIM, HEAD_HIDDEN_DIM, HEAD_BOTTLENECK_DIM,
    HEAD_NLAYERS, HIGHAM_TOL, EMA_MOMENTUM, TEACHER_TEMP,
    RELATION_WEIGHT, LOCAL_CROPS, GRAM_SIZE, GLOBAL_SIZE,
    DINO_WEIGHT, IBOT_WEIGHT, KOLEO_WEIGHT,
)
from .gram_loss import OriginalGramLoss
from .aga_loss import AmplifiedGramLoss

def resize_patch_tokens(tokens, src_size, dst_size, patch_size=14):
    src_n = src_size // patch_size
    dst_n = dst_size // patch_size

    assert tokens.shape[1] == src_n * src_n

    x = tokens.transpose(1, 2).reshape(
        tokens.shape[0],
        tokens.shape[2],
        src_n,
        src_n,
    )

    x = F.interpolate(
        x,
        size=(dst_n, dst_n),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )

    return x.flatten(2).transpose(1, 2)

class DINOv2Refiner(nn.Module):
    def __init__(self, relation_mode="none", aga_b=0.4, aga_alpha=8.0, higham_max_iter=10):
        super().__init__()
        assert relation_mode in {"none", "gram", "aga"}

        self.relation_mode = relation_mode

        self.student_backbone = dinov2_vits14(pretrained=True)

        self.teacher_backbone = dinov2_vits14(pretrained=False)
        self.teacher_backbone.load_state_dict(
            self.student_backbone.state_dict(),
            strict=True,
        )

        # Only relation runs need a third ViT.
        if relation_mode in {"gram", "aga"}:
            self.gram_teacher = dinov2_vits14(pretrained=False)
            self.gram_teacher.load_state_dict(
                self.student_backbone.state_dict(),
                strict=True,
            )
        else:
            self.gram_teacher = None

        self.student_dino_head = DINOHead(
            in_dim=384,
            out_dim=DINO_OUT_DIM,
            hidden_dim=HEAD_HIDDEN_DIM,
            bottleneck_dim=HEAD_BOTTLENECK_DIM,
            nlayers=HEAD_NLAYERS,
        )
        self.student_ibot_head = DINOHead(
            in_dim=384,
            out_dim=IBOT_OUT_DIM,
            hidden_dim=HEAD_HIDDEN_DIM,
            bottleneck_dim=HEAD_BOTTLENECK_DIM,
            nlayers=HEAD_NLAYERS,
        )
        self.teacher_dino_head = DINOHead(
            in_dim=384,
            out_dim=DINO_OUT_DIM,
            hidden_dim=HEAD_HIDDEN_DIM,
            bottleneck_dim=HEAD_BOTTLENECK_DIM,
            nlayers=HEAD_NLAYERS,
        )
        self.teacher_ibot_head = DINOHead(
            in_dim=384,
            out_dim=IBOT_OUT_DIM,
            hidden_dim=HEAD_HIDDEN_DIM,
            bottleneck_dim=HEAD_BOTTLENECK_DIM,
            nlayers=HEAD_NLAYERS,
        )

        self.teacher_dino_head.load_state_dict(
            self.student_dino_head.state_dict(),
            strict=True,
        )
        self.teacher_ibot_head.load_state_dict(
            self.student_ibot_head.state_dict(),
            strict=True,
        )

        self.dino_loss = DINOLoss(DINO_OUT_DIM)
        self.ibot_loss = iBOTPatchLoss(IBOT_OUT_DIM)
        self.koleo_loss = KoLeoLoss()

        if relation_mode == "gram":
            self.relation_loss = OriginalGramLoss()
        elif relation_mode == "aga":
            self.relation_loss = AmplifiedGramLoss(
                b=aga_b,
                alpha=aga_alpha,
                max_iter=higham_max_iter,
                tol=HIGHAM_TOL,
            )
        else:
            self.relation_loss = None

        self.freeze_teachers()

    def freeze_teachers(self):
        modules = [
            self.teacher_backbone,
            self.teacher_dino_head,
            self.teacher_ibot_head,
        ]
        if self.gram_teacher is not None:
            modules.append(self.gram_teacher)

        for module in modules:
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher_backbone.eval()
        self.teacher_dino_head.eval()
        self.teacher_ibot_head.eval()
        if self.gram_teacher is not None:
            self.gram_teacher.eval()
        return self

    def set_backbone_trainable(self, trainable):
        self.student_backbone.requires_grad_(trainable)

    @torch.no_grad()
    def reset_gram_teacher_from_student(self):
        if self.gram_teacher is not None:
            self.gram_teacher.load_state_dict(
                self.student_backbone.state_dict(),
                strict=True,
            )
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()

    @torch.no_grad()
    def update_teacher(self, m=EMA_MOMENTUM):
        pairs = [
            (self.student_backbone, self.teacher_backbone),
            (self.student_dino_head, self.teacher_dino_head),
            (self.student_ibot_head, self.teacher_ibot_head),
        ]

        for student, teacher in pairs:
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.mul_(m).add_(ps, alpha=1.0 - m)

    def forward_losses(
        self,
        batch,
        teacher_temp=TEACHER_TEMP,
        relation_weight=RELATION_WEIGHT,
        calibration=False,
    ):
        device = next(self.student_backbone.parameters()).device

        global_crops = batch["collated_global_crops"].to(
            device, non_blocking=True
        )
        local_crops = batch["collated_local_crops"].to(
            device, non_blocking=True
        )
        masks = batch["collated_masks"].to(
            device, non_blocking=True
        )
        mask_indices = batch["mask_indices_list"].to(
            device, non_blocking=True
        )
        masks_weight = batch["masks_weight"].to(
            device, non_blocking=True
        )

        n_global = 2
        n_local = LOCAL_CROPS
        n_masked = mask_indices.shape[0]

        with torch.no_grad():
            t = self.teacher_backbone(
                global_crops,
                is_training=True,
            )
            t_cls = t["x_norm_clstoken"]
            t_patch = t["x_norm_patchtokens"]

            c0, c1 = t_cls.chunk(2)
            t_cls_rev = torch.cat([c1, c0], dim=0)

            t_cls_logits = self.teacher_dino_head(t_cls_rev)
            t_cls_probs = self.dino_loss.softmax_center_teacher(
                t_cls_logits,
                teacher_temp=teacher_temp,
            ).view(2, -1, DINO_OUT_DIM)
            self.dino_loss.update_center(t_cls_logits)

            t_masked_patch = torch.index_select(
                t_patch.flatten(0, 1),
                dim=0,
                index=mask_indices,
            )

            t_patch_logits = self.teacher_ibot_head(t_masked_patch)
            t_patch_logits_3d = t_patch_logits.unsqueeze(0)

            t_patch_probs = self.ibot_loss.softmax_center_teacher(
                t_patch_logits_3d,
                teacher_temp=teacher_temp,
            ).squeeze(0)
            self.ibot_loss.update_center(t_patch_logits_3d)

            gt_patch = None
            if (not calibration) and self.gram_teacher is not None:
                gram_crops = batch[
                    "collated_gram_teacher_crops"
                ].to(device, non_blocking=True)

                gt = self.gram_teacher(
                    gram_crops,
                    is_training=True,
                )
                gt_patch = resize_patch_tokens(
                    gt["x_norm_patchtokens"],
                    src_size=GRAM_SIZE,
                    dst_size=GLOBAL_SIZE,
                    patch_size=14,
                )

        s_global = self.student_backbone(
            global_crops,
            masks=masks,
            is_training=True,
        )
        s_local = self.student_backbone(
            local_crops,
            is_training=True,
        )

        s_g_cls = s_global["x_norm_clstoken"]
        s_l_cls = s_local["x_norm_clstoken"]
        s_g_patch = s_global["x_norm_patchtokens"]

        s_g_cls_logits = self.student_dino_head(s_g_cls)
        s_l_cls_logits = self.student_dino_head(s_l_cls)

        s_masked_patch = torch.index_select(
            s_g_patch.flatten(0, 1),
            dim=0,
            index=mask_indices,
        )
        s_masked_patch_logits = self.student_ibot_head(
            s_masked_patch
        )

        n_local_terms = max(n_local * n_global, 1)
        n_global_terms = (n_global - 1) * n_global
        denom = n_local_terms + n_global_terms

        dino_local = (
            self.dino_loss(
                student_output_list=s_l_cls_logits.chunk(n_local),
                teacher_out_softmaxed_centered_list=t_cls_probs,
            )
            / denom
        )

        dino_global = (
            self.dino_loss(
                student_output_list=[s_g_cls_logits],
                teacher_out_softmaxed_centered_list=[
                    t_cls_probs.flatten(0, 1)
                ],
            )
            * 2.0
            / denom
        )

        dino = dino_local + dino_global

        ibot = self.ibot_loss.forward_masked(
            s_masked_patch_logits,
            t_patch_probs,
            student_masks_flat=masks,
            n_masked_patches=n_masked,
            masks_weight=masks_weight,
        )

        koleo = sum(
            self.koleo_loss(x)
            for x in s_g_cls.chunk(2)
        )

        relation_stats = {}

        if calibration:
            relation = dino.new_zeros(())
            total = (
                DINO_WEIGHT * dino
                + IBOT_WEIGHT * ibot
            )
        else:
            if self.relation_loss is not None:
                relation, relation_stats = self.relation_loss(
                    s_g_patch,
                    gt_patch,
                )
            else:
                relation = dino.new_zeros(())

            total = (
                DINO_WEIGHT * dino
                + IBOT_WEIGHT * ibot
                + KOLEO_WEIGHT * koleo
                + relation_weight * relation
            )

        stats = {
            "total": float(total.detach()),
            "dino": float(dino.detach()),
            "ibot": float(ibot.detach()),
            "koleo": float(koleo.detach()),
            "relation": float(relation.detach()),
        }
        stats.update(relation_stats)

        return total, stats
