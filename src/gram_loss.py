"""Original Gram Anchoring loss (source: clean training notebook, cell 13)."""
import torch
from torch import nn
import torch.nn.functional as F


class OriginalGramLoss(nn.Module):
    def forward(self, student_feats, teacher_feats):
        with torch.autocast(device_type="cuda", enabled=False):
            s = F.normalize(student_feats.float(), dim=-1)
            t = F.normalize(teacher_feats.float(), dim=-1)

            Gs = s @ s.transpose(-1, -2)

            with torch.no_grad():
                Gt = t @ t.transpose(-1, -2)

            return F.mse_loss(Gs, Gt), {}
