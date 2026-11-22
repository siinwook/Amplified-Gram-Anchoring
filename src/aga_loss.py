import torch
from torch import nn
import torch.nn.functional as F

class AmplifiedGramLoss(nn.Module):
    def __init__(
        self,
        b=0.4,
        alpha=8.0,
        max_iter=10,
        tol=1e-4,
        eps=1e-8,
    ):
        super().__init__()
        self.b = b
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.last_iters = 0

    def nonlin_proj(self, G):
        denom = torch.tanh(
            torch.as_tensor(
                self.alpha * (1.0 - self.b),
                device=G.device,
                dtype=G.dtype,
            )
        ).clamp_min(self.eps)

        amp = (
            self.b
            + (1.0 - self.b)
            * torch.tanh(self.alpha * (G - self.b))
            / denom
        )

        return torch.where(G > self.b, amp, G)

    @staticmethod
    def psd_proj(R):
        R = 0.5 * (R + R.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(R)
        eigvals = eigvals.clamp_min(0.0)
        X = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        return 0.5 * (X + X.transpose(-1, -2))

    @staticmethod
    def unitdiag_proj(X):
        Y = X.clone()
        Y.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        return Y

    @torch.no_grad()
    def higham(self, A):
        Y = A.clone()
        S = torch.zeros_like(A)
    
        for i in range(self.max_iter):
            old = Y
    
            R = Y - S
            X = self.psd_proj(R)
            S = X - R
            Y = self.unitdiag_proj(X)
    
            rel = (
                torch.linalg.matrix_norm(
                    Y - old,
                    ord="fro",
                    dim=(-2, -1),
                )
                /
                (
                    torch.linalg.matrix_norm(
                        old,
                        ord="fro",
                        dim=(-2, -1),
                    )
                    + self.eps
                )
            )
    
            # 현재 iteration의 worst convergence error 저장
            self.last_conv_error = rel.max().item()
    
            if rel.max() < self.tol:
                self.last_iters = i + 1
                break
    
        else:
            self.last_iters = self.max_iter
    
        return Y

    @torch.no_grad()
    def stats(self, G, G_amp, G_hat):
        tri = torch.triu(
            torch.ones_like(G, dtype=torch.bool),
            diagonal=1,
        )
        strong = tri & (G > self.b)
        weak = tri & (G <= self.b)

        def mean_or_nan(x, mask):
            if mask.any():
                return x[mask].mean()
            return torch.tensor(float("nan"), device=x.device)

        d_amp = mean_or_nan(G_amp - G, strong)
        d_hat = mean_or_nan(G_hat - G, strong)
        d_weak = mean_or_nan((G_hat - G).abs(), weak)

        if torch.isfinite(d_amp) and d_amp.abs() > self.eps:
            retention = d_hat / d_amp
        else:
            retention = torch.tensor(float("nan"), device=G.device)

        return {
            "d_amp": float(d_amp),
            "d_hat": float(d_hat),
            "retention": float(retention),
            "d_weak": float(d_weak),
            "strong_ratio": float(strong.sum() / tri.sum().clamp_min(1)),
            "higham_iters": self.last_iters,
        }

    def forward(self, student_feats, teacher_feats):
        with torch.autocast(device_type="cuda", enabled=False):
            s = F.normalize(student_feats.float(), dim=-1)
            Gs = s @ s.transpose(-1, -2)

            with torch.no_grad():
                t = F.normalize(teacher_feats.float(), dim=-1)
                G = t @ t.transpose(-1, -2)
                G_amp = self.nonlin_proj(G)
                G_hat = self.higham(G_amp)
                stats = self.stats(G, G_amp, G_hat)

            return F.mse_loss(Gs, G_hat), stats
