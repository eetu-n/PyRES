import torch
import torch.nn as nn
import scipy.integrate as integral

# MSE scaled up by 1000
class ScaledMSELoss(nn.Module):
    def __init__(self, scale=1000):
        super().__init__()
        self.scale = scale
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.scale * self.mse(input, target)

# MAE scaled up by 1000
class ScaledMAELoss(nn.Module):
    def __init__(self, scale=1000):
        super().__init__()
        self.scale = scale
        self.mae = nn.L1Loss()

    def forward(self, input, target):
        return self.scale * self.mae(input, target)

class EDCLoss(nn.Module):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale
        #self.input_edc = integral.quad(lambda)

class ESRLoss(nn.Module):
    """Error-to-Signal Ratio (ESR) loss.
    
    Reference:
        Wright & Välimäki, 2019 — "Perceptual Audio Losses"
        (https://arxiv.org/abs/1911.08922)

    Args:
        reduction (str): 'none' | 'mean' | 'sum'
        eps (float): Small constant for numerical stability
    Shape:
        - input:  (batch, channels, samples)
        - target: (batch, channels, samples)
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ESR per element
        num = (target - input) ** 2
        denom = (target ** 2) + self.eps
        esr = num / denom  # shape: (B, C, N)

        esr_per_ch = esr.mean(dim=-1)  # (B, C)

        # Average over channels
        esr_per_sample = esr_per_ch.mean(dim=-1)  # (B,)

        # Reduction across batch
        if self.reduction == "mean":
            return esr_per_sample.mean() / 1000
        elif self.reduction == "sum":
            return esr_per_sample.sum() / 1000
        elif self.reduction == "none":
            return esr_per_sample / 1000
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")