import torch
import torch.nn as nn
import scipy.integrate as integral
import numpy as np

def apply_mask(input, target, mask_length=2000):
    fade_length = 960  # 10 ms at 48 kHz
    
    fade_out = torch.hann_window(2 * fade_length, device=input.device)[fade_length:]
    protected_region = torch.ones(mask_length - fade_length, device=input.device)
    window = torch.cat([protected_region, fade_out]).unsqueeze(0).unsqueeze(-1)
    window = window[:, :mask_length, :]

    masked_input = input.clone()
    masked_input[:, :mask_length, :] = (
        target[:, :mask_length, :] * window + 
        input[:, :mask_length, :] * (1 - window)
    )
    
    return masked_input

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

import torch
import torch.nn as nn

class EDCLoss(nn.Module):
    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def compute_edc(self, ir):
        """
        Compute the Energy Decay Curve (EDC) from an impulse response (IR).
        EDC[n] = sum_{k=n}^{N-1} ir[k]^2 / sum_{k=0}^{N-1} ir[k]^2
        """

        energy = ir ** 2

        cumulative_energy = torch.flip(torch.cumsum(torch.flip(energy, dims=[-1]), dim=-1), dims=[-1])

        edc = cumulative_energy / (torch.sum(energy, dim=-1, keepdim=True) + self.eps)
        return edc

    def forward(self, input_ir, target_ir):
        input_edc = self.compute_edc(input_ir)
        target_edc = self.compute_edc(target_ir)

        num = torch.sum((input_edc - target_edc) ** 2, dim=-1)
        denom = torch.sum(target_edc ** 2, dim=-1) + self.eps
        ratio = num / denom

        loss = torch.log1p(ratio)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ThresholdedEDCLoss(nn.Module):
    def __init__(self, threshold_db):
        super().__init__()
        self.threshold_linear = 10**(threshold_db / 20)
    
    def forward(self, output, target):
        output_thresh = torch.where(torch.abs(output) < self.threshold_linear, 
                                torch.zeros_like(output), output)
        target_thresh = torch.where(torch.abs(target) < self.threshold_linear, 
                                torch.zeros_like(target), target)
        
        return EDCLoss()(output_thresh, target_thresh)
    
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