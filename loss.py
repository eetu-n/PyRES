import torch
import torch.nn as nn

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
