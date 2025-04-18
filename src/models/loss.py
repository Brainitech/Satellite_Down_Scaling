import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, mode='bce'):
        super().__init__()
        if mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'mse':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unsupported GAN loss type")

    def forward(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)

# Optional: perceptual loss could be added here if using VGG features
