import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def PSNR(pred: torch.Tensor, target: torch.Tensor):
    pred_np = pred.squeeze().cpu().detach().numpy()
    target_np = target.squeeze().cpu().detach().numpy()
    return torch.tensor(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0))

def SSIM(pred: torch.Tensor, target: torch.Tensor):
    pred_np = pred.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    target_np = target.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    return torch.tensor(structural_similarity(target_np, pred_np, multichannel=True, data_range=1.0))
