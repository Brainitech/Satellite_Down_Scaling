import matplotlib.pyplot as plt
import numpy as np
import torch

def tensor_to_image(tensor: torch.Tensor):
    tensor = tensor.detach().cpu().squeeze()
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    return tensor.numpy()

def save_image_grid(pred, target, path):
    pred_img = tensor_to_image(pred)
    target_img = tensor_to_image(target)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(pred_img)
    axs[0].set_title("Prediction")
    axs[0].axis('off')
    axs[1].imshow(target_img)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
