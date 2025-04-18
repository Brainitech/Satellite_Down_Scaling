import logging
import torch
from pathlib import Path

def get_logger(name="default"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)
    return logger

def save_checkpoint(G, D, epoch, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict()
    }, save_dir / f"model_epoch_{epoch}.pth")
