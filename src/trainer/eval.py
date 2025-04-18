import torch
from pathlib import Path
from src.data.dataset import MODISLandsatDataset
from torch.utils.data import DataLoader
from src.models import Generator
from src.utils.metrics import PSNR, SSIM
from src.utils.logger import get_logger

def evaluate(config):
    cfg = config
    logger = get_logger("eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MODISLandsatDataset(Path(cfg["data"]["modis"]),
                                  Path(cfg["data"]["landsat"]))
    dataloader = DataLoader(dataset, batch_size=1)

    G = Generator().to(device)
    checkpoint = torch.load(Path(cfg["eval"]["checkpoint"]), map_location=device)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()

    psnr_total, ssim_total, count = 0, 0, 0

    with torch.no_grad():
        for modis_seq, landsat_img in dataloader:
            modis_seq, landsat_img = modis_seq.to(device), landsat_img.to(device)
            sr_img = G(modis_seq)

            psnr_total += PSNR(sr_img, landsat_img).item()
            ssim_total += SSIM(sr_img, landsat_img).item()
            count += 1

    logger.info(f"Avg PSNR: {psnr_total / count:.2f}, Avg SSIM: {ssim_total / count:.4f}")

if __name__ == "__main__":
    evaluate()
