import torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.dataset import MODISLandsatDataset
from src.models import Generator, Discriminator, GANLoss, L1Loss
from src.utils.logger import get_logger
from src.utils.metrics import PSNR, SSIM
from src.utils.logger import save_checkpoint


def train_one_epoch(G, D, dataloader, optimizer_G, optimizer_D, gan_loss, pixel_loss, device, logger, epoch):
    G.train()
    D.train()

    for batch_idx, (modis, landsat) in enumerate(dataloader):
        modis, landsat = modis.to(device), landsat.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        fake_landsat = G(modis)
        real_loss = gan_loss(D(landsat), True)
        fake_loss = gan_loss(D(fake_landsat.detach()), False)
        d_loss = (real_loss + fake_loss) * 0.5
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_landsat = G(modis)
        g_gan_loss = gan_loss(D(fake_landsat), True)
        g_pixel_loss = pixel_loss(fake_landsat, landsat)
        g_loss = g_gan_loss + g_pixel_loss
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % 10 == 0:
            logger.info(f"Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], "
                        f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


def train(config):
    cfg = config
    logger = get_logger("train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MODISLandsatDataset(Path(cfg["data"]["modis"]),
                                  Path(cfg["data"]["landsat"]))
    dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=cfg["train"]["lr"])
    optimizer_D = torch.optim.Adam(D.parameters(), lr=cfg["train"]["lr"])

    gan_loss = GANLoss()
    pixel_loss = L1Loss()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_one_epoch(G, D, dataloader, optimizer_G, optimizer_D,
                        gan_loss, pixel_loss, device, logger, epoch)

        if epoch % cfg["train"]["save_every"] == 0:
            save_checkpoint(G, D, epoch, Path(cfg["log"]["ckpt_dir"]))

if __name__ == "__main__":
    train()
