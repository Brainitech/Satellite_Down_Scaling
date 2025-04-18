from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

class MODISLandsatDataset(Dataset):
    def __init__(self, modis_dir: Path, landsat_dir: Path, transform=None):
        self.modis_paths = sorted(modis_dir.glob("*.npy"))
        self.landsat_paths = sorted(landsat_dir.glob("*.npy"))
        self.transform = transform

        assert len(self.modis_paths) == len(self.landsat_paths), \
            "Mismatch in MODIS and Landsat sequence counts."

    def __len__(self):
        return len(self.modis_paths)

    def __getitem__(self, idx):
        modis_seq = np.load(self.modis_paths[idx])
        landsat_img = np.load(self.landsat_paths[idx])

        modis_tensor = torch.from_numpy(modis_seq).float()
        landsat_tensor = torch.from_numpy(landsat_img).float()

        if self.transform:
            modis_tensor, landsat_tensor = self.transform(modis_tensor, landsat_tensor)

        return modis_tensor, landsat_tensor
