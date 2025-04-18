from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm

def read_tif(file_path):
    with rasterio.open(file_path) as src:
        return src.read(), src.profile

def save_numpy_array(array, save_path):
    np.save(save_path, array)

def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def preprocess_modis(modis_dir: Path, output_dir: Path, sequence_length=5):
    files = sorted(modis_dir.glob("*.tif"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(len(files) - sequence_length + 1)):
        seq = []
        for j in range(sequence_length):
            img, _ = read_tif(files[i + j])
            img = normalize(img)
            seq.append(img)
        
        seq_arr = np.stack(seq, axis=0)  # (T, C, H, W)
        save_path = output_dir / f"seq_{i:04d}.npy"
        save_numpy_array(seq_arr, save_path)

def preprocess_landsat(landsat_dir: Path, output_dir: Path):
    files = sorted(landsat_dir.glob("*.tif"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(tqdm(files)):
        img, _ = read_tif(path)
        img = normalize(img)
        save_numpy_array(img, output_dir / f"target_{i:04d}.npy")

if __name__ == "__main__":
    preprocess_modis(Path("data/raw/MODIS"), Path("data/processed/modis_sequences"))
    preprocess_landsat(Path("data/raw/LANSAT"), Path("data/processed/landsat_aligned"))
