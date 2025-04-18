import torch
from pathlib import Path
from src.models import Generator
from src.utils.logger import get_logger
import numpy as np
def load_modis_sequence(modis_path):
    """
    Load a MODIS sequence from a .npy file.

    Args:
        modis_path (Path): Path to the .npy file.

    Returns:
        torch.Tensor: Loaded MODIS sequence as a PyTorch tensor.
    """
    data = np.load(modis_path)
    return torch.from_numpy(data).float()


def save_prediction(prediction, save_path):
    """
    Save the prediction as a GeoTIFF file.

    Args:
        prediction (torch.Tensor): The prediction tensor to save.
        save_path (Path): Path to save the GeoTIFF file.
    """
    import rasterio
    from rasterio.transform import from_origin

    # Assuming the prediction is in (C, H, W) format
    prediction = prediction.cpu().numpy()
    height, width = prediction.shape[1], prediction.shape[2]
    transform = from_origin(0, 0, 1, 1)  # Placeholder transform, adjust as needed

    with rasterio.open(
        save_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=prediction.shape[0],
        dtype=prediction.dtype.name,
    ) as dst:
        for i in range(prediction.shape[0]):
            dst.write(prediction[i], i + 1)

def inference(config):
    cfg = config
    logger = get_logger("inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator().to(device)
    checkpoint = torch.load(cfg["inference"]["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()

    input_dir = Path(cfg["inference"]["modis_input_dir"])
    output_dir = Path(cfg["inference"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for modis_path in input_dir.glob("*.npy"):
        seq = load_modis_sequence(modis_path).unsqueeze(0).to(device)  # (1, T, C, H, W)
        with torch.no_grad():
            pred = model(seq)
        save_prediction(pred.squeeze(0), output_dir / modis_path.name.replace(".npy", "_sr.tif"))
        logger.info(f"Saved prediction for {modis_path.name}")

if __name__ == "__main__":
    inference()
