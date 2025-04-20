import rasterio
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
def show_metadata(dataset):
    print("\n--- Metadata ---")
    print(f"Driver: {dataset.driver}")
    print(f"CRS: {dataset.crs}")
    print(f"Width, Height: {dataset.width}, {dataset.height}")
    print(f"Bounds: {dataset.bounds}")
    print(f"Transform: {dataset.transform}")
    print(f"Count (bands): {dataset.count}")
    print(f"Data Type: {dataset.dtypes}")
    print(f"Metadata Tags: {dataset.tags()}")

    # Additional metadata details
    print("\n--- Band Information ---")
    for i in range(1, dataset.count + 1):
        band = dataset.read(i)
        print(f"Band {i}:")
        print(f"  Min: {band.min()}")
        print(f"  Max: {band.max()}")
        print(f"  Mean: {band.mean()}")
        print(f"  Std Dev: {band.std()}")

    print("\n--- Coordinate Reference System Details ---")
    if dataset.crs:
        print(f"  EPSG: {dataset.crs.to_epsg()}")
        print(f"  WKT: {dataset.crs.to_wkt()}")

    print("\n--- Area Information ---")
    area = dataset.tags().get('Area', 'N/A')
    print(f"  Area: {area} square meters")

def show_band(dataset, band=1):
    data = dataset.read(band)
    plt.imshow(data, cmap='hot')  # Use 'hot' colormap for heatmap
    plt.title(f"Band {band} (Heatmap)")
    plt.colorbar(label="Temperature")
    plt.show()

def interactive_pixel_reader(dataset):
    print("\nClick on the image to get pixel values")

    fig, ax = plt.subplots()
    img = dataset.read(1)
    cax = ax.imshow(img, cmap='gray')
    fig.colorbar(cax)

    def onclick(event):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            values = [dataset.read(b+1)[y, x] for b in range(dataset.count)]
            print(f"\nPixel at ({x}, {y}) -> Band values: {values}")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click to inspect pixel values")
    plt.show()


def save_pixel_values_to_npz(dataset, output_npz):
    print("\nSaving pixel values to NPZ...")

    # Read the first band to get the dimensions
    band_data = dataset.read(1)
    height, width = band_data.shape

    # Create a dictionary to store pixel values for each band
    pixel_data = {}

    # Iterate over each band and replace sentinel values with NaN
    for b in range(dataset.count):
        band = dataset.read(b + 1).astype(float)  # Convert to float for NaN support
        band[band == dataset.nodata] = 0  # Replace sentinel value with NaN
        pixel_data[f"Band_{b + 1}"] = band

    # Save the pixel data to an NPZ file
    np.savez_compressed(output_npz, **pixel_data)

    print(f"Pixel values saved to {output_npz}")

def main(file_path):
    if not os.path.isfile(file_path):
        print("File not found.")
        return

    with rasterio.open(file_path) as dataset:
        show_metadata(dataset)

        for i in range(1, min(dataset.count + 1, 4)):  # Show up to 3 bands
            show_band(dataset, i)

        interactive_pixel_reader(dataset)

        output_npz=os.path.splitext(file_path)[0] + "_pixel_values.npz"
        save_pixel_values_to_npz(dataset, output_npz)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_tif.py path_to_file.tif")
    else:
        main(sys.argv[1])
