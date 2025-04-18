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

def show_band(dataset, band=1):
    data = dataset.read(band)
    plt.imshow(data, cmap='gray')
    plt.title(f"Band {band}")
    plt.colorbar()
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

def main(file_path):
    if not os.path.isfile(file_path):
        print("File not found.")
        return

    with rasterio.open(file_path) as dataset:
        show_metadata(dataset)

        for i in range(1, min(dataset.count + 1, 4)):  # Show up to 3 bands
            show_band(dataset, i)

        interactive_pixel_reader(dataset)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_tif.py path_to_file.tif")
    else:
        main(sys.argv[1])
