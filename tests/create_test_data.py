"""Script to create small standalone OME-Zarr test images from an HCS zarr.

Run once to generate test data:
    pixi run python tests/create_test_data.py

Outputs two OME-Zarr containers (512x512 crops) for use with
microsam_segmentation_task tests.
"""

from pathlib import Path

import numpy as np
import zarr
from ngio import create_ome_zarr_from_array

# --- Configuration -----------------------------------------------------------
HCS_ZARR = Path("C:/Repos/test_data/exp164-diff8.zarr")
OUT_DIR = Path("C:/Repos/test_data/test_microsam")
CROP_SIZE = 512  # px; large enough for microSAM, small enough to be fast

# Two wells/fields to extract
SOURCES = [
    ("C", "03", "0"),
    ("C", "04", "0"),
]
# -----------------------------------------------------------------------------


# def crop_center(arr: np.ndarray, size: int) -> np.ndarray:
#     """Return a (size, size) crop from the centre of a 2-D array."""
#     cy, cx = arr.shape[-2] // 2, arr.shape[-1] // 2
#     half = size // 2
#     return arr[..., cy - half : cy + half, cx - half : cx + half]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_group = zarr.open_group(str(HCS_ZARR), mode="r")

    for i, (row, col, field) in enumerate(SOURCES):
        # Read full-resolution image array
        img_arr = src_group[row][col][field]["0"]
        data = np.array(img_arr)  # shape: (y, x) or (c, y, x)

        # Crop to a manageable size
        # data = crop_center(data, CROP_SIZE)

        # Ensure data is 2-D (y, x) before stacking channels
        if data.ndim == 3:
            data = data[0]  # take first channel if already multi-channel

        # Build a 2-channel array by duplicating the single plane
        data = np.stack([data, data], axis=0)  # shape: (2, y, x)
        axes = "cyx"
        channels_meta = ["channel_0", "channel_1"]

        out_path = OUT_DIR / f"image_{i + 1:02d}.zarr"

        create_ome_zarr_from_array(
            store=str(out_path),
            array=data,
            axes_names=list(axes),
            channels_meta=channels_meta,
            pixelsize=1.0,
            levels=3,
            overwrite=True,
        )
        print(f"Created {out_path}  shape={data.shape}  dtype={data.dtype}")

    print("\nDone. Test zarr paths:")
    for p in sorted(OUT_DIR.glob("image_*.zarr")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
