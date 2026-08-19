"""Rescale bounding box annotations to match resized images.

Usage:
    python rescale_images.py \
        --annotations dataset/annotations_train.csv \
        --sizes dataset/train_meta.csv \
        --out-dir dataset
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Original bbox annotations CSV")
    parser.add_argument("--sizes", required=True, help="Original image dimensions CSV (train_meta.csv)")
    parser.add_argument("--out-dir", default="dataset", help="Directory to write scaled CSVs to")
    args = parser.parse_args()

    annotations_df = pd.read_csv(args.annotations)
    sizes_df = pd.read_csv(args.sizes)

    # Merge both on image_id
    df = annotations_df.merge(sizes_df, on="image_id", how="inner")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Scale to 256x256 ---
    scale_256 = df.copy()
    scale_256["x_min"] = df["x_min"] * 256 / df["dim0"]
    scale_256["x_max"] = df["x_max"] * 256 / df["dim0"]
    scale_256["y_min"] = df["y_min"] * 256 / df["dim1"]
    scale_256["y_max"] = df["y_max"] * 256 / df["dim1"]
    scale_256 = scale_256.drop(columns=["dim0", "dim1"])
    scale_256.to_csv(os.path.join(args.out_dir, "annotations_scaled_256.csv"), index=False)

    # --- Scale to 512 x original height (maintain aspect ratio) ---
    scale_512 = df.copy()
    scale_512["x_min"] = df["x_min"] * 512 / df["dim0"]
    scale_512["x_max"] = df["x_max"] * 512 / df["dim0"]
    scale_512["y_min"] = df["y_min"]  # height unchanged
    scale_512["y_max"] = df["y_max"]
    scale_512 = scale_512.drop(columns=["dim0", "dim1"])
    scale_512.to_csv(os.path.join(args.out_dir, "annotations_scaled_512.csv"), index=False)


if __name__ == "__main__":
    main()
