import pandas as pd

# Load original annotations and image sizes
annotations_df = pd.read_csv(r"C:\Users\jsayed\Downloads\DHBW\lung-disease-detection\dataset\annotations_train.csv")  # original bbox coords
sizes_df = pd.read_csv(r"C:\Users\jsayed\Downloads\DHBW\lung-disease-detection\dataset\train_meta.csv")        # original image dimensions

# Merge both on image_id
df = annotations_df.merge(sizes_df, on="image_id", how="inner")

# --- Scale to 256x256 ---
scale_256 = df.copy()
scale_256["x_min"] = df["x_min"] * 256 / df["dim0"]
scale_256["x_max"] = df["x_max"] * 256 / df["dim0"]
scale_256["y_min"] = df["y_min"] * 256 / df["dim1"]
scale_256["y_max"] = df["y_max"] * 256 / df["dim1"]
scale_256 = scale_256.drop(columns=["dim0", "dim1"])
scale_256.to_csv("dataset/annotations_scaled_256.csv", index=False)

# --- Scale to 512 × original height (maintain aspect ratio) ---
scale_512 = df.copy()
scale_512["x_min"] = df["x_min"] * 512 / df["dim0"]
scale_512["x_max"] = df["x_max"] * 512 / df["dim0"]
scale_512["y_min"] = df["y_min"]  # height unchanged
scale_512["y_max"] = df["y_max"]
scaled_512 = scale_512.drop(columns=['dim0', 'dim1'])
scaled_512.to_csv("dataset/annotations_scaled_512.csv", index=False)
