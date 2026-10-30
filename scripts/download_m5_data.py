"""Helper script to download the official M5 dataset from Kaggle.

Prerequisites:
- Kaggle API token (~/.kaggle/kaggle.json)
- Run: pip install kaggle
"""

import os
import sys
import subprocess
import zipfile


def download_m5_dataset(dest_dir: str = "data/raw"):
    os.makedirs(dest_dir, exist_ok=True)
    print("Checking Kaggle CLI installation...")
    try:
        subprocess.run(["kaggle", "--version"], check=True)
    except Exception:
        print("Kaggle CLI not installed or kaggle.json missing.")
        print("To download manually, visit: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
        print(f"Place 'calendar.csv', 'sell_prices.csv', and 'sales_train_validation.csv' into {dest_dir}")
        return

    print("Downloading M5 Forecasting - Accuracy competition data...")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "m5-forecasting-accuracy",
        "-p", dest_dir
    ], check=True)

    zip_path = os.path.join(dest_dir, "m5-forecasting-accuracy.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        os.remove(zip_path)
        print("Extraction complete.")


if __name__ == "__main__":
    download_m5_dataset()

