
import os
import h5py
import numpy as np
from scipy.io import loadmat
from pathlib import Path

# Project dataset directory
DATA_DIR = Path(__file__).resolve().parent / "dataset"

print("=" * 60)
print("          FRACNET MATLAB DATASET INSPECTOR")
print("=" * 60)

# Required MATLAB variables
required_variables = [
    "XTrain",
    "YTrain",
    "XValidation",
    "YValidation",
    "XTest",
    "YTest"
]

# Find all files in dataset folder
files = list(DATA_DIR.rglob("*"))

mat_files = [
    f for f in files
    if f.is_file() and f.suffix.lower() == ".mat"
]

print(f"\nDataset folder: {DATA_DIR}")
print(f"MATLAB files found: {len(mat_files)}")

if not mat_files:
    print("\nNo .mat files found!")
    print("Place your MATLAB dataset inside the dataset folder.")

else:

    for file_path in mat_files:

        print("\n" + "=" * 60)
        print("FILE:", file_path.name)
        print("SIZE:", round(file_path.stat().st_size / (1024**2), 2), "MB")

        data = None

        # Try MATLAB v7 format
        try:
            data = loadmat(file_path)
            print("Format: MATLAB v7 or compatible")

        except NotImplementedError:

            print("Format: MATLAB v7.3 (HDF5)")

            with h5py.File(file_path, "r") as f:
                data = {
                    key: np.array(f[key])
                    for key in f.keys()
                    if not key.startswith("#")
                }

        except Exception as e:
            print("Could not read file:", e)
            continue

        print("\nVARIABLES FOUND:")

        available = []

        for key, value in data.items():

            if key.startswith("__"):
                continue

            available.append(key)

            if isinstance(value, np.ndarray):
                print(
                    f"{key:20} "
                    f"Shape: {str(value.shape):25} "
                    f"Type: {value.dtype}"
                )
            else:
                print(key, type(value))

        print("\nREQUIRED VARIABLE CHECK:")

        for variable in required_variables:

            if variable in available:
                print(f"[FOUND] {variable}")
            else:
                print(f"[MISSING] {variable}")

        print("\nDATASET VALIDATION:")

        if all(v in available for v in required_variables):

            print("All six required variables are present.")

            for x_name, y_name in [
                ("XTrain", "YTrain"),
                ("XValidation", "YValidation"),
                ("XTest", "YTest")
            ]:

                x = np.asarray(data[x_name])
                y = np.asarray(data[y_name])

                print(f"\n{x_name} vs {y_name}")

                print("Input shape :", x.shape)
                print("Target shape:", y.shape)

                print("Input finite:", np.isfinite(x).all())
                print("Target finite:", np.isfinite(y).all())

                if x.shape == y.shape:
                    print("Shapes match: YES")
                else:
                    print("Shapes match: NO")

        else:
            print("This file does not contain all required variables.")

print("\n" + "=" * 60)
print("INSPECTION FINISHED")
print("=" * 60)
