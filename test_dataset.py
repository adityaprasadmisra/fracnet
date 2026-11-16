
import h5py
import numpy as np

# Path to MATLAB v7.3 dataset
file_path = "dataset/otfs_dataset.mat"

# Read MATLAB v7.3 file
with h5py.File(file_path, "r") as data:

    X_train = np.array(data["XTrain"])
    Y_train = np.array(data["YTrain"])

    X_val = np.array(data["XValidation"])
    Y_val = np.array(data["YValidation"])

    X_test = np.array(data["XTest"])
    Y_test = np.array(data["YTest"])

# MATLAB v7.3 dimensions are generally reversed
# by h5py, so check before transposing.

print("Raw training input:", X_train.shape)
print("Raw training target:", Y_train.shape)

# Convert only if needed
if X_train.shape[0:3] == (12, 12, 2):
    X_train = np.transpose(X_train, (3, 2, 0, 1))
    Y_train = np.transpose(Y_train, (3, 2, 0, 1))

    X_val = np.transpose(X_val, (3, 2, 0, 1))
    Y_val = np.transpose(Y_val, (3, 2, 0, 1))

    X_test = np.transpose(X_test, (3, 2, 0, 1))
    Y_test = np.transpose(Y_test, (3, 2, 0, 1))

print("\nFinal dataset dimensions:")
print("Training input:", X_train.shape)
print("Training target:", Y_train.shape)

print("Validation input:", X_val.shape)
print("Validation target:", Y_val.shape)

print("Testing input:", X_test.shape)
print("Testing target:", Y_test.shape)

# Check for invalid values
print("\nNaN in training input:", np.isnan(X_train).any())
print("NaN in training target:", np.isnan(Y_train).any())

print("\nDataset loading completed!")
