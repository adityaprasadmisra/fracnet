
import h5py
import numpy as np

file_path = "dataset/fracnet_train.mat"


def read_complex(dataset):
    """Convert MATLAB HDF5 real/imag fields into NumPy complex arrays."""
    data = np.array(dataset)

    if data.dtype.names and "real" in data.dtype.names:
        return data["real"] + 1j * data["imag"]

    return data


with h5py.File(file_path, "r") as f:

    print("Available datasets:")
    for key in f.keys():
        print(key, f[key].shape, f[key].dtype)

    H_est = read_complex(f["H_est"])
    Yclean = read_complex(f["Yclean"])
    Xpilot = read_complex(f["Xpilot"])
    Ynoisy = read_complex(f["Ynoisy"])

    SNR = np.array(f["SNR_dB"]).flatten()

    print("\n--- SHAPES ---")
    print("H_est:", H_est.shape)
    print("Yclean:", Yclean.shape)
    print("Xpilot:", Xpilot.shape)
    print("Ynoisy:", Ynoisy.shape)

    print("\n--- DATA TYPES ---")
    print("H_est:", H_est.dtype)
    print("Yclean:", Yclean.dtype)

    print("\n--- INVALID VALUES ---")
    print("H_est NaN:", np.isnan(H_est).sum())
    print("H_est Inf:", np.isinf(H_est).sum())

    print("Yclean NaN:", np.isnan(Yclean).sum())
    print("Yclean Inf:", np.isinf(Yclean).sum())

    print("\n--- SNR DISTRIBUTION ---")
    print(np.unique(SNR))

    print("\n--- MAGNITUDE STATISTICS ---")
    print("H_est mean:", np.abs(H_est).mean())
    print("Yclean mean:", np.abs(Yclean).mean())

    print("\nDataset inspection completed!")
