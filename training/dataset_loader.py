
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split


def read_complex(dataset):
    data = np.array(dataset)

    if data.dtype.names and "real" in data.dtype.names:
        return data["real"] + 1j * data["imag"]

    return data


class FracNetDataset(Dataset):

    def __init__(self, file_path):

        with h5py.File(file_path, "r") as f:

            H_est = read_complex(f["H_est"])
            Yclean = read_complex(f["Yclean"])

        # Convert complex arrays into real and imaginary channels.
        # Initial shape: [samples, frames, height, width]
        # Final shape:   [samples, frames * 2, height, width]

        X = np.stack(
            [H_est.real, H_est.imag],
            axis=2
        )

        Y = np.stack(
            [Yclean.real, Yclean.imag],
            axis=2
        )

        samples = X.shape[0]

        X = X.reshape(samples, 8, 12, 12)
        Y = Y.reshape(samples, 8, 12, 12)

        self.X = torch.from_numpy(
            X.astype(np.float32)
        )

        self.Y = torch.from_numpy(
            Y.astype(np.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_dataloaders(
    file_path="dataset/fracnet_train.mat",
    batch_size=16
):

    dataset = FracNetDataset(file_path)

    total = len(dataset)

    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
