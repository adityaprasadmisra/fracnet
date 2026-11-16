
import h5py
import os

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "fracnet_train.mat"
)

with h5py.File(DATA_PATH, "r") as f:

    print("\n========== MATLAB DATASET VARIABLES ==========")

    for key in f.keys():
        print("\nVariable:", key)

        item = f[key]

        if isinstance(item, h5py.Dataset):
            print("Shape:", item.shape)
            print("Data type:", item.dtype)

            if item.size < 20:
                print("Values:", item[()])
        else:
            print("Group:", list(item.keys()))

    print("\n========== COMPLETED ==========")
