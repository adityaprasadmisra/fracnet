
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Allow importing dataset_loader
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "training"))

from dataset_loader import load_channel_data


# Load dataset
(
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test
) = load_channel_data()


# Select one sample
index = 0

H_int = X_train[index].numpy()
H_frac = Y_train[index].numpy()

# Reconstruct complex matrices
H_int_complex = H_int[0] + 1j * H_int[1]
H_frac_complex = H_frac[0] + 1j * H_frac[1]


# Calculate magnitudes
H_int_mag = np.abs(H_int_complex)
H_frac_mag = np.abs(H_frac_complex)


# Plot
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

im1 = axes[0, 0].imshow(H_int_mag, aspect="auto")
axes[0, 0].set_title("Integer Doppler - Input Magnitude")
axes[0, 0].set_xlabel("Doppler index")
axes[0, 0].set_ylabel("Delay index")
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(H_frac_mag, aspect="auto")
axes[0, 1].set_title("Fractional Doppler - Ground Truth")
axes[0, 1].set_xlabel("Doppler index")
axes[0, 1].set_ylabel("Delay index")
plt.colorbar(im2, ax=axes[0, 1])

im3 = axes[1, 0].imshow(H_int_complex.real, aspect="auto")
axes[1, 0].set_title("Input Real Component")
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(H_frac_complex.real, aspect="auto")
axes[1, 1].set_title("Target Real Component")
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(PROJECT_DIR / "results" / "channel_comparison.png")
plt.show()


print("\nChannel inspection:")
print("Input magnitude range:", H_int_mag.min(), H_int_mag.max())
print("Target magnitude range:", H_frac_mag.min(), H_frac_mag.max())

print("Input mean magnitude:", H_int_mag.mean())
print("Target mean magnitude:", H_frac_mag.mean())

print("Input-target difference:",
      np.mean(np.abs(H_int_complex - H_frac_complex)))

print("\nSaved channel_comparison.png")
