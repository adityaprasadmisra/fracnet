
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "training"))

from dataset_loader import load_channel_data

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_channel_data()


def analyse(name, data):

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    # Convert two channels into complex matrix
    real = data[:, 0]
    imag = data[:, 1]

    magnitude = np.sqrt(real**2 + imag**2)

    print("Shape:", data.shape)
    print("Real range:", real.min().item(), real.max().item())
    print("Imag range:", imag.min().item(), imag.max().item())
    print("Magnitude range:", magnitude.min().item(), magnitude.max().item())
    print("Mean magnitude:", magnitude.mean().item())
    print("Std magnitude:", magnitude.std().item())

    nonzero = (magnitude > 1e-7).sum(dim=(1, 2))

    print("Average nonzero coefficients:", nonzero.float().mean().item())
    print("Minimum nonzero coefficients:", nonzero.min().item())
    print("Maximum nonzero coefficients:", nonzero.max().item())


analyse("INTEGER DOPPLER - TRAINING", X_train)
analyse("FRACTIONAL DOPPLER - TRAINING", Y_train)


# Compare input and target for the same samples
X_mag = torch_magnitude = np.sqrt(
    X_train[:, 0].numpy()**2 + X_train[:, 1].numpy()**2
)

Y_mag = np.sqrt(
    Y_train[:, 0].numpy()**2 + Y_train[:, 1].numpy()**2
)

print("\n" + "=" * 55)
print("INPUT VS TARGET")
print("=" * 55)

print("Mean input magnitude:", X_mag.mean())
print("Mean target magnitude:", Y_mag.mean())

print("Mean absolute magnitude difference:",
      np.mean(np.abs(X_mag - Y_mag)))

print("Input-target pairs:", len(X_train))
