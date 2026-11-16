
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.fracnet import FracNet
from training.dataset_loader import get_dataloaders

DEVICE = torch.device("cpu")

# Load the same deterministic train/validation/test split
_, val_loader, _ = get_dataloaders(
    file_path=str(PROJECT_DIR / "dataset" / "fracnet_train.mat"),
    batch_size=16
)

# Load best checkpoint
model = FracNet().to(DEVICE)

checkpoint = torch.load(
    PROJECT_DIR / "checkpoints" / "fracnet_best.pth",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

inputs_all = []
targets_all = []
corrections_all = []

with torch.no_grad():

    for inputs, targets in val_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(inputs)

        correction = predictions - inputs

        inputs_all.append(inputs)
        targets_all.append(targets)
        corrections_all.append(correction)

X = torch.cat(inputs_all)
Y = torch.cat(targets_all)
D = torch.cat(corrections_all)

# Actual correction required
true_correction = Y - X

# Least-squares scalar calibration
numerator = torch.sum(D * true_correction)
denominator = torch.sum(D ** 2)

alpha = (numerator / (denominator + 1e-12)).item()

# Baseline validation NMSE
baseline_nmse = (
    torch.sum((X - Y) ** 2) /
    torch.sum(Y ** 2)
).item()

# Calibrated prediction
calibrated = X + alpha * D

calibrated_nmse = (
    torch.sum((calibrated - Y) ** 2) /
    torch.sum(Y ** 2)
).item()

# Original prediction
original_nmse = (
    torch.sum((X + D - Y) ** 2) /
    torch.sum(Y ** 2)
).item()

print("\n--- VALIDATION CALIBRATION ---")

print("Optimal alpha:", alpha)

print("Baseline NMSE:", baseline_nmse)

print("Original FracNet NMSE:", original_nmse)

print("Calibrated FracNet NMSE:", calibrated_nmse)

print("\n--- NMSE IN dB ---")

print("Baseline:",
      10 * torch.log10(torch.tensor(baseline_nmse)).item())

print("Original FracNet:",
      10 * torch.log10(torch.tensor(original_nmse)).item())

print("Calibrated FracNet:",
      10 * torch.log10(torch.tensor(calibrated_nmse)).item())

if calibrated_nmse < baseline_nmse:
    print("\nCalibration improves validation performance.")
else:
    print("\nCalibration does not beat the baseline.")
