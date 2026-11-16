
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.fracnet import FracNet
from training.dataset_loader import get_dataloaders

DEVICE = torch.device("cpu")

# Alpha obtained exclusively from validation data
ALPHA = 0.0018385910661891103

_, _, test_loader = get_dataloaders(
    file_path=str(PROJECT_DIR / "dataset" / "fracnet_train.mat"),
    batch_size=16
)

model = FracNet().to(DEVICE)

checkpoint = torch.load(
    PROJECT_DIR / "checkpoints" / "fracnet_best.pth",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_baseline = []
all_original = []
all_calibrated = []
all_targets = []

with torch.no_grad():

    for inputs, targets in test_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(inputs)

        correction = predictions - inputs

        calibrated = inputs + ALPHA * correction

        all_baseline.append(inputs)
        all_original.append(predictions)
        all_calibrated.append(calibrated)
        all_targets.append(targets)


baseline = torch.cat(all_baseline)
original = torch.cat(all_original)
calibrated = torch.cat(all_calibrated)
targets = torch.cat(all_targets)


def nmse(prediction, target):

    error = torch.sum((prediction - target) ** 2)
    power = torch.sum(target ** 2)

    value = error / (power + 1e-12)

    return value.item()


def to_db(value):
    return 10 * np.log10(value + 1e-12)


baseline_nmse = nmse(baseline, targets)
original_nmse = nmse(original, targets)
calibrated_nmse = nmse(calibrated, targets)

print("\n========================================")
print("CALIBRATED FRACNET TEST RESULTS")
print("========================================")

print("Test samples:", len(targets))
print("Alpha:", ALPHA)

print("\nBaseline NMSE:", baseline_nmse)
print("Baseline dB:", to_db(baseline_nmse))

print("\nOriginal FracNet NMSE:", original_nmse)
print("Original FracNet dB:", to_db(original_nmse))

print("\nCalibrated FracNet NMSE:", calibrated_nmse)
print("Calibrated FracNet dB:", to_db(calibrated_nmse))

improvement = to_db(baseline_nmse) - to_db(calibrated_nmse)

print("\nNMSE improvement:", improvement, "dB")

if improvement > 0:
    print("Calibrated FracNet beats the baseline.")
else:
    print("Calibrated FracNet does not beat the baseline.")

print("\nEvaluation completed!")
