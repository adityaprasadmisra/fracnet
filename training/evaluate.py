
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.fracnet import FracNet
from training.dataset_loader import get_dataloaders


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = PROJECT_DIR / "checkpoints" / "fracnet_best.pth"


def calculate_nmse(prediction, target):

    error_power = torch.sum((prediction - target) ** 2)
    signal_power = torch.sum(target ** 2)

    return (error_power / (signal_power + 1e-12)).item()


def calculate_sample_nmse(prediction, target):

    error_power = torch.sum(
        (prediction - target) ** 2,
        dim=(1, 2, 3)
    )

    signal_power = torch.sum(
        target ** 2,
        dim=(1, 2, 3)
    )

    return error_power / (signal_power + 1e-12)


# Load test data
_, _, test_loader = get_dataloaders(
    file_path=str(PROJECT_DIR / "dataset" / "fracnet_train.mat"),
    batch_size=16
)

# Load model
model = FracNet().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["val_loss"])

all_predictions = []
all_targets = []
all_inputs = []

# Evaluate
with torch.no_grad():

    for inputs, targets in test_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(inputs)

        all_inputs.append(inputs.cpu())
        all_targets.append(targets.cpu())
        all_predictions.append(predictions.cpu())


inputs = torch.cat(all_inputs, dim=0)
targets = torch.cat(all_targets, dim=0)
predictions = torch.cat(all_predictions, dim=0)

# Baseline: coarse estimator
baseline_nmse = calculate_nmse(inputs, targets)

# FracNet prediction
fracnet_nmse = calculate_nmse(predictions, targets)

baseline_db = 10 * np.log10(baseline_nmse)
fracnet_db = 10 * np.log10(fracnet_nmse)

# Per-sample NMSE
baseline_per_sample = calculate_sample_nmse(inputs, targets)
fracnet_per_sample = calculate_sample_nmse(predictions, targets)

print("\n" + "=" * 50)
print("FRACNET TEST EVALUATION")
print("=" * 50)

print("Test samples:", len(targets))

print("\n--- BASELINE: H_est ---")
print("Global NMSE:", baseline_nmse)
print("Global NMSE (dB):", baseline_db)
print("Average sample NMSE:",
      baseline_per_sample.mean().item())

print("\n--- FRACNET ---")
print("Global NMSE:", fracnet_nmse)
print("Global NMSE (dB):", fracnet_db)
print("Average sample NMSE:",
      fracnet_per_sample.mean().item())

improvement = baseline_db - fracnet_db

print("\n--- COMPARISON ---")
print("NMSE improvement (dB):", improvement)

if improvement > 0:
    print("FracNet improves over the baseline.")
elif improvement < 0:
    print("FracNet performs worse than the baseline.")
else:
    print("Both methods have equal global NMSE.")

print("\nEvaluation completed!")
