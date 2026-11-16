
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.fracnet import FracNet
from training.dataset_loader import get_dataloaders

DEVICE = torch.device("cpu")

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

inputs_all = []
targets_all = []
outputs_all = []

with torch.no_grad():

    for inputs, targets in test_loader:

        inputs = inputs.to(DEVICE)

        outputs = model(inputs)

        inputs_all.append(inputs)
        targets_all.append(targets)
        outputs_all.append(outputs)

X = torch.cat(inputs_all)
Y = torch.cat(targets_all)
P = torch.cat(outputs_all)

# Network's learned correction
correction = P - X

# Actual required correction
true_correction = Y - X

print("\n--- RESIDUAL DIAGNOSTICS ---")

print("Input RMS:",
      torch.sqrt(torch.mean(X**2)).item())

print("True correction RMS:",
      torch.sqrt(torch.mean(true_correction**2)).item())

print("Predicted correction RMS:",
      torch.sqrt(torch.mean(correction**2)).item())

print("Correction MSE:",
      torch.mean((correction - true_correction)**2).item())

print("Baseline MSE:",
      torch.mean((X-Y)**2).item())

print("FracNet MSE:",
      torch.mean((P-Y)**2).item())

print("Correction mean:",
      correction.mean().item())

print("Correction std:",
      correction.std().item())

print("\nDiagnostics completed!")
