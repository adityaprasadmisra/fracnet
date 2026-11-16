
import torch
import numpy as np

from training.dataset_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    file_path="dataset/fracnet_train.mat",
    batch_size=16
)

all_inputs = []
all_targets = []

for inputs, targets in test_loader:
    all_inputs.append(inputs)
    all_targets.append(targets)

X = torch.cat(all_inputs)
Y = torch.cat(all_targets)

# Direct baseline error
baseline_mse = torch.mean((X - Y) ** 2)

# Target power
target_power = torch.mean(Y ** 2)

# Baseline NMSE
baseline_nmse = torch.sum((X-Y)**2) / torch.sum(Y**2)

print("\n--- DATA DIAGNOSTICS ---")

print("Input mean:", X.mean().item())
print("Target mean:", Y.mean().item())

print("Input std:", X.std().item())
print("Target std:", Y.std().item())

print("Baseline MSE:", baseline_mse.item())
print("Target power:", target_power.item())

print("Baseline NMSE:", baseline_nmse.item())
print(
    "Baseline NMSE dB:",
    10 * np.log10(baseline_nmse.item())
)

print(
    "Mean absolute input-target difference:",
    torch.mean(torch.abs(X-Y)).item()
)

print("\nDiagnostics completed!")
