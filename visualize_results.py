
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.fracnet import FracNet
from training.dataset_loader import get_dataloaders


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "otfs_dataset.mat"
)

CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "checkpoints", "best_fracnet.pth"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================
# 2. LOAD TEST DATA
# =====================================

_, _, test_loader = get_dataloaders(
    DATASET_PATH,
    batch_size=1
)

# Select one test sample
inputs, targets = next(iter(test_loader))

inputs = inputs.to(DEVICE)
targets = targets.to(DEVICE)


# =====================================
# 3. LOAD TRAINED FRACNET
# =====================================

model = FracNet().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# =====================================
# 4. GENERATE PREDICTION
# =====================================

with torch.no_grad():
    predictions = model(inputs)


# =====================================
# 5. CONVERT TO NUMPY
# =====================================

input_data = inputs[0].cpu().numpy()
true_data = targets[0].cpu().numpy()
pred_data = predictions[0].cpu().numpy()

# Reconstruct complex channel matrices
H_input = input_data[0] + 1j * input_data[1]
H_true = true_data[0] + 1j * true_data[1]
H_pred = pred_data[0] + 1j * pred_data[1]

# Magnitude representations
input_magnitude = np.abs(H_input)
true_magnitude = np.abs(H_true)
pred_magnitude = np.abs(H_pred)

error_magnitude = np.abs(H_true - H_pred)


# =====================================
# 6. CALCULATE SAMPLE NMSE
# =====================================

nmse = (
    np.sum(np.abs(H_pred - H_true) ** 2)
    / (np.sum(np.abs(H_true) ** 2) + 1e-12)
)

nmse_db = 10 * np.log10(nmse + 1e-12)


# =====================================
# 7. VISUALIZE CHANNELS
# =====================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

images = [
    (input_magnitude, "Integer-Doppler Input"),
    (true_magnitude, "True Fractional-Doppler Channel"),
    (pred_magnitude, "FracNet Prediction"),
    (error_magnitude, "Absolute Reconstruction Error")
]

for ax, (matrix, title) in zip(axes.flat, images):

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower"
    )

    ax.set_title(title)
    ax.set_xlabel("Delay Bin")
    ax.set_ylabel("Doppler Bin")

    fig.colorbar(im, ax=ax)

fig.suptitle(
    f"OTFS Channel Reconstruction using FracNet | NMSE = {nmse_db:.2f} dB",
    fontsize=14
)

plt.tight_layout()

# Save figure
output_path = os.path.join(
    RESULTS_DIR,
    "fracnet_visualization.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nVisualization completed!")
print("Sample NMSE:", nmse)
print("Sample NMSE (dB):", nmse_db)
print("Saved figure:", output_path)
