
import os
import sys
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT, "dataset", "fracnet_train.mat")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "fracnet_best.pth")
RESULTS_DIR = os.path.join(ROOT, "results", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.append(ROOT)

from models.fracnet import FracNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# LOAD MATLAB COMPLEX DATA (MATLAB V7.3)
# --------------------------------------------------

def read_complex(x):
    x = np.asarray(x)

    if x.dtype.fields is not None:
        return x["real"] + 1j * x["imag"]

    return x


with h5py.File(DATA_PATH, "r") as f:
    H_est = read_complex(np.array(f["H_est"]))
    Yclean = read_complex(np.array(f["Yclean"]))
    snr_values = np.array(f["SNR_dB"]).flatten()

print("Dataset loaded successfully.")
print("H_est shape:", H_est.shape)
print("Yclean shape:", Yclean.shape)
print("SNR shape:", snr_values.shape)

# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

# MATLAB storage: samples, frames, N, M
# PyTorch format: samples, 8, N, M

X = np.stack([H_est.real, H_est.imag], axis=2)
Y = np.stack([Yclean.real, Yclean.imag], axis=2)

X = X.reshape(X.shape[0], 8, 12, 12)
Y = Y.reshape(Y.shape[0], 8, 12, 12)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

if len(snr_values) != len(X):
    raise ValueError("SNR array does not match dataset size.")

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = FracNet().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint)
model.eval()

X_device = X.to(DEVICE)

with torch.no_grad():
    predictions = model(X_device).cpu().numpy()

X = X.numpy()
Y = Y.numpy()

print("Model loaded successfully.")
print("Prediction shape:", predictions.shape)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_nmse(pred, target):
    error = np.sum((pred - target) ** 2)
    power = np.sum(target ** 2) + 1e-12
    return error / power


def calculate_rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def calculate_mae(pred, target):
    return np.mean(np.abs(pred - target))


def magnitude(data):
    real = data[:, 0::2]
    imag = data[:, 1::2]
    return np.sqrt(real ** 2 + imag ** 2)


pred_mag = magnitude(predictions)
true_mag = magnitude(Y)
input_mag = magnitude(X)

# --------------------------------------------------
# 1. 3D CHANNEL SCATTER PLOT
# --------------------------------------------------

sample = 0
frame = 0

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

n, m = np.meshgrid(
    np.arange(12),
    np.arange(12),
    indexing="ij"
)

ax.scatter(
    m.flatten(),
    n.flatten(),
    true_mag[sample, frame].flatten(),
    label="Ground Truth",
    marker="o"
)

ax.scatter(
    m.flatten(),
    n.flatten(),
    pred_mag[sample, frame].flatten(),
    label="FracNet Prediction",
    marker="^"
)

ax.set_xlabel("Doppler Index")
ax.set_ylabel("Delay Index")
ax.set_zlabel("Channel Magnitude")
ax.set_title("3D Channel Response Comparison")

ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "3d_channel_scatter.png"))
plt.close()

# --------------------------------------------------
# 2. CHANNEL HEATMAPS
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

images = [
    input_mag[sample, frame],
    true_mag[sample, frame],
    pred_mag[sample, frame]
]

titles = [
    "Coarse Estimate",
    "Ground Truth",
    "FracNet Prediction"
]

for ax, data, title in zip(axes, images, titles):
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Doppler Index")
    ax.set_ylabel("Delay Index")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "channel_heatmaps.png"))
plt.close()

# --------------------------------------------------
# 3. ERROR HEATMAP
# --------------------------------------------------

error_map = np.abs(
    pred_mag[sample, frame] -
    true_mag[sample, frame]
)

plt.figure(figsize=(7, 6))

plt.imshow(error_map, cmap="hot", aspect="auto")
plt.colorbar(label="Absolute Magnitude Error")

plt.xlabel("Doppler Index")
plt.ylabel("Delay Index")
plt.title("FracNet Channel Estimation Error")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_heatmap.png"))
plt.close()

# --------------------------------------------------
# 4. NMSE VS SNR
# --------------------------------------------------

unique_snr = np.sort(np.unique(snr_values))

baseline_nmse = []
fracnet_nmse = []

for snr in unique_snr:
    indices = np.where(snr_values == snr)[0]

    baseline_nmse.append(
        calculate_nmse(X[indices], Y[indices])
    )

    fracnet_nmse.append(
        calculate_nmse(predictions[indices], Y[indices])
    )

baseline_db = 10 * np.log10(np.maximum(baseline_nmse, 1e-12))
fracnet_db = 10 * np.log10(np.maximum(fracnet_nmse, 1e-12))

plt.figure(figsize=(8, 6))

plt.plot(
    unique_snr,
    baseline_db,
    marker="o",
    label="MATLAB Coarse Estimator"
)

plt.plot(
    unique_snr,
    fracnet_db,
    marker="s",
    label="FracNet"
)

plt.xlabel("SNR (dB)")
plt.ylabel("NMSE (dB)")
plt.title("NMSE vs SNR")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "nmse_vs_snr.png"))
plt.close()

# --------------------------------------------------
# 5. RMSE VS SNR
# --------------------------------------------------

baseline_rmse = []
fracnet_rmse = []

for snr in unique_snr:
    indices = np.where(snr_values == snr)[0]

    baseline_rmse.append(
        calculate_rmse(X[indices], Y[indices])
    )

    fracnet_rmse.append(
        calculate_rmse(predictions[indices], Y[indices])
    )

plt.figure(figsize=(8, 6))

plt.plot(
    unique_snr,
    baseline_rmse,
    marker="o",
    label="MATLAB Coarse Estimator"
)

plt.plot(
    unique_snr,
    fracnet_rmse,
    marker="s",
    label="FracNet"
)

plt.xlabel("SNR (dB)")
plt.ylabel("RMSE")
plt.title("RMSE vs SNR")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "rmse_vs_snr.png"))
plt.close()

# --------------------------------------------------
# 6. PREDICTION VS GROUND TRUTH
# --------------------------------------------------

plt.figure(figsize=(7, 7))

plt.scatter(
    true_mag.flatten(),
    pred_mag.flatten(),
    alpha=0.25
)

max_value = max(true_mag.max(), pred_mag.max())

plt.plot(
    [0, max_value],
    [0, max_value],
    "r--",
    label="Ideal Prediction"
)

plt.xlabel("Ground Truth Magnitude")
plt.ylabel("Predicted Magnitude")
plt.title("FracNet Prediction vs Ground Truth")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "prediction_scatter.png"))
plt.close()

# --------------------------------------------------
# 7. ERROR DISTRIBUTION
# --------------------------------------------------

errors = np.abs(predictions - Y)

plt.figure(figsize=(8, 6))

plt.hist(errors.flatten(), bins=50)

plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("FracNet Error Distribution")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))
plt.close()

# --------------------------------------------------
# 8. PRINT EVALUATION METRICS
# --------------------------------------------------

print("\n========== FRACNET EVALUATION ==========")

print("Samples:", len(X))

print("Baseline NMSE:",
      calculate_nmse(X, Y))

print("FracNet NMSE:",
      calculate_nmse(predictions, Y))

print("Baseline NMSE (dB):",
      10 * np.log10(calculate_nmse(X, Y) + 1e-12))

print("FracNet NMSE (dB):",
      10 * np.log10(calculate_nmse(predictions, Y) + 1e-12))

print("Baseline RMSE:",
      calculate_rmse(X, Y))

print("FracNet RMSE:",
      calculate_rmse(predictions, Y))

print("FracNet MAE:",
      calculate_mae(predictions, Y))

print("\nPlots saved in:")
print(RESULTS_DIR)

print("\n========== COMPLETED ==========")
