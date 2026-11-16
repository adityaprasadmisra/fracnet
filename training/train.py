
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ======================================
# PROJECT PATH
# ======================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.fracnet import FracNet
from training.dataset_loader import get_dataloaders


# ======================================
# CONFIGURATION
# ======================================

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_PATH = PROJECT_DIR / "dataset" / "fracnet_train.mat"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ======================================
# LOAD DATA
# ======================================

print("=" * 55)
print("FRACNET TRAINING")
print("=" * 55)

print("\nLoading dataset...")

train_loader, val_loader, test_loader = get_dataloaders(
    file_path=str(DATASET_PATH),
    batch_size=BATCH_SIZE
)

print("Training samples:", len(train_loader.dataset))
print("Validation samples:", len(val_loader.dataset))
print("Test samples:", len(test_loader.dataset))

print("Device:", DEVICE)


# ======================================
# INITIALIZE MODEL
# ======================================

model = FracNet().to(DEVICE)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nModel parameters:",
      sum(p.numel() for p in model.parameters()))


# ======================================
# TRAINING
# ======================================

train_losses = []
val_losses = []

best_val_loss = float("inf")

print("\nStarting FracNet training...")
print("=" * 55)


for epoch in range(EPOCHS):

    # ------------------------------
    # TRAINING
    # ------------------------------

    model.train()

    running_train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)

    epoch_train_loss = (
        running_train_loss / len(train_loader.dataset)
    )

    # ------------------------------
    # VALIDATION
    # ------------------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            running_val_loss += loss.item() * inputs.size(0)

    epoch_val_loss = (
        running_val_loss / len(val_loader.dataset)
    )

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {epoch_train_loss:.8f} "
        f"Val Loss: {epoch_val_loss:.8f}"
    )

    # ------------------------------
    # SAVE BEST CHECKPOINT
    # ------------------------------

    if epoch_val_loss < best_val_loss:

        best_val_loss = epoch_val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss
            },
            CHECKPOINT_DIR / "fracnet_best.pth"
        )

        print("Best model checkpoint saved!")


# ======================================
# SAVE FINAL MODEL
# ======================================

torch.save(
    model.state_dict(),
    CHECKPOINT_DIR / "fracnet_final.pth"
)


# ======================================
# PLOT LOSS CURVES
# ======================================

plt.figure(figsize=(9, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("FracNet Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_curve.png",
    dpi=300
)

plt.close()


# ======================================
# TRAINING SUMMARY
# ======================================

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)

print("Best validation loss:", best_val_loss)

print("Best checkpoint:",
      CHECKPOINT_DIR / "fracnet_best.pth")

print("Final model:",
      CHECKPOINT_DIR / "fracnet_final.pth")

print("Loss curve:",
      RESULTS_DIR / "loss_curve.png")
