
import torch

from dataset_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

X, Y = next(iter(train_loader))

print("\nInput shape:", X.shape)
print("Target shape:", Y.shape)

print("Input dtype:", X.dtype)
print("Target dtype:", Y.dtype)

print("Input NaN:", torch.isnan(X).sum().item())
print("Target NaN:", torch.isnan(Y).sum().item())
