
import torch

from models.stnet import STNet
from models.srnet import SRNet

# Create both networks
stnet = STNet()
srnet = SRNet()

# Sample OTFS input
x = torch.randn(64, 2, 12, 12)

# Extract spatial features
features = stnet(x)

# Refine features and reconstruct channel
output = srnet(features)

print("Input shape:", x.shape)
print("STNet output:", features.shape)
print("SRNet output:", output.shape)
