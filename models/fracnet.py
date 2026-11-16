
import torch
import torch.nn as nn

from models.srnet import SRNet
from models.stnet import STNet


class FracNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Stage 1: Feature extraction and reconstruction
        self.srnet = SRNet(
            in_channels=8,
            out_channels=8
        )

        # Stage 2: Learn the channel correction
        self.stnet = STNet(
            in_channels=8,
            out_channels=8
        )

    def forward(self, x):

        # Preserve original coarse estimate
        original = x

        # Stage 1
        features = self.srnet(x)

        # Stage 2: Predict correction
        correction = self.stnet(features)

        # Global residual learning
        output = original + correction

        return output


# =====================================
# TEST FRACNET
# =====================================

if __name__ == "__main__":

    model = FracNet()

    dummy_input = torch.randn(16, 8, 12, 12)

    output = model(dummy_input)

    print("FracNet Verification")
    print("-" * 40)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )

    print("\nFracNet working successfully!")
