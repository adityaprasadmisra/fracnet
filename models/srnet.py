
import torch
import torch.nn as nn


class SRNet(nn.Module):

    def __init__(self, in_channels=8, out_channels=8):
        super().__init__()

        self.height = 12
        self.width = 12
        self.out_channels = out_channels

        # Convolutional feature extraction
        self.features = nn.Sequential(

            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Two fully connected layers
        self.fc1 = nn.Linear(32 * 12 * 12, 512)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(
            512,
            out_channels * 12 * 12
        )

    def forward(self, x):

        # [B, 8, 12, 12]
        x = self.features(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Dense layer 1
        x = self.relu(self.fc1(x))

        # Dense layer 2
        x = self.fc2(x)

        # Reshape
        x = x.view(
            -1,
            self.out_channels,
            self.height,
            self.width
        )

        return x


# Test SRNet
if __name__ == "__main__":

    model = SRNet()

    dummy_input = torch.randn(16, 8, 12, 12)

    output = model(dummy_input)

    print("SRNet verification")
    print("-" * 40)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", parameters)

    print("\nSRNet working successfully!")
