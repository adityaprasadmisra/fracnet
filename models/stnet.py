
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, features=64):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                features, features,
                kernel_size=3, padding=1
            ),

            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features, features,
                kernel_size=3, padding=1
            ),

            nn.BatchNorm2d(features)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        x = self.block(x)

        x = x + residual

        return self.relu(x)


class STNet(nn.Module):

    def __init__(self, in_channels=8, out_channels=8):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial feature extraction
        self.initial = nn.Sequential(

            nn.Conv2d(
                in_channels, 64,
                kernel_size=3, padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # Residual feature learning
        self.residual_layers = nn.Sequential(

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # Correction reconstruction
        self.final = nn.Conv2d(
            64,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        features = self.initial(x)

        features = self.residual_layers(features)

        correction = self.final(features)

        return correction


# =====================================
# TEST STNET
# =====================================

if __name__ == "__main__":

    model = STNet()

    dummy_input = torch.randn(16, 8, 12, 12)

    output = model(dummy_input)

    print("STNet Verification")
    print("-" * 40)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", parameters)

    print("\nSTNet working successfully!")
