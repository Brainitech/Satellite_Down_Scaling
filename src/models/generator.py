import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels, act=False)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_frames=5, base_channels=64, num_res_blocks=8):
        super().__init__()
        self.conv3d = nn.Conv3d(1, base_channels, kernel_size=(input_frames, 3, 3), padding=(0, 1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.encoder = ConvBlock(base_channels, base_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_res_blocks)])
        self.decoder = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, 3, padding=1)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.relu(self.conv3d(x)).squeeze(2)  # (B, F, H, W)
        x = self.encoder(x)
        x = self.res_blocks(x)
        out = self.decoder(x)
        return out
