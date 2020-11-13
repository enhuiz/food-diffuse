import numpy as np
import torch
import torch.nn as nn

from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, dim, linear_scale=5000):
        super().__init__()
        self.dim = dim

        half_dim = dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32)
        exponents = exponents / half_dim

        omegas = linear_scale * 1e-4 ** exponents
        omegas = omegas.unsqueeze(0)

        self.register_buffer("omegas", omegas)

    def forward(self, positions):
        """
        Args:
            positions: (b) or (b 1)
        Returns:
            embeddings: (b d)
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(1)
        angles = positions * self.omegas
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoding = PositionalEncoding(in_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels * 2, 3, padding=1)

    def forward(self, x, noise_level):
        pe = self.encoding(noise_level)
        pe = rearrange(pe, "b d -> b d 1 1")
        x = self.conv1(x)
        x = self.conv2(x + pe)
        scale, shift = x.chunk(2, dim=1)
        return scale, shift


class FiLMBlock(nn.Sequential):
    def __init__(self, n_channels, dilation):
        super().__init__(
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
            ),
        )

    @staticmethod
    def featurewise_affine(x, scale, shift):
        return x * scale + shift

    def forward(self, x, scale, shift):
        x = self.featurewise_affine(x, scale, shift)
        return super().forward(x)


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor, dilations):
        super().__init__()
        self.block1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Upsample(
                        scale_factor=factor,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilations[0],
                        dilation=dilations[0],
                    ),
                ),
                FiLMBlock(out_channels, dilations[1]),
            ]
        )
        self.block2 = nn.Sequential(
            nn.Upsample(
                scale_factor=factor,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.block3 = nn.ModuleList(
            [
                FiLMBlock(
                    out_channels,
                    dilations[2 + i],
                )
                for i in range(2)
            ]
        )

    def forward(self, input, scale, shift):
        x = self.block1[0](input)
        x = self.block1[1](x, scale, shift)
        x = x + self.block2(input)

        identity = x
        x = self.block3[0](x, scale, shift)
        x = self.block3[1](x, scale, shift)
        x = x + identity

        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor, dilations):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=1 / factor, mode="nearest"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=1 / factor, mode="nearest"),
        )

    def forward(self, x):
        return self.block1(x) + self.block2(x)


class Network(nn.Module):
    def __init__(
        self,
        num_classes,
        class_dim,
        factors,
        up_pre_dim,
        up_post_dim,
        up_dims,
        up_dilations,
        down_pre_dim,
        down_dims,
        down_dilations,
        in_channels=3,
    ):
        super().__init__()

        # for noise size calculation
        self.up_post_dim = up_post_dim
        self.factors = factors

        assert len(up_dims) == len(down_dims) + 1

        assert class_dim % 16 == 0, "class_dim should be divisible by 16."

        input_nc = class_dim // 16

        self.embedding = nn.Embedding(num_classes, class_dim)

        self.downsamples = nn.ModuleList()
        self.downsamples.append(nn.Conv2d(up_post_dim, down_pre_dim, 3, padding=1))
        for i, out_channels in enumerate(down_dims):
            in_channels = down_pre_dim if i == 0 else down_dims[i - 1]
            self.downsamples.append(
                DBlock(in_channels, out_channels, factors[-i - 1], down_dilations[i])
            )

        self.upsamples = nn.ModuleList()
        self.upsamples.append(nn.Conv2d(input_nc, up_pre_dim, 3, padding=1))
        for i, out_channels in enumerate(up_dims):
            in_channels = up_pre_dim if i == 0 else up_dims[i - 1]
            self.upsamples.append(
                UBlock(in_channels, out_channels, factors[i], up_dilations[i])
            )
        self.upsamples.append(nn.Conv2d(up_dims[-1], up_post_dim, 3, padding=1))

        self.films = nn.ModuleList()
        for in_channels, out_channels in zip([down_pre_dim] + down_dims, up_dims[::-1]):
            self.films.append(FiLM(in_channels, out_channels))

    def generate_noise(self, n):
        total_factor = np.prod(self.factors)
        return torch.randn(n, self.up_post_dim, 4 * total_factor, 4 * total_factor)

    def forward(self, labels, yt, noise_level):
        labels = self.embedding(labels)  # (b) => (b, d)
        labels = rearrange(labels, "b (c h w) -> b c h w", h=4, w=4)

        x = yt
        stats = []
        for downsample, film in zip(self.downsamples, self.films):
            x = downsample(x)
            stats.append(film(x, noise_level))

        x = self.upsamples[0](labels)
        for upsample, (scale, shift) in zip(self.upsamples[1:-1], reversed(stats)):
            x = upsample(x, scale, shift)
        x = self.upsamples[-1](x)

        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = Network(
        num_classes=1,
        class_dim=128,
        factors=[2, 2, 2, 2, 1],
        up_pre_dim=384,
        up_post_dim=3,
        up_dims=[256, 256, 128, 64, 64],
        up_dilations=[
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 4, 8],
            [1, 2, 4, 8],
            [1, 2, 4, 8],
        ],
        down_pre_dim=32,
        down_dims=[64, 64, 128, 256],
        down_dilations=[
            [1, 2, 4],
            [1, 2, 4],
            [1, 2, 4],
            [1, 2, 4],
        ],
    ).cuda()
    output = model(
        torch.zeros(2).long().cuda(),
        torch.randn(2, 3, 64, 64).cuda(),
        torch.rand(2).cuda(),
    )
    print(output.shape)
