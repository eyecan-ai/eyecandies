import typing as t
from torch import nn


def make_encoder_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


def make_decoder_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels, in_channels, 3, padding=1),
        nn.BatchNorm2d(in_channels),
        nn.GELU(),
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class AutoEncoder(nn.Module):
    def __init__(
        self,
        image_size: t.Union[int, t.Tuple[int, int]],
        image_channels: int = 3,
        n_blocks: int = 4,
        hidden_channels: int = 32,
        latent_size: int = 256,
    ):
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        encoder_blocks = [make_encoder_block(image_channels, hidden_channels)]
        image_size = (image_size[0] // 2, image_size[1] // 2)
        for i in range(n_blocks - 1):
            encoder_blocks.append(
                make_encoder_block(hidden_channels, hidden_channels * 2)
            )
            hidden_channels *= 2
            image_size = (image_size[0] // 2, image_size[1] // 2)

        self.encoder = nn.Sequential(*encoder_blocks)

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * image_size[0] * image_size[1], latent_size),
            nn.Linear(latent_size, hidden_channels * image_size[0] * image_size[1]),
            nn.Unflatten(1, (hidden_channels, image_size[0], image_size[1])),
        )

        decoder_blocks = []
        for i in range(n_blocks - 1):
            decoder_blocks.append(
                make_decoder_block(hidden_channels, hidden_channels // 2)
            )
            hidden_channels //= 2
            image_size = (image_size[0] * 2, image_size[1] * 2)
        decoder_blocks.append(make_decoder_block(hidden_channels, image_channels))

        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
