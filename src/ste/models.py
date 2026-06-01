from typing import Optional, Tuple

from diffusers import UNet2DModel


_DEFAULT_BLOCK_OUT_CHANNELS: Tuple[int, ...] = (128, 256, 256, 256)
_DOWN_BLOCK_TYPES = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
_UP_BLOCK_TYPES = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")


def build_unet(
    model_type: str = "unet",
    image_size: int = 32,
    in_channels: int = 3,
    out_channels: int = 3,
    num_class_embeds: int = 10,
    block_out_channels: Optional[Tuple[int, ...]] = None,
    dropout: float = 0.1,
) -> UNet2DModel:
    """Build a UNet2DModel matching the STE paper architecture.

    The default block configuration matches the legacy scripts exactly.
    Pass block_out_channels to override for custom experiments.
    """
    if model_type != "unet":
        raise ValueError(f"Unsupported model_type: {model_type!r}. Only 'unet' is supported.")
    if block_out_channels is None:
        block_out_channels = _DEFAULT_BLOCK_OUT_CHANNELS
    return UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=_DOWN_BLOCK_TYPES,
        up_block_types=_UP_BLOCK_TYPES,
        add_attention=True,
        num_class_embeds=num_class_embeds,
        dropout=dropout,
    )
