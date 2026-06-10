import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MobileViTModel


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([features, skip], dim=1))


class MobileViTUNet(nn.Module):
    """U-Net decoder on top of the five multi-scale MobileViT encoder stages."""

    def __init__(
        self,
        num_classes: int,
        pretrained_model_name: str = "apple/mobilevit-x-small",
        local_files_only: bool = False,
    ):
        super().__init__()
        self.encoder = MobileViTModel.from_pretrained(
            pretrained_model_name,
            local_files_only=local_files_only,
            expand_output=False,
        )
        channels = list(self.encoder.config.neck_hidden_sizes[1:6])
        if len(channels) != 5:
            raise ValueError(f"Expected five MobileViT encoder stages, got channels={channels}")

        self.up4 = UpBlock(channels[4], channels[3], 128)
        self.up3 = UpBlock(128, channels[2], 96)
        self.up2 = UpBlock(96, channels[1], 64)
        self.up1 = UpBlock(64, channels[0], 32)
        self.refine = ConvBlock(32, 32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        stages = output.hidden_states
        if stages is None or len(stages) != 5:
            raise RuntimeError(f"Expected five MobileViT hidden states, got {0 if stages is None else len(stages)}")

        features = self.up4(stages[4], stages[3])
        features = self.up3(features, stages[2])
        features = self.up2(features, stages[1])
        features = self.up1(features, stages[0])
        features = F.interpolate(features, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.refine(features))

    def parameter_groups(self, encoder_lr: float, decoder_lr: float):
        encoder_ids = {id(parameter) for parameter in self.encoder.parameters()}
        decoder_parameters = [parameter for parameter in self.parameters() if id(parameter) not in encoder_ids]
        return [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": decoder_parameters, "lr": decoder_lr},
        ]
