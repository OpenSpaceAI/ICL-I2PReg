from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import ConvBlock, build_act_layer

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        norm_cfg: Union[str, dict] = "GroupNorm",
        act_cfg: Union[str, dict] = "LeakyReLU",
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            dilation=dilation,
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dilation=dilation,
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg="None",
        )

        if stride == 1:
            self.identity = nn.Identity()
        else:
            self.identity = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                dilation=dilation,
                conv_cfg="Conv2d",
                norm_cfg=norm_cfg,
                act_cfg="None",
            )

        self.act = build_act_layer(act_cfg)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        identity = self.identity(x)
        output = self.act(identity + residual)
        return output

class ImageOverlap(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        dilation: int = 1,
        norm_cfg: Union[str, dict] = "GroupNorm",
        act_cfg: Union[str, dict] = "LeakyReLU",
    ):
        super().__init__()

        self.encoder1 = ConvBlock(
            in_channels,
            base_channels * 1,
            kernel_size=7,
            padding=3,
            stride=2,
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.encoder2 = nn.Sequential(
            BasicBlock(
                base_channels * 1,
                base_channels * 1,
                stride=1,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            BasicBlock(
                base_channels * 1,
                base_channels * 1,
                stride=1,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        self.encoder3 = nn.Sequential(
            BasicBlock(
                base_channels * 1,
                base_channels * 2,
                stride=2,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            BasicBlock(
                base_channels * 2,
                base_channels * 2,
                stride=1,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        self.encoder4 = nn.Sequential(
            BasicBlock(
                base_channels * 2,
                base_channels * 4,
                stride=2,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            BasicBlock(
                base_channels * 4,
                base_channels * 4,
                stride=1,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        self.decoder4_1 = ConvBlock(
            base_channels * 4,
            base_channels * 4,
            kernel_size=1,
            conv_cfg="Conv2d",
            norm_cfg="None",
            act_cfg="None",
        )

    def forward(self, x):
        feats_list = []
        # encoder
        feats_s1 = self.encoder1(x)  # 1/2
        feats_s2 = self.encoder2(feats_s1)  # 1/2
        feats_s3 = self.encoder3(feats_s2)  # 1/4
        feats_s4 = self.encoder4(feats_s3)  # 1/8

        # decoder
        latent_s4 = self.decoder4_1(feats_s4)  # (1/8)

        return latent_s4