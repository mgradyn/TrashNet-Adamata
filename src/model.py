import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttentionModule(nn.Module):
    """
    Module to apply channel-wise attention by focusing on important channels in the input feature map.
    """
    def __init__(self, channels, r):
        super(ChannelAttentionModule, self).__init__()
        reduced_channels = channels // r
        self.linear = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
        )
    def forward(self, x):
        b, c, _, _ = x.size()

        # Global pooling to summarize channel importance
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # Apply fc to compute attention
        attention = self.linear(max_pool) + self.linear(avg_pool)
        attention = torch.sigmoid(attention).view(b, c, 1, 1)

        return x * attention

class SpatialAttentionModule(nn.Module):
    """
    Module to apply spatial attention by focusing on important spatial regions in the feature map.
    """
    def __init__(self, bias=False):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=bias)
    def forward(self, x):

        # Compute max pooling and average pooling along the channel axis
        max_pool, _ = torch.max(x, dim=1, keepdim=True) # Max pooling (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True) # Average pooling (B, 1, H, W)

        # Concatenate along the channel axis to form a 2-channel tensor
        concat = torch.cat((max_pool, avg_pool), dim=1)

        # Apply convolution and sigmoid to compute attention map
        attention = torch.sigmoid(self.conv(concat))

        return x * attention

class AttentionModule(nn.Module):
    """
    Combined attention module that applies both channel and spatial attention.
    """
    def __init__(self, channels, r=8, bias=False):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttentionModule(channels, r)
        self.spatial_attention = SpatialAttentionModule(bias)

    def forward(self, x):
        residual = x
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x + residual

class LightModule(nn.Module):
    """
    A lightweight convolutional module that generates efficient feature maps by combining
    a traditional convolution and a depthwise convolution to expand the feature space.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, stride=1, dw_size=3, relu=True):
        super(LightModule, self).__init__()
        self.out_channels = out_channels

        # Calculate the number of channels for the two convolution stages
        init_channels = math.ceil(out_channels / ratio)
        new_channels = out_channels - init_channels

        self.traditional_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        # Traditional convolution
        self.traditional_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        # Depthwise convolution to generate additional feature maps
        self.dw_conv = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=dw_size, stride=1, padding=dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        x1 = self.traditional_conv(x)
        x2 = self.dw_conv(x1)
        out = torch.cat([x1, x2], dim=1)

        return out[:, :self.out_channels, :, :]

class LightBottleneck(nn.Module):
    """
    Lightweight bottleneck module with optional attention.
    """
    def __init__(self, in_channels, mid_channels, out_channels, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, r=8):
        super(LightBottleneck, self).__init__()
        self.stride = stride

        # First LightModule
        self.light1 = LightModule(in_channels, mid_channels, relu=True)

        # Optional depthwise convolution for stride > 1
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride=stride,
                      padding=(dw_kernel_size - 1) // 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels)
        ) if stride > 1 else nn.Identity()

        # Optional attention module
        self.attention = AttentionModule(mid_channels, r=r) if r > 1 else nn.Identity()

        # Second LightModule
        self.light2 = LightModule(mid_channels, out_channels, relu=False)

        # Shortcut connection
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        residual = x

        x = self.light1(x)
        x = self.dw_conv(x)
        x = self.attention(x)
        x = self.light2(x)

        return x + self.shortcut(residual)

    def _make_divisible(v, divisor, min_value=None):
        min_value = min_value or divisor
        half_divisor = divisor / 2

        new_v = max(min_value, int(v + half_divisor) - (int(v + half_divisor) % divisor))

        # to avoid significant reduction in value
        if new_v < (v * 0.9):
            new_v += divisor
        return new_v

class TrashNet(nn.Module):
    """
    TrashNet: A lightweight convolutional neural network designed with flexible
    bottleneck blocks and an efficient attention mechanism.
    """
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super(TrashNet, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # Initial stem layer
        out_channels = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        in_channels = out_channels

        # Build inverted residual blocks
        stages = []
        block = LightBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, r, s in cfg:
                out_channels = _make_divisible(c * width, 4)
                hidden_channels = _make_divisible(exp_size * width, 4)
                layers.append(
                    block(in_channels, hidden_channels, out_channels, dw_kernel_size=k, stride=s, r=r)
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*layers))

        # Final convolution stage
        out_channels = _make_divisible(exp_size * width, 4)
        stages.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        in_channels = out_channels
        self.blocks = nn.Sequential(*stages)

        # Classification head
        out_channels = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):

        # Stem
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Bottleneck blocks
        x = self.blocks(x)

        # Classification head
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1) # Flatten
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x