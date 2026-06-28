# model/custom_cnn.py
"""
Custom CNN for brain MRI tumor classification with optional attention (SE or CBAM)
and Dropout layers that can be toggled for MC Dropout at inference.

Usage:
    from custom_cnn import CustomCNN, enable_mc_dropout
    model = CustomCNN(num_classes=2, attention="cbam", dropout_p=0.3)
    # ... train ...
    model.eval()
    enable_mc_dropout(model)  # keep dropout active during eval for MC sampling
"""
import torch
import torch.nn as nn

# ------------------------------
# Attention Blocks
# ------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        inter = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, inter, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ------------------------------
# Convolutional Block
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, attention: str = "none", dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.attention_type = attention.lower()
        if self.attention_type == "se":
            self.attn = SEBlock(out_ch)
        elif self.attention_type == "cbam":
            self.attn = CBAM(out_ch)
        else:
            self.attn = nn.Identity()

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attn(x)
        x = self.dropout(x)
        return x

# ------------------------------
# Custom CNN
# ------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2, attention: str = "none",
                 dropout_p: float = 0.3, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.block1 = ConvBlock(in_channels, c1, attention=attention, dropout_p=dropout_p)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = ConvBlock(c1, c2, attention=attention, dropout_p=dropout_p)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = ConvBlock(c2, c3, attention=attention, dropout_p=dropout_p)
        self.pool3 = nn.MaxPool2d(2)
        self.block4 = ConvBlock(c3, c4, attention=attention, dropout_p=dropout_p)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def enable_mc_dropout(model: nn.Module):
    """Enable dropout layers during evaluation for MC Dropout."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
