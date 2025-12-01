import torch.nn as nn
import torch.nn.functional as F

# ================================
# Learnable Down Sampling
# ================================
class LDS(nn.Module):
    def __init__(self, c1, c2, kernel_size=2, stride=2):
        super().__init__()
        '''Unclear document: Resort to reduce by two times (?)'''
        self.c1 = c1
        self.c2 = c2

        self.down_conv = nn.Conv2d(self.c1, self.c2, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(self.c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.bn(self.down_conv(x)))
        return x

# ================================
# Global Attention Mechanism
# ================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, r: int = 16):
        super().__init__()
        self.channels = channels
        self.hidden_channels = max(channels // r, 1)

        self.ffn1 = nn.Linear(self.channels, self.hidden_channels)
        self.act = nn.SiLU()
        
        self.ffn2 = nn.Linear(self.hidden_channels, self.channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]
        
        # Change from chw -> whc
        x_perm = x.permute(0, 3, 2, 1)
        x_perm = x_perm.reshape(-1, w*h, self.channels) # (batch, wh, channel)

        # MLP
        attn_w = self.ffn1(x_perm) # (batch, wh, hidden channel)
        attn_w = self.act(attn_w)
        attn_w = self.ffn2(attn_w)

        # Convert back to chw
        attn_w = attn_w.reshape(-1, w, h, self.channels)
        attn_w = attn_w.permute(0, 3, 2, 1)
        attn_w = self.sigmoid(attn_w) # (batch, channel, w, h)
        return (attn_w * x) + (0.01 * x)

class SpatialAttention(nn.Module):
    def __init__(self, channels, pool_kernel=7, r: int = 16):
        super().__init__()
        self.channels = channels
        self.hidden_channels = max(channels // r, 1)

        self.conv1 = nn.Conv2d(self.channels, self.hidden_channels, kernel_size=pool_kernel, padding=pool_kernel//2)
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        self.act = nn.SiLU()
        
        self.conv2 = nn.Conv2d(self.hidden_channels, self.channels, kernel_size=pool_kernel, padding=pool_kernel//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attn_weight = self.conv1(x)
        attn_weight = self.act(self.bn(attn_weight))

        attn_weight = self.sigmoid(self.conv2(attn_weight))
        return (attn_weight * x) + (0.01 * x)

class GlobalAttentionMechanism(nn.Module):
    def __init__(self, c1, c2=None, pool_kernel=7, r: int = 16):
        if c2 is None:
            c2 = c1
        assert c1 == c2

        self.c1 = c1
        self.c2 = c2
        
        super().__init__()
        self.channel_attn = ChannelAttention(self.c1, r=r)
        self.spatial_attn = SpatialAttention(self.c1, pool_kernel=pool_kernel, r=r)
        
    def forward(self, x):
        x = self.channel_attn(x)
        return self.spatial_attn(x)