import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.softmax = nn.Softmax(dim=1)

        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, inputs):
        scores = torch.matmul(inputs, self.weight).squeeze(2) + self.bias
        attn_weights = self.softmax(scores).unsqueeze(2)

        attn_output = torch.sum(attn_weights * inputs, dim=1)
        return attn_output


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 256))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 256))

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))   # b, 1, 256
        maxout = self.shared_MLP(self.max_pool(x))  # b, 1, 256
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = out.permute(1, 0, 2)
        out = self.sigmoid(self.conv2d(out))
        return out.permute(1, 0, 2)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
