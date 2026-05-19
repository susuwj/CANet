import math

import torch
import torch.nn as nn
from einops import rearrange

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos

class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4) #3,b,num_heads,n,C // self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1).contiguous()
        return x

class GCA(nn.Module):
    def __init__(self, feat_dim=64, cost_dim=8, qkv_dim=4, heads=4, eta=1.):
        super().__init__()
        dim = qkv_dim * heads

        self.norm_layer1 = nn.LayerNorm(feat_dim)
        self.norm_layer2 = nn.LayerNorm(dim)
        self.to_qk = nn.Conv2d(feat_dim, dim*2, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim)

        self.cost_conv = nn.Conv2d(cost_dim, dim, 1, bias=False)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

        self.norm_layer3 = nn.LayerNorm(dim)
        self.local_mp = LPI(in_features=dim, act_layer=nn.GELU)

        self.norm_layer4 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim))

        self.cost_out = nn.Conv2d(dim, cost_dim, 1, bias=False)

    def forward(self, cost, feat):
        B, C, D, H, W = cost.shape

        q, k = self.to_qk(self.norm_layer1(feat.reshape(B, -1, H*W).permute(0, 2, 1).contiguous()).permute(0, 2,1).contiguous().reshape(B, -1, H, W)).chunk(2, dim=1)
        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), (q,k))
        q = q.transpose(-2, -1).contiguous()
        k = k.transpose(-2, -1).contiguous()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = attn.unsqueeze(1).repeat(1, D, 1, 1, 1).view(B*D, attn.shape[1], attn.shape[2], attn.shape[3])

        cost = cost.permute(0, 2, 1, 3, 4).contiguous()
        cost = cost.reshape(B*D, C, H, W)
        cost = self.cost_conv(cost) #TODO
        v = cost.reshape(B*D, -1, H*W).permute(0, 2, 1).contiguous()
        v = self.to_v(self.norm_layer2(v).permute(0, 2, 1).reshape(B*D, -1, H, W).contiguous()) #b, c, h, w
        v = v.reshape(B*D, self.heads, -1, H*W).permute(0, 1, 3, 2).contiguous()
        v = v.transpose(-2, -1).contiguous()

        x = (attn @ v)
        x = x.permute(0, 3, 1, 2).contiguous().reshape(B*D, H*W, -1)
        x = self.proj(x)   
        
        cost = cost.reshape(B*D, -1, H*W).permute(0, 2, 1).contiguous() + self.gamma1 * x
        cost = cost + self.gamma2 * self.local_mp(self.norm_layer3(cost), H, W)
        cost = cost + self.gamma3 * self.mlp(self.norm_layer4(cost)) 
        cost = self.cost_out(cost.permute(0, 2, 1).contiguous().view(B*D, -1, H, W))

        return cost.view(B, D, -1, H, W).permute(0, 2, 1, 3, 4).contiguous()
