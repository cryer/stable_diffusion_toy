# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class TimestepEmbedding(nn.Module):
    """时间步嵌入层"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        # 第一个 Linear 输入应为 half_dim，输出 dim
        self.emb = nn.Sequential(
            nn.Linear(self.half_dim, dim),  # 输入是 half_dim
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        # t: [B]
        t = t.float()
        half_dim = self.half_dim
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)  # [half_dim]
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        # 转换 emb 的 dtype 以匹配 Linear 层
        emb = emb.to(self.emb[0].weight.dtype)  # 自动匹配 Linear 的 dtype
        # 保留 [B, half_dim]，让 Linear 层映射到 dim
        return self.emb(emb)  # [B, half_dim] → Linear → [B, dim]

class CrossAttention(nn.Module):
    """简化交叉注意力层"""
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        self.dim_head = dim_head  # 保存 dim_head
        self.inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, query_dim)

    def forward(self, x, context=None):
        B, L, _ = x.shape  # x: [B, L, C]
        h = self.heads
        dim_head = self.dim_head  # 显式使用保存的 dim_head

        q = self.to_q(x)  # [B, L, 512]
        context = context if context is not None else x
        k = self.to_k(context)  # [B, 77, 512]
        v = self.to_v(context)  # [B, 77, 512]

        # 分头: [B, L, 512] -> [B, L, h, dim_head] -> [B, h, L, dim_head]
        q = q.view(B, L, h, dim_head).transpose(1, 2)
        k = k.view(B, context.shape[1], h, dim_head).transpose(1, 2) 
        v = v.view(B, context.shape[1], h, dim_head).transpose(1, 2)  

        # 计算注意力
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [B, h, L, 77]
        attn = sim.softmax(dim=-1)  # [B, h, L, 77]

        # 加权求和
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, h, L, dim_head]

        # 合并头
        out = out.transpose(1, 2).contiguous().view(B, L, self.inner_dim)  # [B, L, 512]
        return self.to_out(out)

class ResnetBlock(nn.Module):
    """带时间条件的残差块"""
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_c)
        )
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, in_c), in_c)  # 最多8组，但不超过通道数
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t):
        if x.shape[1] != self.norm1.num_channels:
            raise ValueError(f"通道数不匹配！输入: {x.shape[1]}, 期望: {self.norm1.num_channels}")

        def block1(x_in):
            return self.conv1(F.silu(self.norm1(x_in)))

        def block2(x_in):
            return self.conv2(F.silu(self.norm2(x_in)))

        h = checkpoint(block1, x)
        h += self.mlp(t)[:, :, None, None]
        h = checkpoint(block2, h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """空间自注意力块"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h

class UNet(nn.Module):
    """简化版条件U-Net，支持文本交叉注意力"""
    def __init__(self, in_channels=4, out_channels=4, text_embed_dim=512):
        super().__init__()

        # 时间步嵌入
        time_embed_dim = 256
        self.time_embed = TimestepEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # 编码器层
        self.enc1 = ResnetBlock(in_channels, 64, time_embed_dim)
        self.enc2 = ResnetBlock(64, 128, time_embed_dim)
        self.enc3 = ResnetBlock(128, 256, time_embed_dim)

        # 瓶颈层
        self.mid1 = ResnetBlock(256, 256, time_embed_dim)
        self.mid_attn = AttentionBlock(256)
        self.mid_cross_attn = CrossAttention(256, text_embed_dim)
        self.mid2 = ResnetBlock(256, 256, time_embed_dim)

        # 解码器层
        self.dec1 = ResnetBlock(256 + 128, 128, time_embed_dim)  
        self.dec2 = ResnetBlock(128 + 64, 64, time_embed_dim) 
        self.dec3 = ResnetBlock(64, 64, time_embed_dim) 

        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

        # 下采样 & 上采样
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, t, context):
        # x: [B, C, H, W], t: [B], context: [B, L, 768]
        t_emb = self.time_mlp(self.time_embed(t))

        # 编码器
        h1 = self.enc1(x, t_emb)  # [B, 64, H, W]
        h2 = self.enc2(self.downsample(h1), t_emb)  # [B, 128, H/2, W/2]
        h3 = self.enc3(self.downsample(h2), t_emb)  # [B, 256, H/4, W/4]

        # 瓶颈
        h = self.mid1(h3, t_emb)
        h = self.mid_attn(h)
        B, C, H, W = h.shape
        h_flat = h.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        h_flat = self.mid_cross_attn(x=h_flat, context=context)
        h = h_flat.transpose(1, 2).view(B, C, H, W)
        h = self.mid2(h, t_emb)

        # 解码器
        h = self.upsample(h)  
        h = torch.cat([h, h2], dim=1)               
        h = self.dec1(h, t_emb)                     

        h = self.upsample(h)                         
        h = torch.cat([h, h1], dim=1)                
        h = self.dec2(h, t_emb)                  

        h = self.dec3(h, t_emb) 
        out = self.out(h)        
        return out