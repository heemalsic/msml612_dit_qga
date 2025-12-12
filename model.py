# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(t, dim: int):
    """
    t: (B,) int/float timesteps
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class GQAAttention(nn.Module):
    """
    Grouped Query Attention:
    - q has num_heads
    - k/v have num_kv_heads (<= num_heads)
    - each kv head is shared among a group of q heads
    """
    def __init__(self, dim, num_heads=8, num_kv_heads=2, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N, Hd)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, N, Hd)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # expand kv to match q heads by repeating groups
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)  # (B, H, N, Hd)
        v = v.repeat_interleave(repeat, dim=1)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, Hd)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaLNBlock(nn.Module):
    """
    AdaLN-Zero style conditioning:
    - conditioning vector produces scale/shift for LayerNorm per block
    """
    def __init__(self, dim, num_heads, num_kv_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = GQAAttention(dim, num_heads=num_heads, num_kv_heads=num_kv_heads, proj_drop=drop)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

        # produce (shift1, scale1, gate1, shift2, scale2, gate2)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        # "zero-init" last linear for stability
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    def forward(self, x, c):
        """
        x: (B, N, D)
        c: (B, D) conditioning
        """
        B, N, D = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.cond(c).chunk(6, dim=-1)

        h = self.ln1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h = self.attn(h)
        x = x + gate1.unsqueeze(1) * h

        h = self.ln2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h
        return x


class DiT(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=4,
        in_ch=1,
        dim=512,
        depth=6,
        num_heads=8,
        num_kv_heads=2,
        num_classes=10,
        drop=0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.dim = dim
        self.num_classes = num_classes

        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_ch * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.class_embed = nn.Embedding(num_classes, dim)

        self.blocks = nn.ModuleList([
            AdaLNBlock(dim, num_heads=num_heads, num_kv_heads=num_kv_heads, mlp_ratio=4.0, drop=drop)
            for _ in range(depth)
        ])

        self.final_ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def patchify(self, x):
        # x: (B, C, H, W) -> (B, N, patch_dim)
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, (H // p) * (W // p), C * p * p)
        return x

    def unpatchify(self, x):
        # x: (B, N, patch_dim) -> (B, C, H, W)
        B, N, D = x.shape
        p = self.patch_size
        H = W = self.image_size
        C = self.in_ch
        x = x.view(B, H // p, W // p, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, x, t, y):
        """
        x: (B,C,H,W) noisy image at timestep t
        t: (B,) timesteps (int)
        y: (B,) class labels
        returns predicted noise eps: (B,C,H,W)
        """
        B = x.size(0)
        tok = self.patchify(x)                  # (B,N,patch_dim)
        tok = self.patch_embed(tok)             # (B,N,dim)
        tok = tok + self.pos_embed

        t_emb = sinusoidal_timestep_embedding(t, self.dim)
        t_emb = self.time_mlp(t_emb)            # (B,dim)
        y_emb = self.class_embed(y)             # (B,dim)

        c = t_emb + y_emb                       # conditioning vector (B,dim)

        for blk in self.blocks:
            tok = blk(tok, c)

        tok = self.final_ln(tok)
        tok = self.out(tok)                     # (B,N,patch_dim)
        out = self.unpatchify(tok)              # (B,C,H,W)
        return out
