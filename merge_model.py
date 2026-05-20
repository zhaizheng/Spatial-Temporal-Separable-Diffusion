import torch
import torch.nn as nn
import math

# =========================================================
# Time Embedding (统一)
# =========================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)

# =========================================================
# ResBlock
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()

        self.use_time = time_dim is not None

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if self.use_time:
            self.time_mlp = nn.Linear(time_dim, out_ch)

        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)

        if self.use_time:
            h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)

# =========================================================
# TimeNet（deterministic）
# =========================================================
class ResBlock_time(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class TimeNet(nn.Module):
    def __init__(self, time_dim, C):
        super().__init__()

        hidden = time_dim * 4

        self.input = nn.Linear(time_dim, hidden)

        self.blocks = nn.Sequential(
            ResBlock_time(hidden),
            ResBlock_time(hidden),
            ResBlock_time(hidden)
        )

        self.out = nn.Linear(hidden, C)

    def forward(self, t_emb):
        x = self.input(t_emb)
        x = self.blocks(x)
        return self.out(x)
'''
class TimeNet(nn.Module):
    def __init__(self, time_dim, C):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(time_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, C)
        )

    def forward(self, t_emb):
        return self.net(t_emb)  # (B, C)
'''
# =========================================================
# UNet Backbone
# =========================================================
class UNet(nn.Module):
    def __init__(self, C=16, mode="M1"):
        super().__init__()

        """
        mode:
        M1 = standard diffusion
        M2 = basis + time weight
        M3 = full time-aware + basis
        """

        base = 128
        time_dim = 128

        self.mode = mode
        self.time_embed = TimeEmbedding(time_dim)

        # 🔥 only M3 injects time into UNet
        use_time = (mode == "M3") or (mode == "M1")

        self.enc1 = ResBlock(1, base, time_dim if use_time else None)
        self.down1 = nn.Conv2d(base, base, 4, 2, 1)

        self.enc2 = ResBlock(base, base*2, time_dim if use_time else None)
        self.down2 = nn.Conv2d(base*2, base*2, 4, 2, 1)

        self.mid = ResBlock(base*2, base*2, time_dim if use_time else None)

        self.up2 = nn.ConvTranspose2d(base*2, base*2, 4, 2, 1)
        self.dec2 = ResBlock(base*4, base, time_dim if use_time else None)

        self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)
        self.dec1 = ResBlock(base*2, base, time_dim if use_time else None)

        # 🔥 输出统一为 C channel
        self.out = nn.Conv2d(base, C, 1)
        #if self.mode == "M1":
        #    self.out = nn.Conv2d(base, 1, 1)
        #else:
        #    self.out = nn.Conv2d(base, C, 1)

        # 🔥 only M2 / M3 use TimeNet
        if mode in ["M2", "M3", "M2orth","M2orth_smooth"]:
            self.time_net = TimeNet(time_dim, C)

    def forward(self, x, t):

        t = t.float() / 1000
        t_emb = self.time_embed(t)

        # -----------------------
        # UNet forward
        # -----------------------
        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(self.down1(x1), t_emb)

        h = self.mid(self.down2(x2), t_emb)

        h = self.up2(h)
        if h.shape[-1] != x2.shape[-1]:
            h = nn.functional.interpolate(h, size=x2.shape[2:])
        h = torch.cat([h, x2], dim=1)
        h = self.dec2(h, t_emb)

        h = self.up1(h)
        if h.shape[-1] != x1.shape[-1]:
            h = nn.functional.interpolate(h, size=x1.shape[2:])
        h = torch.cat([h, x1], dim=1)
        h = self.dec1(h, t_emb)

        f = self.out(h)  # (B, C, H, W)

        # =====================================================
        # 🔵 M1: Standard diffusion
        # =====================================================
        if self.mode == "M1":
        #    return f
            return f.mean(dim=1, keepdim=True), f
            # 👉 统一到 1 channel（保证公平）

        # =====================================================
        # 🟢 M2: Basis + time weight
        # =====================================================
        elif self.mode == "M2" or self.mode == "M2orth" or self.mode =="M2orth_smooth":
            w = self.time_net(t_emb)[:, :, None, None]
            out = (f * w).sum(dim=1, keepdim=True)
            return out, f

        # =====================================================
        # 🔴 M3: Full time-aware + basis
        # =====================================================
        elif self.mode == "M3":
            w = self.time_net(t_emb)[:, :, None, None]
            out = (f * w).sum(dim=1, keepdim=True)
            return out, f