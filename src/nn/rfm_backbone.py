# nn/tfm_backbone.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [B] or [B,1]
    return: [B, dim]
    """
    if t.dim() == 2:
        t = t.squeeze(1)
    device = t.device
    half = dim // 2
    denom = (half - 1) if half > 1 else 1
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / denom
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """
    ResBlock with FiLM conditioning: h -> h*(1+gamma) + beta
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_ch * 2)  # gamma, beta
        )

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        gamma_beta = self.cond_proj(cond)  # [B, 2*out_ch]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        h = self.norm2(h)
        h = h * (1.0 + gamma) + beta
        h = self.conv2(self.dropout(F.silu(h)))

        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class RFMUNet(nn.Module):
    """
    Mask-conditioned UNet backbone for RFM (noise -> x):
        v = f(x_s, m_prob, s)

    Inputs:
        x_img : [B, C, H, W]
        m_prob: [B, 1, H, W]
        s     : [B, 1] (or [B]) in [0,1]

    Output:
        v_img : [B, C, H, W]
    """
    def __init__(
        self,
        img_channels: int = 1,          # C
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_channels = int(img_channels)
        self.in_channels = self.img_channels + 1   # image + mask_prob
        self.out_channels = self.img_channels      # velocity only
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.cond_dim = int(cond_dim)

        # time embedding (only s for RFM)
        self.raw_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        self.input = nn.Conv2d(self.in_channels, self.base_channels, 3, padding=1)

        # -------------------------
        # Down path (store skip after each ResBlock)
        # -------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = self.base_channels
        self._skip_ch = []

        for i, mult in enumerate(self.channel_mults):
            out_ch = self.base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, cond_dim=self.cond_dim, dropout=dropout))
                ch = out_ch
                self._skip_ch.append(ch)  # one skip per ResBlock output
            self.down_blocks.append(blocks)

            if i != len(self.channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # -------------------------
        # -------------------------
        self.mid1 = ResBlock(ch, ch, cond_dim=self.cond_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, cond_dim=self.cond_dim, dropout=dropout)

        # -------------------------
        # Up path (consume exactly one skip per ResBlock)
        # -------------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        skip_ch_stack = list(self._skip_ch)

        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_ch = self.base_channels * mult
            blocks = nn.ModuleList()

            for _ in range(self.num_res_blocks):
                skip_ch = skip_ch_stack.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, cond_dim=self.cond_dim, dropout=dropout))
                ch = out_ch

            self.up_blocks.append(blocks)

            if i != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        assert len(skip_ch_stack) == 0, "Internal error: skips not fully consumed."

        g_out = min(8, ch)
        self.out_norm = nn.GroupNorm(g_out, ch)
        self.out = nn.Conv2d(ch, self.out_channels, 3, padding=1)

    def make_cond(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: [B,1] or [B] in [0,1]
        """
        if s.dim() == 1:
            s_in = s
        else:
            s_in = s.squeeze(1)

        s_emb = sinusoidal_embedding(s_in, self.cond_dim)  # [B, cond_dim]
        cond = self.raw_proj(s_emb)
        cond = self.time_mlp(cond)
        return cond

    def forward(self, x_img: torch.Tensor, m_prob: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        cond = self.make_cond(s)

        x = torch.cat([x_img, m_prob], dim=1)  # [B, C+1, H, W]
        h = self.input(x)
        skips = []

        for blocks, down in zip(self.down_blocks, self.downsamples):
            for rb in blocks:
                h = rb(h, cond)
                skips.append(h)  # store after each ResBlock
            h = down(h)

        h = self.mid1(h, cond)
        h = self.mid2(h, cond)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for rb in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = rb(h, cond)
            h = up(h)

        h = F.silu(self.out_norm(h))
        v_img = self.out(h)  # [B, C, H, W]
        return v_img
