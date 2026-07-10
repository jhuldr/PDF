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


class PDFUNet(nn.Module):
    """
    Mask-conditioned version based on the PDF U-Net structure:

    Input channels:
        x_img: [B, C, H, W]
        m_prob: [B, 1, H, W]
        -> concatenate and feed into the U-Net: [B, C+1, H, W]

    Output channels:
        v_img: [B, C, H, W]
        v_l  : [B, 1, H, W]  (mask logit velocity)
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
        self.img_channels = img_channels
        self.in_channels = img_channels + 1        # image + mask_prob
        self.out_channels = img_channels + 1       # v_img + v_l
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.cond_dim = int(cond_dim)

        # (s, dt) embedding -> cond
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

        self.input = nn.Conv2d(self.in_channels, base_channels, 3, padding=1)

        # -------------------------
        # Down path
        # -------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        self._skip_ch = []

        for i, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, cond_dim=cond_dim, dropout=dropout))
                ch = out_ch
                self._skip_ch.append(ch)
            self.down_blocks.append(blocks)

            if i != len(self.channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # -------------------------
        # -------------------------
        self.mid1 = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)

        # -------------------------
        # Up path
        # -------------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        skip_ch_stack = list(self._skip_ch)  # local copy

        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()

            for _ in range(self.num_res_blocks):
                skip_ch = skip_ch_stack.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, cond_dim=cond_dim, dropout=dropout))
                ch = out_ch

            self.up_blocks.append(blocks)

            if i != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        assert len(skip_ch_stack) == 0, "Internal error: skips not fully consumed."

        self.out_norm = nn.GroupNorm(8, ch)
        self.out = nn.Conv2d(ch, self.out_channels, 3, padding=1)

    def make_cond(self, s: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        s_emb = sinusoidal_embedding(s, self.cond_dim // 2)
        dt_emb = sinusoidal_embedding(delta_t, self.cond_dim // 2)
        cond = torch.cat([s_emb, dt_emb], dim=1)
        cond = self.raw_proj(cond)
        cond = self.time_mlp(cond)
        return cond

    def forward(self, x_img: torch.Tensor, m_prob: torch.Tensor, s: torch.Tensor, delta_t: torch.Tensor):
        cond = self.make_cond(s, delta_t)

        x = torch.cat([x_img, m_prob], dim=1)  # [B, C+1, H, W]
        h = self.input(x)
        skips = []

        for blocks, down in zip(self.down_blocks, self.downsamples):
            for rb in blocks:
                h = rb(h, cond)
                skips.append(h)
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
        out = self.out(h)  # [B, C+1, H, W]

        v_img = out[:, :self.img_channels, ...]
        v_l   = out[:, self.img_channels:self.img_channels+1, ...]
        return v_img, v_l
    


class PDFUNet_2head(nn.Module):
    """
    Mask-conditioned version based on the PDF U-Net structure:

    Input channels:
        x_img: [B, C, H, W]
        m_prob: [B, 1, H, W]
        -> concatenate and feed into the U-Net: [B, C+1, H, W]

    Output channels:
        v_img: [B, C, H, W]
        v_l  : [B, 1, H, W]  (mask logit velocity)
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
        self.img_channels = img_channels
        self.in_channels = img_channels + 1        # image + mask_prob
        self.out_channels = img_channels + 1       # v_img + v_l
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.cond_dim = int(cond_dim)

        # (s, dt) embedding -> cond
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

        self.input = nn.Conv2d(self.in_channels, base_channels, 3, padding=1)

        # -------------------------
        # Down path
        # -------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        self._skip_ch = []

        for i, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, cond_dim=cond_dim, dropout=dropout))
                ch = out_ch
                self._skip_ch.append(ch)
            self.down_blocks.append(blocks)

            if i != len(self.channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # -------------------------
        # -------------------------
        self.mid1 = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)

        # -------------------------
        # Up path
        # -------------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        skip_ch_stack = list(self._skip_ch)  # local copy

        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()

            for _ in range(self.num_res_blocks):
                skip_ch = skip_ch_stack.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, cond_dim=cond_dim, dropout=dropout))
                ch = out_ch

            self.up_blocks.append(blocks)

            if i != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        assert len(skip_ch_stack) == 0, "Internal error: skips not fully consumed."

        self.out_norm = nn.GroupNorm(8, ch)

        self.head_img = nn.Conv2d(ch, self.img_channels, 3, padding=1)
        self.head_mask = nn.Conv2d(ch, 1, 3, padding=1)

    def make_cond(self, s: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        s_emb = sinusoidal_embedding(s, self.cond_dim // 2)
        dt_emb = sinusoidal_embedding(delta_t, self.cond_dim // 2)
        cond = torch.cat([s_emb, dt_emb], dim=1)
        cond = self.raw_proj(cond)
        cond = self.time_mlp(cond)
        return cond

    def forward(self, x_img: torch.Tensor, m_prob: torch.Tensor, s: torch.Tensor, delta_t: torch.Tensor):
        cond = self.make_cond(s, delta_t)

        x = torch.cat([x_img, m_prob], dim=1)  # [B, C+1, H, W]
        h = self.input(x)
        skips = []

        for blocks, down in zip(self.down_blocks, self.downsamples):
            for rb in blocks:
                h = rb(h, cond)
                skips.append(h)
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
        


        v_img = self.head_img(h)
        v_l   = self.head_mask(h)

        return v_img, v_l


class PDFUNet_2head_deep(nn.Module):
    """
    mask-conditioned TFM UNet with DEEP task-specific heads
    """
    def __init__(
        self,
        img_channels: int = 1,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        dropout: float = 0.0,
        head_depth_img: int = 2,   # <<< NEW
        head_depth_mask: int = 1,  # <<< NEW
    ):
        super().__init__()
        self.img_channels = img_channels
        self.in_channels = img_channels + 1
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.cond_dim = int(cond_dim)

        # -------------------------
        # time embedding
        # -------------------------
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

        self.input = nn.Conv2d(self.in_channels, base_channels, 3, padding=1)

        # -------------------------
        # Down path
        # -------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        self._skip_ch = []

        for i, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(
                    ResBlock(ch, out_ch, cond_dim=cond_dim, dropout=dropout)
                )
                ch = out_ch
                self._skip_ch.append(ch)
            self.down_blocks.append(blocks)

            self.downsamples.append(
                Downsample(ch) if i != len(self.channel_mults) - 1 else nn.Identity()
            )

        # -------------------------
        # -------------------------
        self.mid1 = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)

        # -------------------------
        # Up path
        # -------------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        skip_ch_stack = list(self._skip_ch)

        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()

            for _ in range(self.num_res_blocks):
                skip_ch = skip_ch_stack.pop()
                blocks.append(
                    ResBlock(ch + skip_ch, out_ch, cond_dim=cond_dim, dropout=dropout)
                )
                ch = out_ch

            self.up_blocks.append(blocks)
            self.upsamples.append(Upsample(ch) if i != 0 else nn.Identity())

        assert len(skip_ch_stack) == 0

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act = nn.SiLU()

        # =========================================================
        # 🔴 Deep task-specific heads
        # =========================================================

        # ---- Intensity head (DEEPER) ----
        img_head_blocks = []
        img_ch = ch
        for _ in range(head_depth_img):
            img_head_blocks.append(
                ResBlock(img_ch, img_ch, cond_dim=cond_dim, dropout=dropout)
            )
        img_head_blocks.append(
            nn.Conv2d(img_ch, img_channels, 3, padding=1)
        )
        self.head_img = nn.ModuleList(img_head_blocks)

        # ---- Mask-logit velocity head (SHALLOWER) ----
        mask_head_blocks = []
        mask_ch = ch
        for _ in range(head_depth_mask):
            mask_head_blocks.append(
                ResBlock(mask_ch, mask_ch, cond_dim=cond_dim, dropout=dropout)
            )
        mask_head_blocks.append(
            nn.Conv2d(mask_ch, 1, 3, padding=1)
        )
        self.head_mask = nn.ModuleList(mask_head_blocks)

    # -------------------------------------------------
    def make_cond(self, s: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        s_emb = sinusoidal_embedding(s, self.cond_dim // 2)
        dt_emb = sinusoidal_embedding(delta_t, self.cond_dim // 2)
        cond = torch.cat([s_emb, dt_emb], dim=1)
        cond = self.raw_proj(cond)
        cond = self.time_mlp(cond)
        return cond

    # -------------------------------------------------
    def forward(
        self,
        x_img: torch.Tensor,
        m_prob: torch.Tensor,
        s: torch.Tensor,
        delta_t: torch.Tensor,
    ):
        cond = self.make_cond(s, delta_t)

        x = torch.cat([x_img, m_prob], dim=1)
        h = self.input(x)
        skips = []

        for blocks, down in zip(self.down_blocks, self.downsamples):
            for rb in blocks:
                h = rb(h, cond)
                skips.append(h)
            h = down(h)

        h = self.mid1(h, cond)
        h = self.mid2(h, cond)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for rb in blocks:
                h = torch.cat([h, skips.pop()], dim=1)
                h = rb(h, cond)
            h = up(h)

        h = self.out_act(self.out_norm(h))

        # -------------------------
        # intensity head
        # -------------------------
        h_img = h
        for layer in self.head_img:
            if isinstance(layer, ResBlock):
                h_img = layer(h_img, cond)
            else:
                h_img = layer(h_img)
        v_img = h_img

        # -------------------------
        # mask head
        # -------------------------
        h_mask = h
        for layer in self.head_mask:
            if isinstance(layer, ResBlock):
                h_mask = layer(h_mask, cond)
            else:
                h_mask = layer(h_mask)
        v_l = h_mask

        return v_img, v_l



class PDFUNet_mask(nn.Module):
    """
    Mask-conditioned version based on the PDF U-Net structure:

    Input channels:
        x_img: [B, C, H, W]
        m_prob: [B, 1, H, W]
        -> concatenate and feed into the U-Net: [B, C+1, H, W]

    Output channels:
        v_img: [B, C, H, W]
        v_l  : [B, 1, H, W]  (mask logit velocity)
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
        self.img_channels = img_channels
        self.in_channels = img_channels + 1        # image + mask_prob
        self.out_channels = img_channels       # v_l
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.cond_dim = int(cond_dim)

        # (s, dt) embedding -> cond
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

        self.input = nn.Conv2d(self.in_channels, base_channels, 3, padding=1)

        # -------------------------
        # Down path
        # -------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        self._skip_ch = []

        for i, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, cond_dim=cond_dim, dropout=dropout))
                ch = out_ch
                self._skip_ch.append(ch)
            self.down_blocks.append(blocks)

            if i != len(self.channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # -------------------------
        # -------------------------
        self.mid1 = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)

        # -------------------------
        # Up path
        # -------------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        skip_ch_stack = list(self._skip_ch)  # local copy

        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()

            for _ in range(self.num_res_blocks):
                skip_ch = skip_ch_stack.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, cond_dim=cond_dim, dropout=dropout))
                ch = out_ch

            self.up_blocks.append(blocks)

            if i != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        assert len(skip_ch_stack) == 0, "Internal error: skips not fully consumed."

        self.out_norm = nn.GroupNorm(8, ch)
        self.out = nn.Conv2d(ch, self.out_channels, 3, padding=1)

    def make_cond(self, s: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        s_emb = sinusoidal_embedding(s, self.cond_dim // 2)
        dt_emb = sinusoidal_embedding(delta_t, self.cond_dim // 2)
        cond = torch.cat([s_emb, dt_emb], dim=1)
        cond = self.raw_proj(cond)
        cond = self.time_mlp(cond)
        return cond

    def forward(self, x_img: torch.Tensor, m_prob: torch.Tensor, s: torch.Tensor, delta_t: torch.Tensor):
        cond = self.make_cond(s, delta_t)

        x = torch.cat([x_img, m_prob], dim=1)  # [B, C+1, H, W]
        h = self.input(x)
        skips = []

        for blocks, down in zip(self.down_blocks, self.downsamples):
            for rb in blocks:
                h = rb(h, cond)
                skips.append(h)
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
        out = self.out(h)  # [B, C+1, H, W]

        v_l = out[:, :self.img_channels, ...]
        
        return v_l
    
