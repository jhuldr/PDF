# nn/rectified_flow_matching.py
import os
import torch
import torch.nn as nn
from nn.rfm_backbone import RFMUNet


class PDF_intensity(nn.Module):
    naming = "PDF_intensity"

    def __init__(
        self,
        device: torch.device,
        in_channels: int,           # image channels C
        num_filters: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels

        if depth <= 1:
            channel_mults = (1,)
        else:
            channel_mults = tuple(min(2 ** i, 8) for i in range(depth))

        self.net = RFMUNet(
            img_channels=in_channels,
            base_channels=num_filters,
            channel_mults=channel_mults,
            num_res_blocks=2,
            cond_dim=256,
            dropout=dropout,
        ).to(device)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def save_weights(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device=None):
        if device is None:
            device = self.device
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)

    def freeze_time_independent(self):
        pass

    def forward(self, x_img: torch.Tensor, m_prob: torch.Tensor, s: torch.Tensor):
        return self.net(x_img, m_prob, s)
