from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def init_state(self, batch_size: int, height: int, width: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        i, f, o, g = torch.chunk(self.gates(combined), chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class CloudForwardConvLSTM(nn.Module):
    """Small dynamic-only cloud/radiation forward model.

    Input:
      x: [B, T, C_in, H, W]

    Output:
      y_hat: [B, C_out, H, W]
    """

    def __init__(
        self,
        input_channels: int = 20,
        output_channels: int = 4,
        encoder_channels: int = 32,
        hidden_channels: int = 24,
        kernel_size: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=encoder_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=encoder_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

        self.recurrent = ConvLSTMCell(
            input_channels=encoder_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, encoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=encoder_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(encoder_channels, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x [B,T,C,H,W], got {tuple(x.shape)}")

        b, t_steps, _, hgt, wid = x.shape
        h, c = self.recurrent.init_state(b, hgt, wid, x.device)

        for t in range(t_steps):
            features = self.encoder(x[:, t])
            h, c = self.recurrent(features, (h, c))

        return self.decoder(h)
