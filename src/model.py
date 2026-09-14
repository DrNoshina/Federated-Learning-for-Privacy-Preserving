"""
Sequential model (paper Section 4.6, Table 1).

A stacked two-layer LSTM followed by a dense classification layer. The final
hidden state after L steps is passed to a fully connected layer, giving

    y_hat = softmax(W . h_L + b)                              (Eq. 4)

Training uses cross-entropy loss (Eq. 5) and the Adam optimiser.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KTLSTM(nn.Module):
    def __init__(self, n_features: int = 10, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, L, H)
        h_L = out[:, -1, :]            # final hidden state
        return self.fc(self.drop(h_L))  # logits; softmax applied in the loss / at eval


def make_model(cfg, n_features: int, device) -> KTLSTM:
    model = KTLSTM(n_features, cfg.hidden, cfg.layers, cfg.dropout).to(device)
    return model


def model_size_report(model: nn.Module) -> dict:
    """Parameter count and on-the-wire size of one model transfer."""
    n_params = sum(p.numel() for p in model.parameters())
    bytes_fp32 = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    return {
        "parameters": int(n_params),
        "update_mb_fp32": bytes_fp32 / 1e6,
        "update_mb_fp16": bytes_fp32 / 2e6,
    }
