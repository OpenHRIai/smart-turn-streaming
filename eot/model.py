"""Stage B1 model: tau-aware heads on a frozen encoder's pooled vector plus a GRU over fresh mel.

Inputs per window
    pooled   (384,)   frozen encoder, attention-pooled (cached, fp16)
    mel20    (20, 80) last 1 s of Whisper log-mel at 50 ms steps (cached, fp16)
    tau_feat (3,)     [tau / 2.5, log1p(tau), speaking]  (speaking = 1 while the user talks,
                      i.e. for 'pre' and 'mid' windows; tau = 0 then)
Outputs
    logits (2,): p_end, p_eager
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def tau_features(kind: np.ndarray, value: np.ndarray) -> np.ndarray:
    """(N,3) float32 from the item kind/value columns."""
    is_tau = kind == "tau"
    tau = np.where(is_tau, value, 0.0).astype(np.float32)
    speaking = (~is_tau).astype(np.float32)
    return np.stack([tau / 2.5, np.log1p(tau), speaking], axis=1).astype(np.float32)


class Heads(nn.Module):
    def __init__(self, d_pooled: int = 384, d_mel: int = 80, h_mel: int = 128, h: int = 128,
                 use_mel: bool = True, use_tau: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_mel, self.use_tau = use_mel, use_tau
        self.pool_proj = nn.Sequential(nn.LayerNorm(d_pooled), nn.Linear(d_pooled, h), nn.GELU())
        d_in = h
        if use_mel:
            self.mel_norm = nn.LayerNorm(d_mel)
            self.gru = nn.GRU(d_mel, h_mel, batch_first=True)
            d_in += h_mel
        if use_tau:
            d_in += 3
        self.fuse = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(h, 64), nn.GELU())
        self.out = nn.Linear(64, 2)

    def forward(self, pooled: torch.Tensor, mel20: torch.Tensor, tau_feat: torch.Tensor) -> torch.Tensor:
        parts = [self.pool_proj(pooled.float())]
        if self.use_mel:
            _, hN = self.gru(self.mel_norm(mel20.float()))
            parts.append(hN[-1])
        if self.use_tau:
            parts.append(tau_feat.float())
        return self.out(self.fuse(torch.cat(parts, dim=1)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_bce(logits: torch.Tensor, y_end: torch.Tensor, y_eager: torch.Tensor,
               w_eager: float = 1.0) -> tuple[torch.Tensor, dict]:
    """BCE on p_end for every item; BCE on p_eager only where it is defined (y_eager >= 0).
    Positive weights from the batch, clamped to [0.1, 10] as upstream train.py does."""
    le, lg = logits[:, 0], logits[:, 1]
    y_end = y_end.float()
    pw = ((y_end == 0).sum() / (y_end == 1).sum().clamp(min=1)).clamp(0.1, 10.0)
    loss_end = nn.functional.binary_cross_entropy_with_logits(le, y_end, pos_weight=pw)
    m = y_eager >= 0
    if m.any():
        ye = y_eager[m].float()
        pwe = ((ye == 0).sum() / (ye == 1).sum().clamp(min=1)).clamp(0.1, 10.0)
        loss_eager = nn.functional.binary_cross_entropy_with_logits(lg[m], ye, pos_weight=pwe)
    else:
        loss_eager = torch.zeros((), device=logits.device)
    return loss_end + w_eager * loss_eager, {"end": float(loss_end), "eager": float(loss_eager)}
