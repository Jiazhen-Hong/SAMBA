from __future__ import annotations

import torch


def temporal_statistics(features: torch.Tensor) -> torch.Tensor:
    """Paper-defined nine-statistic summary for [B, T, D] representations."""
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, time, features]")
    quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=features.device)
    parts = [
        features.amin(dim=1),
        features.amax(dim=1),
        features.mean(dim=1),
        features.std(dim=1, unbiased=False),
        torch.quantile(features, quantiles, dim=1).permute(1, 2, 0).flatten(1),
    ]
    return torch.cat(parts, dim=1)
