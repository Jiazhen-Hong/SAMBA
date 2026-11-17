from __future__ import annotations

import math

import torch


def temporal_semantic_random_mask(
    signals: torch.Tensor,
    mask_ratio: float = 0.5,
    num_visible_blocks: int = 8,
    min_block_ratio: float = 0.5,
    max_block_ratio: float = 1.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask time spans while preserving exactly the requested number of samples.

    Returns the masked signal and a boolean mask where True denotes a masked value.
    The same time mask is applied to all channels in one recording.
    """
    if signals.ndim != 3:
        raise ValueError("signals must have shape [batch, channels, time]")
    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio must be in [0, 1)")
    if not 0 < min_block_ratio <= 1 <= max_block_ratio:
        raise ValueError("block ratios must satisfy 0 < min <= 1 <= max")
    batch, channels, length = signals.shape
    visible = length - int(round(length * mask_ratio))
    blocks = max(1, min(num_visible_blocks, visible))
    average = visible / blocks
    minimum = max(1, math.floor(min_block_ratio * average))
    maximum = max(minimum, math.ceil(max_block_ratio * average))
    block_lengths = torch.empty(
        (batch, blocks), dtype=torch.long, device=signals.device
    )
    remaining = torch.full(
        (batch,), visible, dtype=torch.long, device=signals.device
    )
    for index in range(blocks - 1):
        remaining_blocks = blocks - index - 1
        low = torch.maximum(
            torch.full_like(remaining, minimum),
            remaining - remaining_blocks * maximum,
        )
        high = torch.minimum(
            torch.full_like(remaining, maximum),
            remaining - remaining_blocks * minimum,
        )
        draw = torch.rand(batch, device=signals.device, generator=generator)
        sampled = low + torch.floor(draw * (high - low + 1)).long()
        block_lengths[:, index] = sampled
        remaining -= sampled
    block_lengths[:, -1] = remaining

    gap_total = length - visible
    raw_gaps = torch.rand((batch, blocks + 1), device=signals.device, generator=generator)
    gaps = torch.floor(raw_gaps / raw_gaps.sum(dim=1, keepdim=True) * gap_total).long()
    gaps[:, -1] += gap_total - gaps.sum(dim=1)

    starts = torch.empty_like(block_lengths)
    starts[:, 0] = gaps[:, 0]
    if blocks > 1:
        starts[:, 1:] = gaps[:, :1] + torch.cumsum(
            block_lengths[:, :-1] + gaps[:, 1:blocks], dim=1
        )
    ends = starts + block_lengths
    time = torch.arange(length, device=signals.device)[None, None, :]
    visible_mask = ((time >= starts[..., None]) & (time < ends[..., None])).any(dim=1)
    mask = (~visible_mask)[:, None, :].expand(batch, channels, length)
    return signals.masked_fill(mask, 0), mask
