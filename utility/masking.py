import torch
import math

#Random Masking
def apply_mask_random(input_tensor, time_mask_ratio=0.5):
    B, C, T = input_tensor.shape
    masked_input = input_tensor.clone()
    mask = torch.ones_like(input_tensor)
    num_mask = int(time_mask_ratio * T)

    for b in range(B):
        mask_indices = torch.randperm(T)[:num_mask]
        masked_input[b, :, mask_indices] = 0
        mask[b, :, mask_indices] = 0
    return masked_input, mask

#Block-wise Random Masking
def apply_mask_block_random(input_tensor, time_mask_ratio=0.5, block_num=8): 
    """
    Mask the input EEG tensor in both time and channel dimensions.
    input_tensor: [B, C, T]
    """
    B, C, T = input_tensor.shape
    masked_input = input_tensor.clone()
    mask = torch.ones_like(input_tensor)

    # Mask time (block-wise masking)
    block_size = T // block_num
    for b in range(B):
        for _ in range(int(T * time_mask_ratio / block_size)):
            start = torch.randint(0, T - block_size, (1,))
            masked_input[b, :, start:start+block_size] = 0
            mask[b, :, start:start+block_size] = 0

    return masked_input, mask

def apply_mask_ssp(input_tensor, time_mask_ratio=0.5, num_preserved_blocks=8):
    """
    SSP-style masking over time axis for all channels equally.
    input_tensor: [B, C, T]
    Returns:
        masked_input: input with masked time points zeroed across all channels
        mask: binary mask tensor (1: visible, 0: masked), same shape
    """
    B, C, T = input_tensor.shape
    masked_input = torch.zeros_like(input_tensor)
    mask = torch.zeros_like(input_tensor)

    total_unmasked_len = int((1 - time_mask_ratio) * T)
    block_len = math.ceil(total_unmasked_len / num_preserved_blocks)

    for b in range(B):
        used_ranges = []
        preserved_starts = []

        for _ in range(num_preserved_blocks):
            valid = False
            trial = 0
            while not valid and trial < 50:
                start = torch.randint(0, T - block_len, (1,)).item()
                end = start + block_len
                # Ensure no overlap
                if all(end <= s or start >= e for s, e in used_ranges):
                    preserved_starts.append((start, end))
                    used_ranges.append((start, end))
                    valid = True
                trial += 1

        for start, end in preserved_starts:
            masked_input[b, :, start:end] = input_tensor[b, :, start:end]
            mask[b, :, start:end] = 1  # 1 = visible

    return masked_input, mask


def apply_mask_ssp(input_tensor, time_mask_ratio=0.5, num_preserved_blocks=8, evenly_spaced=False):
    """
    Semantic Subsequence Preserving (SSP) Masking for EEG input.
    
    Args:
        input_tensor: [B, C, T] EEG tensor
        time_mask_ratio: float, proportion of time to mask (e.g. 0.5 means 50% masked)
        num_preserved_blocks: int, number of semantic blocks to preserve
        evenly_spaced: bool, whether to enforce even spacing of preserved blocks
    
    Returns:
        masked_input: [B, C, T], masked version of input
        mask: [B, C, T], binary mask (1 = unmasked, 0 = masked)
    """
    B, C, T = input_tensor.shape
    masked_input = torch.zeros_like(input_tensor)
    mask = torch.zeros_like(input_tensor)

    total_unmasked_len = int((1 - time_mask_ratio) * T)
    block_len = math.ceil(total_unmasked_len / num_preserved_blocks)

    for b in range(B):
        preserved_ranges = []

        if evenly_spaced: 
            interval = T // num_preserved_blocks
            for i in range(num_preserved_blocks):
                base = i * interval
                start = base + torch.randint(0, max(1, interval - block_len), (1,)).item()
                end = min(start + block_len, T)
                preserved_ranges.append((start, end))
        else:
            used_ranges = []
            trial = 0
            while len(preserved_ranges) < num_preserved_blocks and trial < 100:
                start = torch.randint(0, T - block_len, (1,)).item()
                end = start + block_len
                if all(end <= s or start >= e for s, e in used_ranges):
                    preserved_ranges.append((start, end))
                    used_ranges.append((start, end))
                trial += 1

        for start, end in preserved_ranges:
            masked_input[b, :, start:end] = input_tensor[b, :, start:end]
            mask[b, :, start:end] = 1  # 1 = visible

    return masked_input, mask

# Temporal Semantic Random Masking（TSRM）
def apply_mask_tsr(input_tensor, time_mask_ratio=0.5, num_preserved_blocks=8, min_ratio=0.5, max_ratio=1.5):
    """
    Optimized TSRM (Temporal Semantic Random Masking): strict preserved length, variable block length, no overlap.
    This version is fully vectorized for significant performance improvement on GPU.

    Args:
        input_tensor: [B, C, T]
        time_mask_ratio: fraction of time to mask (e.g., 0.5 means 50% masked)
        num_preserved_blocks: number of preserved semantic blocks
        min_ratio: min scaling factor for block length
        max_ratio: max scaling factor for block length

    Returns:
        masked_input: [B, C, T], masked EEG
        mask: [B, C, T], 1=visible, 0=masked
    """
    B, C, T = input_tensor.shape
    device = input_tensor.device

    # --- Step 1: Batch-Generate All Block Lengths at Once ---
    # Calculate base lengths
    total_preserve_len = int((1 - time_mask_ratio) * T)
    avg_block_len = total_preserve_len // num_preserved_blocks

    # Generate random lengths for the first N-1 blocks for the entire batch
    num_to_rand = num_preserved_blocks - 1
    if num_to_rand > 0:
        rand_scales = torch.rand(B, num_to_rand, device=device) * (max_ratio - min_ratio) + min_ratio
        block_lengths_prefix = (avg_block_len * rand_scales).long().clamp_(min=4)

        # Calculate the length of the last block to ensure the total preserved length is met
        # This respects the logic from your original code
        last_block_len = (total_preserve_len - block_lengths_prefix.sum(dim=1)).clamp_(min=4)
        block_lengths = torch.cat([block_lengths_prefix, last_block_len.unsqueeze(1)], dim=1)
    else:
        # Handle edge case where there is only one block
        block_lengths = torch.full((B, 1), total_preserve_len, device=device, dtype=torch.long)

    # The actual preserved length might slightly differ from total_preserve_len due to clamping,
    # just like in the original implementation.
    actual_preserve_len = block_lengths.sum(dim=1)

    # --- Step 2: Vectorize Block Placement by Allocating Gaps ---
    total_masked_len = T - actual_preserve_len

    # Generate random proportions for the N+1 gaps between preserved blocks
    rand_gaps = torch.rand(B, num_preserved_blocks + 1, device=device)
    # Prevent division by zero if all random numbers are zero
    rand_gaps.add_(1e-6)
    gap_proportions = rand_gaps / rand_gaps.sum(dim=1, keepdim=True)

    # Calculate integer gap lengths and distribute rounding error
    gap_lengths = (gap_proportions * total_masked_len.unsqueeze(1)).floor().long()
    remainder = total_masked_len - gap_lengths.sum(dim=1)
    gap_lengths[:, -1] += remainder  # Add remainder to the last gap

    # --- Step 3: Calculate Block Positions Using Cumulative Sum ---
    # Interleave gaps and blocks and find start/end indices with cumulative sum
    c_block_padded = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.long), block_lengths.cumsum(dim=1)[:, :-1]], dim=1)
    c_gap = gap_lengths.cumsum(dim=1)

    starts = c_gap[:, :-1] + c_block_padded
    ends = starts + block_lengths

    # --- Step 4: Create Final Mask in a Single Vectorized Operation ---
    # Create a time index tensor [0, 1, ..., T-1]
    time_indices = torch.arange(T, device=device).view(1, T)

    # Use broadcasting to create a mask for each block and combine them
    # (B, num_blocks, 1) vs (1, 1, T) -> (B, num_blocks, T)
    block_masks = (time_indices >= starts.unsqueeze(2)) & (time_indices < ends.unsqueeze(2))

    # Combine masks for all blocks with a logical OR
    # (B, num_blocks, T) -> (B, T)
    mask = torch.any(block_masks, dim=1)

    # Expand mask to the final shape [B, C, T]
    mask = mask.unsqueeze(1).expand(-1, C, -1)

    # Apply the mask
    masked_input = input_tensor * mask.float()

    return masked_input, mask


def split_eeg_for_prediction(x, ratio=0.3):
    """
    Args:
        x: Tensor of shape [B, C, T]
        ratio: float, the fraction of future to predict
    Returns:
        past:   [B, C, T_past]
        future: [B, C, T_future]
        mask:   [B, C, T], bool tensor, future part is 1
    """
    B, C, T = x.shape
    T_future = max(1, int(T * ratio))
    T_past = T - T_future

    past = x[:, :, :T_past]
    future = x[:, :, T_past:]

    mask = torch.zeros((B, C, T), dtype=torch.bool, device=x.device)
    mask[:, :, T_past:] = True  # 标记未来区域为 True

    return past, future, mask