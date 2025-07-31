import numpy as np
def concatenate_trials(data, labels, target_length):
    """
    Concatenate trials into longer sequences, grouped by label.
    Ensure that sequences are only formed by trials with the same label.

    Args:
        data: shape (N, C, T)
        labels: shape (N,)
        target_length: desired sequence length (T')
    
    Returns:
        new_data: shape (N', C, T')
        new_labels: shape (N',)
    """
    from collections import defaultdict
    grouped_data = defaultdict(list)

    # Group trials by label
    for x, y in zip(data, labels):
        grouped_data[y].append(x)

    new_data, new_labels = [], []

    for label, trials in grouped_data.items():
        trials = np.array(trials)  # shape (n, C, T)
        n, C, T = trials.shape
        trials_per_segment = target_length // T
        usable_segments = n // trials_per_segment

        for i in range(usable_segments):
            start = i * trials_per_segment
            end = start + trials_per_segment
            segment = np.concatenate(trials[start:end], axis=-1)  # shape (C, target_length)
            new_data.append(segment)
            new_labels.append(label)

    return np.stack(new_data), np.array(new_labels)