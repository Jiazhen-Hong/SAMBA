import os
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset, Dataset
from scipy.signal import iirdesign, zpk2sos, sosfiltfilt, cheb2ord


class DataUtils:

    @staticmethod
    def EmotivLoader(train_data, train_label, val_data=None, val_label=None, test_data=None, test_label=None,
                     batch_size=128, num_workers=8):
        """Create DataLoaders for Emotiv-style labeled datasets."""
        def make_loader(data, label, shuffle):
            dataset = TensorDataset(torch.tensor(data, dtype=torch.float32),
                                    torch.tensor(label, dtype=torch.long))
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        train_loader = make_loader(train_data, train_label, shuffle=True)
        val_loader = make_loader(val_data, val_label, shuffle=False) if val_data is not None else None
        test_loader = make_loader(test_data, test_label, shuffle=False) if test_data is not None else None
        return train_loader, val_loader, test_loader

    @staticmethod
    def TUABLoader(tuab_root, batch_size=128, num_workers=8):
        """Create DataLoaders for TUAB EEG datasets from directory structure."""
        def get_loader(split):
            split_path = os.path.join(tuab_root, split)
            files = os.listdir(split_path)
            dataset = TUABprocess((split_path, files))
            return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)

        return get_loader("train"), get_loader("val"), get_loader("test")

    @staticmethod
    def chebyBandpassFilter(data, cutoff, gstop=40, gpass=0.5, fs=128):
        """Apply a Chebyshev Type-II bandpass filter to EEG data."""
        wp = [cutoff[1] / (fs / 2), cutoff[2] / (fs / 2)]
        ws = [cutoff[0] / (fs / 2), cutoff[3] / (fs / 2)]
        z, p, k = iirdesign(wp=wp, ws=ws, gstop=gstop, gpass=gpass, ftype='cheby2', output='zpk')
        sos = zpk2sos(z, p, k)

        if data.ndim == 1:
            return sosfiltfilt(sos, data)
        elif data.ndim == 2:
            return np.array([sosfiltfilt(sos, ch) for ch in data])
        elif data.ndim == 3:
            return np.array([[sosfiltfilt(sos, data[t, c]) for c in range(data.shape[1])]
                             for t in range(data.shape[0])])
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")


class TUABprocess(Dataset):
    def __init__(self, tuab_data):
        self.root, self.files = tuab_data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)

        X = sample["X"][:16, :]  
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        Y = sample["y"]
        return torch.FloatTensor(X), Y


def compute_erp_template(train_loader, target_label=1, device="cuda"):
    '''
    used for P300ERP loss
    '''
    erp_accumulator = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        mask = y == target_label
        if mask.any():
            erp_accumulator.append(x[mask])
    if len(erp_accumulator) == 0:
        raise ValueError("No target samples found in training data.")
    erp_template = torch.cat(erp_accumulator, dim=0).mean(dim=0)  # shape: [C, T]
    return erp_template
