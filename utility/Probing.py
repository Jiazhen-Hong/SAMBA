import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
)
import numpy as np
from collections import Counter

def count_parameters_up_to(model, target_layer_name):
    total_params = 0
    found_layer = False
    for name, module in model.named_modules():
        if hasattr(module, "parameters"):
            for param in module.parameters(recurse=False):
                total_params += param.numel()
        if target_layer_name in name:
            found_layer = True
            break
    if not found_layer:
        print(f"Warning: Layer {target_layer_name} not found in the model!")
    return total_params


def locate_submodule_by_name(model, submodule_name):
    for name in submodule_name.split("."):
        if not hasattr(model, name):
            raise ValueError(f"Submodule '{submodule_name}' not found in model.")
        model = getattr(model, name)
    return model

# Option [1] Convinience for experiment => target layer can be selected pool3, decoder3, bottleneck
def extract_pooled_representation(model, loader, target_submodule_name, device="cuda"):
    model.eval()
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    module = locate_submodule_by_name(core_model, target_submodule_name)
    captured_features = []

    def hook(_, __, output):
        captured_features.append(output)

    handle = module.register_forward_hook(hook)
    all_feats, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            captured_features.clear()
            _ = model(x)
            feats = captured_features[0]
            if feats.dim() == 3:
                feats = compute_time_quantile_stats(feats)
            all_feats.append(feats.cpu())
            all_labels.append(y)

    handle.remove()
    return torch.cat(all_feats), torch.cat(all_labels)

# Option [2] Convinience for compile training => saving time
def extract_decoder3_representation(model, loader, device="cuda", pool_type="quantile"):
    """
    extract decoder 3 of EEGMS for linear probing。
    compare to hook more stable for torch.compile

    Args:
        model: EEGMS 
        loader: DataLoader
        device: .
        pool_type: default quantile, "mean" / "max" / "quantile"

    Returns:
        feats: Tensor, shape [N, F]
        labels: Tensor, shape [N]
    """
    model.eval()
    all_feats, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, d3 = model(x, return_decoder3=True)  

            # d3: [B, C, T] => feats: [B, F]
            if pool_type == "mean":
                feats = d3.mean(dim=2)
            elif pool_type == "max":
                feats = d3.max(dim=2).values
            elif pool_type == "quantile":
                feats = compute_time_quantile_stats(d3)
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}")

            all_feats.append(feats.cpu())
            all_labels.append(y)

    return torch.cat(all_feats), torch.cat(all_labels)

    
# Mannual selected EEG signatures.
def compute_time_quantile_stats(features, q_list=[0.05, 0.25, 0.5, 0.75, 0.95]):
    stats = [
        features.min(dim=2).values,
        features.max(dim=2).values,
        features.mean(dim=2),
        features.std(dim=2),
        torch.quantile(features, torch.tensor(q_list, device=features.device), dim=2).permute(1, 2, 0).reshape(features.size(0), -1)
    ]
    return torch.cat(stats, dim=1)



def fit_lr(features, labels, seed=3407, max_samples=None):
    features, labels = shuffle(features, labels, random_state=seed)
    if max_samples is not None and features.shape[0] > max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
    pipe = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(max_iter=1_000_000, random_state=seed, class_weight='balanced'))
    )
    pipe.fit(features, labels)
    return pipe

def MLP_for_nonlinear_probe(train_feats, train_labels, val_feats, val_labels, test_feats, test_labels,
                            device, log_path, logger, num_classes=2, num_epochs=20, batch_size=64, lr=1e-3):

    def get_loader(x, y):
        ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    train_loader = get_loader(train_feats, train_labels)
    val_loader = get_loader(val_feats, val_labels)
    test_loader = get_loader(test_feats, test_labels)

    class SimpleMLP(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(32, out_dim)
            )
        def forward(self, x): return self.net(x)

    model = SimpleMLP(train_feats.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # crit = nn.CrossEntropyLoss()

    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    class_weights = [total / label_counts[i] for i in range(num_classes)]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=weights_tensor)


    def eval_model(model, loader):
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                probs = F.softmax(out, dim=1)
                preds = probs.argmax(dim=1)
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()
        return y_true, y_prob, y_pred

    # === Training Loop ===
    best_val_loss = float('inf')
    best_path = os.path.join(log_path, "best_linear_probe_mlp.pth")

    for ep in range(num_epochs):
        model.train()
        total_loss = 0.
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        # Eval on val set
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = crit(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        logger.info(f"[Epoch {ep+1}] ValLoss={val_loss:.4f}")

    # === Final test result ===
    model.load_state_dict(torch.load(best_path))
    test_labels, probs, preds = eval_model(model, test_loader)

    logger.info(f"[Test] Final nonlinear probe results exported.")
    torch.save(model.state_dict(), os.path.join(log_path, "final_linear_probe_mlp.pth"))

    return test_labels, probs, preds