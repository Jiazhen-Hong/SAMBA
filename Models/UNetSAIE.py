"""
Created on Mon May 14 04:12:39 2025

@author: jiazhen@emotiv.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAIE_Learned_Global(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, input_coords, target_coords, input_channel_names=None):
        B, C_in, T = x.shape
        device = x.device

        if input_channel_names is None:
            input_channel_names = list(input_coords.keys())[:C_in]

        in_pos = torch.tensor([input_coords[ch] for ch in input_channel_names], dtype=torch.float32, device=device)
        out_pos = torch.tensor([target_coords[ch] for ch in target_coords], dtype=torch.float32, device=device)

        diff = out_pos[:, None, :] - in_pos[None, :, :]  # [C_out, C_in, 3]
        weights = self.net(diff).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        x_out = torch.einsum('oc,bct->bot', weights, x)
        return x_out, weights

class MultiBranchInputEmbedding(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.fuse = nn.Conv1d(3 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b7 = self.branch7(x)
        return self.fuse(torch.cat([b1, b3, b7], dim=1))

class UNetSAIE(nn.Module):
    def __init__(self, out_channels, input_coords=None, target_coords=None, logger=None):
        super(UNetSAIE, self).__init__()
        self.logger = logger
        self.logged_input_shapes = False
        base_channels = 64

        assert input_coords is not None and target_coords is not None
        self.saie = SAIE_Learned_Global()
        self.input_coords = input_coords
        self.target_coords = target_coords

        self.input_embedding = MultiBranchInputEmbedding(len(target_coords), base_channels)

        self.encoder1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv1d(base_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv1d(192, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.output_layer = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x, return_weights=False):
        if self.logger and not self.logged_input_shapes:
            self.logger.info(f"Input shape before embedding: {x.shape}")

        x, ch_weights = self.saie(x, self.input_coords, self.target_coords)
        x = self.input_embedding(x)
        if self.logger and not self.logged_input_shapes:
            self.logger.info(f"Input embedding shape: {x.shape}")
            self.logged_input_shapes = True

        x1 = self.encoder1(x)
        x1p = self.pool1(x1)

        x2 = self.encoder2(x1p)
        x2p = self.pool2(x2)

        x3 = self.encoder3(x2p)
        x3p = self.pool3(x3)

        bottleneck = self.bottleneck(x3p)
        bottleneck = F.interpolate(bottleneck, size=x3.size(2), mode='linear', align_corners=False)

        d3 = torch.cat([x3, bottleneck], dim=1)
        d3 = self.decoder3(d3)
        d3 = F.interpolate(d3, size=x2.size(2), mode='linear', align_corners=False)

        d2 = torch.cat([x2, d3], dim=1)
        d2 = self.decoder2(d2)
        d2 = F.interpolate(d2, size=x1.size(2), mode='linear', align_corners=False)

        d1 = torch.cat([x1, d2], dim=1)
        d1 = self.decoder1(d1)

        out = self.output_layer(d1)
        if return_weights:
            return out, ch_weights
        else:
            return out
