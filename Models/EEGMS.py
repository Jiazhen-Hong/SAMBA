import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2


#3D x, y, z
class SAIE_Learned_Global(nn.Module):
    def __init__(self, hidden_dim=32):
        """
        Channel-agnostic SAIE: maps (x_in, x_out, z_in, z_out) → weight using a shared MLP.

        Args:
            hidden_dim (int): hidden dimension of the MLP.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 从2改为3
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, input_coords, target_coords):
        """
        Spatio-Adaptive Input Embedding (SAIE) forward pass.

        Args:
            x: Tensor of shape [B, C_in, T], where C_in is the number of input channels.
            **Assumption**: The order of channels in `x` must match the order of `input_coords`.
            input_coords: dict or ordered mapping from channel name to 3D coordinates (X, Y, Z) for input space.
            target_coords: dict or ordered mapping from channel name to 3D coordinates (X, Y, Z) for target space.

        Returns:
            x_out: Tensor of shape [B, C_out, T], projected into the target coordinate space.
            weights: Tensor of shape [C_out, C_in], representing learned spatial projection weights.
        """
        B, C_in, T = x.shape
        device = x.device

        # Convert coordinates to tensors [C_in, 3] and [C_out, 3]
        in_pos = torch.tensor(list(input_coords.values()), dtype=torch.float32, device=device)
        out_pos = torch.tensor(list(target_coords.values()), dtype=torch.float32, device=device)

        # Compute pairwise 3D differences for all (out, in) channel pairs → [C_out, C_in, 3]
        diff = out_pos[:, None, :] - in_pos[None, :, :]

        # Predict weights via MLP and normalize → [C_out, C_in]
        weights = self.net(diff).squeeze(-1)
        weights = torch.softmax(weights, dim=1)

        # Apply spatial projection → [B, C_out, T]
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


'''
def DiffAttn(X, W_q, W_k, W_v, λ):
    Q1, Q2 = split(X @ W_q)
    K1, K2 = split(X @ W_k)
    V = X @ W_v

    s = 1 / sqrt(d)
    A1 = Q1 @ K1.T * s
    A2 = Q2 @ K2.T * s
    return (softmax(A1) - λ * softmax(A2)) @ V


def MultiHead(X, W_qi, W_ki, W_vi, W_o, λ):
    O = GroupNorm([DiffAttn(X, W_qi, W_ki, W_vi, λ) for i in range(h)])
    O = O * (1 - λ_init)
    return Concat(O) @ W_o
'''

class DiffMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, headdim=16):
        super().__init__()
        assert d_model % headdim == 0, "d_model must be divisible by headdim"
        self.num_heads = d_model // headdim
        self.headdim = headdim
        self.d_model = d_model

        self.team1 = nn.ModuleList([
            Mamba2(d_model=headdim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
            for _ in range(self.num_heads)
        ])
        self.team2 = nn.ModuleList([
            Mamba2(d_model=headdim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
            for _ in range(self.num_heads)
        ])

        self.lam = nn.Parameter(torch.ones(d_model))
        self.head_norms = nn.ModuleList([
            nn.GroupNorm(num_groups=1, num_channels=headdim) for _ in range(self.num_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        B, T, D = x.shape
        split_x = torch.chunk(x, self.num_heads, dim=-1)  # list of [B, T, headdim]

        head_outputs = []
        for i in range(self.num_heads):
            xh = split_x[i].contiguous()
            # xh = split_x[i]
            x1 = self.team1[i](xh)
            x2 = self.team2[i](xh)
            lam_i = self.lam[i * self.headdim : (i + 1) * self.headdim]
            diff = x1 - lam_i * x2
            diff = self.head_norms[i](diff.transpose(1, 2)).transpose(1, 2)
            head_outputs.append(diff)

        out = torch.cat(head_outputs, dim=-1)  # [B, T, d_model]
        return x + self.out_proj(out)
    
class DMBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, headdim=16):
        super().__init__()
        self.block = DiffMamba(     
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim
        )

    def forward(self, x):  # x: [B, T, d_model]
        return self.block(x)
  
  
class SingleMambaModel(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.RMSNorm(d_model)
        
    def forward(self, x):
        return x + self.mamba(x)

class EEGMS(nn.Module):
    def __init__(self, out_channels, d_state, d_conv, expand, input_coords=None, target_coords=None, logger=None):
        super(EEGMS, self).__init__()

        # Initial parameters
        self.logger = logger
        self.logged_input_shapes = False
        base_channels = 64

        # SAIE: Spatio-Adaptive Input Embedding
        assert input_coords is not None and target_coords is not None, "SAIE requires input and target coordinates"
        self.saie = SAIE_Learned_Global()
        self.input_coords = input_coords
        self.target_coords = target_coords

        # Multi-branch input embedding
        self.input_embedding = MultiBranchInputEmbedding(len(target_coords), base_channels)

        # Encoder1: 64 → 64
        self.encoder1 = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            SingleMambaModel(d_model=base_channels, d_state=d_state, d_conv=d_conv, expand=expand),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder2: 64 → 64
        self.encoder2 = nn.Conv1d(base_channels, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder3: 64 → 128
        self.encoder3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck: 128 → 128
        self.bottleneck = nn.Sequential(
            nn.Linear(128, 128),
            DMBlock(d_model=128, d_state=d_state, d_conv=d_conv, expand=expand),
            nn.Linear(128, 128),
        )

        # Decoder3: 128 + 128 → 128
        self.decoder3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.decodeMamba3 = SingleMambaModel(d_model=128, d_state=d_state, d_conv=d_conv, expand=expand)

        # Decoder2: 64 + 128 → 64
        self.decoder2 = nn.Conv1d(192, 64, kernel_size=3, padding=1)
        self.decodeMamba2 = SingleMambaModel(d_model=64, d_state=d_state, d_conv=d_conv, expand=expand)

        # Decoder1: 64 + 64 → 64
        self.decoder1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)

        # Output embedding
        self.onput_embedding = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x, return_weights=False, return_decoder3=False):
        if self.logger and not self.logged_input_shapes:
            self.logger.info(f"Input data shape before multi-branch embedding: {x.shape}")

        # SAIE projection: [B, C_in, T] → [B, C_out, T]
        x, ch_weights = self.saie(x, self.input_coords, self.target_coords)

        # Input embedding: [B, C_out, T] → [B, 64, T]
        x = self.input_embedding(x)
        if self.logger and not self.logged_input_shapes:
            self.logger.info(f"Input embedding shape after multi-branch: {x.shape}")
            self.logged_input_shapes = True

        # ---- Encoder 1 ----
        x = x.permute(0, 2, 1)               # => (B, T, 64)
        x1 = self.encoder1(x)                                # => (B, T, 64)
        x1 = x1.permute(0, 2, 1)             # => (B, 64, T)
        x1p = self.pool1(x1)                 # => (B, 64, T/2)

        # ---- Encoder 2 ----
        x2 = self.encoder2(x1p)              # => (B, 64, T/2)
        x2p = self.pool2(x2)                 # => (B, 64, T/4)

        # ---- Encoder 3 ----
        x3 = self.encoder3(x2p)              # => (B, 128, T/4)
        x3p = self.pool3(x3)                 # => (B, 128, T/8)

        # ---- Bottleneck ----
        x3p = x3p.permute(0, 2, 1)                      # => (B, T/8, 128)
        bottleneck = self.bottleneck(x3p)                           # => (B, T/8, 128)
        bottleneck = bottleneck.permute(0, 2, 1)        # => (B, 128, T/8)

        # Upsample bottleneck to match x3: T/8 → T/4
        bottleneck = F.interpolate(bottleneck, size=x3.size(2), mode='linear', align_corners=False)

        # ---- Decoder 3 ----
        d3 = torch.cat([x3, bottleneck], dim=1)              # => (B, 256, T/4)
        d3 = self.decoder3(d3)                                # => (B, 128, T/4)
        d3 = d3.permute(0, 2, 1)                  # => (B, T/4, 128)
        d3 = self.decodeMamba3(d3)                              # => (B, T/4, 128)
        d3 = d3.permute(0, 2, 1)                  # => (B, 128, T/4)

        # Upsample to match x2: T/4 → T/2
        d3 = F.interpolate(d3, size=x2.size(2), mode='linear', align_corners=False)

        # ---- Decoder 2 ----
        d2 = torch.cat([x2, d3], dim=1)                        # => (B, 192, T/2)
        d2 = self.decoder2(d2)                                 # => (B, 64, T/2)
        d2 = d2.permute(0, 2, 1)                  # => (B, T/2, 64)
        d2 = self.decodeMamba2(d2)                             # => (B, T/2, 64)
        d2 = d2.permute(0, 2, 1)                  # => (B, 64, T/2)

        # Upsample to match x1: T/2 → T
        d2 = F.interpolate(d2, size=x1.size(2), mode='linear', align_corners=False)

        # ---- Decoder 1 ----
        d1 = torch.cat([x1, d2], dim=1)           # => (B, 128, T)
        d1 = self.decoder1(d1)                    # => (B, 64, T)

        # ---- Output embedding ----
        out = self.onput_embedding(d1)            # => (B, out_channels, T)

        if return_weights and return_decoder3:
            return out, ch_weights, d3
        elif return_weights:
            return out, ch_weights
        elif return_decoder3:
            return out, d3
        else:
            return out


# Prediction Head CNN only
class EEGMSPredictionHead(nn.Module):
    def __init__(self):
        super(EEGMSPredictionHead, self).__init__()
        self.head = None
        self.built = False

    def forward(self, x, target_length=None, target_channels=None):
        """
        Args:
            x: Tensor of shape [B, C_in, T] - decoder features (e.g., decoder3_feat)
            target_length: int - desired temporal length (usually x_future.shape[-1])
            target_channels: int - number of output channels (usually x_future.shape[1])
        Returns:
            pred: Tensor of shape [B, target_channels, target_length]
        """
        if not self.built:
            in_channels = x.shape[1]
            if target_channels is None:
                raise ValueError("target_channels must be provided on first forward pass.")
            self.head = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels, target_channels, kernel_size=1)
            )
            self.head.to(x.device)
            self.built = True

        out = self.head(x)
        if target_length is not None and out.shape[-1] != target_length:
            out = F.interpolate(out, size=target_length, mode='linear', align_corners=False)
        return out


# Prediction Head CNN+Mamba
class EEGMSPredictionHead(nn.Module):
    def __init__(self, d_state=16, d_conv=4, expand=2):
        super(EEGMSPredictionHead, self).__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.proj = None         # 1x1 Conv projection
        self.temporal_block = None
        self.built = False

    def forward(self, x, target_length=None, target_channels=None):
        """
        x: [B, C_in, T] - decoder feature
        target_length: int - output time length (e.g., T_future)
        target_channels: int - output channel size (e.g., EEG channel count)
        """
        if not self.built:
            in_channels = x.shape[1]
            if target_channels is None:
                raise ValueError("target_channels must be provided on first forward pass.")

            self.proj = nn.Conv1d(in_channels, target_channels, kernel_size=1)
            self.temporal_block = SingleMambaModel(
                d_model=target_channels,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand
            )
            self.proj.to(x.device)
            self.temporal_block.to(x.device)
            self.built = True

        out = self.proj(x)                    # [B, target_channels, T_in]
        out = out.permute(0, 2, 1)            # → [B, T_in, C]
        out = self.temporal_block(out)        # → [B, T_in, C]
        out = out.permute(0, 2, 1)            # → [B, C, T_in]

        if target_length is not None and out.shape[-1] != target_length:
            out = F.interpolate(out, size=target_length, mode='linear', align_corners=False)

        return out
  