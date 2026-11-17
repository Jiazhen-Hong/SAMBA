from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .configuration_samba import SambaConfig


class _GRUMixer(nn.Module):
    """Portable development backend. Released checkpoints use Mamba-2."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(width, width, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


def mamba2_fused_kernels_available() -> bool:
    """Return True only when every function required by Mamba-2's fused path exists."""
    try:
        from causal_conv1d import causal_conv1d_fn
        from mamba_ssm.ops.triton import ssd_combined

        required = (
            causal_conv1d_fn,
            ssd_combined.causal_conv1d_fwd_function,
            ssd_combined.causal_conv1d_bwd_function,
        )
        return all(callable(function) for function in required)
    except Exception:
        return False


def _mixer(config: SambaConfig, width: int, head_dim: int | None = None) -> nn.Module:
    if config.mixer_backend == "gru":
        return _GRUMixer(width)
    if config.mixer_backend != "mamba2":
        raise ValueError(f"Unsupported mixer_backend: {config.mixer_backend}")
    try:
        from mamba_ssm import Mamba2
    except ImportError as error:
        raise ImportError("Install SAMBA with `pip install -e '.[mamba]'` to use Mamba-2") from error
    kwargs = dict(
        d_model=width,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        use_mem_eff_path=mamba2_fused_kernels_available(),
    )
    if head_dim is not None:
        kwargs["headdim"] = head_dim
    return Mamba2(**kwargs)


class CoordinateDrivenSpatialEmbedding(nn.Module):
    def __init__(self, target_positions: torch.Tensor, hidden_dim: int) -> None:
        super().__init__()
        self.register_buffer("target_positions", target_positions.float(), persistent=True)
        self.net = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        signals: torch.Tensor,
        input_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_positions.ndim == 2:
            input_positions = input_positions.unsqueeze(0).expand(signals.shape[0], -1, -1)
        if input_positions.shape != (signals.shape[0], signals.shape[1], 3):
            raise ValueError("input_positions must have shape [C, 3] or [B, C, 3]")
        finite_channels = torch.isfinite(input_positions).all(dim=-1)
        channel_mask = finite_channels if channel_mask is None else channel_mask & finite_channels
        input_positions = torch.nan_to_num(input_positions)
        delta = self.target_positions[None, :, None, :] - input_positions[:, None, :, :]
        logits = self.net(delta.to(dtype=self.net[0].weight.dtype)).squeeze(-1)
        if channel_mask is not None:
            if channel_mask.shape != signals.shape[:2]:
                raise ValueError("channel_mask must have shape [B, C]")
            if not channel_mask.any(dim=1).all():
                raise ValueError("each sample must contain at least one valid channel")
            logits = logits.masked_fill(~channel_mask[:, None, :], torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=-1)
        return torch.einsum("boc,bct->bot", weights, signals), weights


class CoordinateDrivenSpatialDecoder(nn.Module):
    """Map the shared montage back to each sample's fixed raw EEG channels.

    Unlike the encoder CDSE, this module predicts output weights for the
    runtime input positions.  Reconstruction can therefore use the original
    EEG as a stationary target even when pretraining mixes montages.
    """

    def __init__(self, source_positions: torch.Tensor, hidden_dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "source_positions", source_positions.float(), persistent=True
        )
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        signals: torch.Tensor,
        output_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_positions.ndim == 2:
            output_positions = output_positions.unsqueeze(0).expand(
                signals.shape[0], -1, -1
            )
        if output_positions.ndim != 3 or output_positions.shape[0] != signals.shape[0]:
            raise ValueError("output_positions must have shape [C, 3] or [B, C, 3]")
        finite_channels = torch.isfinite(output_positions).all(dim=-1)
        output_mask = (
            finite_channels
            if channel_mask is None
            else channel_mask & finite_channels
        )
        output_positions = torch.nan_to_num(output_positions)
        delta = (
            output_positions[:, :, None, :]
            - self.source_positions[None, None, :, :]
        )
        logits = self.net(delta.to(dtype=self.net[0].weight.dtype)).squeeze(-1)
        weights = logits.softmax(dim=-1)
        decoded = torch.einsum("boc,bct->bot", weights, signals)
        decoded = decoded.masked_fill(~output_mask[..., None], 0)
        return decoded, weights


class _ResidualMixer(nn.Module):
    def __init__(self, config: SambaConfig, width: int) -> None:
        super().__init__()
        self.mixer = _mixer(config, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(x)


class _DifferentialMamba(nn.Module):
    def __init__(self, config: SambaConfig) -> None:
        super().__init__()
        width, head = config.bottleneck_channels, config.head_dim
        self.head_dim = head
        self.first = nn.ModuleList([_mixer(config, head, head) for _ in range(width // head)])
        self.second = nn.ModuleList([_mixer(config, head, head) for _ in range(width // head)])
        self.lam = nn.Parameter(torch.ones(width))
        self.norms = nn.ModuleList([nn.GroupNorm(1, head) for _ in self.first])
        self.projection = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for index, (first, second, norm, head) in enumerate(zip(self.first, self.second, self.norms, x.split(self.head_dim, -1))):
            lam = self.lam[index * self.head_dim : (index + 1) * self.head_dim]
            difference = first(head.contiguous()) - lam * second(head.contiguous())
            outputs.append(norm(difference.transpose(1, 2)).transpose(1, 2))
        return x + self.projection(torch.cat(outputs, dim=-1))


@dataclass
class SambaOutput:
    reconstruction: torch.Tensor
    last_hidden_state: torch.Tensor
    spatial_weights: torch.Tensor


class SambaModel(nn.Module):
    """SAMBA v2 with runtime montage support and portable checkpoint I/O."""

    def __init__(self, config: SambaConfig) -> None:
        super().__init__()
        self.config = config
        base, bottleneck = config.base_channels, config.bottleneck_channels
        self.cdse = CoordinateDrivenSpatialEmbedding(torch.tensor(config.target_positions), config.cdse_hidden_dim)
        target_channels = config.target_channels
        self.input_embedding = nn.Sequential(
            _MultiBranchEmbedding(target_channels, base),
        )
        self.encoder1 = nn.Sequential(nn.Linear(base, base), _ResidualMixer(config, base))
        self.encoder2 = nn.Conv1d(base, base, 3, padding=1)
        self.encoder3 = nn.Conv1d(base, bottleneck, 3, padding=1)
        self.bottleneck = nn.Sequential(nn.Linear(bottleneck, bottleneck), _DifferentialMamba(config), nn.Linear(bottleneck, bottleneck))
        self.decoder3 = nn.Conv1d(bottleneck * 2, bottleneck, 3, padding=1)
        self.decoder3_mixer = _ResidualMixer(config, bottleneck)
        self.decoder2 = nn.Conv1d(base + bottleneck, base, 3, padding=1)
        self.decoder2_mixer = _ResidualMixer(config, base)
        self.decoder1 = nn.Conv1d(base * 2, base, 3, padding=1)
        self.output_embedding = nn.Conv1d(base, config.output_channels, 1)
        self.spatial_decoder = (
            CoordinateDrivenSpatialDecoder(
                torch.tensor(config.target_positions), config.cdse_hidden_dim
            )
            if config.reconstruction_mode == "input_coordinates"
            else None
        )

    def _encode_all(self, signals, input_positions, channel_mask=None):
        if signals.ndim != 3 or signals.shape[-1] < 8:
            raise ValueError("signals must have shape [B, C, T] with T >= 8")
        projected, weights = self.cdse(signals, input_positions, channel_mask)
        embedded = self.input_embedding(projected)
        x1 = self.encoder1(embedded.transpose(1, 2)).transpose(1, 2)
        x2 = self.encoder2(F.max_pool1d(x1, 2))
        x3 = self.encoder3(F.max_pool1d(x2, 2))
        hidden = self.bottleneck(F.max_pool1d(x3, 2).transpose(1, 2))
        return x1, x2, x3, hidden, weights

    def encode(
        self,
        signals: torch.Tensor,
        input_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the paper's post-MDM [B, T/8, D] downstream representation."""
        return self._encode_all(signals, input_positions, channel_mask)[3]

    def forward(self, signals: torch.Tensor, input_positions: torch.Tensor, channel_mask: torch.Tensor | None = None) -> SambaOutput:
        x1, x2, x3, hidden, weights = self._encode_all(signals, input_positions, channel_mask)
        up3 = F.interpolate(hidden.transpose(1, 2), size=x3.shape[-1], mode="linear", align_corners=False)
        d3 = self.decoder3_mixer(self.decoder3(torch.cat([x3, up3], 1)).transpose(1, 2)).transpose(1, 2)
        d3 = F.interpolate(d3, size=x2.shape[-1], mode="linear", align_corners=False)
        d2 = self.decoder2_mixer(self.decoder2(torch.cat([x2, d3], 1)).transpose(1, 2)).transpose(1, 2)
        d2 = F.interpolate(d2, size=x1.shape[-1], mode="linear", align_corners=False)
        reconstruction = self.output_embedding(
            self.decoder1(torch.cat([x1, d2], 1))
        )
        if self.spatial_decoder is not None:
            reconstruction, _ = self.spatial_decoder(
                reconstruction, input_positions, channel_mask
            )
        return SambaOutput(reconstruction, hidden, weights)

    def save_pretrained(self, directory: str | Path, safe_serialization: bool = True) -> None:
        path = Path(directory)
        self.config.save_pretrained(path)
        state = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        if safe_serialization:
            try:
                from safetensors.torch import save_file
                save_file(state, str(path / "model.safetensors"))
                return
            except ImportError:
                pass
        torch.save(state, path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, directory: str | Path, map_location: str | torch.device = "cpu") -> "SambaModel":
        path = Path(directory)
        model = cls(SambaConfig.from_pretrained(path))
        safe_path = path / "model.safetensors"
        if safe_path.exists():
            from safetensors.torch import load_file
            state = load_file(str(safe_path), device=str(map_location))
        else:
            state = torch.load(path / "pytorch_model.bin", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


class _MultiBranchEmbedding(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(input_channels, output_channels, kernel, padding=kernel // 2) for kernel in (1, 3, 7)])
        self.fuse = nn.Conv1d(output_channels * 3, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
