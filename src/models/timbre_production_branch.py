import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimbreProductionBranch(nn.Module):
    """Branch C: descriptor-driven timbre and production encoder.

    The branch stays lightweight by computing simple segment-level descriptors
    directly from the mel spectrogram, then modeling their sequence with a
    small BiGRU. It returns token embeddings so the current classifier can keep
    using the same mean-pooling path as the other custom backbones.
    """

    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        embed_dim,
        segment_frames=None,
        timbre_embed_dim=32,
        descriptor_hidden_dim=128,
        gru_hidden_dim=128,
        gru_layers=2,
        proj_drop_rate=0.1,
    ):
        super().__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.embed_dim = embed_dim
        self.segment_frames = min(
            segment_frames or max(8, input_temp_dim // 8), input_temp_dim
        )
        self.timbre_embed_dim = timbre_embed_dim
        self.descriptor_dim = 8 + timbre_embed_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_directions = 2

        self.timbre_proj = nn.Sequential(
            nn.LayerNorm(input_spec_dim),
            nn.Linear(input_spec_dim, descriptor_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop_rate),
            nn.Linear(descriptor_hidden_dim, timbre_embed_dim),
        )
        self.temporal_model = nn.GRU(
            input_size=self.descriptor_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=proj_drop_rate if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim * self.num_directions),
            nn.Linear(gru_hidden_dim * self.num_directions, embed_dim),
            nn.GELU(),
            nn.Dropout(proj_drop_rate),
        )

    def _segment_spectrogram(self, x):
        num_segments = max(1, math.ceil(self.input_temp_dim / self.segment_frames))
        target_frames = num_segments * self.segment_frames
        if target_frames != x.shape[-1]:
            pad = target_frames - x.shape[-1]
            x = F.pad(x, (0, pad, 0, 0))
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            self.input_spec_dim,
            num_segments,
            self.segment_frames,
        )
        return x.permute(0, 1, 3, 2, 4).contiguous()

    def _compute_descriptor_sequence(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        segments = self._segment_spectrogram(x)
        segments = segments.squeeze(1)  # (B, S, F, T_seg)
        eps = 1e-6

        power = segments.exp()
        frame_energy = power.mean(dim=2)

        freq_positions = torch.linspace(
            0.0, 1.0, self.input_spec_dim, device=x.device, dtype=x.dtype
        ).view(1, 1, self.input_spec_dim, 1)
        freq_mass = power.sum(dim=-1).clamp_min(eps)
        freq_weight = freq_mass / freq_mass.sum(dim=2, keepdim=True).clamp_min(eps)

        spectral_centroid = (freq_weight * freq_positions.squeeze(-1)).sum(dim=2)
        spectral_bandwidth = torch.sqrt(
            ((freq_positions.squeeze(-1) - spectral_centroid.unsqueeze(-1)) ** 2 * freq_weight).sum(dim=2).clamp_min(eps)
        )

        cdf = freq_weight.cumsum(dim=2)
        spectral_rolloff = (cdf < 0.85).sum(dim=2).float() / max(1, self.input_spec_dim - 1)

        rms_mean = torch.sqrt(frame_energy.clamp_min(eps)).mean(dim=-1)
        rms_variability = torch.sqrt(frame_energy.clamp_min(eps)).std(dim=-1)

        temporal_delta = segments[:, :, :, 1:] - segments[:, :, :, :-1]
        onset_density = F.relu(temporal_delta).mean(dim=(2, 3))
        transient_sharpness = temporal_delta.abs().amax(dim=(2, 3))

        harmonic_smooth = F.avg_pool2d(
            power.reshape(-1, 1, self.input_spec_dim, self.segment_frames),
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        ).reshape_as(power)
        noise_residual = (power - harmonic_smooth).abs()
        harmonic_energy = harmonic_smooth.mean(dim=(2, 3))
        noise_energy = noise_residual.mean(dim=(2, 3)).clamp_min(eps)
        harmonic_to_noise = harmonic_energy / noise_energy

        # A learned timbre vector complements the hand-crafted descriptors
        # without changing the repo's mel-spectrogram data path.
        timbre_input = segments.mean(dim=-1)
        timbre_embed = self.timbre_proj(timbre_input)

        descriptor_list = [
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            rms_mean,
            rms_variability,
            onset_density,
            transient_sharpness,
            harmonic_to_noise,
        ]
        descriptors = torch.stack(descriptor_list, dim=-1)
        descriptors = torch.cat([descriptors, timbre_embed], dim=-1)

        desc_mean = descriptors.mean(dim=1, keepdim=True)
        desc_std = descriptors.std(dim=1, keepdim=True).clamp_min(eps)
        return (descriptors - desc_mean) / desc_std

    def forward(self, x):
        descriptor_seq = self._compute_descriptor_sequence(x)
        sequence_tokens, _ = self.temporal_model(descriptor_seq)
        return self.token_proj(sequence_tokens)
