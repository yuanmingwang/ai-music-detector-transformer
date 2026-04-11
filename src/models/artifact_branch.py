import torch
import torch.nn as nn
import torch.nn.functional as F


class ArtifactBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_rate=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(drop_rate),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class ArtifactBranch(nn.Module):
    """Branch B: artifact-focused CNN encoder.

    This branch keeps the repo's existing training path intact by deriving
    artifact-sensitive maps directly from the input spectrogram, then encoding
    them with a compact 2D CNN. The output is a token sequence so the current
    classifier can keep using the same mean-pooling logic as the other models.
    """

    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        embed_dim,
        artifact_channels=(32, 64, 128, 192),
        proj_drop_rate=0.1,
    ):
        super().__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.embed_dim = embed_dim

        c1, c2, c3, c4 = artifact_channels
        self.stem = nn.Sequential(
            nn.Conv2d(6, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ArtifactBlock(c1, c1, stride=1, drop_rate=proj_drop_rate * 0.25),
            ArtifactBlock(c1, c2, stride=2, drop_rate=proj_drop_rate * 0.5),
            ArtifactBlock(c2, c3, stride=2, drop_rate=proj_drop_rate * 0.75),
            ArtifactBlock(c3, c4, stride=2, drop_rate=proj_drop_rate),
        )
        self.token_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.token_proj = nn.Sequential(
            nn.LayerNorm(c4),
            nn.Linear(c4, embed_dim),
            nn.GELU(),
            nn.Dropout(proj_drop_rate),
        )

    def _normalize_map(self, x, eps=1e-6):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (x - mean) / std

    def _build_artifact_maps(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        base = self._normalize_map(x)

        # Emphasize abrupt high-frequency energy that often shows up in aliasing
        # or re-encoding artifacts.
        low_freq_smooth = F.avg_pool2d(base, kernel_size=(5, 1), stride=1, padding=(2, 0))
        high_freq = (base - low_freq_smooth).abs()

        # Highlight sharp local peaks relative to their neighborhood.
        local_max = F.max_pool2d(base, kernel_size=3, stride=1, padding=1)
        local_avg = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        peak_map = F.relu(local_max - local_avg)

        # Short-term temporal changes can expose unstable decoder traces.
        temporal_delta = F.pad(base[:, :, :, 1:] - base[:, :, :, :-1], (1, 0, 0, 0))
        temporal_delta = temporal_delta.abs()

        # Build cheap multi-resolution views so the CNN sees both fine and coarse patterns.
        half_res = F.interpolate(
            F.avg_pool2d(base, kernel_size=2, stride=2),
            size=base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        quarter_res = F.interpolate(
            F.avg_pool2d(base, kernel_size=4, stride=4),
            size=base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Band-ratio maps inject simple frequency-balance cues without changing the pipeline.
        split = max(1, base.shape[-2] // 2)
        low_band = base[:, :, :split, :].mean(dim=(-2, -1), keepdim=True)
        high_band = base[:, :, split:, :].mean(dim=(-2, -1), keepdim=True)
        band_ratio = (high_band - low_band).expand_as(base)

        return torch.cat(
            [base, high_freq, peak_map, temporal_delta, half_res, band_ratio + quarter_res],
            dim=1,
        )

    def forward(self, x):
        artifact_maps = self._build_artifact_maps(x)
        features = self.stem(artifact_maps)
        features = self.blocks(features)
        tokens = self.token_pool(features).flatten(2).transpose(1, 2)
        return self.token_proj(tokens)
