import torch
import torch.nn as nn

from src.models.spectttra import SpecTTTra


class LocalWindowTransformer(nn.Module):
    """Local-window encoder that follows the same encoder contract as SpecTTTra."""

    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        embed_dim,
        t_clip,
        f_clip,
        num_heads,
        num_layers,
        pre_norm=False,
        pe_learnable=False,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        mlp_ratio=4.0,
        local_window_size=None,
        local_num_windows=4,
        local_t_clip=None,
        local_f_clip=None,
        local_num_heads=None,
        local_num_layers=None,
        fusion_drop_rate=0.2,
    ):
        super().__init__()
        self.input_temp_dim = input_temp_dim
        self.local_num_windows = max(1, local_num_windows)
        self.local_window_size = min(
            local_window_size or max(1, input_temp_dim // 4), input_temp_dim
        )
        self.embed_dim = embed_dim

        self.window_encoder = SpecTTTra(
            input_spec_dim=input_spec_dim,
            input_temp_dim=self.local_window_size,
            embed_dim=embed_dim,
            t_clip=local_t_clip or t_clip,
            f_clip=local_f_clip or f_clip,
            num_heads=local_num_heads or num_heads,
            num_layers=local_num_layers or num_layers,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
            pos_drop_rate=pos_drop_rate,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            mlp_ratio=mlp_ratio,
        )
        self.window_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(fusion_drop_rate),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def _get_window_starts(self, device):
        max_start = max(self.input_temp_dim - self.local_window_size, 0)
        if self.local_num_windows == 1 or max_start == 0:
            return torch.zeros(self.local_num_windows, dtype=torch.long, device=device)

        if self.training:
            # Sample one window per region so training sees local variation while
            # still covering early, middle, and late parts of the song.
            boundaries = torch.linspace(
                0, max_start + 1, self.local_num_windows + 1, device=device
            )
            starts = []
            for idx in range(self.local_num_windows):
                low = int(boundaries[idx].item())
                high = max(low, int(boundaries[idx + 1].item()) - 1)
                starts.append(torch.randint(low, high + 1, (1,), device=device))
            return torch.cat(starts, dim=0)

        # Use evenly spaced windows at eval time so inference is deterministic.
        return torch.linspace(
            0, max_start, self.local_num_windows, device=device
        ).round().long()

    def _extract_local_windows(self, x):
        starts = self._get_window_starts(x.device)
        windows = []
        for start in starts.tolist():
            end = start + self.local_window_size
            windows.append(x[:, :, start:end])
        return torch.stack(windows, dim=1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)

        local_windows = self._extract_local_windows(x)
        batch_size, num_windows, spec_dim, window_size = local_windows.shape
        local_windows = local_windows.reshape(
            batch_size * num_windows, spec_dim, window_size
        )

        window_tokens = self.window_encoder(local_windows)
        window_embeds = window_tokens.mean(dim=1).reshape(batch_size, num_windows, -1)
        window_embeds = self.window_proj(window_embeds)

        # Return one token per local window so the classifier can keep using
        # the same mean-pooling path as the existing transformer models.
        return window_embeds
