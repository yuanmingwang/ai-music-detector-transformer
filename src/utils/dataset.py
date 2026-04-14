import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.utils.precomputed import get_precomputed_cfg, get_precomputed_feature_path


def resolve_normalize_mode(normalize):
    if normalize is True:
        return "std"
    if normalize in (False, None):
        return None
    return normalize


def sanitize_skip_time(skip_time):
    if skip_time is None:
        return None
    try:
        if np.isnan(skip_time):
            return None
    except TypeError:
        return None
    return float(skip_time)


def _randint(rng, high):
    if high <= 0:
        return 0
    if rng is None:
        return int(np.random.randint(0, high))
    if hasattr(rng, "integers"):
        return int(rng.integers(0, high))
    return int(rng.randint(0, high))


def crop_or_pad_audio(audio, max_len, random_sampling=True, rng=None):
    audio_len = audio.shape[0]
    if random_sampling:
        diff_len = abs(max_len - audio_len)
        if audio_len < max_len:
            pad1 = _randint(rng, diff_len)
            pad2 = diff_len - pad1
            audio = np.pad(audio, (pad1, pad2), mode="constant")
        elif audio_len > max_len:
            idx = _randint(rng, diff_len)
            audio = audio[idx : (idx + max_len)]
    else:
        if audio_len < max_len:
            audio = np.pad(audio, (0, max_len - audio_len), mode="constant")
        elif audio_len > max_len:
            idx = int((audio_len - max_len) / 4 * 3)
            audio = audio[idx : (idx + max_len)]
    return audio


def normalize_audio(audio, normalize="std"):
    normalize = resolve_normalize_mode(normalize)
    if normalize == "std":
        audio = audio / np.maximum(np.std(audio), 1e-6)
    elif normalize == "minmax":
        audio = audio - np.min(audio)
        audio = audio / np.maximum(np.max(audio), 1e-6)
    return audio


def load_audio_clip(
    filepath,
    skip_time=None,
    max_len=32000,
    random_sampling=True,
    normalize="std",
    rng=None,
):
    import librosa

    audio, sr = librosa.load(filepath, sr=None)
    skip_time = sanitize_skip_time(skip_time)
    if skip_time is not None:
        audio = audio[int(skip_time * sr) :]
    audio = crop_or_pad_audio(audio, max_len, random_sampling=random_sampling, rng=rng)
    audio = normalize_audio(audio, normalize=normalize)
    return audio.astype(np.float32, copy=False)


class AudioDataset(Dataset):
    def __init__(
        self,
        filepaths,
        labels,
        skip_times=None,
        num_classes=1,
        normalize="std",
        max_len=32000,
        random_sampling=True,
        train=False,
        cfg=None,
        use_precomputed=False,
        precomputed_cache_dir=None,
        precomputed_num_views=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filepaths = filepaths
        self.labels = labels
        self.skip_times = skip_times
        self.num_classes = num_classes
        self.random_sampling = random_sampling
        self.normalize = resolve_normalize_mode(normalize)
        self.max_len = max_len
        self.train = train
        self.cfg = cfg
        precomputed_cfg = get_precomputed_cfg(cfg) if cfg is not None else None
        self.use_precomputed = bool(
            use_precomputed or getattr(precomputed_cfg, "enabled", False)
        )
        self.precomputed_cache_dir = (
            precomputed_cache_dir
            or getattr(precomputed_cfg, "cache_dir", None)
        )
        self.precomputed_num_views = max(1, int(precomputed_num_views))
        if not self.train:
            assert (
                not self.random_sampling
            ), "Ensure random_sampling is disabled for val"
        if self.use_precomputed:
            if self.cfg is None:
                raise ValueError("cfg is required when use_precomputed=True")
            if self.precomputed_cache_dir is None:
                raise ValueError(
                    "precomputed_cache_dir is required when use_precomputed=True"
                )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        target = np.array([self.labels[idx]])
        skip_time = self.skip_times[idx] if self.skip_times is not None else None

        if self.use_precomputed:
            view_idx = (
                _randint(None, self.precomputed_num_views)
                if self.train and self.precomputed_num_views > 1
                else 0
            )
            feature_path = get_precomputed_feature_path(
                self.precomputed_cache_dir,
                self.cfg,
                self.filepaths[idx],
                skip_time=skip_time,
                view_idx=view_idx,
                normalize=self.normalize,
            )
            if not os.path.exists(feature_path):
                raise FileNotFoundError(
                    "Precomputed feature file not found: "
                    f"{feature_path}. Run precompute_features.py first."
                )
            cached = torch.load(feature_path, map_location="cpu")
            audio = cached["spec"] if isinstance(cached, dict) else cached
            audio = audio.float()
        else:
            audio = load_audio_clip(
                self.filepaths[idx],
                skip_time=skip_time,
                max_len=self.max_len,
                random_sampling=self.random_sampling,
                normalize=self.normalize,
            )
            audio = torch.from_numpy(audio).float()

        target = torch.from_numpy(target).float().squeeze()
        return {
            "audio": audio,
            "target": target,
        }


def get_dataloader(
    filepaths,
    labels,
    skip_times=None,
    batch_size=8,
    num_classes=1,
    max_len=32000,
    random_sampling=True,
    normalize="std",
    train=False,
    # drop_last=False,
    pin_memory=True,
    worker_init_fn=None,
    collate_fn=None,
    num_workers=0,
    distributed=False,
    cfg=None,
    use_precomputed=False,
    precomputed_cache_dir=None,
    precomputed_num_views=1,
):
    dataset = AudioDataset(
        filepaths,
        labels,
        skip_times=skip_times,
        num_classes=num_classes,
        max_len=max_len,
        random_sampling=random_sampling,
        normalize=normalize,
        train=train,
        cfg=cfg,
        use_precomputed=use_precomputed,
        precomputed_cache_dir=precomputed_cache_dir,
        precomputed_num_views=precomputed_num_views,
    )

    if distributed:
        # drop_last is set to True to validate properly
        # Ref: https://discuss.pytorch.org/t/how-do-i-validate-with-pytorch-distributeddataparallel/172269/8
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=train, drop_last=not train
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) and train,
        # drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    return dataloader
