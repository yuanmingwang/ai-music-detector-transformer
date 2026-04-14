import hashlib
import json
import os
from types import SimpleNamespace


PRECOMPUTED_FEATURE_VERSION = 1


def _sanitize_skip_time(skip_time):
    if skip_time is None:
        return None
    try:
        if skip_time != skip_time:
            return None
    except TypeError:
        return None
    return float(skip_time)


def _resolve_normalize_mode(normalize):
    if normalize is True:
        return "std"
    if normalize in (False, None):
        return None
    return normalize


def get_precomputed_cfg(cfg):
    precomputed = getattr(cfg, "precomputed", None)
    if precomputed is None:
        return SimpleNamespace(
            enabled=False,
            cache_dir="dataset/precomputed_mels",
            num_train_views=1,
        )

    return SimpleNamespace(
        enabled=bool(getattr(precomputed, "enabled", False)),
        cache_dir=getattr(precomputed, "cache_dir", "dataset/precomputed_mels"),
        num_train_views=max(1, int(getattr(precomputed, "num_train_views", 1))),
    )


def get_precomputed_feature_settings(cfg, normalize="std"):
    normalize = _resolve_normalize_mode(normalize)
    return {
        "version": PRECOMPUTED_FEATURE_VERSION,
        "audio": {
            "sample_rate": int(cfg.audio.sample_rate),
            "max_len": int(cfg.audio.max_len),
            "normalize": normalize,
        },
        "melspec": {
            "n_fft": int(cfg.melspec.n_fft),
            "hop_length": int(cfg.melspec.hop_length),
            "win_length": int(cfg.melspec.win_length),
            "n_mels": int(cfg.melspec.n_mels),
            "f_min": float(cfg.melspec.f_min),
            "f_max": float(cfg.melspec.f_max),
            "power": float(cfg.melspec.power),
            "top_db": None
            if getattr(cfg.melspec, "top_db", None) is None
            else float(cfg.melspec.top_db),
            "norm": getattr(cfg.melspec, "norm", None),
        },
    }


def get_precomputed_feature_signature(cfg, normalize="std"):
    settings = get_precomputed_feature_settings(cfg, normalize=normalize)
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def get_precomputed_feature_root(cache_dir, cfg, normalize="std"):
    signature = get_precomputed_feature_signature(cfg, normalize=normalize)
    signature_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, signature_hash)


def get_precomputed_feature_path(
    cache_dir,
    cfg,
    filepath,
    skip_time=None,
    view_idx=0,
    normalize="std",
):
    cache_root = get_precomputed_feature_root(cache_dir, cfg, normalize=normalize)
    identity = {
        "filepath": os.path.normcase(os.path.abspath(filepath)),
        "skip_time": _sanitize_skip_time(skip_time),
        "view_idx": int(view_idx),
    }
    key = hashlib.sha1(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return os.path.join(cache_root, key[:2], f"{key}.pt")


def get_precomputed_view_seed(cfg, filepath, skip_time=None, view_idx=0, normalize="std"):
    signature = get_precomputed_feature_signature(cfg, normalize=normalize)
    identity = (
        f"{signature}|{os.path.normcase(os.path.abspath(filepath))}|"
        f"{_sanitize_skip_time(skip_time)}|{int(view_idx)}"
    )
    return int(hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8], 16)
