import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

import torch

from src.layers.feature import FeatureExtractor
from src.utils.config import dict2cfg
from src.utils.dataset import load_audio_clip
from src.utils.precomputed import (
    get_precomputed_cfg,
    get_precomputed_feature_path,
    get_precomputed_feature_root,
    get_precomputed_feature_settings,
    get_precomputed_view_seed,
)


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Precompute mel spectrogram tensors for faster training."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Dataset splits to precompute",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached tensors",
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(device_name)


def get_split_dataframe(cfg, split):
    path_map = {
        "train": cfg.dataset.train_dataframe,
        "valid": cfg.dataset.valid_dataframe,
        "test": cfg.dataset.test_dataframe,
    }
    return pd.read_csv(path_map[split])


def get_split_num_views(cfg, split, precomputed_cfg):
    if split == "train" and cfg.audio.random_sampling:
        return precomputed_cfg.num_train_views
    return 1


def ensure_feature_settings(cache_dir, cfg):
    os.makedirs(cache_dir, exist_ok=True)
    settings_path = os.path.join(cache_dir, "feature_settings.json")
    if os.path.exists(settings_path):
        return

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(get_precomputed_feature_settings(cfg), f, indent=2)


def main():
    args = arg_parser()
    dict_ = yaml.safe_load(open(args.config, encoding="utf-8").read())
    cfg = dict2cfg(dict_)
    precomputed_cfg = get_precomputed_cfg(cfg)
    device = resolve_device(args.device)

    feature_extractor = FeatureExtractor(cfg).to(device)
    feature_extractor.eval()

    cache_root = get_precomputed_feature_root(precomputed_cfg.cache_dir, cfg)
    ensure_feature_settings(cache_root, cfg)

    total_items = 0
    split_dfs = {}
    for split in args.splits:
        df = get_split_dataframe(cfg, split)
        split_dfs[split] = df
        total_items += len(df) * get_split_num_views(cfg, split, precomputed_cfg)

    written = 0
    skipped = 0

    progress = tqdm(total=total_items, desc="Precompute", ncols=150)
    with torch.no_grad():
        for split in args.splits:
            df = split_dfs[split]
            num_views = get_split_num_views(cfg, split, precomputed_cfg)
            use_skip_time = bool(getattr(cfg.audio, "skip_time", False))
            if use_skip_time and "skip_time" not in df.columns:
                raise KeyError(
                    f"Expected a skip_time column in {split} split because cfg.audio.skip_time is enabled."
                )

            for row in df.itertuples(index=False):
                filepath = row.filepath
                skip_time = getattr(row, "skip_time", None) if use_skip_time else None

                for view_idx in range(num_views):
                    feature_path = get_precomputed_feature_path(
                        precomputed_cfg.cache_dir,
                        cfg,
                        filepath,
                        skip_time=skip_time,
                        view_idx=view_idx,
                    )
                    os.makedirs(os.path.dirname(feature_path), exist_ok=True)

                    if os.path.exists(feature_path) and not args.overwrite:
                        skipped += 1
                        progress.update(1)
                        progress.set_postfix(written=written, skipped=skipped, split=split)
                        continue

                    rng = None
                    if split == "train" and cfg.audio.random_sampling:
                        seed = get_precomputed_view_seed(
                            cfg,
                            filepath,
                            skip_time=skip_time,
                            view_idx=view_idx,
                        )
                        rng = np.random.default_rng(seed)

                    audio = load_audio_clip(
                        filepath,
                        skip_time=skip_time,
                        max_len=cfg.audio.max_len,
                        random_sampling=(split == "train" and cfg.audio.random_sampling),
                        normalize="std",
                        rng=rng,
                    )
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
                    spec = feature_extractor(audio_tensor).squeeze(0).cpu().float().contiguous()
                    torch.save(
                        {
                            "spec": spec,
                            "meta": {
                                "source_filepath": filepath,
                                "skip_time": skip_time,
                                "split": split,
                                "view_idx": view_idx,
                            },
                        },
                        feature_path,
                    )
                    written += 1
                    progress.update(1)
                    progress.set_postfix(written=written, skipped=skipped, split=split)

    progress.close()
    print(f"> Cache directory: {cache_root}")
    print(f"> Features written: {written}")
    print(f"> Features skipped: {skipped}")


if __name__ == "__main__":
    main()
