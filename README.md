# AI Music Detector Transformer

This repository contains a PyTorch training and evaluation pipeline for **detecting AI-generated music from audio**. It treats the task as a binary classification problem:

- `0` = real / human-made song
- `1` = fake / AI-generated song

The code works on raw audio files, converts them into mel spectrograms, and trains a classifier using one of several backbones:

- `SpecTTTra`
- `ArtifactBranch`
- `TimbreProductionBranch`
- `LocalWindowTransformer`
- `ViT`
- `timm-convnext_tiny`
- `timm-efficientvit_b2`

The project is structured for experiments on datasets similar to **SONICS**, where metadata CSVs describe real and fake songs and the training scripts build train/validation/test splits from those files.

## What This Project Does

At a high level, the pipeline is:

1. Load waveform audio from `.mp3` files.
2. Crop or pad each clip to a fixed duration such as 5 seconds or 120 seconds.
3. Convert the waveform into a mel spectrogram.
4. Apply optional spectrogram-level augmentation during training.
5. Feed the spectrogram into a classifier backbone.
6. Train with binary supervision and report detection metrics.

This repo is focused on **audio-only classification**, not lyrics-based or multimodal detection.

### End-to-End Code Path

The core inference/training path is implemented through these functions and classes:

1. [`AudioDataset.__getitem__()`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/utils/dataset.py)
   - Loads an `.mp3` file with `librosa.load(...)`
   - Optionally trims the front using `skip_time`
   - Calls `crop_or_pad(...)` to force a fixed waveform length
   - Returns a tensor named `audio`

2. [`FeatureExtractor.forward()`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/layers/feature.py)
   - Receives the waveform tensor
   - Calls `self.audio2melspec(...)`, where `self.audio2melspec` is a `torchaudio.transforms.MelSpectrogram`
   - Calls `self.amplitude_to_db(...)`, where `self.amplitude_to_db` is a `torchaudio.transforms.AmplitudeToDB`
   - Applies the configured normalization
   - Returns the mel spectrogram tensor

3. [`AudioClassifier.forward()`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/models/model.py)
   - Calls `self.ft_extractor(audio)` to convert waveform to mel spectrogram
   - Optionally applies augmentation with `self.augment(...)`
   - Adds a channel dimension with `unsqueeze(1)`
   - Resizes the spectrogram with `torch.nn.functional.interpolate(...)`
   - Sends the spectrogram into the selected backbone with `self.encoder(spec)`
   - Passes the extracted features into the final linear layer `self.classifier(...)`
   - Returns the prediction logits

So in short, the spectrogram is created by `FeatureExtractor.forward()`, and it is fed into the classifier by `AudioClassifier.forward()`.

## Methods

### Input Representation

The model does not consume precomputed features from disk. Instead, it computes them on the fly:

- Input audio is loaded from the file path listed in the split CSV.
- Audio is resampled/handled according to the config.
- A mel spectrogram is created using `torchaudio.transforms.MelSpectrogram`.
- The spectrogram is converted to decibel scale with `AmplitudeToDB`.
- One of several normalization modes is applied:
  - `mean_std`
  - `min_max`
  - `simple`
  - or no normalization

The feature extraction logic lives in [`src/layers/feature.py`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/layers/feature.py).

### How MP3 Audio Becomes a Mel Spectrogram

The conversion from `.mp3` audio to mel spectrogram happens in two stages.

First, the audio file is loaded in [`AudioDataset.__getitem__()`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/utils/dataset.py):

- `librosa.load(self.filepaths[idx], sr=None)` reads the `.mp3` file into a waveform array
- `crop_or_pad(...)` makes the waveform exactly `cfg.audio.max_len` samples long
- optional normalization is applied to the waveform before it is converted to a tensor

Second, the waveform tensor is transformed into a mel spectrogram in [`FeatureExtractor.forward()`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/layers/feature.py):

- `self.audio2melspec(x.float())`
  - this is the actual mel spectrogram step
  - `self.audio2melspec` is created in `FeatureExtractor.__init__()` as `MelSpectrogram(...)`
- `self.amplitude_to_db(melspec)`
  - converts the mel power spectrogram into a log-like decibel representation
- `self.normalizer(melspec)`
  - applies the configured normalization method

The `MelSpectrogram(...)` parameters are controlled by the config file:

- `n_fft`
- `hop_length`
- `win_length`
- `n_mels`
- `sample_rate`
- `f_min`
- `f_max`
- `power`

These values come from the `melspec` and `audio` sections of the YAML config.

### How the Mel Spectrogram Is Fed Into the Classifier

This happens in [`AudioClassifier.forward()`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/models/model.py).

The sequence is:

1. `spec = self.ft_extractor(audio)`
   - calls `FeatureExtractor.forward()`
   - output shape is roughly `(batch_size, n_mels, n_frames)`

2. `spec, y = self.augment(spec, y)` during training
   - applies spectrogram augmentation when the model is in training mode

3. `spec = spec.unsqueeze(1)`
   - adds the channel dimension expected by image-style backbones
   - shape becomes `(batch_size, 1, n_mels, n_frames)`

4. `spec = F.interpolate(spec, size=tuple(self.input_shape), mode="bilinear")`
   - resizes the spectrogram to the configured model input size

5. `features = self.encoder(spec)`
   - this is the step where the mel spectrogram is actually fed into the backbone model
   - `self.encoder` is created by `AudioClassifier.get_encoder(...)`
  - depending on the config, it will be `SpecTTTra`, `ArtifactBranch`, `TimbreProductionBranch`, `LocalWindowTransformer`, `ViT`, or a `timm` model

6. `preds = self.classifier(embeds)`
   - the backbone output is pooled if needed
   - a final `nn.Linear(...)` layer maps features to the binary output logit

So the exact function that feeds the mel spectrogram into the classifier backbone is:

- [`AudioClassifier.forward()`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/models/model.py)

and the exact line of logic is conceptually:

```python
features = self.encoder(spec)
preds = self.classifier(embeds)
```

### Augmentation

During training, the code can apply spectrogram augmentation such as:

- mixup
- time masking
- frequency masking

These are controlled by the `augment` section in the config YAML files.

### Supported Model Backbones

The main model wrapper is in [`src/models/model.py`](d:/GitHub/Project/Python/ai-music-detector-transformer/src/models/model.py). It supports several backbone families:

- `SpecTTTra`
  - A spectro-temporal transformer-style architecture designed for this task.
  - Variants are controlled through config parameters such as `f_clip`, `t_clip`, `embed_dim`, `num_heads`, and `num_layers`.

- `ViT`
  - A Vision Transformer adapted to single-channel spectrogram inputs.

- `timm-*`
  - Any supported timm image backbone can be used as a spectrogram classifier.
  - This repo currently provides configs for:
    - ConvNeXt Tiny
    - EfficientViT B2

## Our models

- `ArtifactBranch`
  - A compact CNN branch deliberately narrow and focuses on artifact-sensitive evidence. The purpose is to detect subtle periodic peaks, high-frequency inconsistencies.
  - It derives artifact-sensitive maps from the input spectrogram, including high-frequency emphasis, local peak maps, temporal deltas, and multi-resolution views.
  - Important config parameters include `artifact_channels`, `embed_dim`, and `proj_drop_rate`.

- `TimbreProductionBranch`
  - A lightweight descriptor-sequence branch based on the timbre and production encoder.
  - It computes segment-level descriptors such as spectral centroid, bandwidth, rolloff, RMS variability, onset density, transient sharpness, and a simple harmonic-to-noise proxy directly from the mel spectrogram.
  - Those descriptors are combined with a compact learned timbre embedding per segment and modeled with a small BiGRU.
  - Important config parameters include `segment_frames`, `timbre_embed_dim`, `descriptor_hidden_dim`, `gru_hidden_dim`, and `gru_layers`.

- `LocalWindowTransformer`
  - A local-window transformer that slices several short time windows from the mel spectrogram and encodes them with a shared-weight `SpecTTTra` backbone.
  - It is useful when you want the model to focus on short-range local artifacts while still training through the same `train.py` pipeline and classifier wrapper.
  - Important config parameters include `local_window_size`, `local_num_windows`, `local_t_clip`, `local_f_clip`, `local_num_heads`, and `local_num_layers`.

### Training Objective

The code currently supports:

- `BCEWithLogitsLoss`
- `SigmoidFocalLoss`

Training quality is tracked with:

- balanced accuracy
- F1 score
- sensitivity
- specificity

The primary metric used to choose the best checkpoint is controlled by `logger.primary_metric` in the config.

## Repository Structure

```text
ai-music-detector-transformer/
+-- configs/                 # Experiment configs
+-- dataset/                 # Metadata CSVs and audio folders
+-- output/                  # Checkpoints, predictions, logs, profiles
+-- scripts/                 # Helper scripts for data download/prep
+-- src/                     # Models, layers, utilities
+-- split_data.py            # Build train.csv, valid.csv, test.csv
+-- test.py                  # Evaluation script
+-- train.py                 # Training script
```

## Installation

### Recommended Environment

This project is intended for Python 3 and PyTorch. A GPU is strongly recommended for training, especially for long-context runs such as 120-second clips.

### Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Main Dependencies

The repo depends on:

- PyTorch
- torchaudio
- librosa
- pandas
- scikit-learn
- tqdm
- timm
- fvcore

See [`requirements.txt`](d:/GitHub/Project/Python/ai-music-detector-transformer/requirements.txt) for the exact package list.

## Data

### Expected Dataset Format

This code expects a dataset folder with:

- a metadata file for real songs
- a metadata file for fake songs
- an audio folder for real songs
- an audio folder for fake songs

Expected layout:

```text
dataset/
├── real_songs.csv
├── fake_songs.csv
├── real_songs/
│   ├── song_a.mp3
│   ├── song_b.mp3
│   └── ...
└── fake_songs/
    ├── fake_a.mp3
    ├── fake_b.mp3
    └── ...
```

The generated split files used by the training code are written at the repository root:

- `train.csv`
- `valid.csv`
- `test.csv`

### Metadata Requirements

The training and split-generation code assumes the metadata CSVs include at least these columns:

- `filename`
  - Stem of the audio filename without `.mp3`
- `split`
  - One of `train`, `valid`, or `test`
- `duration`
  - Song duration in seconds
- `no_vocal`
  - Boolean-like field used by the split script filter

The training/evaluation pipeline also expects the final split CSVs to contain:

- `filepath`
- `target`

If `cfg.audio.skip_time` is enabled, the split CSV also needs:

- `skip_time`

### How to Get the Data

This repo does not automatically download the full dataset for you. The intended workflow is:

1. Obtain metadata and audio from your chosen source.
2. Place the files into the expected `dataset/` structure.
3. Generate `train.csv`, `valid.csv`, and `test.csv`.
4. Point your config to those split CSVs.

In practice, you need:

- `dataset/real_songs.csv`
- `dataset/fake_songs.csv`
- actual `.mp3` files under `dataset/real_songs/`
- actual `.mp3` files under `dataset/fake_songs/`

The split-generation script only keeps rows whose audio files actually exist on disk.

## Generating Train/Validation/Test Splits

Use [`split_data.py`](d:/GitHub/Project/Python/ai-music-detector-transformer/split_data.py) to build the CSV files that `train.py` and `test.py` consume.

### Full Usable Dataset

```bash
python split_data.py
```

### Limited Subset for Quick Experiments

```bash
python split_data.py --limit 100
```

This keeps up to:

- the first 100 usable real songs
- the first 100 usable fake songs

according to the metadata row order.

### Custom Dataset Directory

```bash
python split_data.py --data-dir ./dataset --limit 500
```

### What the Split Script Filters

The script:

1. Reads `real_songs.csv` and `fake_songs.csv`.
2. Checks whether the referenced `.mp3` files actually exist.
3. Filters rows using:
   - `duration >= 30`
   - `no_vocal == False`
4. Adds:
   - `filepath`
   - `target`
5. Splits rows using the existing `split` column.
6. Saves:
   - `train.csv`
   - `valid.csv`
   - `test.csv`

## Configuration

All experiments are driven by YAML config files under [`configs/`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs).

Examples:

- [`configs/convnext-5s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/convnext-5s.yaml)
- [`configs/convnext-120s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/convnext-120s.yaml)
- [`configs/efficientvit-5s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/efficientvit-5s.yaml)
- [`configs/efficientvit-120s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/efficientvit-120s.yaml)
- [`configs/vit-5s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/vit-5s.yaml)
- [`configs/vit-120s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/vit-120s.yaml)
- [`configs/spectttra_f1t3-5s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/spectttra_f1t3-5s.yaml)
- [`configs/spectttra_f1t3-120s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/spectttra_f1t3-120s.yaml)
- [`configs/artifact_branch-5s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/artifact_branch-5s.yaml)
- [`configs/artifact_branch-120s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/artifact_branch-120s.yaml)
- [`configs/timbre_production_branch-5s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/timbre_production_branch-5s.yaml)
- [`configs/timbre_production_branch-120s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/timbre_production_branch-120s.yaml)
- [`configs/local_window_transformer_f1t3-5s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/local_window_transformer_f1t3-5s.yaml)
- [`configs/local_window_transformer_f1t3-120s.yaml`](d:/GitHub/Project/Python/ai-music-detector-transformer/configs/local_window_transformer_f1t3-120s.yaml)

### Important Config Sections

#### `experiment_name`

Used to create the output folder:

```text
output/<experiment_name>/
```

#### `dataset`

Defines the CSVs to use:

```yaml
dataset:
  train_dataframe: "train.csv"
  valid_dataframe: "valid.csv"
  test_dataframe: "test.csv"
```

#### `audio`

Controls waveform length and clip sampling:

- `sample_rate`
- `max_time`
- `random_sampling`
- `normalize`
- `skip_time`

`max_time` is converted internally into `audio.max_len`.

#### `melspec`

Controls spectrogram generation:

- `n_fft`
- `hop_length`
- `win_length`
- `n_mels`
- `f_min`
- `f_max`
- `power`
- `top_db`
- `norm`

#### `model`

Defines the backbone and its hyperparameters.

Examples:

```yaml
model:
  name: "SpecTTTra"
```

```yaml
model:
  name: "ArtifactBranch"
```

```yaml
model:
  name: "TimbreProductionBranch"
```

```yaml
model:
  name: "LocalWindowTransformer"
```

```yaml
model:
  name: "ViT"
```

```yaml
model:
  name: "timm-convnext_tiny"
```

#### `training` and `validation`

Controls:

- batch size
- number of epochs

#### `optimizer`

Controls:

- optimizer type
- weight decay
- gradient accumulation
- gradient clipping

#### `scheduler`

Controls:

- scheduler type
- learning rate
- warmup
- minimum LR

#### `loss`

Selects the training objective.

#### `logger.primary_metric`

Determines which validation metric chooses the best checkpoint:

- `f1`
- `acc`
- `sens`
- `spec`

## How to Run Training

### Example

```bash
python train.py --config ./configs/convnext-5s.yaml
```

### What Training Does

The training script:

1. Loads the config.
2. Builds the output directory.
3. Loads train/validation/test CSVs.
4. Builds dataloaders.
5. Creates the model, optimizer, scheduler, and loss.
6. Trains for the configured number of epochs.
7. Selects the best checkpoint using the configured primary metric.
8. Evaluates the best checkpoint on the test set.
9. Saves predictions and checkpoints.

### Distributed Training

If multiple CUDA devices are visible, `train.py` will automatically enable distributed training using `torch.multiprocessing.spawn`.

### Resume Training

To resume from a checkpoint, set:

```yaml
model:
  resume: "output/<experiment_name>/last_checkpoint.pth"
```

Then run the same training command again.

## How to Run Evaluation

Use [`test.py`](d:/GitHub/Project/Python/ai-music-detector-transformer/test.py) to evaluate a saved checkpoint.

```bash
python test.py --config ./configs/convnext-5s.yaml --ckpt_path ./output/convnext-t=5/best_checkpoint.pth
```

This script:

- loads the test CSV
- loads the checkpoint
- runs inference
- prints summary metrics
- saves per-example predictions

## How to Profile a Model

Use [`model_profile.py`](d:/GitHub/Project/Python/ai-music-detector-transformer/model_profile.py) to estimate profile information such as parameter count and compute.

```bash
python model_profile.py --config ./configs/convnext-5s.yaml --batch_size 12
```

Profile output is saved to:

```text
output/<experiment_name>/model_profile.csv
```

## Outputs

Each experiment writes outputs under:

```text
output/<experiment_name>/
```

Typical files include:

- `best_checkpoint.pth`
- `last_checkpoint.pth`
- `valid_predictions.csv`
- `test_predictions.csv`
- `model_profile.csv`
- `logs/train-YYYYMMDD-HHMMSS.txt`

### Training Logs

Each training run now saves a text log file under:

```text
output/<experiment_name>/logs/
```

The log captures the same console output you see during training, including:

- config printout
- epoch progress
- checkpoint updates
- validation/test summaries
- total training time

## Metrics

The code reports the following metrics:

- `loss`
- `acc`
  - balanced accuracy
- `f1`
- `sens`
  - sensitivity / recall for the fake class
- `spec`
  - specificity / recall for the real class

The best checkpoint is chosen using `logger.primary_metric`.

## Available Experiment Presets

The repo currently includes presets for:

- 5-second clips
- 120-second clips

and for multiple backbones:

- ConvNeXt
- EfficientViT
- ViT
- ArtifactBranch
- TimbreProductionBranch
- LocalWindowTransformer alpha
- SpecTTTra alpha
- SpecTTTra beta
- SpecTTTra gamma

The different SpecTTTra variants mostly differ in tokenization/window settings such as `f_clip` and `t_clip`.
The LocalWindowTransformer presets reuse the same style of settings, but add local-window controls such as `local_window_size` and `local_num_windows`.

## Example Workflow

### 1. Prepare data

Put your metadata and audio files under:

```text
dataset/
```

### 2. Build split CSVs

```bash
python split_data.py --limit 100
```

### 3. Train a model

```bash
python train.py --config ./configs/convnext-5s.yaml
```

### 4. Evaluate the best checkpoint

```bash
python test.py --config ./configs/convnext-5s.yaml --ckpt_path ./output/convnext-t=5/best_checkpoint.pth
```

### 5. Inspect outputs

Check:

- checkpoints in `output/convnext-t=5/`
- predictions in `output/convnext-t=5/`
- logs in `output/convnext-t=5/logs/`

## Notes and Practical Tips

### Clip Duration Matters

The configs include both short-context and long-context settings:

- `5s` configs are useful for quick iteration and lower memory usage.
- `120s` configs are much heavier but can capture more long-range musical context.

### Dataset CSVs Must Match Reality

The split script discards rows whose `.mp3` files do not exist. If you expected more training examples than you see in `train.csv`, check that:

- filenames in metadata match filenames on disk
- the files are really under `dataset/real_songs/` or `dataset/fake_songs/`
- the rows satisfy the duration and `no_vocal` filters

### `skip_time`

If your CSV includes a `skip_time` column and you enable `audio.skip_time`, the loader will trim that many seconds from the beginning before cropping the audio clip.

## Troubleshooting

### Training is very slow

Possible reasons:

- you are running on CPU
- clip duration is too long
- batch size is too large for your hardware
- your audio files are stored on a slow disk

Try:

- using a 5-second config first
- lowering batch size
- reducing `num_workers` if dataloading becomes unstable

### Out of memory

Try:

- smaller batch size
- a 5-second config instead of 120-second
- a lighter backbone
- mixed precision enabled in the config

### No samples appear in split CSVs

Check:

- metadata file paths
- `filename` values
- `.mp3` file existence
- `split` column contents
- `duration` and `no_vocal` filters

## Reference

SONICS project
