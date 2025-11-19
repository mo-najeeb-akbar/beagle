# Optical Profilometry: Wavelet VAE for Surface Analysis

This example demonstrates a complete machine learning pipeline for optical profilometry data:
converting proprietary formats, training a wavelet-based VAE, and running inference.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                                │
└─────────────────────────────────────────────────────────────────────────┘

   .datx files                numpy arrays                TFRecord files
   (proprietary)              (.npy + .json)              (.tfrecord)
   ┌──────────┐               ┌──────────┐               ┌──────────┐
   │ Defective│               │ surface  │               │  shard   │
   │  *.datx  │──┬────────────┤  .npy    │──┬────────────┤  -0000   │
   └──────────┘  │            │ metadata │  │            │  -0001   │
                 │            │  .json   │  │            │  ...     │
   ┌──────────┐  │            └──────────┘  │            │  -0009   │
   │  Normal  │  │                           │            └──────────┘
   │  *.datx  │──┤                           │            ┌──────────┐
   └──────────┘  │                           │            │ polymer  │
                 │                           │            │  .json   │
   ┌──────────┐  │                           │            │(parser)  │
   │   More   │  │                           │            └──────────┘
   │  *.datx  │──┘                           │
   └──────────┘                              │
                                             │
    datx_to_np.py                            │  np_to_tfrecord.py
    ─────────────►                           │  ──────────────────►
                                             │


┌─────────────────────────────────────────────────────────────────────────┐
│                            TRAINING                                     │
└─────────────────────────────────────────────────────────────────────────┘

   TFRecord Dataset                        Training Outputs
   ┌──────────────┐                        ┌────────────────────┐
   │  shard-0000  │                        │ checkpoint_0001    │
   │  shard-0001  │                        │ checkpoint_0002    │
   │    ...       │────────────────────────┤      ...           │
   │  shard-0009  │                        │ checkpoint_final   │
   │              │                        ├────────────────────┤
   │ polymer.json │                        │ polymer_stats.json │
   └──────────────┘                        ├────────────────────┤
                                           │ config.json        │
        train.py                           ├────────────────────┤
        ────────►                          │ metrics.json       │
                                           ├────────────────────┤
                                           │ viz/               │
                                           │  epoch_001.png     │
                                           │  epoch_002.png     │
                                           │      ...           │
                                           └────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE                                     │
└─────────────────────────────────────────────────────────────────────────┘

   Checkpoint + Stats          TFRecord Data           Reconstruction Images
   ┌────────────────┐          ┌──────────┐            ┌───────────────────┐
   │ checkpoint_    │          │  shard   │            │ 000000.png        │
   │   final        │──┐       │  -0000   │──┐         │ [Orig|Recon|Err]  │
   ├────────────────┤  │       │  -0001   │  │         ├───────────────────┤
   │ polymer_stats  │  │       │  ...     │  │         │ 000001.png        │
   │   .json        │  │       └──────────┘  │         │ [Orig|Recon|Err]  │
   └────────────────┘  └───────────────┬─────┘         ├───────────────────┤
                                       │               │ 000002.png        │
                                       │               │      ...          │
                                       │               └───────────────────┘
                                       │
                                       │      inference.py
                                       │      ─────────────►
```

## Quick Start

### Prerequisites

All commands must be run inside Docker containers:

```bash
# Build Docker images (first time only)
make build

# Open development shell with example dependencies
make examples
```

### Step 1: Convert DATX to NumPy

Convert proprietary `.datx` files to NumPy arrays with metadata:

```bash
# Edit datx_to_np.py to set your input/output paths, then run:
make run CMD='python examples/optical_profilometry/datx_to_np.py'
```

**Inputs:**
- `/data/optical_prof/Defective/*.datx` (proprietary format)

**Outputs:**
- `/data/defective_pe_processed/*/surface.npy` (depth maps)
- `/data/defective_pe_processed/*/dataset_metadata.json` (metadata)

### Step 2: Convert NumPy to TFRecord

Convert NumPy arrays to efficient TFRecord format for training:

```bash
make run CMD='python examples/optical_profilometry/np_to_tfrecord.py \
  /data/defective_pe_processed \
  /data/polymer_tfrecords'
```

**Inputs:**
- `/data/defective_pe_processed/*/surface.npy`
- `/data/defective_pe_processed/*/dataset_metadata.json`

**Outputs:**
- `/data/polymer_tfrecords/shard-000[0-9].tfrecord` (10 shards)
- `/data/polymer_tfrecords/polymer.json` (parser configuration)

### Step 3: Train Wavelet VAE

Train a variational autoencoder with wavelet loss on the dataset:

```bash
make run CMD='python examples/optical_profilometry/train.py \
  /data/polymer_tfrecords'
```

**Inputs:**
- `/data/polymer_tfrecords/*.tfrecord`
- `/data/polymer_tfrecords/polymer.json`

**Outputs:**
- `/data/experiments/wavelet_vae_YYYYMMDD_HHMMSS/checkpoint_*` (model weights)
- `/data/experiments/wavelet_vae_YYYYMMDD_HHMMSS/polymer_stats.json` (normalization)
- `/data/experiments/wavelet_vae_YYYYMMDD_HHMMSS/config.json` (training config)
- `/data/experiments/wavelet_vae_YYYYMMDD_HHMMSS/metrics.json` (loss history)
- `/data/experiments/wavelet_vae_YYYYMMDD_HHMMSS/viz/*.png` (training progress)

**Optional: Compute dataset statistics only**

```bash
make run CMD='python examples/optical_profilometry/train.py \
  /data/polymer_tfrecords --compute-stats'
```

### Step 4: Run Inference

Generate reconstructions and error maps for all images:

```bash
make run CMD='python examples/optical_profilometry/inference.py \
  /data/experiments/wavelet_vae_YYYYMMDD_HHMMSS \
  /data/inference_output \
  /data/polymer_tfrecords'
```

**Inputs:**
- `/data/experiments/wavelet_vae_YYYYMMDD_HHMMSS/checkpoint_final` (or latest)
- `/data/experiments/wavelet_vae_YYYYMMDD_HHMMSS/polymer_stats.json`
- `/data/polymer_tfrecords/*.tfrecord`

**Outputs:**
- `/data/inference_output/NNNNNN.png` (comparison images)

Each output image contains three panels: `[Original | Reconstruction | Error (5x)]`

## Complete Example Workflow

```bash
# 1. Open Docker shell
make examples

# 2. Convert proprietary data to NumPy (edit paths in script first)
python examples/optical_profilometry/datx_to_np.py

# 3. Convert NumPy to TFRecord
python examples/optical_profilometry/np_to_tfrecord.py \
  /data/defective_pe_processed \
  /data/polymer_tfrecords

# 4. Train model (creates timestamped experiment directory)
python examples/optical_profilometry/train.py \
  /data/polymer_tfrecords

# 5. Run inference (replace YYYYMMDD_HHMMSS with your experiment timestamp)
python examples/optical_profilometry/inference.py \
  /data/experiments/wavelet_vae_20231119_143022 \
  /data/inference_results \
  /data/polymer_tfrecords
```

## Configuration

### Training Configuration

Edit `CONFIG` in `train.py`:

```python
CONFIG = {
    "learning_rate": 0.001,    # AdamW learning rate
    "num_epochs": 10,          # Training epochs
    "batch_size": 32,          # Batch size
    "base_features": 32,       # Network width (channels)
    "latent_dim": 128,         # Latent space dimensionality
    "crop_size": 256,          # Crop size for training
    "crop_overlap": 192,       # Stride for cropping (192 = 64px step)
    "val_split": 0.2,          # Validation split (0.2 = 20%)
    "split_seed": 42,          # Random seed for train/val split
}
```

### Data Conversion Configuration

Edit `config` dict in `datx_to_np.py`:

```python
config = {
    "input_folder": "/data/optical_prof/Defective",
    "output_folder": "/data/defective_pe_processed",
    "datasets": ["intensity", "quality", "surface"],  # Which layers to extract
    "save_dataset_metadata": True,
    "save_instrument_params": True,
}
```

### Filtering Data

Use the `filters` dict in `datx_to_np.py` to select specific samples:

```python
template = "date_polymer_simulant_trial_replicate_magnification_level"
filters = {
    "date": None,              # Any date
    "polymer": "PE",           # Only polyethylene
    "simulant": None,          # Any simulant
    "trial": None,             # Any trial
    "replicate": None,         # Any replicate
    "magnification": None,     # Any magnification
    "level": None,             # Any level
}
```

## Output Format

### Training Visualizations

Training progress images in `viz/` show three panels per sample:
- **Original**: Input depth map
- **Reconstruction**: VAE output
- **Error (5x)**: Absolute difference scaled 5x

### Inference Outputs

Each inference image (`NNNNNN.png`) is a horizontal concatenation:

```
┌─────────────┬─────────────┬─────────────┐
│  Original   │ Reconstruct │  Error (5x) │
│  Depth Map  │    VAE      │  |Orig-Rec| │
└─────────────┴─────────────┴─────────────┘
```

Images are 8-bit grayscale, normalized to [0, 255] range.

## GPU Usage

By default, all GPUs are available. Control GPU visibility:

```bash
# Use specific GPU(s)
NVIDIA_VISIBLE_DEVICES=0 make run CMD='python examples/optical_profilometry/train.py ...'

# CPU only
NVIDIA_VISIBLE_DEVICES="" make run CMD='python examples/optical_profilometry/train.py ...'
```

## Mounting External Data

Mount external directories for data access:

```bash
# Mount host directory to /data in container
MOUNT_DIR=~/my_datasets make examples

# Then access at /data inside container
ls /data
```

## File Structure

```
examples/optical_profilometry/
├── README.md                    # This file
├── datx_to_np.py               # Convert .datx → .npy + .json
├── np_to_tfrecord.py           # Convert .npy → .tfrecord
├── train.py                    # Train wavelet VAE
├── inference.py                # Run inference on trained model
├── data_loader.py              # Dataset loading utilities
└── utils/
    ├── datx.py                 # DATX format parser
    └── filter_utils.py         # Sample filtering utilities
```

## Troubleshooting

**Q: No checkpoints found**
- Ensure training completed successfully
- Check experiment directory exists: `/data/experiments/wavelet_vae_*`

**Q: Missing polymer_stats.json**
- Stats are saved during training in experiment directory
- Inference looks for stats in checkpoint dir first, then data dir

**Q: Out of memory during training**
- Reduce `batch_size` in `train.py`
- Reduce `crop_size` (must be power of 2: 128, 256, 512)
- Use fewer GPUs: `NVIDIA_VISIBLE_DEVICES=0`

**Q: Slow data loading**
- Increase number of TFRecord shards
- Ensure data is on fast storage (SSD)

## References

- Wavelet VAE architecture: `beagle/network/wavelet_vae.py`
- Training loop: `beagle/training/train_loop.py`
- Dataset API: `beagle/dataset/`

