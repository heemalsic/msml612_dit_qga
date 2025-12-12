# DiT (Diffusion Transformer) for MNIST

A PyTorch implementation of Diffusion Transformers (DiT) with class conditioning for MNIST digit generation, supporting both single-GPU and distributed training.

## 🎯 Features

- ✅ **Diffusion Transformer (DiT)** architecture with multi-head attention
- ✅ **Class-conditioned generation** - generate specific digits (0-9)
- ✅ **DDIM sampling** - fast inference with configurable step counts
- ✅ **Multi-GPU support** - DistributedDataParallel (DDP) training
- ✅ **Checkpoint management** - resume training and save periodic checkpoints
- ✅ **FID evaluation** - find optimal DDIM step counts
- ✅ **YAML configuration** - easy hyperparameter tuning
- ✅ **AWS deployment** - included scripts for cloud training

---

## 📁 Project Structure

```
msml_612_project/
├── configs/
│   └── dit_config.yaml           # Model and training hyperparameters
├── data_loader/
│   └── dataset.py                # MNIST/CIFAR-10 data loader with DDP support
├── models/
│   ├── attention.py              # Multi-head attention mechanism
│   ├── dit_model.py              # DiT architecture implementation
│   └── embeddings.py             # Sinusoidal position embeddings
├── train/
│   ├── trainer.py                # Training loop with checkpointing
│   └── dit_training/             # Training outputs (checkpoints, losses)
├── inference/
│   └── mnist_inference/          # Generated inference images
├── aws_scripts/
│   ├── launch_1gpu_instance.sh   # AWS instance setup script
│   └── run_training_single_gpu.sh # Remote training script
├── run.py                        # Main training script
├── inference.py                  # DDIM inference with CLI args
├── evaluate.py                   # FID evaluation across step counts
├── utils.py                      # Utility functions (seed, logging)
├── requirements.txt              # Python dependencies
├── dit_training_colab.ipynb      # Standalone Colab training notebook
└── README.md                     # This file
```

### 📄 File Descriptions

#### Core Files

- **`run.py`**: Main training script with DDP support. Initializes model, data loader, and trainer.
- **`inference.py`**: Generate images using DDIM sampling. Supports command-line arguments for digit and checkpoint selection.
- **`evaluate.py`**: Calculate FID scores for different DDIM step counts to find optimal sampling configuration.
- **`utils.py`**: Helper functions for reproducibility (random seed setting) and logging.

#### Configuration

- **`configs/dit_config.yaml`**: Complete model, training, diffusion, and inference configuration. Modify hyperparameters here.

#### Models

- **`models/dit_model.py`**: DiT architecture with patch embedding, transformer blocks, and noise prediction.
- **`models/attention.py`**: Multi-head self-attention with LayerNorm.
- **`models/embeddings.py`**: Sinusoidal timestep embeddings for diffusion process.

#### Data

- **`data_loader/dataset.py`**: PyTorch DataLoader with support for MNIST/CIFAR-10 and DistributedSampler for multi-GPU training.

#### Training

- **`train/trainer.py`**: Training loop with:
  - DDPM forward/reverse diffusion process
  - Checkpoint saving and loading
  - Loss tracking and visualization
  - DDP-aware operations

#### AWS Deployment

- **`aws_scripts/launch_1gpu_instance.sh`**: Automates EC2 instance setup with GPU support.
- **`aws_scripts/run_training_single_gpu.sh`**: Runs training on remote instance and downloads artifacts.

#### Notebooks

- **`dit_training_colab.ipynb`**: Self-contained Colab notebook for single-GPU training (no external file dependencies).

---

## 🚀 Setup

### 1. Clone or Navigate to Project

```bash
cd /Users/sarveshkhetan/work/msml_612_project
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**

- `torch` - PyTorch deep learning framework
- `torchvision` - Vision datasets and transforms
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `Pillow` - Image processing
- `einops` - Tensor operations
- `pyyaml` - YAML configuration parsing
- `matplotlib` - Plotting and visualization
- `scipy` - FID calculation (matrix operations)

### 4. Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

---

## 🎓 Training

### Configuration

Edit `configs/dit_config.yaml` to customize:

```yaml
# Model architecture
model:
  image_size: 28 # MNIST image size
  image_channels: 1 # Grayscale
  depth: 4 # Transformer layers
  num_heads: 4 # Attention heads
  hidden_dim: 768 # Hidden dimension

# Training parameters
training:
  num_epochs: 50
  learning_rate: 1.0e-4
  batch_size: 128
  save_every_n_epochs: 5

# Diffusion process
diffusion:
  num_steps: 100 # Training timesteps
  beta_start: 1.0e-4
  beta_end: 0.02
  beta_schedule: "linear"
```

### Single GPU Training

```bash
python3 run.py
```

Or with torchrun for consistency:

```bash
torchrun --nproc_per_node=1 run.py
```

### Multi-GPU Training (Linux with CUDA)

```bash
# For 4 GPUs
torchrun --nproc_per_node=4 run.py

# For 8 GPUs
torchrun --nproc_per_node=8 run.py
```

### Resume Training from Checkpoint

Edit `configs/dit_config.yaml`:

```yaml
training:
  resume_from: "models/dit_models/checkpoint_epoch_50.pth"
```

Then run:

```bash
python3 run.py
```

### Training Outputs

During training, the following will be generated:

- **Checkpoints**: `models/dit_models/checkpoint_epoch_X.pth` (every N epochs)
- **Latest checkpoint**: `models/dit_models/latest_checkpoint.pth` (updated each epoch)
- **Loss curve**: `outputs/training_loss.png`
- **Console logs**: Real-time loss and timing information

---

## 🎨 Inference

### Generate Images with DDIM Sampling

The `inference.py` script supports command-line arguments for flexible generation:

```bash
python3 inference.py --digit <0-9> --epoch <checkpoint_epoch>
```

#### Examples

Generate digit **3** using checkpoint from epoch **50**:

```bash
python3 inference.py --digit 3 --epoch 50
```

Generate digit **7** using checkpoint from epoch **160**:

```bash
python3 inference.py --digit 7 --epoch 160
```

Generate all digits from epoch 50:

```bash
for digit in {0..9}; do
  python3 inference.py --digit $digit --epoch 50
done
```

### Command-Line Arguments

| Argument  | Type | Default | Description              |
| --------- | ---- | ------- | ------------------------ |
| `--digit` | int  | 3       | Digit to generate (0-9)  |
| `--epoch` | int  | 50      | Checkpoint epoch to load |

### Inference Behavior

- **Step counts**: Tests multiple step counts (25, 50, 75, 100)
- **Output**: Generates comparison grid saved to `inference/mnist_inference/digit_{digit}_checkpoint_{epoch}.png`
- **Timing**: Reports inference time for each step count
- **Device**: Defaults to CPU (modify in script for GPU)

### Inference Outputs

Generated images are saved to:

```
inference/mnist_inference/
├── digit_0_checkpoint_50.png
├── digit_1_checkpoint_50.png
├── digit_2_checkpoint_50.png
└── ...
```

Each output shows the same digit generated with different DDIM step counts for quality comparison.

---

## 📊 Evaluation

### Calculate FID Scores

The `evaluate.py` script calculates Fréchet Inception Distance (FID) scores to find optimal DDIM step counts:

```bash
python3 evaluate.py
```

### What It Does

1. **Loads real MNIST data** - 10,000 samples for reference statistics
2. **Generates samples** - Creates images for all digits (0-9) with different step counts
3. **Calculates FID** - Compares generated vs real distributions
4. **Finds optimal steps** - Identifies best quality/speed tradeoff

### Configuration

Edit the following in `evaluate.py` to customize:

```python
# Line 129: Checkpoint path
checkpoint_path = Path('new_model_artifacts/checkpoints/checkpoint_epoch_50.pth')

# Line 141: Step counts to test
step_counts = [25, 50, 75, 100]

# Line 142: Samples per digit
num_samples_per_class = 1
```

### Example Output

```
================================================================================
RESULTS SUMMARY
================================================================================

Steps      FID Score       Time (s)        Time/Sample (s)
--------------------------------------------------------------------------------
25         892.45          12.3            0.12
50         645.23          23.1            0.23               ← BEST
75         658.91          34.5            0.35
100        661.34          45.2            0.45

================================================================================
🏆 OPTIMAL CONFIGURATION
================================================================================
Best FID Score: 645.23
Optimal Steps: 50
================================================================================
```

### Interpreting Results

- **Lower FID** = Better quality (closer to real MNIST distribution)
- **Trade-off**: More steps generally improve quality but increase inference time
- **Optimal steps**: Usually 50-100 for MNIST with this architecture

---

## ☁️ AWS Training

### Launch EC2 Instance and Train

```bash
cd aws_scripts
./launch_1gpu_instance.sh
```

This script will:

1. Launch a GPU instance (e.g., g4dn.xlarge)
2. Install dependencies
3. Run training
4. Download checkpoints and training plots

### Manual Remote Training

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone/upload your code
git clone <your-repo>
cd msml_612_project

# Install dependencies
pip install -r requirements.txt

# Run training
python3 run.py

# Download results
scp -i your-key.pem ubuntu@<instance-ip>:~/msml_612_project/models/dit_models/*.pth ./models/dit_models/
```

---

## 📝 Model Architecture

### DiT (Diffusion Transformer)

The model uses a Vision Transformer architecture adapted for diffusion models:

```
Input Image (28x28x1)
    ↓
Patch Embedding (7x7 patches, patch_size=4)
    ↓
Position Embedding (49 patches)
    ↓
Time Embedding (sinusoidal) + Class Embedding
    ↓
[DiT Block 1]
    Multi-Head Attention
    Layer Norm
    MLP
    ↓
[DiT Block 2]
    ...
    ↓
[DiT Block N]
    ↓
Final Layer Norm
    ↓
Linear Projection → Noise Prediction (28x28x1)
```

### Key Components

1. **Patch Embedding**: Converts image into sequence of patch embeddings
2. **Positional Encoding**: Adds spatial information to patches
3. **Time Conditioning**: Sinusoidal embeddings encode diffusion timestep
4. **Class Conditioning**: Learned embeddings for digit labels (0-9)
5. **Transformer Blocks**: Self-attention + MLP with residual connections
6. **Noise Prediction**: Outputs predicted noise at given timestep

### Hyperparameters

| Parameter     | Default | Description                   |
| ------------- | ------- | ----------------------------- |
| `image_size`  | 28      | Input image resolution        |
| `patch_size`  | 4       | Size of each patch (4x4)      |
| `hidden_dim`  | 768     | Transformer hidden dimension  |
| `depth`       | 4       | Number of transformer layers  |
| `num_heads`   | 4       | Multi-head attention heads    |
| `num_classes` | 10      | Number of digit classes (0-9) |

---

## 🔧 Advanced Usage

### Custom Dataset

To use CIFAR-10 instead of MNIST, edit `configs/dit_config.yaml`:

```yaml
data:
  dataset: "cifar10" # Change from "mnist"
  batch_size: 64

model:
  image_size: 32 # CIFAR-10 is 32x32
  image_channels: 3 # RGB instead of grayscale
```

### Modify Model Depth

```yaml
model:
  depth: 8 # Increase transformer layers
  hidden_dim: 1024 # Increase model capacity
  num_heads: 8 # More attention heads
```

### Change Diffusion Schedule

```yaml
diffusion:
  num_steps: 1000 # More timesteps (higher quality, slower)
  beta_schedule: "cosine" # Try cosine schedule
```

### Environment Variables

Set custom seed for reproducibility:

```bash
SEED=123 python3 run.py
```
