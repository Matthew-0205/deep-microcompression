Here is the README for the experiment reproduction based on the provided notebook.

---

# LeNet-5 Experiment Reproduction: Deep Microcompression

This guide details how to reproduce the LeNet-5 experiments (Baseline, DMC Ultra, and DMC Tiny) from the "Deep Microcompression" paper. The provided script trains a baseline fp32 model and two compressed variants optimized for constrained hardware (like the ATmega32).

## Experiment Overview

This experiment generates and evaluates three models:

1. **Baseline (fp32):** Standard LeNet-5 implementation.
2. **DMC Ultra:** Maximally compressed model optimized for extreme storage reduction (e.g., ~2.6KB).
3. **DMC Tiny:** Optimized for deployment on ultra-constrained hardware (ATmega32) with a strict 2KB SRAM runtime limit.

## Prerequisites & File Structure

This script relies on the `development` module from the main project. Ensure your directory structure is set up as follows:

```text
project_root/
├── development/           # Core library (imported in the script)
├── Datasets/              # Directory for MNIST dataset
├── experiments/
│   └── lenet_dmc_experment/
│       └── script.ipynb  # This notebook
└── examples/
    └── arduino_uno_lenet5/ # Output directory for C deployment code

```

### Dependencies

* Python 3.12+
* PyTorch
* Torchvision
* tqdm

If missing, install via pip:

```bash
pip install torch torchvision tqdm

```

## Configuration Details

### 1. Hardware Determinism

The script enforces determinism for reproducibility:

* **Random Seed:** `25`
* **CUDNN:** Deterministic mode enabled
* **Device:** CUDA (if available), else CPU.

### 2. Model Configurations

| Model | Optimization Goal | Configuration |
| --- | --- | --- |
| **Baseline** | Reference Accuracy | Standard LeNet-5 (fp32) |
| **DMC Ultra** | **Max Compression** | *Pruning:* `{conv0: 0, conv1: 9, lin0: 64}`<br>

<br>*Quantization:* 4-bit static (mixed granularity) |
| **DMC Tiny** | **2KB SRAM Limit** | *Pruning:* `{conv0: 4, conv1: 2, lin0: 31}`<br>

<br>*Quantization:* Mixed 8-bit/4-bit parameters, 4-bit activations |

## Execution Steps

### Step 1: Baseline Training

The script first trains the Baseline fp32 model or loads existing weights (`lenet5_state_dict.pth`).

* **Epochs:** 30 (or until early stopping)
* **Expected Accuracy:** ~99.42%

### Step 2: Compression & Retraining (DMC)

It then applies the DMC pipeline (Pruning + Quantization Aware Training) to the baseline.

* **Method:** Two-step training (Pruning → Quantization)
* **DMC Ultra Epochs:** 40
* **DMC Tiny Epochs:** 40

### Step 3: Deployment (C Generation)

The final compressed models are converted into C++ code suitable for Arduino/embedded deployment.

* **Output Path:** `../../examples/arduino_uno_lenet5`
* **Generated Files:** `src/` (weights/logic) and `include/` (headers).

## Expected Results

Based on the provided notebook execution:

| Model | Accuracy | Size (KB) | Compression Ratio |
| --- | --- | --- | --- |
| **Baseline** | 99.42% | 144.95 KB | 1.0x |
| **DMC Ultra** | 97.93% | 2.60 KB | 55.75x |
| **DMC Tiny** | 98.04% | 19.57 KB | ~7.4x |

*> **Note:** DMC Tiny is larger than Ultra because it prioritizes **runtime activation memory** (SRAM) reduction over pure storage (Flash) size to fit the ATmega32's 2KB RAM limit.*

## Usage

To run the reproduction:

1. Open `reproduce_table1.ipynb` in Jupyter.
2. Ensure the `development` module is in the Python path (handled by `sys.path.append("../../")` in the notebook).
3. Run all cells.

To deploy a specific model to the board (generates C code), modify the last cell:

```python
model_name = "dmc_tiny" # or "dmc_ultra"
save_model_to_board(model_family[model_name], PROJECT_BASE_DIR)

```
