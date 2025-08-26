# CMDBA : Detection and Mitigation of Backdoor Attacks by Contribution Measurement in Vertical Federated Learning

This repository implements a **Vertical Federated Learning (VFL)** training pipeline with pluggable backdoor **attacks** and **defenses**. The main entry point is **`train_new.py`**, which trains a VFL model (clean or poisoned), optionally applies a defense, and evaluates on clean/poisoned test sets.

---

## ✅ Quick Start

### 1) Environment setup

```bash
# (optional) create & activate a virtual environment
python3 -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

> GPU is strongly recommended. The provided `requirements.txt` targets PyTorch **2.4.0** with CUDA **12.4** wheels.

### 2) Datasets

- **CIFAR10 / CINIC10**: auto-downloaded on first run.
- **Bank Marketing (BM)**: automatically handled from CSV (no manual download required).

### 3) Run
* If you want to use our defense strategy, please use `SHAP_MAE` in `DM` field.
* If you want to run `He et.al attack`, please use `TIFS` in `AM` field

#### Attack: VILLIAN, Defense: VFLIP defense
```bash
python3 train_new.py --D CIFAR10 --P 4 --AM VILLIAN --AN 1 --DM VFLIP --epochs 50 --label 0
```

#### Attack: He et.al, Defense: SHAP_MAE
```bash
python3 train_new.py --D CIFAR10 --P 4 --AM TIFS --AN 1 --DM SHAP_MAE --epochs 50 --label 0
```

#### Attack: VILLIAN, Defense: VFLMonitor
```bash
python3 train_new.py --D CINIC10 --P 4 --AM VILLIAN --AN 1 --DM VFLMonitor --epochs 50 --label 0
```

#### Bank Marketing (BM) example
```bash
python3 train_new.py --D BM --P 4 --AM VILLIAN --AN 1 --DM SHAP_MAE --epochs 40 --label 0
```
---

## 🔧 Command‑line Arguments (main ones)

> See `train_new.py` for the full list. Below are the commonly used flags.

| Argument   | Type / Default   | Choices / Range                                                                                            | Description                                       |
| ---------- | ---------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `--D`      | **required**     | `CIFAR10`, `CINIC10`, `BM`                                                                                 | Dataset                                           |
| `--P`      | int / `2`        | `2`, `4`, `8`                                                                                              | Number of parties                                 |
| `--epochs` | int / `50`       | `40 or 50`                                                                                                 | Training epochs (CIFAR10,CINIC usually 50, BM 40) |
| `--AM`     | str / `VILLIAN`  | `TIFS`, `VILLIAN`                                                                                          | Attack method                                     |
| `--DM`     | str / `SHAP_MAE` | `NONE`, `VFLIP`, `SHAP_MAE`, `SHAP_MAE_D`, `SHAP_MAE_CVPR`, `NEURAL_CLEANSE`, `CoPur`, `VFLMonitor`, `ABL` | Defense method                                    |
| `--label`  | int / `0`        | dataset classes                                                                                            | Target class for attack                           |

**Defense-specific (All are set, you don't have to change)**
- **CoPur**
  - `--copur_delta` (float), `--copur_tau` (float), `--copur_iter` (int)

- **VFLMonitor**
  - `--swd_proj` (int): sliced‑Wasserstein projection count  
  - `--tie_rand_seed` (int): randomized decision seed  
  - `center_loss_weight` is set internally when `--DM VFLMonitor`


## 🗺️ Repository Map

```
train_new.py                 # main entry point
eval.py                      # evaluation (metrics, timings)
Defense.py                   # SHAP utilities, thresholds, helper logic
defense_model_structure.py   # defense models (MAE, SHAP_MAE, CoPur, etc.)
train_model_structure.py     # VFL architectures (bottom/top)
vflmonitor.py                # VFLMonitor (bank builder, detector)
data.py                      # dataset loaders & party feature slicing
Attack.py                    # attacks (TIFS, VILLIAN)
generate_vec.py              # vector generation utilities
search_vec.py                # design/search of trigger vectors
cal_centers.py               # center computations used by defenses
requirements.txt             # dependencies
```

---