# Source Modules

`src/` contains code extracted from the **cleaned** experiment notebooks, with imports and
entry points adapted where necessary. The actual notebooks remain the authoritative runs.

| File | Source notebook | Purpose |
|---|---|---|
| `config.py`, `augmentation.py` | `notebooks/refinement/` | Refinement settings, streaming and multi-crop data |
| `gram_loss.py`, `aga_loss.py` | `notebooks/refinement/` | Original Gram, threshold amplification and Higham/Dykstra |
| `refinement.py`, `training.py` | `notebooks/refinement/` | DINO/iBOT/KoLeo integration, calibration and 1,500-step refinement |
| `evaluation.py` | `notebooks/evaluation/` | Six final frozen-backbone benchmarks; KITTI-only code excluded |
| `visualization.py` | `notebooks/visualization/` | Five-model similarity and AGA-H−Gram difference maps |

**Settings:** `(b, alpha)=(0.4, 8)`; `H=0` for AGA and `H=10` for AGA-H.
The clean training notebook was uploaded with `RUN_MODES=["aga"]` and `HIGHAM_MAX_ITER=0`;
these defaults are preserved in `config.py`. Switch settings for the other variants.

External DINOv2 code, ImageNet access credentials, Kaggle checkpoint paths and benchmark data
are prerequisites. No claim is made that a full training run was reproduced in this package.
