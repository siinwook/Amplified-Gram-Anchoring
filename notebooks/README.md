# Notebooks

The notebooks are **cleaned copies of the user-supplied final files**. Model calculations,
training loops and remaining evaluation metrics are preserved; outputs were cleared, Korean
markdown replaced by short English section headings, and report-unused experiments removed.

| Notebook | Scope |
|---|---|
| `refinement/aga_refinement.ipynb` | ImageNet-1K streaming, head calibration, `none` / `gram` / `aga` refinement and checkpoint output |
| `evaluation/aga_full_split_frozen_backbone_evaluation.ipynb` |  VOC 2012, ADE20K, Oxford Pet, NYUv2, CIFAR-100 and Flowers-102 frozen probes |
| `visualization/aga_post_training_analysis.ipynb` | Figure 7 five-model patch cosine similarities and Figure 1 AGA-H−Gram difference maps |

In the final evaluation notebook, `TASKS` lists all six reported benchmarks; running them can
trigger large downloads. `MODEL_PATHS` must point to the five actual checkpoints.

To produce AGA-H in the training notebook, set `HIGHAM_MAX_ITER=10` and choose output names
that distinguish it from H=0. Figures already included under `results/figures/` are extracted
from the final research report; notebook plotting requires the same input image/query patches.

The original uploaded evaluation notebooks are not a complete automated launcher for the nine-setting `(b, alpha)` sweep in the report; the preliminary tables must not be presented as if all ablations can be reproduced from a single notebook run.
