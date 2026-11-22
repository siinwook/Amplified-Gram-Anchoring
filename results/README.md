# Experimental Results

This document summarizes the full AGA comparison and the main findings from the accompanying research report.

## 1. Experimental Setup

Five backbone variants are compared:

| Variant | Refinement objective |
|---|---|
| **Pretrained** | Off-the-shelf DINOv2-S/14, no refinement |
| **None** | DINO + iBOT + KoLeo refinement, without Gram loss |
| **Gram** | Same objectives with the original teacher Gram target |
| **AGA** | Nonlinearly amplified target, **H=0** |
| **AGA-H** | Same amplified target with **H=10** Higham iterations |

The refinement runs share backbone initialization, ImageNet-1K data stream, optimization conditions, and 1,500 training iterations. Batch size is 80 (120,000 image samples processed). The Gram teacher is a frozen copy of the pretrained DINOv2 backbone. For AGA and AGA-H, the full-evaluation settings are **`b=0.4, alpha=8`**.

Downstream backbones are **frozen**. Segmentation uses a 1×1 convolution on normalized final patch features; depth concatenates four intermediate-layer feature maps with a 1×1 prediction head; classification uses a linear classifier on four CLS features plus the mean final patch feature.

**Evaluation split caveat:** VOC 2012 and ADE20K report the official validation split used for epoch selection. NYUv2 reports the evaluation split also used for selection. Oxford-IIIT Pet, CIFAR-100 and Flowers-102 use held-out test sets after selecting the probe epoch using training-data holdouts or the official Flowers validation set.

## 2. Semantic Segmentation

| Model | VOC 2012 mIoU (%) ↑ | ADE20K mIoU (%) ↑ | Oxford-IIIT Pet mIoU (%) ↑ |
|---|---:|---:|---:|
| Pretrained | 77.25 | 40.63 | 74.61 |
| None | 76.28 | 40.05 | 74.54 |
| Gram | 77.19 | 40.12 | 75.73 |
| AGA | 76.88 | 41.17 | 75.81 |
| **AGA-H** | **78.13** | **41.17** | **76.03** |

### Observation

- AGA-H has the highest displayed mIoU on all three segmentation benchmarks; on ADE20K, AGA and AGA-H are equal at two-decimal precision.
- Relative to Gram, the full-report improvements for AGA-H are **+0.93 pp** (VOC), **+1.04 pp** (ADE20K), and **+0.30 pp** (Pet), calculated using unrounded measurements.
- The importance of Higham correction varies by dataset: the improvement is most noticeable on VOC 2012.

## 3. Monocular Depth Estimation

| Model | NYUv2 RMSE ↓ |
|---|---:|
| Pretrained | 0.534 |
| None | 0.529 |
| Gram | 0.531 |
| **AGA** | **0.524** |
| **AGA-H** | **0.524** |

AGA and AGA-H have the lowest displayed RMSE. The uncorrected AGA variant reaches similar NYUv2 performance at lower refinement cost. NYUv2 evaluation is selected and reported on the same evaluation split.

## 4. Image Classification

| Model | CIFAR-100 top-1 (%) ↑ | Oxford Flowers-102 top-1 (%) ↑ |
|---|---:|---:|
| Pretrained | 86.35 | **99.53** |
| None | **87.04** | 99.38 |
| Gram | 86.80 | 99.37 |
| AGA | 86.57 | 99.41 |
| AGA-H | 86.51 | 99.37 |

The AGA variants remain close to the Gram baseline in classification. This is not evidence of improved global classification accuracy; the primary observed improvements are in dense prediction under this evaluation protocol.

## 5. Qualitative Patch Similarity Analysis

We compare cosine-similarity maps at identical query patches across **Pretrained, None, Gram, AGA, and AGA-H**. Selected examples show strengthened within-object or semantically related patch associations after AGA refinement. These are qualitative observations rather than quantitative segmentation scores.

Relevant report figures: patch similarity comparison (Figure 7), AGA-H minus Gram difference maps (Figure 2), and DINO/iBOT training loss curves (Figure 4).

## 6. Ablation Studies

### 6.1 Threshold and Amplification Strength

A preliminary grid uses **H=20** and a **shorter evaluation protocol than the main results above**. The scores must not be compared as if they were produced under the same probe budget.

| Threshold b | Strength alpha | VOC mIoU ↑ | ADE20K mIoU ↑ | Pet mIoU ↑ | NYUv2 RMSE ↓ |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 2 | 77.37 | 28.53 | 76.96 | 0.6293 |
| 0.4 | 4 | 77.70 | 29.03 | 77.04 | 0.6267 |
| **0.4** | **8** | **77.98** | **29.43** | **77.17** | **0.6247** |
| 0.5 | 2 | 77.16 | 28.37 | 76.91 | 0.6305 |
| 0.5 | 4 | 77.43 | 29.21 | 76.91 | 0.6285 |
| 0.5 | 8 | 77.59 | 28.96 | 77.00 | 0.6271 |
| 0.6 | 2 | 77.09 | 28.07 | 76.88 | 0.6310 |
| 0.6 | 4 | 77.12 | 28.64 | 76.92 | 0.6308 |
| 0.6 | 8 | 77.29 | 28.56 | 76.94 | 0.6295 |

Within this limited grid, `(b, alpha) = (0.4, 8)` performed best on the four displayed preliminary tasks.

### 6.2 Higham / Nearest Correlation Matrix

The elementwise amplification preserves Gram symmetry and its unit diagonal, but may violate positive semidefiniteness. We use Higham's alternating projections with Dykstra's correction to reduce this violation. The finite-iteration output may still have negative eigenvalues and is not guaranteed to be the exact nearest correlation matrix.

The report analyzes **H=0, 10, 20, and 30**, including eigenvalue behavior, realized amplification, and weak-pair distortion. `AGA-H` in the full evaluation uses **H=10**, while the parameter grid above uses **H=20**.

## 7. Limitations

- One backbone scale (**DINOv2-S/14**), **1,500 refinement iterations**, and **one seed**.
- Lightweight frozen probes do not represent the best possible dense decoders or full backbone fine-tuning.
- VOC, ADE20K and NYUv2 selection/reporting use the same evaluation split, so estimates may be optimistic.
- The contribution of correlation-matrix validity versus other changes in the amplified target remains uncertain.
- Further work needs repeated seeds, longer refinement, independent test evaluation, stronger dense heads and improved projections.
