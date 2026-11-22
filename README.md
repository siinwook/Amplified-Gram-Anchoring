# Amplified Gram Anchoring: Selectively Strengthening Patch Relationships for Dense Visual Representations

This project proposes **Amplified Gram Anchoring (AGA)**, a post-training objective that selectively strengthens patch relationships already exhibiting high cosine similarity in a frozen teacher's Gram matrix. The method modifies the **teacher target** rather than introducing trainable parameters.

AGA builds on the Gram Anchoring idea presented in [DINOv3](https://arxiv.org/abs/2508.10104). We study whether reinforcing strong teacher patch relationships can improve dense visual representations while preserving image-level recognition performance.

## Motivation

During extended self-supervised training, image-level representation quality can improve while spatial consistency in patch-level features deteriorates. Gram Anchoring aims to preserve localized patch relationships using a teacher Gram matrix.

Our hypothesis is that **selected strong patch similarities can be emphasized**, rather than matching all teacher relations with equal emphasis. Because this nonlinear modification may make the target indefinite, we also investigate a finite-iteration nearest-correlation-matrix approximation based on Higham's alternating projections.

## What I Implemented

### Amplified Gram Anchoring

- Pairwise cosine-similarity Gram matrices from normalized patch tokens
- Threshold-based nonlinear amplification of strong teacher similarities
- Optional **Higham / Dykstra correction** of the amplified Gram target
- AGA Gram alignment loss applied to the student patch representations
- Controlled refinement with DINO, iBOT, KoLeo, and Gram / AGA objectives

### Experiments

- Controlled comparison: **Pretrained / None / Gram / AGA / AGA-H**
- Semantic segmentation: PASCAL VOC 2012, ADE20K, Oxford-IIIT Pet
- Monocular depth estimation: NYUv2
- Image classification: CIFAR-100, Oxford Flowers-102
- Qualitative patch cosine-similarity comparisons
- Ablations of the similarity threshold, amplification strength, and Higham iteration budget

## Method

We compute a Gram matrix from L2-normalized teacher patch features and amplify entries above the threshold `b` using a normalized `tanh` mapping with strength `alpha`. Entries at or below `b` remain unchanged before correction. The student is trained to match either the original or the transformed teacher Gram matrix.

- **Gram:** unmodified teacher Gram target
- **AGA:** amplified target, without Higham correction (`H=0`)
- **AGA-H:** amplified target with `H=10` iterations of Higham / Dykstra correction

The final-report amplification setting is **`b=0.4`, `alpha=8`**. Finite Higham iterations can reduce, but do **not** guarantee elimination of, negative eigenvalues.

## Experimental Setup

All refinement runs start from the same **pretrained DINOv2-S/14** backbone. The study uses the ImageNet-1K training stream, **1,500 refinement iterations**, batch size **80**, and shared training conditions. We evaluate **frozen backbones** with lightweight downstream heads rather than jointly fine-tuning each backbone for every task.

The training pipeline is an adapted, resource-constrained study—not an exact reproduction of DINOv3 pretraining.

## Key Results

### Dense Prediction

| Benchmark | Metric | Gram | AGA-H |
|---|---|---:|---:|
| PASCAL VOC 2012 | mIoU (%) ↑ | 77.19 | **78.13** |
| ADE20K | mIoU (%) ↑ | 40.12 | **41.17** |
| Oxford-IIIT Pet | mIoU (%) ↑ | 75.73 | **76.03** |
| NYUv2 | RMSE ↓ | 0.531 | **0.524** |

AGA-H improves on Gram across these four dense evaluations **under our frozen-probe protocol**. For VOC 2012, ADE20K, and NYUv2, the reported evaluation split is also used for checkpoint/epoch selection; these figures should not be interpreted as independent held-out test estimates.

### Image Classification

| Benchmark | Gram | AGA | AGA-H |
|---|---:|---:|---:|
| CIFAR-100 top-1 (%) ↑ | 86.80 | 86.57 | 86.51 |
| Flowers-102 top-1 (%) ↑ | 99.37 | 99.41 | 99.37 |

Classification accuracy remains close to the Gram baseline, with small differences in both directions.

**See [`results/README.md`](results/README.md)** for all five model variants, evaluation protocols, ablations, and limitations.

## Repository Structure

```text
Amplified-Gram-Anchoring/
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── augmentation.py
│   ├── gram_loss.py
│   ├── aga_loss.py
│   ├── refinement.py
│   ├── training.py
│   └── probes.py
├── results/
│   └── README.md
├── notebooks/                       # Upload original training/evaluation notebooks
└── report/                          # Place the full technical report PDF here
```

## Reproduction Notes

The `src/` implementation was separated from the completed AGA notebooks with the training/loss logic preserved where possible. It relies on the [official DINOv2 repository](https://github.com/facebookresearch/dinov2) for the ViT backbone, DINO/iBOT/KoLeo losses, masking and the projection head.

Use a CUDA-enabled PyTorch environment with DINOv2 and its dependencies importable; configure `HF_TOKEN` for authorized access to the ImageNet-1K stream. The default refinement pipeline can be run as:

```bash
python -m src.training --mode gram
python -m src.training --mode aga --b 0.4 --alpha 8
python -m src.training --mode aga-h --b 0.4 --alpha 8 --higham-iterations 10
```

The head-calibration checkpoint is reused between runs. The full downstream dataset adapters and validation/test split rules are retained in the original evaluation notebook, rather than reimplemented in `src/probes.py`. A reproducible release should publish the original notebooks, dataset access instructions, environment versions, and the precise DINOv2 revision.

## Limitations

This study uses one backbone scale, 1,500 refinement iterations, and a single training seed. The reported improvements are specific to the frozen-probe evaluation protocol. The finite Higham projection is approximate, and a causal link between target correlation-matrix validity and downstream improvements remains unestablished.

## References

- [DINOv3](https://arxiv.org/abs/2508.10104)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [Higham (2002), Computing the Nearest Correlation Matrix](https://doi.org/10.1093/imanum/22.3.329)
