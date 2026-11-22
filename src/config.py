"""Refinement settings from the clean AGA notebook, with final report AGA defaults."""
import os
from pathlib import Path

SEED = 42
HF_DATASET = "ILSVRC/imagenet-1k"
HF_TOKEN = os.environ.get("HF_TOKEN")
OUTPUT_ROOT = Path(os.environ.get("AGA_OUTPUT_ROOT", "./outputs"))

# ImageNet stream / crops
BATCH_SIZE = 80
LOCAL_CROPS = 4
GLOBAL_SIZE = 224
LOCAL_SIZE = 98
GRAM_SIZE = 448
SHUFFLE_BUFFER = 2048
NUM_WORKERS = 0

# DINOv2 heads
DINO_OUT_DIM = 8192
IBOT_OUT_DIM = 8192
HEAD_HIDDEN_DIM = 2048
HEAD_BOTTLENECK_DIM = 256
HEAD_NLAYERS = 3

# Loss weights
DINO_WEIGHT = 1.0
IBOT_WEIGHT = 1.0
KOLEO_WEIGHT = 0.1
RELATION_WEIGHT = 1.0
HIGHAM_TOL = 1e-4

# Refinement
CALIB_STEPS = 20
REFINE_STEPS = 1500
HEAD_CALIB_LR = 1e-4
BACKBONE_LR = 1e-5
HEAD_REFINE_LR = 1e-4
WEIGHT_DECAY = 0.04
GRAD_CLIP = 3.0
TEACHER_TEMP = 0.07
EMA_MOMENTUM = 0.999
USE_AMP = True
PRINT_EVERY = 50
SAVE_EVERY = 500

# Final-report configurations: AGA = H=0, AGA-H = H=10.
AGA_B = 0.4
AGA_ALPHA = 8.0
AGA_HIGHAM_ITER = 10
