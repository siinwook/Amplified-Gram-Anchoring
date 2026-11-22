"""Experiment configuration copied from the clean Kaggle training notebook.
Change HIGHAM_MAX_ITER to 0 (AGA) or 10 (AGA-H) before a run.
All official DINOv2 imports are resolved from a locally cloned dinov2 repository.
"""
import os
import json
from pathlib import Path

WORK = Path(os.environ.get("AGA_WORK", "/kaggle/working"))
OUTPUT_ROOT = WORK / "aga_server_run"
DINO_REPO = Path(os.environ.get("DINOV2_REPO", str(WORK / "dinov2")))
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = "ILSVRC/imagenet-1k"

SEED = 42

# Data
BATCH_SIZE = 80
LOCAL_CROPS = 4
GLOBAL_SIZE = 224
LOCAL_SIZE = 98
GRAM_SIZE = 448
SHUFFLE_BUFFER = 2048
NUM_WORKERS = 0  # HF IterableDataset 중복 worker 방지를 위해 유지

# Heads
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

# AGA — uploaded notebook의 현재 값 유지
AGA_B = 0.4
AGA_ALPHA = 8.0
HIGHAM_MAX_ITER = 0
HIGHAM_TOL = 1e-4

# Training
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

# unattended run controls
#RUN_MODES = ["none", "gram", "aga"]
RUN_MODES = ["aga"]
PRINT_EVERY = 5
SAVE_EVERY = 500
