"""Figure 7 patch similarity maps and Figure 1 AGA-H minus Gram difference maps.
Notebook-derived script: visualization/aga_post_training_analysis.ipynb.
Change MODEL_PATHS, IMAGE_PATH / VIS_INDEX, and QUERY_ROW / QUERY_COL before running.
"""
import subprocess
from IPython.display import display

import os, sys, gc, re, json
from pathlib import Path
from itertools import islice

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/kaggle/working")
DINO_REPO = WORK / "dinov2"



if not DINO_REPO.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/facebookresearch/dinov2.git", str(DINO_REPO)], check=True)

if str(DINO_REPO) not in sys.path:
    sys.path.insert(0, str(DINO_REPO))

from torchvision import transforms
from PIL import Image

from dinov2.hub.backbones import dinov2_vits14
from dinov2.data.transforms import make_normalize_transform

print("DINOv2 import OK")


from pathlib import Path
import pandas as pd


# ============================================================
# ============================================================

MODEL_PATHS = {
    "Pretrained": Path(
        "/kaggle/input/models/siinwook/dinov2-vits14-pretrain/pytorch/default/1/dinov2_vits14_pretrain.pth"
    ),

    "None": Path(
         "/kaggle/input/models/siinwook/none-1500iter/pytorch/default/1/student_step_01500.pt"
    ),
    
    "Gram": Path(
        "/kaggle/input/models/siinwook/gram-1500iter/pytorch/default/1/student_step_01500.pt"
    ),

    "AGA": Path(
        "/kaggle/input/models/siinwook/aga-1500iter-h0-b04-a8/pytorch/default/1/aga_student_step_01500_H0_b04_a8.pt"
    ),

    "AGA-H": Path(
        "/kaggle/input/models/siinwook/aga-1500iter-h10-b04-a8/pytorch/default/1/aga_student_step_01500_H10_b04_a8.pt"
    ),
}

# ============================================================
# ============================================================

for mode, path in MODEL_PATHS.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{mode} backbone 파일을 찾지 못했습니다:\n{path}"
        )

    print(
        f"{mode:10s} -> {path}"
    )


# ============================================================
# ============================================================

FINAL_CHECKPOINTS = {
    mode: path
    for mode, path in MODEL_PATHS.items()
}


print("\nFinal checkpoints:")

for name, path in FINAL_CHECKPOINTS.items():
    print(
        f"{name:10s} -> {path}"
    )


IMAGE_PATH = None

VIS_SEED = 123
VIS_INDEX = 1
HF_DATASET = "ILSVRC/imagenet-1k"


def load_analysis_image():
    if IMAGE_PATH is not None:
        path = Path(IMAGE_PATH)
        if not path.exists():
            raise FileNotFoundError(path)

        image = Image.open(path).convert("RGB")
        print("Loaded local image:", path)
        return image

    # HF path
    from datasets import load_dataset

    HF_TOKEN = None
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

    if not HF_TOKEN:
        raise RuntimeError(
            "IMAGE_PATH=None인데 HF_TOKEN을 찾지 못했습니다.\n"
            "Kaggle Secret HF_TOKEN을 등록하거나 IMAGE_PATH를 지정하세요."
        )

    stream = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    ).shuffle(
        seed=VIS_SEED,
        buffer_size=1000,
    )

    iter_stream = iter(stream)
    for _ in range(VIS_INDEX):
        example = next(iter_stream, VIS_INDEX)
        display(example["image"].convert("RGB"))
    return example["image"].convert("RGB")


pil_img = load_analysis_image()
display(pil_img)



PATCH_SIZE = 14
GRID = 16
IMAGE_SIZE = GRID * PATCH_SIZE  # 224

display_transform = transforms.Compose([
    transforms.Resize(
        256,
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(IMAGE_SIZE),
])

model_transform = transforms.Compose([
    transforms.Resize(
        256,
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    make_normalize_transform(),
])

display_img = display_transform(pil_img)
x = model_transform(pil_img).unsqueeze(0).to(DEVICE)

print("model input:", tuple(x.shape))
display(display_img)



def show_patch_grid(image):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    for i in range(GRID + 1):
        ax.axhline(i * PATCH_SIZE, linewidth=0.5, alpha=0.7)
        ax.axvline(i * PATCH_SIZE, linewidth=0.5, alpha=0.7)

    centers = [
        i * PATCH_SIZE + PATCH_SIZE / 2
        for i in range(GRID)
    ]

    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(range(GRID), fontsize=7)
    ax.set_yticklabels(range(GRID), fontsize=7)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title("Choose query patch: (row, col)")
    plt.show()


show_patch_grid(display_img)



QUERY_ROW = 7
QUERY_COL = 8

assert 0 <= QUERY_ROW < GRID
assert 0 <= QUERY_COL < GRID

print("query:", (QUERY_ROW, QUERY_COL))



def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_backbone_from_checkpoint(path):
    model = dinov2_vits14(pretrained=False)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE).eval()
    return model


@torch.no_grad()
def extract_patch_tokens(backbone, x):
    out = backbone(
        x,
        is_training=True,
    )

    z = out["x_norm_patchtokens"]

    z = F.normalize(z, p=2, dim=-1)

    return z


@torch.no_grad()
def get_similarity(backbone, x, row, col):
    z = extract_patch_tokens(backbone, x)

    idx = row * GRID + col
    q = z[:, idx]

    sim = torch.einsum(
        "bd,bnd->bn",
        q,
        z,
    ).reshape(GRID, GRID)

    return sim.detach().cpu()


def evaluate_checkpoint(path, row=QUERY_ROW, col=QUERY_COL):
    model = load_backbone_from_checkpoint(path)

    try:
        sim = get_similarity(
            model,
            x,
            row,
            col,
        )
    finally:
        del model
        cleanup()

    return sim



def upsample_similarity(
    sim_map,
    upsample=False,
):
    mode = "bilinear" if upsample else "nearest"

    kwargs = {}
    if mode == "bilinear":
        kwargs["align_corners"] = False

    return F.interpolate(
        sim_map[None, None].float(),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode=mode,
        **kwargs,
    )[0, 0]



def collect_final_similarity_maps():
    maps = {}

    for name, path in FINAL_CHECKPOINTS.items():
        print("Evaluating:", name)
        sim = evaluate_checkpoint(path)
        maps[name] = sim

    return maps


final_maps = collect_final_similarity_maps()



def compare_similarity_maps(
    maps,
    threshold=None,
    overlap=True,
    upsample=False,
    vmin=-1,
    vmax=1,
    alpha=0.82,
):
    names = list(maps.keys())

    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(5 * len(names), 5),
        squeeze=False,
    )
    axes = axes[0]

    last_im = None

    for ax, name in zip(axes, names):
        sim_up = upsample_similarity(
            maps[name],
            upsample=upsample,
        )

        arr = sim_up.numpy()
        if threshold is not None:
            arr = np.ma.masked_less(
                arr,
                threshold,
            )

        if overlap:
            ax.imshow(display_img)

        last_im = ax.imshow(
            arr,
            alpha=alpha if overlap else 1.0,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

        qx = QUERY_COL * PATCH_SIZE + PATCH_SIZE / 2
        qy = QUERY_ROW * PATCH_SIZE + PATCH_SIZE / 2

        ax.scatter(
            qx,
            qy,
            marker="x",
            s=100,
        )

        ax.set_title(name)
        ax.set_xlim(0, IMAGE_SIZE)
        ax.set_ylim(IMAGE_SIZE, 0)
        ax.axis("off")

    title = (
        f"Same image / same query ({QUERY_ROW}, {QUERY_COL})"
    )
    if threshold is not None:
        title += f" | sim ≥ {threshold}"

    fig.suptitle(title)
    fig.colorbar(
        last_im,
        ax=axes.tolist(),
        fraction=0.02,
        pad=0.02,
        label="Cosine similarity",
    )
    plt.show()


compare_similarity_maps(
    final_maps,
    threshold=None,
    overlap=False,
    upsample=True,
    vmin=-1,
    vmax=1,
)


def show_similarity_difference(
    map_a,
    map_b,
    name_a,
    name_b,
    overlap=True,
    upsample=False,
    alpha=0.6,
):
    # Positive: B has higher similarity than A
    diff = map_b - map_a

    diff_up = upsample_similarity(
        diff,
        upsample=upsample,
    )

    arr = diff_up.detach().cpu().numpy()

    bound = max(
        float(np.abs(arr).max()),
        1e-6,
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    extent = (0, IMAGE_SIZE, IMAGE_SIZE, 0)

    if overlap:
        ax.imshow(
            display_img,
            extent=extent,
            origin="upper",
        )

    im = ax.imshow(
        arr,
        cmap="coolwarm",
        vmin=-bound,
        vmax=bound,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        alpha=alpha if overlap else 1.0,
    )

    qx = (QUERY_COL + 0.5) * PATCH_SIZE
    qy = (QUERY_ROW + 0.5) * PATCH_SIZE

    ax.scatter(
        qx,
        qy,
        marker="x",
        s=120,
        color="deepskyblue",
        linewidths=2,
    )

    ax.set_title(
        f"{name_b} - {name_a}"
    )

    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.axis("off")

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Δ cosine similarity",
    )

    plt.tight_layout()
    plt.show()

name_a = 'Gram'
name_b = 'AGA-H'

show_similarity_difference(
    final_maps[name_a],
    final_maps[name_b],
    name_a,
    name_b,
    overlap=False,
    upsample=True,
)
