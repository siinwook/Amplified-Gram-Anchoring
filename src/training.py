"""Notebook-derived DINOv2 controlled refinement entry point.

Run using: PYTHONPATH=/path/to/dinov2:. python -m src.training
The clean notebook is retained under notebooks/refinement/.
"""
import os, sys, gc, math, random, json, time
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import psutil
from . import config
from .config import *
from .augmentation import build_stream_loader
from .refinement import DINOv2Refiner

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def memory_status(prefix=""):
    p = psutil.Process(os.getpid())
    cpu = p.memory_info().rss / 1024**3
    gpu_alloc = torch.cuda.memory_allocated() / 1024**3
    gpu_res = torch.cuda.memory_reserved() / 1024**3
    print(
        f"{prefix} CPU RSS={cpu:.2f} GB | "
        f"GPU allocated={gpu_alloc:.2f} GB | "
        f"GPU reserved={gpu_res:.2f} GB"
    )


def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def make_calibration_optimizer(model):
    return torch.optim.AdamW(
        [
            {
                "params": model.student_dino_head.parameters(),
                "lr": HEAD_CALIB_LR,
            },
            {
                "params": model.student_ibot_head.parameters(),
                "lr": HEAD_CALIB_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


def make_refinement_optimizer(model):
    return torch.optim.AdamW(
        [
            {
                "params": model.student_backbone.parameters(),
                "lr": BACKBONE_LR,
            },
            {
                "params": model.student_dino_head.parameters(),
                "lr": HEAD_REFINE_LR,
            },
            {
                "params": model.student_ibot_head.parameters(),
                "lr": HEAD_REFINE_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


def save_student_backbone(model, mode, step):
    mode_dir = OUTPUT_ROOT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    path = mode_dir / f"student_step_{step:05d}.pt"
    torch.save(
        model.student_backbone.state_dict(),
        path,
    )
    print("checkpoint:", path)


def run_steps(
    model,
    loader,
    optimizer,
    steps,
    mode,
    calibration=False,
    relation_weight=RELATION_WEIGHT,
    print_every=PRINT_EVERY,
    save_every=None,
):
    model.train()
    iterator = iter(loader)

    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP
    )
    logs = []

    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            loss, stats = model.forward_losses(
                batch,
                relation_weight=relation_weight,
                calibration=calibration,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        trainable_params = [
            p for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()
        model.update_teacher(EMA_MOMENTUM)

        logs.append({"step": step, **stats})

        if step == 1 or step % print_every == 0 or step == steps:
            msg = (
                f"[{mode} {step:04d}/{steps}] "
                f"total={stats['total']:.4f} "
                f"dino={stats['dino']:.4f} "
                f"ibot={stats['ibot']:.4f} "
                f"koleo={stats['koleo']:.4f} "
                f"rel={stats['relation']:.5f}"
            )

            if "retention" in stats:
                msg += (
                    f" | d_amp={stats['d_amp']:.4f}"
                    f" d_hat={stats['d_hat']:.4f}"
                    f" R={stats['retention']:.3f}"
                    f" weak={stats['d_weak']:.4f}"
                    f" strong={stats['strong_ratio']:.3f}"
                    f" H={stats['higham_iters']}"
                )

            print(msg)
            memory_status("  ")

        if (
            (not calibration)
            and save_every is not None
            and step % save_every == 0
        ):
            save_student_backbone(
                model,
                mode,
                step,
            )

    return pd.DataFrame(logs)


def load_calibrated_model(mode):
    model = DINOv2Refiner(
        relation_mode=mode
    ).to(device)

    ckpt = torch.load(
        CALIB_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    if mode == "none":
        model.load_state_dict(
            ckpt["model"],
            strict=True,
        )
    else:
        incompatible = model.load_state_dict(
            ckpt["model"],
            strict=False,
        )

        # Calibration checkpoint에는 의도적으로 gram_teacher가 없음.
        unexpected = list(incompatible.unexpected_keys)
        missing_non_gram = [
            k for k in incompatible.missing_keys
            if not k.startswith("gram_teacher.")
        ]

        if unexpected or missing_non_gram:
            raise RuntimeError(
                "Unexpected checkpoint mismatch:\n"
                f"unexpected={unexpected}\n"
                f"missing_non_gram={missing_non_gram}"
            )

        # Calibration 중 student backbone은 frozen이었으므로
        # 이것이 공통 frozen Gram teacher init.
        model.reset_gram_teacher_from_student()

    model.set_backbone_trainable(True)
    return model


def run_refinement(mode):
    assert mode in {"none", "gram", "aga"}

    print("\n" + "=" * 72)
    print("START:", mode)
    print("=" * 72)

    # Build objects first.
    model = load_calibrated_model(mode)
    loader = build_stream_loader(
        mode=mode,
        seed=SEED,
    )
    optimizer = make_refinement_optimizer(
        model
    )

    # Reset again after mode-specific model construction so that
    # student stochasticity / augmentation / masking start identically.
    seed_everything(SEED)

    memory_status(f"before {mode}:")

    try:
        df = run_steps(
            model,
            loader,
            optimizer,
            steps=REFINE_STEPS,
            mode=mode,
            calibration=False,
            relation_weight=RELATION_WEIGHT,
            print_every=PRINT_EVERY,
            save_every=SAVE_EVERY,
        )

        mode_dir = OUTPUT_ROOT / mode
        mode_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Final checkpoint is always written even if REFINE_STEPS
        # is not divisible by SAVE_EVERY.
        final_path = (
            mode_dir
            / f"student_step_{REFINE_STEPS:05d}.pt"
        )
        if not final_path.exists():
            torch.save(
                model.student_backbone.state_dict(),
                final_path,
            )

        df.to_csv(
            mode_dir / "training_log.csv",
            index=False,
        )

        summary_cols = [
            "total",
            "dino",
            "ibot",
            "koleo",
            "relation",
        ]
        summary = {
            "mode": mode,
            **{
                col: float(df[col].mean())
                for col in summary_cols
            },
        }

        for col in [
            "d_amp",
            "d_hat",
            "retention",
            "d_weak",
            "strong_ratio",
            "higham_iters",
        ]:
            if col in df.columns:
                summary[col] = float(
                    df[col].mean()
                )

        with open(
            mode_dir / "summary.json",
            "w",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

        print("DONE:", mode)
        print(summary)

    finally:
        # Crucial: no model / optimizer / loader / iterator survives.
        try:
            del df
        except Exception:
            pass
        del optimizer
        del loader
        del model
        cleanup_cuda()
        memory_status(f"after {mode} cleanup:")

    return summary


def main():
    global device, CALIB_CKPT
    assert torch.cuda.is_available(), "CUDA GPU required for this training configuration"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_snapshot = {name: globals()[name] for name in [
        "SEED", "BATCH_SIZE", "LOCAL_CROPS", "GLOBAL_SIZE", "LOCAL_SIZE",
        "GRAM_SIZE", "SHUFFLE_BUFFER", "NUM_WORKERS", "DINO_OUT_DIM", "IBOT_OUT_DIM",
        "HEAD_HIDDEN_DIM", "HEAD_BOTTLENECK_DIM", "HEAD_NLAYERS", "DINO_WEIGHT",
        "IBOT_WEIGHT", "KOLEO_WEIGHT", "RELATION_WEIGHT", "AGA_B", "AGA_ALPHA",
        "HIGHAM_MAX_ITER", "HIGHAM_TOL", "CALIB_STEPS", "REFINE_STEPS",
        "HEAD_CALIB_LR", "BACKBONE_LR", "HEAD_REFINE_LR", "WEIGHT_DECAY",
        "GRAD_CLIP", "TEACHER_TEMP", "EMA_MOMENTUM", "USE_AMP", "RUN_MODES",
        "PRINT_EVERY", "SAVE_EVERY",
    ]}
    (OUTPUT_ROOT / "config.json").write_text(json.dumps(config_snapshot,indent=2))

    device = torch.device("cuda")

    seed_everything(SEED)

    calib_loader = build_stream_loader(
        mode="none",
        seed=SEED,
    )

    calib_model = DINOv2Refiner(
        relation_mode="none"
    ).to(device)

    calib_model.set_backbone_trainable(False)

    calib_optimizer = make_calibration_optimizer(
        calib_model
    )

    # Model construction이 소비한 RNG와 관계없이
    # data/masking/training RNG의 시작점을 고정.
    seed_everything(SEED)

    memory_status("before calibration:")

    calib_logs = run_steps(
        calib_model,
        calib_loader,
        calib_optimizer,
        steps=CALIB_STEPS,
        mode="calibration",
        calibration=True,
        print_every=max(10, PRINT_EVERY),
        save_every=None,
    )

    CALIB_CKPT = OUTPUT_ROOT / "head_calibrated.pt"

    torch.save(
        {
            "model": calib_model.state_dict(),
            "seed": SEED,
            "calib_steps": CALIB_STEPS,
        },
        CALIB_CKPT,
    )

    calib_logs.to_csv(
        OUTPUT_ROOT / "calibration_logs.csv",
        index=False,
    )

    # Student backbone is still original pretrained DINOv2-S
    # because calibration freezes it. Save as t=0 patch-sim reference.
    torch.save(
        calib_model.student_backbone.state_dict(),
        OUTPUT_ROOT / "student_pretrained_reference.pt",
    )

    print("saved:", CALIB_CKPT)

    # IMPORTANT: no calibration objects survive into refinement.
    del calib_logs
    del calib_optimizer
    del calib_model
    del calib_loader
    cleanup_cuda()

    memory_status("after calibration cleanup:")



    all_summaries = []

    for mode in RUN_MODES:
        summary = run_refinement(mode)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(
        OUTPUT_ROOT / "summary.csv",
        index=False,
    )

    print(summary_df.to_string(index=False))

    print("\nAll requested runs finished.")
    print("Outputs:", OUTPUT_ROOT)
    memory_status("final:")


if __name__ == "__main__":
    main()
