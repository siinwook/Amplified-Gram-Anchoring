import os, gc, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from .config import *
from .augmentation import build_stream_loader
from .refinement import DINOv2Refiner

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CALIB_CKPT = Path(os.environ.get("AGA_CALIB_CKPT", str(OUTPUT_ROOT / "head_calibrated.pt")))

device = DEVICE

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

def save_student_backbone(model, run_name, step):
    mode_dir = OUTPUT_ROOT / run_name
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
    run_name=None,
    calibration=False,
    relation_weight=RELATION_WEIGHT,
    print_every=PRINT_EVERY,
    save_every=None,
):
    if run_name is None:
        run_name = mode

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
                f"[{run_name} {step:04d}/{steps}] "
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
                run_name,
                step,
            )

    return pd.DataFrame(logs)

def load_calibrated_model(mode, aga_config=None):
    aga_config = aga_config or {}

    model = DINOv2Refiner(
        relation_mode=mode,
        aga_b=float(aga_config.get("b", AGA_B)),
        aga_alpha=float(aga_config.get("alpha", AGA_ALPHA)),
        higham_max_iter=int(aga_config.get("H", AGA_HIGHAM_ITER)),
    ).to(device)

    ckpt = torch.load(
        CALIB_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    if mode == "none":
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        incompatible = model.load_state_dict(ckpt["model"], strict=False)
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
        model.reset_gram_teacher_from_student()

    model.set_backbone_trainable(True)
    return model

def _fmt_value(x):
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    return str(xf).replace(".", "p")

def make_run_name(mode, aga_config=None):
    if mode != "aga":
        return mode
    aga_config = aga_config or {}
    return (
        f"aga_b{_fmt_value(aga_config['b'])}"
        f"_a{_fmt_value(aga_config['alpha'])}"
        f"_H{int(aga_config['H'])}"
    )

def run_refinement(mode, aga_config=None, run_name=None):
    assert mode in {"none", "gram", "aga"}
    aga_config = aga_config or {}
    run_name = run_name or make_run_name(mode, aga_config)

    print("\n" + "=" * 72)
    print("START:", run_name)
    if mode == "aga":
        print("AGA config:", aga_config)
    print("=" * 72)

    model = load_calibrated_model(mode, aga_config=aga_config)
    loader = build_stream_loader(mode=mode, seed=SEED)
    optimizer = make_refinement_optimizer(model)

    seed_everything(SEED)
    memory_status(f"before {run_name}:")

    try:
        df = run_steps(
            model, loader, optimizer,
            steps=REFINE_STEPS,
            mode=mode,
            run_name=run_name,
            calibration=False,
            relation_weight=RELATION_WEIGHT,
            print_every=PRINT_EVERY,
            save_every=SAVE_EVERY,
        )

        run_dir = OUTPUT_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        final_path = run_dir / f"student_step_{REFINE_STEPS:05d}.pt"
        if not final_path.exists():
            torch.save(model.student_backbone.state_dict(), final_path)

        df.to_csv(run_dir / "training_log.csv", index=False)

        summary_cols = ["total", "dino", "ibot", "koleo", "relation"]
        summary = {
            "run_name": run_name,
            "mode": mode,
            **{col: float(df[col].mean()) for col in summary_cols},
        }
        if mode == "aga":
            summary.update({
                "b": float(aga_config["b"]),
                "alpha": float(aga_config["alpha"]),
                "H": int(aga_config["H"]),
            })

        for col in [
            "d_amp", "d_hat", "retention", "d_weak",
            "strong_ratio", "higham_iters",
        ]:
            if col in df.columns:
                summary[col] = float(df[col].mean())

        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("DONE:", run_name)
        print(summary)

    finally:
        try:
            del df
        except Exception:
            pass
        del optimizer, loader, model
        cleanup_cuda()
        memory_status(f"after {run_name} cleanup:")

    return summary

def ensure_calibration_checkpoint(reuse=True):
    """Reproduce the clean notebook's head-calibration stage, or reuse its checkpoint."""
    global CALIB_CKPT
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if reuse and CALIB_CKPT.exists():
        ckpt = torch.load(CALIB_CKPT, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "seed" in ckpt and ckpt["seed"] != SEED:
                raise ValueError("Calibration checkpoint seed does not match SEED")
            if "calib_steps" in ckpt and ckpt["calib_steps"] != CALIB_STEPS:
                raise ValueError("Calibration checkpoint steps do not match CALIB_STEPS")
        print("Reusing calibration checkpoint:", CALIB_CKPT)
        return CALIB_CKPT

    seed_everything(SEED)
    calib_loader = build_stream_loader(mode="none", seed=SEED)
    calib_model = DINOv2Refiner(relation_mode="none").to(device)
    calib_model.set_backbone_trainable(False)
    calib_optimizer = make_calibration_optimizer(calib_model)
    seed_everything(SEED)
    calib_logs = run_steps(
        calib_model, calib_loader, calib_optimizer,
        steps=CALIB_STEPS, mode="calibration", run_name="calibration",
        calibration=True, print_every=max(10, PRINT_EVERY), save_every=None,
    )
    torch.save({"model": calib_model.state_dict(), "seed": SEED, "calib_steps": CALIB_STEPS}, CALIB_CKPT)
    calib_logs.to_csv(OUTPUT_ROOT / "calibration_logs.csv", index=False)
    torch.save(calib_model.student_backbone.state_dict(), OUTPUT_ROOT / "student_pretrained_reference.pt")
    del calib_logs, calib_optimizer, calib_model, calib_loader
    cleanup_cuda()
    return CALIB_CKPT


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AGA controlled DINOv2-S refinement")
    parser.add_argument("--mode", choices=["none", "gram", "aga", "aga-h", "all"], default="all")
    parser.add_argument("--b", type=float, default=AGA_B)
    parser.add_argument("--alpha", type=float, default=AGA_ALPHA)
    parser.add_argument("--higham-iterations", type=int, default=AGA_HIGHAM_ITER)
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This notebook-derived training pipeline expects a CUDA GPU.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_calibration_checkpoint(reuse=not args.recalibrate)

    if args.mode == "all":
        experiments = [("none", None), ("gram", None),
                       ("aga", {"b": args.b, "alpha": args.alpha, "H": 0}),
                       ("aga", {"b": args.b, "alpha": args.alpha, "H": args.higham_iterations})]
    elif args.mode in ("none", "gram"):
        experiments = [(args.mode, None)]
    else:
        h = args.higham_iterations if args.mode == "aga-h" else 0
        experiments = [("aga", {"b": args.b, "alpha": args.alpha, "H": h})]

    summaries = [run_refinement(mode, aga_config=aga_config) for mode, aga_config in experiments]
    pd.DataFrame(summaries).to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    print("Saved:", OUTPUT_ROOT / "summary.csv")


if __name__ == "__main__":
    main()
