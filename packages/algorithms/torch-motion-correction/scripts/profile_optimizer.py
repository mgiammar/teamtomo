"""Profiling script for the gradient-based motion estimation optimizer.

Measures wall-clock time, per-iteration timing, and peak GPU memory for a
configurable synthetic workload. Edit the constants below to match your target
use case before running.

Usage
-----
    # CPU (always works):
    DEVICE=cpu uv run python scripts/profile_optimizer.py

    # GPU:
    uv run python scripts/profile_optimizer.py

    # With full torch.profiler kernel trace:
    # Set EMIT_TORCH_PROFILER = True below, then run on GPU.
"""

import time
import os

import torch

from torch_motion_correction.types import FourierFilterConfig, PatchSamplingConfig
from torch_motion_correction.utils import normalize_image, prepare_patch_filters

# ── Image parameters ──────────────────────────────────────────────────────────
IMAGE_HEIGHT: int = 1024
IMAGE_WIDTH: int = 1024
N_FRAMES: int = 40
PIXEL_SPACING: float = 1.0  # Angstroms

# ── Patch parameters ──────────────────────────────────────────────────────────
PATCH_SHAPE: tuple[int, int] = (512, 512)
PATCH_OVERLAP: float = 0.5

# ── Deformation field ─────────────────────────────────────────────────────────
DEFORMATION_FIELD_RESOLUTION: tuple[int, int, int] = (5, 3, 3)
GRID_TYPE: str = "catmull_rom"

# ── Optimizer ─────────────────────────────────────────────────────────────────
OPTIMIZER_TYPE: str = "adam"
LOSS_TYPE: str = "mse"
MAX_ITERATIONS: int = 20
BATCH_SIZE: int = 8
PRECOMPUTE_PATCHES: bool = True  # no-op until Step 3 of optimizations is implemented
USE_COMPILE: bool = False  # no-op until Step 4 of optimizations is implemented

# ── Profiling ─────────────────────────────────────────────────────────────────
DEVICE: str = "cuda"
N_WARMUP_RUNS: int = 1
EMIT_TORCH_PROFILER: bool = False  # enables torch.profiler kernel trace table


def make_synthetic_image() -> torch.Tensor:
    """Return a (N_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH) float32 Gaussian noise tensor."""
    return torch.randn(N_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_setup_phase(device: torch.device) -> float:
    """Time the pre-optimization setup: iterator build, filter prep, and (if implemented)
    patch FFT precomputation.

    Returns
    -------
    float
        Wall-clock seconds for the setup phase.
    """
    image = make_synthetic_image().to(device)

    _sync(device)
    t0 = time.perf_counter()

    image_norm = normalize_image(image)
    patch_sampling = PatchSamplingConfig(patch_shape=PATCH_SHAPE, overlap=PATCH_OVERLAP)
    _iterator = patch_sampling.get_patch_iterator(image_norm, device=device)

    fourier_filter = FourierFilterConfig()
    _circle_mask, _b_envelope, _bandpass = prepare_patch_filters(
        shape=PATCH_SHAPE,
        pixel_spacing=PIXEL_SPACING,
        fourier_filter=fourier_filter,
        mask_smoothing_fraction=1.0,
        device=device,
    )

    # Precompute FFT patches if the method exists (added in Step 3)
    if PRECOMPUTE_PATCHES and hasattr(_iterator, "precompute_fft_patches"):
        _precomputed = _iterator.precompute_fft_patches(
            circle_mask=_circle_mask,
            store_device=torch.device("cpu"),
        )

    _sync(device)
    return time.perf_counter() - t0


def run_full_optimization(device: torch.device) -> dict:
    """Run one complete optimization trial and return timing + memory metrics.

    Returns
    -------
    dict with keys:
        total_opt_time_s, n_iterations, mean_iter_time_s,
        peak_memory_gb, loss_trajectory
    """
    from torch_motion_correction.estimate_motion_optimizer import estimate_local_motion
    from torch_motion_correction.types import (
        FourierFilterConfig,
        OptimizationConfig,
        PatchSamplingConfig,
    )

    image = make_synthetic_image().to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Build OptimizationConfig, passing new fields only if OptimizationConfig supports them
    # (gracefully degrades before Step 2 lands)
    import dataclasses

    opt_fields = {f.name for f in dataclasses.fields(OptimizationConfig)}
    opt_kwargs: dict = dict(
        max_iterations=MAX_ITERATIONS,
        optimizer_type=OPTIMIZER_TYPE,
        loss_type=LOSS_TYPE,
        grid_type=GRID_TYPE,
    )
    if "batch_size" in opt_fields:
        opt_kwargs["batch_size"] = BATCH_SIZE
    if "precompute_patches" in opt_fields:
        opt_kwargs["precompute_patches"] = PRECOMPUTE_PATCHES
    if "use_compile" in opt_fields:
        opt_kwargs["use_compile"] = USE_COMPILE

    _sync(device)
    t0 = time.perf_counter()

    result, tracker = estimate_local_motion(
        image=image,
        pixel_spacing=PIXEL_SPACING,
        deformation_field_resolution=DEFORMATION_FIELD_RESOLUTION,
        patch_sampling=PatchSamplingConfig(
            patch_shape=PATCH_SHAPE,
            overlap=PATCH_OVERLAP,
        ),
        fourier_filter=FourierFilterConfig(),
        optimization=OptimizationConfig(**opt_kwargs),
        device=device,
        trajectory_kwargs={
            "sample_every_n_steps": 1,
            "total_steps": MAX_ITERATIONS,
        },
    )

    _sync(device)
    total_time_s = time.perf_counter() - t0

    peak_memory_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    loss_trajectory = [cp.loss for cp in tracker.checkpoints]
    n_iters = len(loss_trajectory)

    return {
        "total_opt_time_s": total_time_s,
        "n_iterations": n_iters,
        "mean_iter_time_s": total_time_s / max(n_iters, 1),
        "peak_memory_gb": peak_memory_bytes / 1e9,
        "loss_trajectory": loss_trajectory,
    }


def _print_table(setup_time_s: float, metrics: dict) -> None:
    sep = "─" * 54
    print(sep)
    print(f"{'Metric':<38} {'Value':>14}")
    print(sep)
    print(f"{'Setup / precompute time (s)':<38} {setup_time_s:>14.3f}")
    print(f"{'Total optimization time (s)':<38} {metrics['total_opt_time_s']:>14.3f}")
    print(f"{'Iterations completed':<38} {metrics['n_iterations']:>14d}")
    print(f"{'Mean time per iteration (s)':<38} {metrics['mean_iter_time_s']:>14.4f}")
    print(f"{'Peak GPU memory (GB)':<38} {metrics['peak_memory_gb']:>14.3f}")
    losses = metrics["loss_trajectory"]
    if losses:
        print(f"{'Loss (first iteration)':<38} {losses[0]:>14.6f}")
        print(f"{'Loss (last iteration)':<38} {losses[-1]:>14.6f}")
    print(sep)


if __name__ == "__main__":
    device_str = os.environ.get("DEVICE", DEVICE)
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        device = torch.device("cpu")

    n_patches_h = max(
        1,
        (IMAGE_HEIGHT - PATCH_SHAPE[0]) // int(PATCH_SHAPE[0] * (1 - PATCH_OVERLAP))
        + 1,
    )
    n_patches_w = max(
        1,
        (IMAGE_WIDTH - PATCH_SHAPE[1]) // int(PATCH_SHAPE[1] * (1 - PATCH_OVERLAP)) + 1,
    )
    approx_patches = n_patches_h * n_patches_w

    print()
    print("torch-motion-correction optimizer profiler")
    print(f"  Device          : {device}")
    print(f"  Image           : {N_FRAMES} x {IMAGE_HEIGHT} x {IMAGE_WIDTH}")
    print(
        f"  Patches         : {PATCH_SHAPE}, overlap={PATCH_OVERLAP} (~{approx_patches} patches)"
    )
    print(
        f"  Optimizer       : {OPTIMIZER_TYPE}, loss={LOSS_TYPE}, max_iter={MAX_ITERATIONS}"
    )
    print(f"  Precompute      : {PRECOMPUTE_PATCHES}  |  Compile: {USE_COMPILE}")
    print()

    for i in range(N_WARMUP_RUNS):
        print(f"Warmup {i + 1}/{N_WARMUP_RUNS}...")
        run_full_optimization(device)

    print("Measuring setup phase...")
    setup_time = time_setup_phase(device)

    print("Measuring optimization...")
    metrics = run_full_optimization(device)

    print()
    _print_table(setup_time, metrics)

    if EMIT_TORCH_PROFILER:
        if device.type != "cuda":
            print("\nEMIT_TORCH_PROFILER requires CUDA — skipped.")
        else:
            print("\nRunning torch.profiler trace (this adds overhead)...")
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=False,
            ) as prof:
                run_full_optimization(device)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
