#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight timing-only profiler — drop-in replacement for run_ncu_memory /
run_nsys when hardware profiling tools (ncu / nsys) are unavailable.

Provides:
  1) Kernel execution timing via torch.cuda.Event
  2) Kernel launch count via torch.profiler (if available) or counting forward() calls
  3) A stub metrics_to_prompt() that explains NCU metrics are unavailable

Typical usage:
    from run_timing_profiler import profile_timing, timing_metrics_to_prompt

    timing = profile_timing(
        bench_py="bench_ref_inputs_0.py",
        kernel_file="test_kernel_0.py",
        device_idx=0,
        repeat=10,
    )
    prompt_block = timing_metrics_to_prompt(timing)
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

__all__ = [
    "profile_timing",
    "timing_metrics_to_prompt",
    "build_empty_metrics_df",
]


# ---------------------------------------------------------------------------
# Module loader (mirrors bench_ref_inputs_0.py logic)
# ---------------------------------------------------------------------------

def _load_module(path: Path):
    import hashlib
    path = path.resolve()
    mod_name = f"timing_mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Core timing profiler
# ---------------------------------------------------------------------------

def _count_kernel_launches(model, inputs, device) -> int:
    """Try to count CUDA kernel launches in a single forward() call
    using torch.profiler.  Returns 1 as fallback."""
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            with torch.no_grad():
                model(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

        events = prof.key_averages()
        count = sum(1 for ev in events if ev.device_type == torch.autograd.DeviceType.CUDA)
        return max(count, 1)
    except Exception:
        return 1


def profile_timing(
    bench_py: str = "bench_ref_inputs_0.py",
    kernel_file: Optional[Union[str, Path]] = None,
    device_idx: int = 0,
    repeat: int = 100,
    warmup: int = 25,
) -> Dict[str, Any]:
    """Run timing-only profiling.  No ncu / nsys required.

    Returns a dict with:
      - kernel_duration_ns  (float, average)
      - kernel_duration_ms  (dict: avg, min, max, std)
      - kernel_launch_count (int)
    """
    root = Path(__file__).resolve().parent
    ref_py = root / "ref_0.py"
    test_py = Path(kernel_file).resolve() if kernel_file else (root / "test_kernel_0.py")

    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device_idx)

    ref_mod = _load_module(ref_py)
    test_mod = _load_module(test_py)

    get_inputs = getattr(ref_mod, "get_inputs", None)
    get_init_inputs = getattr(ref_mod, "get_init_inputs", None)
    ModelNew = getattr(test_mod, "ModelNew", None)

    if get_inputs is None:
        raise RuntimeError("ref.py must define get_inputs()")
    if ModelNew is None:
        raise RuntimeError(f"{test_py} must define ModelNew")

    init_args = get_init_inputs() if get_init_inputs is not None else []
    if not isinstance(init_args, (list, tuple)):
        init_args = [init_args]

    model = ModelNew(*init_args)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    # --- count kernel launches (single forward) ---
    launch_count = _count_kernel_launches(model, inputs, device)

    # --- warmup ---
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # --- timed runs ---
    times_ms: List[float] = []
    with torch.no_grad():
        if torch.cuda.is_available():
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            for _ in range(repeat):
                start_ev.record()
                model(*inputs)
                end_ev.record()
                end_ev.synchronize()
                times_ms.append(start_ev.elapsed_time(end_ev))
        else:
            for _ in range(repeat):
                t0 = time.perf_counter()
                model(*inputs)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    std_ms = math.sqrt(sum((t - avg_ms) ** 2 for t in times_ms) / max(len(times_ms) - 1, 1))

    return {
        "kernel_duration_ns": avg_ms * 1e6,       # ns
        "kernel_duration_ms": {
            "avg": avg_ms,
            "min": min_ms,
            "max": max_ms,
            "std": std_ms,
        },
        "kernel_launch_count": launch_count,
    }


# ---------------------------------------------------------------------------
# Prompt builder (replaces metrics_to_prompt when NCU is unavailable)
# ---------------------------------------------------------------------------

def timing_metrics_to_prompt(timing: Dict[str, Any]) -> str:
    """Build a metrics prompt block for the LLM when only timing data is
    available (no hardware profiler)."""
    dur = timing.get("kernel_duration_ms", {})
    avg = dur.get("avg", 0)
    min_ = dur.get("min", 0)
    max_ = dur.get("max", 0)
    std = dur.get("std", 0)
    launch = timing.get("kernel_launch_count", 1)

    return (
        "# Kernel Profiling Metrics\n\n"
        "**NOTE**: Hardware profiling tools (Nsight Compute / nsys) are NOT available\n"
        "on this platform.  Only wall-clock kernel timing is provided below.\n"
        "Please analyse the kernel's bottleneck based on **code structure and\n"
        "algorithmic characteristics** rather than hardware counters.\n\n"
        "## Timing\n"
        f"- Average kernel duration: {avg:.4f} ms\n"
        f"- Min: {min_:.4f} ms / Max: {max_:.4f} ms / Std: {std:.4f} ms\n"
        f"- Kernel launch count per forward(): {launch}\n\n"
        "## Hardware Counters\n"
        "Not available — hardware profiler is missing on this platform.\n"
    )


# ---------------------------------------------------------------------------
# Empty metrics DataFrame (for machine_check compatibility)
# ---------------------------------------------------------------------------

def build_empty_metrics_df(
    kernel_duration_ns: float = 0.0,
    kernel_name: str = "unknown_kernel",
):
    """Return a single-row pandas DataFrame with all expected NCU metric
    columns set to NaN, except ``gpu__time_duration.avg`` which is filled
    from the timing profiler result.  This lets *run_machine_check* proceed
    in its fallback-only mode."""
    import pandas as pd

    # Canonical NCU metric columns (mirror config_metrics.ncu-cfg)
    _NCU_COLS = [
        "Kernel Name",
        "gpu__time_duration.avg",
        "sm__cycles_active.avg",
        "sm__cycles_elapsed.avg",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "launch__registers_per_thread",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
        "launch__occupancy_per_register_count",
        "launch__occupancy_per_shared_mem_size",
        "launch__occupancy_per_block_size",
        "launch__block_size",
        "launch__grid_size",
        "launch__shared_mem_per_block",
        "launch__waves_per_multiprocessor",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio",
        "smsp__warp_issue_stalled_no_instruction_per_warp_active.ratio",
        "smsp__warp_issue_stalled_not_selected_per_warp_active.ratio",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.max_rate",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.max_rate",
        "smsp__warp_issue_stalled_no_instruction_per_warp_active.max_rate",
        "smsp__warp_issue_stalled_not_selected_per_warp_active.max_rate",
        "smsp__sass_branch_targets_threads_divergent.avg",
        "smsp__sass_branch_targets_threads_uniform.avg",
        "smsp__thread_inst_executed_pred_on_per_inst_executed.max_rate",
        "smsp__warps_eligible.avg",
    ]

    row: Dict[str, Any] = {col: float("nan") for col in _NCU_COLS}
    row["Kernel Name"] = kernel_name
    row["gpu__time_duration.avg"] = kernel_duration_ns

    return pd.DataFrame([row])
