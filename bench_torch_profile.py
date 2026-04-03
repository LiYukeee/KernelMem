#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_torch_profile.py — torch.profiler-based kernel profiling script.

Run as a subprocess (for CUDA context isolation).
Loads ref.py + test_kernel.py, runs warmup+profiling, and writes a JSON file
with per-op/per-kernel timing, launch counts, and roofline estimates.

CLI usage:
    python bench_torch_profile.py \
        --ref ref_0.py --test test_kernel_0.py \
        --warmup 5 --repeat 3 \
        --out-json torch_profile_temp_0.json \
        --gpu-name A100-80GB \
        --device-idx 0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


# ---------------------------------------------------------------------------
# GPU spec table (peak BW and FP32 TFLOPS for roofline)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_HW_FILE = _ROOT / "prompts" / "hardware" / "gpu_specs.py"


def _load_gpu_specs() -> Dict[str, Any]:
    try:
        spec = importlib.util.spec_from_file_location("gpu_specs", _HW_FILE)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return getattr(module, "GPU_SPEC_INFO", {})
    except Exception:
        return {}


def _parse_float_from_spec(value: str) -> Optional[float]:
    """Extract the first float from a spec string like '1935 GB/s' or '19.5'."""
    if not value:
        return None
    m = re.search(r"[\d]+(?:\.[\d]+)?", str(value))
    if m:
        return float(m.group(0))
    return None


def _get_roofline_params(gpu_name: str) -> Dict[str, float]:
    """Return peak_bw_bytes_s and peak_flops_s. Falls back to A100-80GB if unknown."""
    specs = _load_gpu_specs()
    info = specs.get(gpu_name) or specs.get("A100-80GB") or {}

    bw_gbs = _parse_float_from_spec(info.get("Memory Bandwidth", "0")) or 0.0
    fp32_tflops = _parse_float_from_spec(info.get("FP32 TFLOPS", "0")) or 0.0

    # Fallback values (A100-80GB)
    if bw_gbs == 0.0:
        bw_gbs = 1935.0
    if fp32_tflops == 0.0:
        fp32_tflops = 19.5

    peak_bw = bw_gbs * 1e9       # bytes/s
    peak_flops = fp32_tflops * 1e12  # FLOP/s
    ridge_point = peak_flops / peak_bw  # FLOP/byte

    return {
        "peak_bw_bytes_s": peak_bw,
        "peak_flops_s": peak_flops,
        "ridge_point_flop_per_byte": ridge_point,
        "peak_bw_gbs": bw_gbs,
        "peak_fp32_tflops": fp32_tflops,
    }


# ---------------------------------------------------------------------------
# Dynamic import helpers
# ---------------------------------------------------------------------------

def _import_module(path: Path):
    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _build_inputs(mod, device: torch.device) -> List[Any]:
    """Call get_inputs() from a task module and move tensors to device."""
    inputs = mod.get_inputs()
    moved = []
    for x in inputs:
        if hasattr(x, "to"):
            moved.append(x.to(device))
        else:
            moved.append(x)
    return moved


def _build_model(mod, device: torch.device) -> nn.Module:
    """Build and return ModelNew (or Model if ModelNew not found), moved to device."""
    if hasattr(mod, "ModelNew"):
        cls = mod.ModelNew
    elif hasattr(mod, "Model"):
        cls = mod.Model
    else:
        raise AttributeError(f"Neither 'ModelNew' nor 'Model' found in module")

    init_inputs = []
    if hasattr(mod, "get_init_inputs"):
        raw = mod.get_init_inputs()
        for x in raw:
            if hasattr(x, "to"):
                init_inputs.append(x.to(device))
            else:
                init_inputs.append(x)

    model = cls(*init_inputs).to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Roofline estimation for a single event
# ---------------------------------------------------------------------------

def _estimate_bytes(event) -> int:
    """Roughly estimate data transfer bytes from input_shapes."""
    total_bytes = 0
    try:
        for shape_list in (event.input_shapes or []):
            if not isinstance(shape_list, (list, tuple)):
                continue
            for shape in shape_list:
                if not shape:
                    continue
                n_elem = 1
                for d in shape:
                    n_elem *= d
                total_bytes += n_elem * 4  # assume float32 = 4 bytes
        # output is roughly same size as input → multiply by 1.5 for read+write
        total_bytes = int(total_bytes * 1.5)
    except Exception:
        pass
    return max(total_bytes, 0)


def _build_kernel_entry(event, roofline: Dict[str, float]) -> Dict[str, Any]:
    """Build a single kernel record dict from a FunctionEvent."""
    # Timing (microseconds)
    self_cuda_us = getattr(event, "self_device_time_total", None)
    if self_cuda_us is None:
        self_cuda_us = getattr(event, "self_cuda_time_total", 0)
    cuda_total_us = getattr(event, "device_time_total", None)
    if cuda_total_us is None:
        cuda_total_us = getattr(event, "cuda_time_total", 0)

    num_calls = getattr(event, "count", 1) or 1
    flops = getattr(event, "flops", None) or 0

    # Kernel name from CUDA events list
    cuda_kernel_name = ""
    try:
        cuda_events = getattr(event, "kernels", []) or []
        if cuda_events:
            # Use the first CUDA kernel name
            cuda_kernel_name = str(getattr(cuda_events[0], "name", "")) or ""
    except Exception:
        pass

    # Input shapes
    input_shapes = []
    try:
        input_shapes = [list(s) if isinstance(s, (list, tuple)) else s
                        for s in (event.input_shapes or [])]
    except Exception:
        pass

    # Roofline estimates
    t_s = self_cuda_us * 1e-6
    estimated_bytes = _estimate_bytes(event)
    peak_bw = roofline["peak_bw_bytes_s"]
    peak_flops = roofline["peak_flops_s"]
    ridge_point = roofline["ridge_point_flop_per_byte"]

    ai = flops / max(estimated_bytes, 1)

    est_dram_pct = float("nan")
    est_compute_pct = float("nan")
    if t_s > 1e-9:
        achieved_bw = estimated_bytes / t_s
        est_dram_pct = min(achieved_bw / peak_bw * 100.0, 100.0)
        if flops > 0:
            achieved_flops = flops / t_s
            est_compute_pct = min(achieved_flops / peak_flops * 100.0, 100.0)

    roofline_bound = "compute" if (ai > ridge_point and flops > 0) else "memory"
    if not math.isfinite(ai) or flops == 0:
        roofline_bound = "unknown"

    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    return {
        "op_name": event.key,
        "cuda_kernel_name": cuda_kernel_name,
        "self_cuda_time_us": self_cuda_us,
        "cuda_time_total_us": cuda_total_us,
        "num_calls": num_calls,
        "self_cuda_pct": None,  # filled in later
        "input_shapes": input_shapes,
        "estimated_flops": int(flops),
        "estimated_bytes": estimated_bytes,
        "estimated_ai_flop_per_byte": _clean(round(ai, 3)) if flops > 0 else None,
        "estimated_dram_throughput_pct": _clean(round(est_dram_pct, 2)),
        "estimated_compute_throughput_pct": _clean(round(est_compute_pct, 2)),
        "roofline_bound": roofline_bound,
        "ridge_point_flop_per_byte": round(ridge_point, 3),
    }


# ---------------------------------------------------------------------------
# Main profiling logic
# ---------------------------------------------------------------------------

def run_profile(
    ref_py: Path,
    test_py: Path,
    device_idx: int,
    warmup: int,
    repeat: int,
    out_json: Path,
    gpu_name: str,
) -> None:
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Load modules
    ref_mod = _import_module(ref_py)
    test_mod = _import_module(test_py)

    # Build model and inputs
    model = _build_model(test_mod, device)
    inputs = _build_inputs(ref_mod, device)

    roofline = _get_roofline_params(gpu_name)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize(device)

    # Profile
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=True,
            profile_memory=False,
        ) as prof:
            for _ in range(repeat):
                with record_function("forward"):
                    model(*inputs)
            torch.cuda.synchronize(device)

    # Aggregate
    key_averages = prof.key_averages(group_by_input_shape=True)

    # Total self_cuda for pct calculation
    total_self_cuda = sum(
        (getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0))
        for e in key_averages
    )

    kernels = []
    for event in sorted(
        key_averages,
        key=lambda e: (getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)),
        reverse=True,
    ):
        self_cuda = getattr(event, "self_device_time_total", None) or getattr(event, "self_cuda_time_total", 0)
        if self_cuda <= 0:
            continue
        entry = _build_kernel_entry(event, roofline)
        if total_self_cuda > 0:
            entry["self_cuda_pct"] = round(self_cuda / total_self_cuda * 100.0, 2)
        kernels.append(entry)

    result = {
        "gpu_name": gpu_name,
        "total_cuda_time_us": total_self_cuda,
        "repeat": repeat,
        "peak_bw_gbs": roofline["peak_bw_gbs"],
        "peak_fp32_tflops": roofline["peak_fp32_tflops"],
        "ridge_point_flop_per_byte": round(roofline["ridge_point_flop_per_byte"], 3),
        "kernels": kernels,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[torch_profile] Written: {out_json}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("bench_torch_profile: torch.profiler-based kernel profiler")
    parser.add_argument("--ref", type=Path, required=True, help="PyTorch reference .py (has get_inputs)")
    parser.add_argument("--test", type=Path, required=True, help="Test kernel .py (has ModelNew)")
    parser.add_argument("--device-idx", type=int, default=0, help="CUDA device index")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=3, help="Profiling iterations")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--gpu-name", type=str, default="A100-80GB", help="GPU name for roofline lookup")
    args = parser.parse_args()

    run_profile(
        ref_py=args.ref,
        test_py=args.test,
        device_idx=args.device_idx,
        warmup=args.warmup,
        repeat=args.repeat,
        out_json=args.out_json,
        gpu_name=args.gpu_name,
    )


if __name__ == "__main__":
    main()
