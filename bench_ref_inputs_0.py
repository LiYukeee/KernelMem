"""
bench_ref_inputs_0.py – Benchmark script for ncu / nsys profiling.

Usage (called by profile_bench / nsys_profile_bench):
    python bench_ref_inputs_0.py [--device-idx N] [--test path/to/test_kernel_0.py] [--repeat K]

- Loads ref_0.py  (reference model, get_inputs, get_init_inputs)
- Loads test kernel (--test, default: test_kernel_0.py in same directory)
- Runs ModelNew(*init_inputs).forward(*inputs) for --repeat iterations
  so that ncu / nsys can capture GPU kernel activity.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def _load_module(path: Path):
    import hashlib
    path = path.resolve()
    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description="GPU benchmark for ncu/nsys profiling")
    p.add_argument("--device-idx", type=int, default=0, help="CUDA device index")
    p.add_argument("--test", type=str, default=None, help="Path to test kernel .py file")
    p.add_argument("--repeat", type=int, default=10, help="Number of benchmark iterations")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    ref_py = root / "ref_0.py"
    test_py = Path(args.test).resolve() if args.test else (root / "test_kernel_0.py")

    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_idx)

    ref_mod = _load_module(ref_py)
    test_mod = _load_module(test_py)

    get_inputs = getattr(ref_mod, "get_inputs", None)
    get_init_inputs = getattr(ref_mod, "get_init_inputs", None)
    ModelNew = getattr(test_mod, "ModelNew", None)

    if get_inputs is None:
        raise RuntimeError(f"ref_0.py must define get_inputs()")
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

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profiling runs (captured by ncu / nsys)
    with torch.no_grad():
        for _ in range(args.repeat):
            model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
