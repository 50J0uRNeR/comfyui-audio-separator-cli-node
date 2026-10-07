#!/usr/bin/env python3
"""Uninstall this node's dependencies (audio-separator + friends) from ComfyUI's environment.

Mirrors install.py — pick the SAME profile you installed with:

    CPU (default):  requirements.txt        ->  uninstall.py
    GPU (CUDA ONNX): requirements-gpu.txt   ->  uninstall.py --gpu

Or uninstall manually: `pip uninstall -y audio-separator audioread onnxruntime` (CPU) or
`... onnxruntime-gpu` (GPU).

RUN THIS WITH THE SAME PYTHON THAT RUNS COMFYUI — that is what guarantees the packages are
removed from the environment the node actually uses (see install.py and _resolve_bin() in
__init__.py). If unsure which Python ComfyUI uses, check however you launch it.

Cross-platform: uses `sys.executable -m pip`, so it works on Windows and POSIX.

Only removes what this node's profile installed — direct deps from the requirements file plus
the ONNX runtime its extra pulled in. It does NOT touch your model files (models/) or any other
package, even if audio-separator was a dependency of something else you care about.
"""
import argparse
import importlib.metadata as im
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _req_file(gpu: bool) -> Path:
    return HERE / ("requirements-gpu.txt" if gpu else "requirements.txt")


def _packages(req_file: Path):
    """Direct deps from the requirements file, plus the ONNX runtime its extra installed.

    The [cpu]/[gpu] extra is what pulled in onnxruntime(-gpu); removing audio-separator alone
    would leave a stale runtime behind (the "two ONNX runtimes" problem install.py warns about).
    """
    pkgs, extras = [], set()
    for line in req_file.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?", line)
        if not m:
            continue
        pkgs.append(m.group(1))
        for e in (m.group(2) or "").strip("[]").split(","):
            if e.strip():
                extras.add(e.strip())
    for extra, runtime in (("cpu", "onnxruntime"), ("gpu", "onnxruntime-gpu")):
        if extra in extras and runtime not in pkgs:
            pkgs.append(runtime)
    seen = set()
    return [p for p in pkgs if not (p in seen or seen.add(p))]


def _installed(name):
    try:
        return im.version(name)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Uninstall audio-separator and its dependencies from the environment of this Python interpreter.")
    ap.add_argument("--gpu", action="store_true",
                    help="uninstall the GPU (CUDA ONNX) set from requirements-gpu.txt "
                         "(default is the CPU baseline in requirements.txt)")
    args = ap.parse_args()

    req_file = _req_file(args.gpu)
    if not req_file.is_file():
        print(f"ERROR: {req_file} not found next to uninstall.py.", file=sys.stderr)
        return 1

    pkgs = _packages(req_file)
    py = sys.executable
    label = "GPU (requirements-gpu.txt)" if args.gpu else "CPU baseline (requirements.txt)"
    print(f"Uninstalling [{label}] from the environment of:\n  {py}\n")
    rc = subprocess.call([py, "-m", "pip", "uninstall", "-y", *pkgs])
    if rc != 0:
        print("\nERROR: pip uninstall failed.")
        print("Ensure you ran this with the Python that runs ComfyUI and that it has pip.")
        return rc

    ver = _installed("audio-separator")
    if ver is None:
        print("\nOK: audio-separator is no longer installed.")
    else:
        print(f"\nWARN: audio-separator {ver} is still installed.")
        print("      If ComfyUI runs under a different environment, re-run uninstall.py with THAT Python.")

    runtime = "onnxruntime-gpu" if args.gpu else "onnxruntime"
    rver = _installed(runtime)
    if rver:
        print(f"NOTE: {runtime} {rver} is still present — that's fine if other packages need it;")
        print("      remove it with `pip uninstall -y " + runtime + "` only if nothing else uses it.")

    print("\nNote: your model files in models/ were NOT touched. Delete them manually (or use the")
    print("      node's Remove Models button) to reclaim that disk space too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
