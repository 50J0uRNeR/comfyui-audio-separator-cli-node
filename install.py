#!/usr/bin/env python3
"""Install this node's dependency, audio-separator, into ComfyUI's environment.

Two profiles — pick ONE that matches your hardware:

    CPU (default):  requirements.txt        ->  install.py
    GPU (CUDA ONNX): requirements-gpu.txt   ->  install.py --gpu

Or install manually with either file: `pip install -r requirements[-gpu].txt`.
Do NOT combine both — that leaves two ONNX runtimes and can break the CUDA provider.

RUN THIS WITH THE SAME PYTHON THAT RUNS COMFYUI — that is what guarantees the
dependency lands where the node looks for it (next to ComfyUI's interpreter):

    Linux / macOS:   <comfyui-venv>/bin/python install.py [--gpu]
    Windows:         C:\\ComfyUI\\.venv\\Scripts\\python.exe install.py [--gpu]

This script installs into the environment of whatever interpreter runs IT, and
deliberately does NOT auto-detect or touch any other environment — choosing the
right Python here is what makes it "the right environment" (see _resolve_bin() in
__init__.py). If unsure which Python ComfyUI uses, check however you launch it.

Cross-platform: uses `sys.executable -m pip`, so it works on Windows and POSIX.
"""
import argparse
import importlib.metadata as im
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _req_file(gpu: bool) -> Path:
    return HERE / ("requirements-gpu.txt" if gpu else "requirements.txt")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Install audio-separator into the environment of this Python interpreter.")
    ap.add_argument("--gpu", action="store_true",
                    help="install the GPU (CUDA ONNX) set from requirements-gpu.txt "
                         "(default is the CPU baseline in requirements.txt)")
    args = ap.parse_args()

    req_file = _req_file(args.gpu)
    if not req_file.is_file():
        print(f"ERROR: {req_file} not found next to install.py.", file=sys.stderr)
        return 1

    py = sys.executable
    label = "GPU (requirements-gpu.txt)" if args.gpu else "CPU baseline (requirements.txt)"
    print(f"Installing [{label}] into the environment of:\n  {py}\n")
    rc = subprocess.call([py, "-m", "pip", "install", "-r", str(req_file)])
    if rc != 0:
        print("\nERROR: pip install failed.")
        print("Ensure you ran this with the Python that runs ComfyUI and that it has pip.")
        return rc

    # Verify the CLI landed where _resolve_bin() looks for it (next to this interpreter).
    base = Path(py).parent
    found = next(
        (c for c in (base / "audio-separator", base / "audio-separator.exe") if c.exists()), None)

    try:
        ver = im.version("audio-separator")
    except Exception:
        ver = "(unknown)"
    print(f"\naudio-separator version: {ver}")
    if ver != "(unknown)" and not ver.startswith("0.47"):
        print("NOTE: not the 0.47.x baseline we tested — run smoke_test.py before relying on stems")
        print("      (the node also warns at startup).")

    # beartype pin caveat (audio-separator pins <0.19.0; other packages may want newer)
    print("NOTE: audio-separator pins beartype<0.19.0; if other packages need a newer beartype,")
    print("      run `pip install --upgrade beartype` after (pip warns about the conflict, but it works).")

    if args.gpu and found:
        print("NOTE (GPU): for a clean GPU setup audio-separator recommends removing the CPU runtime too:")
        print("      pip uninstall -y onnxruntime   (only if nothing else in this env needs it).")

    if found:
        print(f"OK: CLI found at {found} — the node will pick it up automatically.")
    else:
        print("WARN: installed, but the audio-separator CLI is not next to this interpreter.")
        print("      If ComfyUI runs under a different environment, re-run install.py with THAT Python,")
        print("      or set AUDIO_SEPARATOR_BIN to the full path of the audio-separator command.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
