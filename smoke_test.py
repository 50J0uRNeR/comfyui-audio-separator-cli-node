#!/usr/bin/env python3
"""Offline upgrade-regression smoke test for audio-separator-cli-node.

Purpose: catch breaking changes in the two audio-separator surfaces this node relies
on (see the "external surface" block at the top of __init__.py) — the CLI contract and
the library's config->stems path — right after an upgrade, before they bite mid-run.

It is fully offline: it inspects the CLI catalog/flags and derives stems from model
configs that already exist under models/. It never downloads weights or runs inference.

Usage:   <venv-python> smoke_test.py
Exit:    0 = all checks passed (skips allowed), 1 = at least one failure.
"""
import importlib.util
import json
import os
import subprocess
import sys

NODE_DIR = os.path.dirname(os.path.abspath(__file__))

# ComfyUI must be importable because __init__.py does `import server` at module load.
_COMFY_CANDIDATES = [
    os.environ.get("COMFYUI_ROOT", ""),
    "/home/ubuntu/comfyui-build/ComfyUI",
]


def _load_node():
    for cand in _COMFY_CANDIDATES:
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    spec = importlib.util.spec_from_file_location("asnode", os.path.join(NODE_DIR, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Stem baselines captured from audio-separator 0.47.0 against the local configs below.
# If an upgrade changes load_model_data_from_yaml() or common_separator's constants so
# that our derivation no longer matches these, check #2 fails and flags the regression.
_STEM_BASELINES = {
    "MDX23C-DrumSep-aufr33-jarredou.ckpt": ("config_drumsep_mdx23c.yaml", ["kick", "snare"]),
    "bs_roformer_karaoke_anvuew.ckpt":     ("config_bs_roformer_karaoke_anvuew.yaml", ["Vocals", "Instrumental"]),
    "htdemucs_ft":                          ("htdemucs_ft.yaml", ["Vocals", "Instrumental"]),
}

_REQUIRED_FLAGS = [
    "--list_models", "--model_filename", "--output_format",
    "--custom_output_names", "--model_file_dir", "--download_model_only",
]


class Report:
    def __init__(self):
        self.rows = []  # (name, status in {PASS,FAIL,SKIP}, detail)

    def add(self, name, status, detail=""):
        self.rows.append((name, status, str(detail)))

    @property
    def failed(self):
        return any(s == "FAIL" for _, s, _ in self.rows)


def check_catalog(mod, rep):
    """CLI --list_models parses and our import-time extraction found at least one yaml."""
    proc = subprocess.run(
        [mod.AUDIO_SEPARATOR_BIN, "--list_models",
         "--model_file_dir", mod.MODELS_DIR, "--list_format", "json"],
        capture_output=True, text=True, timeout=120)
    cat = json.loads(proc.stdout)
    n = with_files = 0
    for arch in cat.values():
        if not isinstance(arch, dict):
            continue
        for info in arch.values():
            if not isinstance(info, dict):
                continue
            n += 1
            if info.get("filename") and info.get("download_files"):
                with_files += 1
    assert n > 0, "catalog returned no models"
    has_yaml = any(any(str(f).lower().endswith((".yaml", ".yml")) for f in v)
                   for v in mod._MODEL_DOWNLOAD_FILES.values())
    assert has_yaml, "_MODEL_DOWNLOAD_FILES contains no .yaml entry (extraction broken?)"
    rep.add("CLI catalog structure + yaml extraction", "PASS", f"{n} models, {with_files} with download_files")


def check_stems(mod, sep, rep):
    """Our config->stems derivation still matches the 0.47.0 baselines for local models."""
    problems = []
    skipped = []
    for model, (yaml_name, expected) in _STEM_BASELINES.items():
        yaml_path = os.path.join(mod.MODELS_DIR, yaml_name)
        if not os.path.isfile(yaml_path):
            skipped.append(model)  # fixture not local; can't verify this arch here
            continue
        got = mod._config_stems(sep, yaml_path, None)
        if got != expected:
            problems.append(f"{model}: got {got}, want {expected}")
    if problems:
        rep.add("stem derivation (MDXC/RoFormer/Demucs)", "FAIL", "; ".join(problems))
    elif skipped:
        rep.add("stem derivation (MDXC/RoFormer/Demucs)", "SKIP", f"verified; fixtures missing locally: {skipped}")
    else:
        rep.add("stem derivation (MDXC/RoFormer/Demucs)", "PASS", "all baselines match")


def check_placeholder(mod, rep):
    """The fake-weight placeholder mechanism can create and clean up a file in MODELS_DIR."""
    p = os.path.join(mod.MODELS_DIR, "__smoke_fake__.ckpt")
    try:
        with open(p, "wb"):
            pass
        assert os.path.isfile(p), "placeholder was not created"
        os.remove(p)
        assert not os.path.exists(p), "placeholder was not cleaned up"
        rep.add("placeholder create/cleanup", "PASS")
    finally:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def check_flags(mod, rep):
    """The CLI flags this node depends on are still present in --help."""
    proc = subprocess.run([mod.AUDIO_SEPARATOR_BIN, "--help"], capture_output=True, text=True, timeout=60)
    help_text = (proc.stdout or "") + (proc.stderr or "")
    missing = [f for f in _REQUIRED_FLAGS if f not in help_text]
    assert not missing, f"missing CLI flags: {missing}"
    rep.add("CLI flag contract (--help)", "PASS", "all required flags present")


def main():
    try:
        mod = _load_node()
    except Exception as e:  # noqa: BLE001 - surface a clear, actionable message
        print(f"FAIL: could not import the node module: {e!r}")
        print("Hint: run with ComfyUI on sys.path (set COMFYUI_ROOT) so `import server` resolves.")
        return 1

    rep = Report()
    sep = mod._get_separator()
    for fn in (lambda: check_catalog(mod, rep),
               lambda: check_stems(mod, sep, rep),
               lambda: check_placeholder(mod, rep),
               lambda: check_flags(mod, rep)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a check error is itself a failure
            name = getattr(e, "msg", None) or repr(e)
            rep.add(fn.__name__.replace("check_", "").replace("_", " "), "FAIL", repr(e))

    print("\n=== audio-separator-cli-node smoke test ===")
    for name, status, detail in rep.rows:
        line = f"[{status}] {name}"
        if detail and status != "PASS":
            line += f" — {detail}"
        elif detail:
            line += f" ({detail})"
        print(line)
    print("RESULT:", "FAILURES PRESENT" if rep.failed else "OK")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
