# audio-separator-cli-node — agent notes

ComfyUI custom node wrapping the `audio-separator` CLI for stem separation, plus a
"List Model Stems" button that resolves a model's stems from its config **without**
downloading weights.

## Layout
- `__init__.py` — the whole backend: node class, API endpoints, stem resolution (single file).
- `web/js/audio-separation.js` — frontend widget (Show Command Help / List Loaded Models / List Model Stems buttons, Remove Models, STOP PROCESS).
- `models/` — audio-separator model files (`ckpt`/`yaml`/`th`). Resolved to `MODELS_DIR`; override with `AUDIO_SEPARATOR_MODEL_DIR`.
- `smoke_test.py` — offline upgrade-regression test (see below).
- `requirements.txt` / `requirements-gpu.txt` — CPU vs GPU dependency sets for audio-separator. ComfyUI-Manager reads `requirements.txt`; the GPU set is opt-in via `install.py --gpu`. Do not install both (two ONNX runtimes can break the CUDA provider).
- `install.py` — cross-platform installer; default → `requirements.txt`, `--gpu` → `requirements-gpu.txt`. Run it with the Python that runs ComfyUI so deps land next to its interpreter.
- `README.md` — user-facing docs; the node wraps and is laid out around audio-separator's CLI arguments.

## Two-surface design (why CLI + library are mixed)
This is deliberate, not accidental — keep it that way:
- **Inference, model listing, and downloads → CLI subprocess.** Gives process isolation (a
   hung/crashed model or download can't take down ComfyUI) and STOP PROCESS via `proc.kill()`.
  Downloads run as a tracked `--download_model_only` subprocess (`_download_model_cli()`); on
  failure/kill it purges that model's catalog files so a retry starts clean.
- **KILL = hard stop.** If the stem-collection download inside `separate()` is killed/fails,
  `_resolve_stems()` re-raises (does NOT swallow into `(None,"error")`) so `separate()` aborts
  *before* launching inference — no second download, no uncleanable partial files. Don't "helpfully"
  catch-and-fallback here; that was the original bug.
- **Config→stems → library**, isolated in the single choke point `_config_stems()`. There is no
  CLI flag for "give me this model's stems", so we parse its config in-process.

Guidance when extending: new inference/listing/download behavior goes through the CLI; any change
to how a model's config becomes stems goes through `_config_stems()` only (do not scatter library calls).

## Upgrading audio-separator — do this, in order
1. Bump `_AUDIO_SEPARATOR_TESTED` at the top of `__init__.py` to the new major.minor.
   (A mismatch logs a warning at startup; it is non-fatal by design.)
2. Run the offline test: `<venv-python> smoke_test.py`. It checks the CLI catalog, our stem
   baselines across MDXC/RoFormer/Demucs, placeholder cleanup, and required CLI flags — no network.
3. If a check fails, fix at the right seam:
   - library-side change (config→stems) → patch `_config_stems()`;
   - a flag was renamed/removed → update `smoke_test._REQUIRED_FLAGS` **and** every call site in `__init__.py`;
   - stems legitimately changed for a baseline model → update `smoke_test._STEM_BASELINES`.
4. Re-run until green before relying on the node.

## Verifying local changes
- `python -m py_compile __init__.py`
- `node --check web/js/audio-separation.js` (only if JS changed)
- `<venv-python> smoke_test.py`

## Naming & destructive ops
- **Output naming** default formula is `{input_stem}-{model_file_stem}-{(stem)}` (e.g.
  `song-htdemucs_ft-(Vocals).wav`), generated in `_identity_names()` and passed via
  `--custom_output_names`. The `else` fallback in `_stem_paths()` mirrors audio-separator's OWN
  built-in default for stems a user left unmapped — do NOT change it to our custom pattern.
- **Remove Models** (`POST /api/audio_separation/remove_models`, helper `_wipe_models_dir()`) is a
  destructive failsafe that deletes the ENTIRE `MODELS_DIR` contents (weights + configs +
  download_checks.json). It is user-confirmed in the UI and path-guarded; it does not touch running
  processes.

## Gotchas for agents
- The backend imports ComfyUI's `server`. To load it standalone (tests/probes), put the ComfyUI
  root on `sys.path` first — see how `smoke_test.py` does it.
- `audio-separator` lives in **ComfyUI's venv**. Its binary is found relative to that interpreter
  (`_resolve_bin()`); do not `Path.resolve()` the symlinked `bin/python`.
- The "List Model Stems" button path (`_resolve_stems_yaml_only`) must stay weight-free: if the
  model's yaml is already on disk it parses it directly (no fake files / subprocess); only a *missing*
  yaml triggers the fake-weight + download path, then deletes the fakes. Never let it pull full weights —
  that is what `_resolve_stems` (used by `separate()`) is for.
