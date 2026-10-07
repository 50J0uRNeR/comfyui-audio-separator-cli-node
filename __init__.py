import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import server
from aiohttp import web


def _resolve_bin():
    # Resolution order (first hit wins):
    # 1. AUDIO_SEPARATOR_BIN env var — explicit override, used as-is.
    # 2. Console script next to ComfyUI's own interpreter: audio-separator is
    #    installed in the same virtualenv that runs ComfyUI, so this finds it
    #    on any machine without hardcoding an absolute path. NOTE: do NOT use
    #    Path.resolve() here — venvs commonly symlink bin/python to the system
    #    interpreter (e.g. .venv/bin/python -> /usr/bin/python), and resolving
    #    would land in /usr/bin where no audio-separator exists.
    # 3. PATH lookup (e.g. venv activated in the launching shell).
    env = os.environ.get("AUDIO_SEPARATOR_BIN")
    if env:
        return env
    cand = Path(sys.executable).parent / "audio-separator"
    if cand.is_file():
        return str(cand)
    found = shutil.which("audio-separator")
    if found:
        return found
    raise RuntimeError(
        "audio-separator executable not found. Install it into the same "
        f"virtualenv as ComfyUI ({sys.executable}) or set AUDIO_SEPARATOR_BIN."
    )


AUDIO_SEPARATOR_BIN = _resolve_bin()
MODELS_DIR = os.environ.get(
    "AUDIO_SEPARATOR_MODEL_DIR", str(Path(__file__).resolve().parent / "models")
)
MODEL_FILE_EXTS = {".yaml", ".yml", ".ckpt", ".pth", ".onnx"}

# Popen objects for audio-separator runs currently active in this server.
# The KILL PID endpoint calls proc.kill() on them (SIGKILL on POSIX,
# TerminateProcess on Windows); entries are removed when each run ends.
_ACTIVE_PROCS = []
_PROC_LOCK = threading.Lock()

# Number of output_stempath_N connectors exposed by the node. Bump this if a
# future model exposes more stems than this (today's max is 6).
MAX_STEM_OUTPUTS = 8

# ---------------------------------------------------------------------------
# audio-separator external surface we depend on
# ---------------------------------------------------------------------------
# This node talks to audio-separator through two surfaces, chosen for stability:
#   * CLI (subprocess) — inference (`separate`), model listing
#     (`--list_models --list_format json`), and model downloads
#     (`--download_model_only`). Gives us process isolation (a hung or crashed
#     model/download can't take down ComfyUI) and STOP PROCESS via proc.kill().
#   * Library (in-process, stem resolution only) — Separator(model_file_dir=...,
#     info_only=True), .load_model_data_from_yaml(path) / .load_model_data_using_hash(path),
#     and the common_separator constants NO_STEM + STEM_PAIR_MAPPER. Used ONLY to turn a
#     model's config into stems (no CLI flag exists for that job).
# If an upgrade changes any of these, _config_stems() is the single place to patch on the
# library side; the CLI flags above are asserted by smoke_test.py.

_AUDIO_SEPARATOR_TESTED = "0.47"  # major.minor verified against (see AGENTS.md)


def _check_audio_separator_version():
    # Early-warning on upgrade: warn (not fail) if the installed audio-separator has a
    # different major.minor than _AUDIO_SEPARATOR_TESTED, so a breaking change is
    # surfaced at startup instead of discovered mid-run. Non-fatal by design — newer
    # versions may still work; the warning just prompts running smoke_test.py.
    try:
        import importlib.metadata as _im
        installed = _im.version("audio-separator")
    except Exception:
        return  # no pip metadata to compare against (e.g. editable install)
    if not installed.startswith(_AUDIO_SEPARATOR_TESTED):
        logging.warning(
            f"audio-separator-cli-node: audio-separator {installed} differs from the "
            f"tested {_AUDIO_SEPARATOR_TESTED}.x — run smoke_test.py before relying on stems."
        )


_check_audio_separator_version()


def _sanitize_filename(name):
    # Mirrors audio-separator's sanitize_filename (common_separator.py) so
    # predicted output paths match what the CLI actually writes to disk.
    s = re.sub(r'[<>:"/\\|?*]', '_', name)
    s = re.sub(r'_+', '_', s)
    return s.strip('_. ')


def _dedup_model_files(filenames):
    # When several files share a base name (name minus extension) and one of them
    # is a .ckpt, list only that .ckpt — audio-separator identifies such models by
    # their checkpoint (see separator.py list_supported_model_files), so the
    # sibling .yaml/.yml config file is redundant in the picker. Groups with no
    # .ckpt keep all their files (e.g. Demucs models are identified by .yaml).
    groups = {}
    for fn in filenames:
        groups.setdefault(os.path.splitext(fn)[0], []).append(fn)
    kept = []
    for fns in groups.values():
        if any(f.lower().endswith(".ckpt") for f in fns):
            fns = [f for f in fns if not f.lower().endswith((".yaml", ".yml"))]
        kept.extend(fns)
    return sorted(kept)


def _list_models():
    # Source of truth is `audio-separator --list_models` (JSON catalog). It carries
    # each model's "stems" array (drives the output_stempath_N connectors) and its
    # "download_files" list (the exact files audio-separator fetches for that model,
    # incl. any .yaml config). Local model files are always merged in so a stale/failed
    # CLI never hides models on disk; those get default 2-stem names. Both maps become
    # module globals — MODEL_STEMS (fn -> stems|None) and _MODEL_DOWNLOAD_FILES
    # (fn -> [files]) — so stem resolution reuses this catalog instead of re-querying
    # the library's list_supported_model_files().
    global _MODEL_DOWNLOAD_FILES
    stems_by_file = {}
    download_files_by_file = {}
    try:
        proc = subprocess.run(
            [AUDIO_SEPARATOR_BIN, "--list_models",
             "--model_file_dir", MODELS_DIR, "--list_format", "json"],
            capture_output=True, text=True, timeout=120,
        )
        catalog = json.loads(proc.stdout)
        for arch in catalog.values():
            for info in arch.values():
                if not isinstance(info, dict):
                    continue
                fn = info.get("filename")
                if not fn:
                    continue
                stems = [str(s) for s in (info.get("stems") or [])]
                # None => the catalog carried no stem data for this model; it is
                # resolved from the model's own config at use time (_resolve_stems).
                # Kept distinct from a real ["Vocals","Instrumental"] so callers can
                # tell curated stems apart from "needs resolution".
                stems_by_file[fn] = stems if stems else None
                download_files_by_file[fn] = [str(f) for f in (info.get("download_files") or [])]
    except Exception as e:
        logging.warning(f"audio-separator-cli-node: --list_models failed ({e}); falling back to local scan")
    try:
        for p in Path(MODELS_DIR).iterdir():
            if p.is_file() and p.suffix.lower() in MODEL_FILE_EXTS and p.name not in stems_by_file:
                stems_by_file[p.name] = None  # local-only file; resolve from config at use time
    except Exception as e:
        logging.warning(f"audio-separator-cli-node: local model scan failed ({e})")
    # Collapse same-name .ckpt/.yaml pairs down to the .ckpt (see _dedup_model_files).
    kept = set(_dedup_model_files(stems_by_file))
    _MODEL_DOWNLOAD_FILES = download_files_by_file
    return dict(sorted((fn, stems) for fn, stems in stems_by_file.items() if fn in kept))


# Populated by _list_models(); maps a model filename -> its [download file names].
_MODEL_DOWNLOAD_FILES = {}
MODEL_STEMS = _list_models()
MODEL_FILES = list(MODEL_STEMS)
logging.info(
    f"audio-separator-cli-node: {len(MODEL_FILES)} models available from "
    f"{MODELS_DIR} (bin={AUDIO_SEPARATOR_BIN})"
)


_SEP_INSTANCE = None
_SEP_LOCK = threading.Lock()
# Serializes model-file downloads across threads (the List Model Stems endpoint runs
# _resolve_stems in a worker thread via asyncio.to_thread, while separate() may resolve
# on the main thread). Prevents two resolutions from double-downloading/corrupting the
# same file. Instance creation itself is guarded separately by _SEP_LOCK.
_DOWNLOAD_LOCK = threading.Lock()


def _get_separator():
    # Lazily build one shared audio-separator Separator. Its __init__ is light
    # (logging + dir setup, no inference weights); importing here keeps the heavy
    # dependency out of module load time and reuses a single instance across calls.
    global _SEP_INSTANCE
    with _SEP_LOCK:
        if _SEP_INSTANCE is None:
            from audio_separator.separator import Separator
            _SEP_INSTANCE = Separator(model_file_dir=MODELS_DIR, info_only=True)
        return _SEP_INSTANCE


def _stems_from_model_data(model_data):
    # Mirror audio-separator's primary/secondary stem derivation exactly so our
    # predicted names match what the CLI writes (common_separator.py: instruments +
    # target_instrument swap, then secondary_stem). Uses the library's own constants
    # to stay in sync across versions.
    from audio_separator.separator.common_separator import CommonSeparator
    NO_STEM = CommonSeparator.NO_STEM
    PAIR = CommonSeparator.STEM_PAIR_MAPPER

    def _secondary(primary):
        p = primary if primary else NO_STEM
        if p in PAIR:
            return PAIR[p]
        return p.replace(NO_STEM, "") if NO_STEM in p else f"{NO_STEM}{p}"

    model_data = model_data or {}
    training = model_data.get("training") or {}
    instruments = training.get("instruments")
    primary = secondary = None
    if instruments:
        target = training.get("target_instrument")
        if (target and len(instruments) >= 2 and instruments[0] != target
                and instruments[1] == target):
            primary, secondary = instruments[1], instruments[0]
        else:
            primary = instruments[0]
            secondary = instruments[1] if len(instruments) > 1 else _secondary(primary)
    if primary is None:
        primary = model_data.get("primary_stem", "Vocals")
        secondary = _secondary(primary)
    return [str(primary), str(secondary)]


def _config_stems(sep, path, yaml_cfg):
    # Single choke point for "model config -> stems". audio-separator exposes this via
    # two internal methods whose signatures we do not control (load_model_data_from_yaml
    # / load_model_data_using_hash); keeping them here means an upgrade that changes
    # either touches exactly one function. `path` is the resolved model file; if it is
    # itself a .yaml, treat it as the config directly (mirrors download_model_and_data).
    if str(path).lower().endswith(".yaml"):
        yaml_cfg = path
    model_data = (sep.load_model_data_from_yaml(yaml_cfg) if yaml_cfg
                  else sep.load_model_data_using_hash(path))
    return _stems_from_model_data(model_data)


def _download_model_cli(model_filename):
    # Download one model through the CLI as a killable subprocess (process isolation +
    # STOP PROCESS), instead of an in-process library call that could hang ComfyUI. On
    # failure/kill we purge ALL of this model's catalog files so a retry starts clean:
    # audio-separator's download_file_if_not_exists skips any file that already exists,
    # so a partial left behind would otherwise block re-download forever. The file list
    # comes from the --list_models catalog (_MODEL_DOWNLOAD_FILES).
    cmd = [AUDIO_SEPARATOR_BIN, "--download_model_only", "-m", model_filename,
           "--model_file_dir", MODELS_DIR]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with _PROC_LOCK:
        _ACTIVE_PROCS.append(proc)
    rc = -1
    try:
        assert proc.stdout is not None
        for line in proc.stdout:  # stream progress to the server console in real time
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
    finally:
        with _PROC_LOCK:
            try:
                _ACTIVE_PROCS.remove(proc)
            except ValueError:
                pass
        if proc.poll() is None:  # exception path: still running -> don't orphan it
            proc.kill()
    if rc != 0:
        for f in (_MODEL_DOWNLOAD_FILES.get(model_filename) or []):
            p = os.path.join(MODELS_DIR, str(f).split("/")[-1])
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        if rc < 0:
            raise ValueError(
                f"audio-separator download was terminated (killed); purged {model_filename} "
                "files for a clean retry")
        raise ValueError(f"audio-separator --download_model_only failed (exit {rc}) for {model_filename}")


def _resolve_stems(model_filename):
    # Shared by separate() and output-connector naming. Returns (stems, status). Prefers
    # the --list_models catalog stems for this model; when those are absent (None) it
    # downloads that single model via `--download_model_only` (killable subprocess) and
    # resolves stems from its config. separate() needs the real weights on disk anyway.
    cat = MODEL_STEMS.get(model_filename)
    if cat:
        return list(cat), "catalog"
    try:
        with _DOWNLOAD_LOCK:
            sep = _get_separator()
            _download_model_cli(model_filename)  # full download (killable); purges on failure
            files = [str(f) for f in (_MODEL_DOWNLOAD_FILES.get(model_filename) or [])]
            yaml_name = next((f.split("/")[-1] for f in files
                              if f.lower().endswith((".yaml", ".yml"))), None)
            path = os.path.join(MODELS_DIR, model_filename)
            yaml_cfg = os.path.join(MODELS_DIR, yaml_name) if yaml_name else None
            return _config_stems(sep, path, yaml_cfg), "config"
    except Exception as e:
        # Propagate (don't swallow): separate() must abort rather than launch inference
        # with unknown stems. A killed/failed download leaves no usable model, and the
        # helper already purged partial files for a clean retry.
        logging.warning(f"audio-separator-cli-node: stem resolution failed for {model_filename}: {e}")
        raise


def _resolve_stems_yaml_only(model_filename):
    # Fast stem resolution for the List Model Stems button: fetch ONLY the model's yaml
    # config (a few KB), never its weights. Returns (stems, status) where status is one of
    # "catalog" | "config" | "must_download" | "error: ...". separate() keeps using
    # _resolve_stems (full download) because inference needs the real weight files on disk.
    cat = MODEL_STEMS.get(model_filename)
    if cat:
        return list(cat), "catalog"
    try:
        with _DOWNLOAD_LOCK:
            sep = _get_separator()
            # Reuse the --list_models catalog fetched at import (_MODEL_DOWNLOAD_FILES)
            # instead of re-querying the library. download_files lists exactly what
            # audio-separator would fetch for this model, including any .yaml config.
            files = [str(f) for f in (_MODEL_DOWNLOAD_FILES.get(model_filename) or [])]
            yaml_name = next((f.split("/")[-1] for f in files
                              if f.lower().endswith((".yaml", ".yml"))), None)
            if not yaml_name:
                # Pure hash-based model (VR/MDX): stems require the real weight file.
                return None, "must_download"
            # Fast path: if the config is already on disk, parse it directly — no fake
            # files, no subprocess. Only download when the yaml is genuinely missing.
            local_yaml = os.path.join(MODELS_DIR, yaml_name)
            if os.path.isfile(local_yaml):
                return _config_stems(sep, local_yaml, None), "config"
            created = []
            try:
                for f in files:
                    name = f.split("/")[-1]
                    if name.lower().endswith((".yaml", ".yml")):
                        continue  # keep the config we actually want to download
                    p = os.path.join(MODELS_DIR, name)
                    if not os.path.isfile(p):
                        with open(p, "wb"):
                            pass  # blank placeholder so audio-separator's download skips it
                        created.append(p)
                _download_model_cli(model_filename)  # CLI fetches only the yaml (fakes skipped)
                return _config_stems(sep, os.path.join(MODELS_DIR, yaml_name), None), "config"
            finally:
                for p in created:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    except Exception as e:
        logging.warning(
            f"audio-separator-cli-node: yaml-only stem resolution failed for {model_filename}: {e}")
        return None, f"error: {e}"


def _default_stems(model_filename):
    # Lightweight stem source (no download) — used at prompt-validation time where we
    # must not trigger a model download. Real stems come from _resolve_stems at run time.
    return MODEL_STEMS.get(model_filename) or ["Vocals", "Instrumental"]


DEFAULT_INPUT_FILE = "/tmp/example.wav"
DEFAULT_OUTPUT_FORMAT = "wav"


def _identity_names(input_files, model_filename, stems=None):
    # Generated --custom_output_names value: maps each of the model's stems to
    # the node's chosen formula {input_stem}-{model_file_stem}-{(stem)}.
    # Pinning names explicitly keeps output_stempath_N resolution exact. `stems` is
    # the resolved stem list (see _resolve_stems); when omitted it falls back to a
    # lightweight default so prompt-validation never triggers a model download.
    stems = (list(stems) if stems else _default_stems(model_filename))[:MAX_STEM_OUTPUTS]
    input_stem = _sanitize_filename(os.path.splitext(os.path.basename(input_files))[0])
    model_stem = _sanitize_filename(os.path.splitext(model_filename)[0])
    return {s: f"{input_stem}-{model_stem}-({s})" for s in stems}


def _find_flag_value(tokens, flag):
    # Locate a user-supplied CLI flag among shlex tokens. Supports both the
    # "--flag value" and "--flag=value" forms (argparse accepts both). Returns
    # None when absent; raises when present without a usable value. Last
    # occurrence wins, matching argparse's own duplicate-flag semantics.
    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i]
        if tok == flag:
            if i + 1 < len(tokens):
                return tokens[i + 1]
            raise ValueError(f"command_options: {flag} is missing its value")
        if tok.startswith(flag + "="):
            val = tok[len(flag) + 1:]
            if not val:
                raise ValueError(f"command_options: {flag}= requires a value")
            return val
    return None


def _parse_custom_output_names(value):
    # Must be a JSON object {stem_name: file_name}; passed to the CLI verbatim
    # and drives output_stempath_N resolution.
    try:
        parsed = json.loads(value)
    except ValueError as e:  # json.JSONDecodeError subclasses ValueError
        raise ValueError(
            'custom_output_names must be valid JSON (a stem -> file name object), '
            f'e.g. {{"Vocals": "my_vocals"}}: {e}'
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            'custom_output_names must be a JSON object mapping stem name -> file name, '
            'e.g. {"Vocals": "my_vocals"}'
        )
    return parsed


def _parse_output_format(value):
    # Becomes the literal file extension, so it must be a simple token.
    fmt = (value or "").strip().lower()
    if not re.fullmatch(r"[A-Za-z0-9]+", fmt):
        raise ValueError("output_format must be a simple format token like wav or flac")
    return fmt


def _resolve_names_and_format(input_files, model_filename, options, stems=None):
    # User-supplied --custom_output_names / --output_format in command_options
    # win; otherwise the node generates them (identity formula + default ext).
    user_names = _find_flag_value(options, "--custom_output_names")
    user_fmt = _find_flag_value(options, "--output_format")
    names = (_parse_custom_output_names(user_names) if user_names is not None
             else _identity_names(input_files, model_filename, stems))
    fmt = (_parse_output_format(user_fmt) if user_fmt is not None
           else DEFAULT_OUTPUT_FORMAT)
    return names, fmt, user_names is not None, user_fmt is not None


def _stem_paths(input_files, model_filename, output_dir, names, fmt, stems=None):
    # Resolve output_stempath_N by parsing custom_output_names — no guessing:
    # slot N is the model's Nth stem; if it has a mapping entry the file name is
    # exactly that value (audio-separator get_stem_output_path), otherwise
    # audio-separator falls back to its standard formula, mirrored here. `stems` is
    # the resolved stem list (see _resolve_stems); omitted => lightweight default.
    names = {str(k).lower(): str(v) for k, v in names.items()}
    stems = (list(stems) if stems else _default_stems(model_filename))[:MAX_STEM_OUTPUTS]
    input_stem = _sanitize_filename(os.path.splitext(os.path.basename(input_files))[0])
    model_stem = _sanitize_filename(os.path.splitext(model_filename)[0])
    paths = []
    for stem in stems:
        if stem.lower() in names:
            fn = f"{_sanitize_filename(names[stem.lower()])}.{fmt}"
        else:
            # audio-separator's OWN built-in default for a stem the user left unmapped —
            # keep this mirroring the library (do NOT switch it to our custom pattern).
            fn = f"{input_stem}_({_sanitize_filename(stem)})_{model_stem}.{fmt}"
        paths.append(os.path.join(output_dir, fn))
    # Pad to a constant width so every declared output slot always resolves.
    return paths + [""] * (MAX_STEM_OUTPUTS - len(paths))


def _wipe_models_dir(root):
    # Destructive failsafe (user-confirmed in the UI dialog): remove every file and
    # subdirectory under `root` so a re-download starts fully clean. Returns
    # (removed, failed) name lists; per-file errors are collected rather than raised
    # so one locked file doesn't abort the rest (open files may resist on Windows).
    # Kept as a pure function so it can be tested without aiohttp.
    removed, failed = [], []
    for p in sorted(Path(root).iterdir()):
        try:
            if p.is_file():
                p.unlink()
                removed.append(p.name)
            elif p.is_dir():
                shutil.rmtree(p)
                removed.append(p.name + "/")
        except OSError as e:
            failed.append(f"{p.name} ({e})")
    return removed, failed


class AudioSeparation:
    """Runs the audio-separator CLI on a single input file.

    Paths are typed verbatim (e.g. /tmp/example.wav) so files can be shared
    with other tools/agents through the container's /tmp; no uploads or
    server-side file listing is involved. The CLI's console output is
    streamed live to ComfyUI's server console AND returned on the STDOUT
    connector.

    Output naming: the node silently passes --custom_output_names (each stem
    mapped to {input_stem}-{model_file_stem}-{(stem)}) and --output_format wav,
    unless command_options already contains those flags — in which case the
    user's values win and drive output_stempath_N resolution instead.

    output_stempath_1..N: slot N is the model's Nth catalog stem resolved to
    its file name (mapping entry if present, else the standard formula);
    unused slots return "".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_files": ("STRING", {"default": DEFAULT_INPUT_FILE}),
            "output_dir": ("STRING", {"default": "/tmp/"}),
            "model_filename": (MODEL_FILES,),
            # Extra CLI arguments verbatim, e.g. "--enable_float32"
            # (see `audio-separator --help`). Blank means no extra options.
            # May include --custom_output_names '{"Vocals": "my_v"}' and/or
            # --output_format flac to override the auto-generated naming.
            "command_options": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING",) * (1 + MAX_STEM_OUTPUTS)
    RETURN_NAMES = (("STDOUT",)
                    + tuple(f"output_stempath_{i}" for i in range(1, MAX_STEM_OUTPUTS + 1)))
    # Terminal node (like SaveImage): a workflow may end here, so outputs can
    # be left unconnected. The log is already streamed to the server console;
    # connecting Text Preview remains optional.
    OUTPUT_NODE = True
    FUNCTION = "separate"
    CATEGORY = "audio"
    DESCRIPTION = (
        "Separates an audio file into stems using the audio-separator CLI. "
        "Type full paths for input/output. Output names are generated "
        "automatically ({input_stem}-{model_file_stem}-{(stem)}, wav) unless command_options "
        "supplies --custom_output_names / --output_format itself. STDOUT "
        f"carries the run's console output; output_stempath_1..{MAX_STEM_OUTPUTS} "
        "carry each stem's full path (unused slots return an empty string)."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, input_files=None, model_filename=None, output_dir=None,
                       command_options=None):
        # A connected (linked) input arrives as None at prompt-validation time —
        # ComfyUI can't resolve an upstream node's output before that node runs, so
        # such fields must be allowed through here. Only a concrete blank STRING
        # widget is ""; the non-empty-path rule is enforced in separate() instead.
        if isinstance(input_files, str) and not input_files.strip():
            return "input_files must be a non-empty path (e.g. /tmp/example.wav)"
        if isinstance(output_dir, str) and not output_dir.strip():
            return "output_dir must be a non-empty directory (e.g. /tmp)"

        opts = command_options if isinstance(command_options, str) else ""
        try:
            options = shlex.split(opts or "")
        except ValueError as e:  # shlex.Error subclasses ValueError
            return f"command_options has unparseable quoting: {e}"

        # Naming/format resolution needs a real path + model; skip while either is
        # still an unresolved link (None) — it's validated at run time.
        if isinstance(input_files, str) and isinstance(model_filename, str):
            try:
                _resolve_names_and_format(input_files, model_filename, options)
            except ValueError as e:
                return str(e)
        return True

    def separate(self, input_files, model_filename, output_dir, command_options=""):
        if (not input_files or not str(input_files).strip()
                or not output_dir or not str(output_dir).strip()):
            raise ValueError("input_files and output_dir must both be non-empty paths")
        # Popen takes a list of argv entries (no shell word-splitting), so the
        # free-text options must be tokenized first. shlex.split keeps quoted
        # values as single arguments and returns [] for blank input, which is
        # exactly the "no extra options" case.
        try:
            options = shlex.split(command_options or "")
        except ValueError as e:  # shlex.Error subclasses ValueError
            raise ValueError(f"command_options has unparseable quoting: {e}") from e
        # Resolve the selected model's real stems (downloads just this model if not
        # already present) so generated names + output_stempath_N match what
        # audio-separator actually writes to disk.
        resolved_stems, stem_status = _resolve_stems(model_filename)
        logging.info("audio-separator-cli-node: stems for %s -> %s (%s)",
                     model_filename, resolved_stems, stem_status)
        names, fmt, user_has_names, user_has_fmt = _resolve_names_and_format(
            input_files, model_filename, options, stems=resolved_stems)
        cmd = [
            AUDIO_SEPARATOR_BIN,
            "--output_dir", output_dir,
            "--model_file_dir", MODELS_DIR,
            "--model_filename", model_filename,
            *options,
        ]
        # Generated flags are appended only when the user did not supply them
        # in command_options, so no flag is ever duplicated and slot resolution
        # (names/fmt above) always matches what the CLI actually receives.
        if not user_has_names:
            cmd += ["--custom_output_names", json.dumps(names)]
        if not user_has_fmt:
            cmd += ["--output_format", fmt]
        cmd.append(input_files)
        logging.info("audio-separator-cli-node: running: %s", " ".join(cmd))
        # Tee the CLI output: stream each line to ComfyUI's server console in
        # real time while accumulating it for the STDOUT connector. stderr is
        # merged into stdout so ordering is preserved.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        with _PROC_LOCK:
            _ACTIVE_PROCS.append(proc)
        try:
            chunks = []
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                chunks.append(line)
            rc = proc.wait()
        finally:
            with _PROC_LOCK:
                try:
                    _ACTIVE_PROCS.remove(proc)
                except ValueError:
                    pass
        out = "".join(chunks)
        if rc != 0:
            if rc < 0:
                raise ValueError(
                    f"audio-separator was terminated by signal {-rc} "
                    "(killed externally or via STOP PROCESS)"
                )
            raise ValueError(
                f"audio-separator failed (exit {rc}):\n{out[-4000:]}"
            )
        paths = _stem_paths(input_files, model_filename, output_dir, names, fmt,
                            stems=resolved_stems)
        return (out, *paths)


_server = getattr(server.PromptServer, "instance", None)
if _server is not None:

    @_server.routes.post("/api/audio_separation/kill")
    async def kill_audio_separator(request):
        with _PROC_LOCK:
            procs = list(_ACTIVE_PROCS)
        killed = []
        for proc in procs:
            if proc.poll() is not None:  # already exited
                continue
            try:
                proc.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
                killed.append(proc.pid)
            except OSError:
                pass
        if killed:
            message = (f"Stopped {len(killed)} audio-separator process(es): "
                       f"{', '.join(map(str, killed))}")
        else:
            message = "No running audio-separator process found"
        logging.info("audio-separator-cli-node: STOP PROCESS -> %s", message)
        return web.json_response({"killed": killed, "message": message})

    @_server.routes.get("/api/audio_separation/models")
    async def get_models(request):
        # Resolve stems for the single currently-selected model (the frontend passes
        # its model_filename). Prefers catalog stems; otherwise resolves from the
        # model's config, downloading that one model if not already present.
        fn = request.query.get("model_filename") or ""
        if not fn:
            return web.json_response(
                {"error": "model_filename query parameter is required"}, status=400)
        # Run in a worker thread so even the (small) yaml fetch doesn't stall aiohttp's
        # event loop. The button path only ever downloads the model's ~KB yaml, never its
        # weights — see _resolve_stems_yaml_only.
        stems, status = await asyncio.to_thread(_resolve_stems_yaml_only, fn)
        payload = {"model_filename": fn, "stems": stems, "status": status}
        if not stems:
            if status == "must_download":
                payload["message"] = ("Model must be downloaded — run it once to fetch "
                                      "the full model.")
            else:
                payload["message"] = (f"Could not resolve stems ({status}). Check the "
                                      "network, or run this model once.")
        return web.json_response(payload)

    @_server.routes.get("/api/audio_separation/local_models")
    async def get_local_models(request):
        # Physical model files present in MODELS_DIR (the /models folder), as
        # opposed to the --list_models catalog which also lists not-yet-
        # downloaded models. Only recognized model extensions are returned so
        # metadata like download_checks.json is excluded.
        try:
            raw = [p.name for p in Path(MODELS_DIR).iterdir()
                   if p.is_file() and p.suffix.lower() in MODEL_FILE_EXTS]
            names = _dedup_model_files(raw)
        except Exception as e:
            logging.warning(f"audio-separator-cli-node: local model scan failed ({e})")
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"models": names, "dir": MODELS_DIR})

    @_server.routes.post("/api/audio_separation/remove_models")
    async def remove_models(request):
        # Destructive failsafe (user-confirmed in the UI dialog): wipe ALL files in
        # MODELS_DIR so unwanted/damaged models are purged and a re-download starts
        # clean. Guards against an unsafe path, then reports exactly what was removed.
        root = Path(MODELS_DIR)
        if not root.is_dir():
            return web.json_response(
                {"error": f"models dir not found: {MODELS_DIR}"}, status=404)
        resolved = str(root.resolve())
        if len(resolved) < 3 or resolved in ("/",):  # refuse to wipe an unsafe/root path
            return web.json_response(
                {"error": "refusing to remove from unsafe path"}, status=400)
        removed, failed = await asyncio.to_thread(_wipe_models_dir, MODELS_DIR)
        logging.info("audio-separator-cli-node: Remove Models -> %d removed, %d failed",
                     len(removed), len(failed))
        message = (f"Removed {len(removed)} item(s) from {MODELS_DIR}"
                   + (f"; FAILED: {', '.join(failed)}" if failed else ""))
        return web.json_response({"removed": removed, "failed": failed, "message": message})

    @_server.routes.get("/api/audio_separation/help")
    async def get_help(request):
        # Runs `audio-separator --help`, mirrors the text to the server console
        # (logging), and returns it for the frontend dialog. argparse prints
        # help on stdout; stderr is appended as a safety net.
        try:
            proc = subprocess.run(
                [AUDIO_SEPARATOR_BIN, "--help"],
                capture_output=True, text=True, timeout=120,
            )
            help_text = (proc.stdout or "") + (proc.stderr or "")
        except Exception as e:
            logging.warning(f"audio-separator-cli-node: --help failed ({e})")
            return web.json_response({"error": str(e)}, status=500)
        if not help_text.strip():
            return web.json_response(
                {"error": "audio-separator --help produced no output"}, status=500
            )
        logging.info("audio-separator-cli-node: --help\n%s", help_text.rstrip())
        return web.json_response({"help": help_text})


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "AudioSeparation": AudioSeparation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSeparation": "Audio Separator CLI",
}
