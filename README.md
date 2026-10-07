# Audio Separator CLI — ComfyUI Node

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node that runs the
[`audio-separator`](https://github.com/Anjok07/audio-separator) command-line tool to split an audio
file into stems (vocals, instrumental, drums, …).

## Design: a GUI around the audio-separator CLI

This node **wraps** `audio-separator` and is **visually designed around its command-line arguments**.
Every input maps 1:1 onto a CLI flag, and every button mirrors an operation you could run from a
terminal — so if you already know the CLI, the node should feel familiar.

| Node field / button | audio-separator equivalent |
| --- | --- |
| `input_files` | positional input-file argument (e.g. `/tmp/song.wav`) |
| `output_dir` | `--output_dir` |
| `model_filename` | `--model_filename` |
| `command_options` | any extra CLI flags, verbatim (see `audio-separator --help`) |
| **Show Command Help** button | `audio-separator --help` |
| **List Loaded Models** button | list the model files present in the models folder |
| **List Model Stems** button | show a model's stems, numbered to match `output_stempath_N` |
| **Remove Models** button | delete everything in the models folder (failsafe) |
| **STOP PROCESS** button | kill the running `audio-separator` subprocess |

Under the hood the node builds and runs exactly this:

```bash
audio-separator --output_dir <output_dir> \
                --model_file_dir <models dir> \
                --model_filename <model_filename> \
                [your command_options…] \
                --custom_output_names '{"Vocals": "song-model-(Vocals)", …}' \
                --output_format wav \
                <input_files>
```

`--custom_output_names` and `--output_format` are added automatically **only** when you haven't
supplied them in `command_options`, so your values always win.

## Inputs

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `input_files` | STRING | `/tmp/example.wav` | Full path to the audio file to separate. |
| `output_dir` | STRING | `/tmp/` | Directory where stems are written. |
| `model_filename` | dropdown | — | A model from the audio-separator catalog (or one already in the models folder). |
| `command_options` | STRING | *(empty)* | Extra CLI flags verbatim, e.g. `--enable_float32`. May include `--custom_output_names '{"Vocals": "my_v"}'` and/or `--output_format flac` to override auto naming. |

## Outputs

| Output | Meaning |
| --- | --- |
| `STDOUT` | The CLI's console output for the run (also streamed live to ComfyUI's server log). |
| `output_stempath_1 … 8` | Full path of each stem file, in model order. Unused slots return an empty string. |

**Chaining:** each `output_stempath_N` is just a file path — feed it as the `input_files` of another
Audio Separator CLI node to run a second separation on that stem. Chain nodes together to build complex
multi-stage graphs (e.g. isolate vocals → then split those vocals into lead/backing).

## Output naming

By default each stem is written as:

```
{input_stem}-{model_file_stem}-{(stem)}.{format}
```

e.g. input `song.wav` with model `htdemucs_ft.yaml` → `song-htdemucs_ft-(Vocals).wav`,
`song-htdemucs_ft-(Instrumental).wav`. The number shown by **List Model Stems** (1, 2, …) is the same
index as `output_stempath_N`. Override with `--custom_output_names` / `--output_format` in
`command_options`.

## Buttons

- **Show Command Help** — runs `audio-separator --help` and shows it.
- **List Loaded Models** — lists the model files currently present in the models folder.
- **List Model Stems** — resolves the selected model's stems (fetching only its small config, never
  its weights) and lists them numbered to match `output_stempath_N`.
- **Remove Models** — *destructive*: deletes **every file** in the models folder so unwanted or
  damaged models are purged and a re-download starts clean. A confirmation dialog guards it.
- **STOP PROCESS** — hard-stops (SIGKILL) any running `audio-separator` subprocess, e.g. one that has
  hung mid-download or mid-inference.

## Installation & setup

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
2. Put this folder in ComfyUI's `custom_nodes/` — or install it via **ComfyUI-Manager**, which reads the
   node's `requirements.txt` and installs [`audio-separator`](https://github.com/Anjok07/audio-separator)
   (the CPU baseline) into ComfyUI's environment for you.
3. If installing manually, run `install.py` **with the same Python that runs ComfyUI** so the dependency
   lands where the node looks for it (next to ComfyUI's interpreter). Pick ONE profile:

   | Profile | Installs | Command |
   | --- | --- | --- |
   | **CPU** (default) | `requirements.txt` — works on any machine, no CUDA needed | `<comfyui-venv>/bin/python install.py` |
   | **GPU** (CUDA ONNX) | `requirements-gpu.txt` — accelerates ONNX stems on a GPU | `<comfyui-venv>/bin/python install.py --gpu` |

   On Windows use the venv's `...\Scripts\python.exe` instead of `.../bin/python`. You can also skip
   `install.py` and run `pip install -r requirements.txt` (or `-r requirements-gpu.txt`) directly. **Do not
   combine both profiles** — that leaves two ONNX runtimes and can break the CUDA provider. To point at a
   specific binary instead, set `AUDIO_SEPARATOR_BIN`.
4. Restart ComfyUI and add an **Audio Separator CLI** node (category *audio*).

> **Dependency notes:** `audio-separator` is a heavy ML package (pulls PyTorch + ~28 others). The node was
> tested against 0.47.x; on any other version it logs a startup warning — run `smoke_test.py` before relying
> on stems. PyTorch-based stems already use your existing CUDA torch, so the CPU profile does not block GPU
> inference — only ONNX stems need the GPU runtime. audio-separator pins `beartype<0.19.0`; if other packages
> need a newer one, run `pip install --upgrade beartype` afterwards (pip warns about the conflict, but it works).

### Models folder

Model weights/configs are stored in the node's `models/` directory by default. Override its location
with the `AUDIO_SEPARATOR_MODEL_DIR` environment variable. Models download on first use; **Remove
Models** clears them all.

## Notes & safety

- The CLI runs as a separate process, so a hung or crashed model can't take down ComfyUI — use
  **STOP PROCESS** to kill it and **Remove Models** to purge bad files.
- `command_options` is passed through verbatim; see the **Show Command Help** button for every flag
  audio-separator supports.
