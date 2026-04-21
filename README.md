# Orbit Decay

Iterative camera rotation over a single source photograph using Qwen-Image-Edit-2509
with dx8152's angle LoRA fused into Phr00t's 4-step rapid transformer. Each generated
frame becomes the input for the next rotation, producing an autoregressive orbit that
gradually decays into a "fever dream." Frames are then assembled into a video with an
accelerating hold-duration curve — long pauses at the start, 25fps at the end.

The loading recipe and inference parameters are chosen to faithfully reproduce the
output of the HuggingFace Space
[`linoyts/Qwen-Image-Edit-Angles`](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles):
specifically, the manual workflow of uploading a photo, generating, downloading the
result, re-uploading it, and generating again.

## Requirements

- An NVIDIA GPU with ≥12 GB VRAM (RTX 4070 or better recommended)
- CUDA 12.1+ drivers
- Python 3.10 – 3.13
- `ffmpeg` on PATH (`winget install ffmpeg` on Windows)
- ~15 GB free disk for the first-time model download (cached to `~/.cache/huggingface/`)

## Install

```powershell
# 1) Create + activate a venv
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install PyTorch with your CUDA build (adjust cu121 to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) Install the rest
pip install -r requirements.txt
```

If the vendored `qwenimage/` module fails to import against your installed
`diffusers`, install diffusers from main:

```powershell
pip install "git+https://github.com/huggingface/diffusers.git"
```

## Quick start

1. Drop a source photograph in the project root (e.g. `portrait.jpg`).
2. Edit `config.yaml`: set `source_image`, `run_name`, `total_frames`, and any
   generation knobs you want to change.
3. Run a single-frame dry run first to confirm the pipeline loads and fits in VRAM:

   ```powershell
   py -3 generate.py --source portrait.jpg --run-name smoke --dry-run
   ```

4. Run a 30-frame burn-in to preview how fast compound drift hits:

   ```powershell
   py -3 generate.py --source portrait.jpg --run-name burn30 --limit 30
   ```

5. When you like what you see, kick off the full run:

   ```powershell
   py -3 generate.py --source portrait.jpg --run-name portrait_v1 --frames 300
   ```

6. Assemble the video:

   ```powershell
   py -3 assemble.py --run-name portrait_v1
   ```

   The output lands in `output/portrait_v1.mp4`.

## Reference: generate.py

```text
--config PATH              YAML config (default: config.yaml)
--source PATH              Override source_image (fresh runs only)
--run-name NAME            Override run_name
--frames N                 Override total_frames
--limit N                  Cap THIS invocation at N frames (scouting compound drift)
--fixed-seed K             Force deterministic seed K on every frame
--randomize-seed           Force randomize_seed=true
--cpu-offload              Force model CPU offload (use if --dry-run OOMs)
--resume                   Continue an interrupted run (reads state.json)
--from-frame N             Fork: use frame N as input for N+1, overwrite N+1..end
--force                    Ignore config-hash mismatch on --resume
--dry-run                  Load pipeline, generate 1 frame, exit
```

### Interrupt / resume semantics

Interrupt with Ctrl+C at any time. The in-progress frame is discarded (never reaches
disk); `state.json` always reflects the last fully written frame.

Resume:

```powershell
py -3 generate.py --run-name portrait_v1 --resume
```

If `config.yaml` has changed since the run started, `--resume` aborts with a diff of
the SHA-256 hashes. Pass `--force` to override (not recommended unless you know what
changed).

### Fork from an arbitrary frame

```powershell
py -3 generate.py --run-name portrait_v1 --from-frame 85
py -3 generate.py --run-name portrait_v1 --from-frame 85 --fixed-seed 99
```

Frame 85 is the *input* for the new frame 86; frames 86..299 are overwritten.

## Reference: assemble.py

```text
--config PATH                  YAML config (default: config.yaml)
--run-name NAME                Override run_name from config
--start-hold SECONDS           Override start_hold_seconds
--end-fps FPS                  Override end_fps
--curve {exponential|linear|custom}   Override curve
--range START END              Assemble only frames [START..END] inclusive
--preview                      720p quick encode
```

The exponential curve is `d(i) = start_hold * (end_hold/start_hold)^(i/(N-1))` where
`end_hold = 1 / end_fps`. For the defaults (5.0s → 25fps, N=300): frame 0 holds 5.00s,
frame 150 holds ~0.44s, frame 299 holds 0.04s.

## Project layout

```
.
├── config.yaml               # single source of truth for both stages
├── generate.py               # Stage 1 CLI
├── assemble.py               # Stage 2 CLI
├── orbit/                    # project-specific modules
│   ├── config.py             # YAML loading + validation + SHA hashing
│   ├── prompts.py            # bilingual rotate prompt (matches the Space)
│   ├── resolution.py         # aspect-preserving, snap-to-8 size calc
│   ├── runs.py               # run directory layout
│   ├── state.py              # state.json read/write, immutable updates
│   ├── seeds.py              # randomize/fixed seed policy
│   ├── logging_setup.py      # file + console logger (UTF-8)
│   ├── pipeline.py           # model load + LoRA fuse (the HF Space recipe)
│   ├── generator.py          # autoregressive generation loop
│   ├── curves.py             # exponential / linear / custom hold curves
│   └── concat.py             # ffmpeg concat file + invocation
├── qwenimage/                # vendored from linoyts/Qwen-Image-Edit-Angles
├── runs/                     # per-run outputs (frames, state, logs, config snapshot)
└── output/                   # final encoded videos
```

## Notes on faithfulness to the HF Space

- bfloat16, not float16
- `QwenImageEditPlusPipeline` (Plus variant), with image passed as a list: `image=[pil]`
- Angle LoRA is *fused* at `lora_scale=1.25`, then adapter is unloaded — this scale is
  part of the look, not a neutral default
- Bilingual prompt: `"将镜头向右旋转15.0度 Rotate the camera 15.0 degrees to the right."`
- Only `true_cfg_scale` is passed to `pipe(...)`; there is no `guidance_scale` kwarg
- `randomize_seed=True` by default — each call gets a fresh seed, matching the Space
- Resolution preserves aspect, longest side = `longest_side`, both snapped to /8
- Zero-GPU-only bits are skipped: `spaces.aoti_blocks_load(...)`, `kernels`,
  FlashAttention 3. We fall back to the default attention processor, which is slower
  but numerically equivalent
