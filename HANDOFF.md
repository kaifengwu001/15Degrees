# Orbit Decay — Handoff Notes (resume on 4090 PC)

Status snapshot as of 2026-04-20. Working on **Windows**, Python 3.13, one
12 GB RTX 4070 SUPER. Current task: migrate to a 24 GB RTX 4090 to run the
pipeline at **reference quality** (bf16, no quantization) at usable speed.

---

## 1. Project at a glance

**Goal:** generate an "orbit decay" video — a sequence of frames where the
camera rotates a fixed number of degrees each step (default 15°) around a
subject, with an accelerating hold curve (long pause on frame 1, snapping
to 1/`end_fps` by the last frame). Two stages:

- `generate.py` — autoregressive frame generator. Feeds previous frame
  back into the model, applies the angle LoRA, writes `frame_NNNN.png`
  and a `state.json`.
- `assemble.py` — turns the frame directory + a duration curve into an
  MP4 via `ffmpeg` concat demuxer.

**Faithful reproduction** of the HF Space
`linoyts/Qwen-Image-Edit-Angles`. Same model, LoRA, prompt format
(bilingual `将镜头向右旋转15.0度 Rotate the camera 15.0 degrees to the right.`),
1.25× fuse scale, bf16, `true_cfg_scale=1.0`, 4 inference steps, random
per-frame seed by default.

The pipeline is in `orbit/pipeline.py`; the vendored transformer/pipeline
classes are under `qwenimage/` (copied from the HF Space because they are
not yet in mainline diffusers).

---

## 2. Current state (what works, what doesn't)

### ✅ Working
- End-to-end generation pipeline. A `--dry-run` (single frame) succeeds
  and produces recognizable output with the 15° rotation applied.
- Vendored `qwenimage` modules loaded correctly.
- LoRA fused-at-bf16 path produces the exact "look" of the HF Space.
- `state.json` atomic writes, resume, fork-from-frame, config hash all in
  place.
- Assembly stage (`assemble.py`) generates valid concat files and invokes
  `ffmpeg` with the exponential / linear / custom curves.

### ⚠️ Compromised (this is the whole reason for the 4090 migration)
- On 12 GB VRAM, bf16 reference pipeline pages weights over PCIe on every
  forward pass → **13m 45s per frame** → ~55 hours for 300 frames.
  Unusable.
- Our only viable local workaround was **4-bit NF4 via bitsandbytes**:
  - ~65 s / frame → ~5.5 hours for 300 frames ✅ speed
  - But visible pixel-level speckle noise in every frame ❌ quality
  - Rapid model is 4-step distilled, so per-layer quantization error
    compounds instead of being smoothed across many steps.

### ❌ Dead ends on Windows 4070 SUPER
- **torchao fp8** (`Float8DynamicActivationFloat8WeightConfig`):
  torchao Windows wheels don't ship compiled cpp/cuda kernels for any
  torch version we have (2.6/2.7/2.8). Python fallback path runs, but is
  ~11 min / frame (same as offload). See
  <https://github.com/pytorch/ao/issues/2919>.
- **bitsandbytes int8** (`Linear8bitLt`): transformer ~9.5 GB barely
  fits; ends up VRAM-capped at 11.9/12.3 GB and pages. Also
  force-casts bf16 → fp16 during quantized matmul, costing precision.

---

## 3. Why a 4090 solves this (summary)

Every problem above traces to one root cause: **bf16 transformer
(~19 GB fused) > 12 GB VRAM**. A 4090 has 24 GB.

On the 4090 with `enable_model_cpu_offload()`:

| Component | bf16 size | Fits? |
|---|---|---|
| Rapid transformer (fused) | ~19 GB | ✅ resident on GPU during diffusion |
| VAE | ~1 GB | ✅ |
| Activations @ 816×1024 | 2–3 GB | ✅ |
| Qwen2.5-VL text encoder | ~15 GB | stays on CPU, moves to GPU only for prompt encoding, once per frame |

Expected speed on 4090 bf16:
- ~3–8 s per step × 4 steps = **~20–40 s per frame**
- **~2–3 hours for 300 frames at reference quality**

No quantization needed. Output will be pixel-faithful to the HF Space.

---

## 4. Migration plan on the 4090 PC

### 4.1 Copy over

Copy the **entire `c:\Kai\15Degrees\` folder** to the 4090 PC, excluding
the following (regenerable or environment-specific):

- `.venv/` — Python venv; rebuild locally.
- `__pycache__/` — cache.
- `runs/` — optional; these are dry-run outputs from the 4070 SUPER
  (smoke, smoke_fp8, smoke_4bit, smoke_4bit_fast, smoke_4bit_v2). Keep
  if you want to compare vs. 4090 outputs; otherwise delete.

Everything else is essential: `orbit/`, `qwenimage/`, `hf-space/` (the
reference clone — kept for comparison), `config.yaml`, `generate.py`,
`assemble.py`, `requirements.txt`, `README.md`, `plan.md`, `TestImage.jpg`,
this file.

### 4.2 Python + venv

Python **3.11 or 3.12** is preferred on the 4090 PC (3.13 works but
binary wheels for some packages, notably torchao, lag). 3.11 LTS is
safest.

```powershell
cd c:\...\15Degrees
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4.3 Install torch (matching the 4090's CUDA)

Ada Lovelace (compute capability 8.9). The stable option that we validated:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

`cu128` wheels run fine on any recent CUDA 12.x driver (PyTorch bundles
its CUDA runtime).

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect something like `2.8.0+cu128 True NVIDIA GeForce RTX 4090`.

### 4.4 Install the rest

```powershell
pip install -r requirements.txt
```

This gives you:

- `diffusers>=0.34` (`0.37.1` is what we tested with)
- `transformers>=4.46` (`5.5.4` tested)
- `accelerate>=1.0`
- `peft>=0.19,<0.20` ← pin matters, see gotcha below
- `bitsandbytes>=0.45` — optional on the 4090, only needed if you ever
  flip `quantize_4bit=true`. Harmless to have installed.

If `diffusers` doesn't expose the symbols `qwenimage/` needs, fall back
to main:

```powershell
pip install "git+https://github.com/huggingface/diffusers.git"
```

### 4.5 Hugging Face authentication (optional but strongly recommended)

Model downloads are MUCH faster with a token.

```powershell
.\.venv\Scripts\huggingface-cli.exe login
# paste your HF token
```

First run will download ~20 GB (rapid transformer + base pipeline VAE
and text encoder) to `%USERPROFILE%\.cache\huggingface\`. If you copied
that cache from the 4070 SUPER PC, drop it into the same location on the
4090 PC and the first run starts instantly.

---

## 5. Config changes for the 4090

Edit `config.yaml`:

```yaml
enable_cpu_offload: true     # leave ON — text encoder (~15 GB) still needs to offload
quantize_4bit: false         # turn OFF — we finally have the VRAM for reference bf16
```

Everything else stays the same (matching HF Space defaults):

- `lora_fuse_scale: 1.25`
- `inference_steps: 4`
- `true_cfg_scale: 1.0`
- `longest_side: 1024`
- `randomize_seed: true`

Alternative tighter config (try second, may be faster): keep transformer
and VAE resident on GPU, encode the prompt on CPU (slower per-prompt
but no VRAM dance):

```yaml
enable_cpu_offload: false
quantize_4bit: false
```

This needs ~20 GB on-GPU. Fits in 24 GB. If it OOMs during text
encoding, go back to `enable_cpu_offload: true`.

---

## 6. Step-by-step validation on the 4090

Run these in order. Don't skip ahead — each step answers "does the new
machine match the old one where it's supposed to, and beat it where it
should".

### 6.1 Dry-run (single frame) — reference quality

```powershell
$env:PYTHONIOENCODING="utf-8"   # needed for the bilingual Chinese prompt in logs
python generate.py --source TestImage.jpg --run-name smoke_4090 --dry-run --no-4bit
```

**What to check:**

1. No `OutOfMemoryError`.
2. Per-step time printed (tqdm line like `1/4 [00:05<00:15, 5.1s/it]`).
   Expect **3–8 s / step**.
3. `frame 0000 ... elapsed=XXs` — expect **20–40 s**.
4. Output in `runs\smoke_4090\frames\frame_0000.png` — should be clean,
   no speckle, matching the HF Space look.

If elapsed < 60 s **and image is clean**, you are in business. Proceed.

If elapsed is closer to 2-5 minutes, something is paging — re-check
config.yaml, verify `quantize_4bit: false` and `enable_cpu_offload: true`,
and run `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`
while it runs to see if you're at the cap.

### 6.2 Small run — 10 frames

```powershell
python generate.py --source TestImage.jpg --run-name test10 --no-4bit --frames 10
```

Produces 10 frames. Lets you:

- Confirm autoregressive feedback looks stable (no drift / color
  artifacts / color banding).
- Verify random-per-frame seeds really do shuffle (check
  `runs\test10\state.json` → `frame_seeds` has 10 unique ints).

### 6.3 Assembly smoke

```powershell
python assemble.py --run-name test10
```

Produces `output\test10.mp4` with the exponential hold curve (first
frame held 5 s, last frame held 1/25 s). Open it and sanity-check the
rotation cadence. Add `--preview` for a fast 720p encode.

### 6.4 Full run — 300 frames

Only after all the above pass. If the 4090 cage matches the math,
expect ~2-3 hours.

```powershell
python generate.py --source TestImage.jpg --run-name orbit_v1 --no-4bit
python assemble.py --run-name orbit_v1
```

The generator supports resume; Ctrl+C at any time, re-run the same
command to continue from the last saved frame.

---

## 7. Known gotchas (Windows-specific)

### 7.1 UTF-8 for the bilingual prompt
Always set `$env:PYTHONIOENCODING="utf-8"` in the shell before running
`generate.py`, otherwise the Chinese prompt in logs triggers
`UnicodeEncodeError` on CP1252 consoles.

### 7.2 Version pinning that matters

The dependency graph has a sharp edge:

- **diffusers ≥ 0.37** requires `peft ≥ 0.17`
- **peft ≥ 0.16** requires `torchao ≥ 0.16` at import time
- **torchao cpp kernels don't ship for Windows** with any recent torch
  version

So if you install peft 0.17+ without torchao 0.16+, peft's LoRA dispatcher
raises `ImportError: Found an incompatible version of torchao`. Fix by
keeping the current `peft>=0.19,<0.20` + `bitsandbytes` combo. You do NOT
need torchao on the 4090 since we won't use fp8 quantization.

(If you ever remove bitsandbytes, also remove any `torchao` install.)

### 7.3 `diffusers` accepts the vendored transformer but warns

Expect this harmless line during pipeline load:
```
Expected types for transformer: (<class 'diffusers.models.transformers.transformer_qwenimage.QwenImageTransformer2DModel'>,), got <class 'qwenimage.transformer_qwenimage.QwenImageTransformer2DModel'>.
```
That's just diffusers noticing we swapped its class for ours. Ignore.

### 7.4 The `local_dir_use_symlinks` deprecation warning
From `huggingface_hub` — harmless, nothing to do.

### 7.5 bitsandbytes "MatMul8bitLt: inputs will be cast from bfloat16 to float16"
Only appears if `quantize_4bit=true` falls down an int8 path. With
`quantize_4bit: false` on the 4090, you won't see this.

### 7.6 ffmpeg on PATH
`assemble.py` shells out to `ffmpeg`. Install from
<https://www.gyan.dev/ffmpeg/builds/> (pick a "release essentials" build)
and add it to PATH, or edit `orbit/concat.py` to use a full path.

---

## 8. Files you might touch

| File | What it does | Touch when... |
|---|---|---|
| `config.yaml` | Single source of truth | changing resolution, step count, curve, seeds |
| `orbit/pipeline.py` | Model loading + LoRA fuse + quantize | changing precision paths |
| `orbit/generator.py` | Autoregressive loop | changing how frames feed back / save |
| `orbit/curves.py` | Exponential/linear/custom hold curves | tuning the orbit decay rhythm |
| `orbit/concat.py` | ffmpeg invocation | changing codec/crf/container |
| `generate.py` | CLI wrapper for stage 1 | adding new CLI flags |
| `assemble.py` | CLI wrapper for stage 2 | adding new CLI flags |

Do **not** edit `qwenimage/*.py` — those are vendored upstream code. If
diffusers main adds native `QwenImageEditPlusPipeline`, we can delete
the whole `qwenimage/` folder and import from `diffusers` directly.

---

## 9. Rollback plan if the 4090 also struggles

In order from most to least likely to help:

1. **Set `enable_cpu_offload: true`** if you had it off — this is the
   single biggest lever.
2. **Drop `longest_side` to 896 or 768** — each side scales quadratically
   in activation VRAM.
3. **Quantize the text encoder only** (not yet implemented, easy add —
   ~10 lines in `orbit/pipeline.py` using `BitsAndBytesConfig` on the
   text encoder, keep transformer in bf16). This is the path to avoid
   ANY CPU offload on a 24 GB card.
4. **Fall back to `quantize_4bit: true`** — you'll get speckle but
   ~65 s/frame is still serviceable.
5. **Cloud.** Modal / RunPod / Lambda H100. ~$2-5 total.

---

## 10. What we'd test next once the 4090 is confirmed good

These are parked ideas, not urgent:

- Real 300-frame `--curve exponential` run, end-to-end video.
- Negative rotation (`rotate_degrees: -15`) to confirm the bilingual
  prompt switches to 向左 / "to the left" correctly.
- Parameter sweep: 3 / 4 / 6 inference steps — does 6 steps with same
  prompt noticeably improve detail stability across frames?
- Try `fixed_seed: 42` for a fully deterministic reference run we can
  compare hash-for-hash across machines.
- Profile where the 4090 per-frame time actually goes (encode vs
  diffuse vs VAE decode). `torch.profiler` for one frame is enough.

---

## 11. Summary numbers (so we don't relearn them)

Measured on this 4070 SUPER (12 GB), TestImage.jpg, `longest_side: 1024`:

| Config | /step | /frame | Quality |
|---|---:|---:|---|
| bf16 + CPU offload | ~206 s | 13m 45s | reference (paging) |
| fp8 (torchao Python fallback) | ~165 s | ~11 min | OK, same paging |
| 4-bit NF4 + CPU offload, skip={proj_out} | ~9 s | 69.7 s | speckle |
| 4-bit NF4 + CPU offload, skip=5 subtrees | ~9 s | 64.6 s | speckle (slightly better) |
| int8 + CPU offload | — | — | VRAM-capped, paging, slow |

Expected on the 4090 (24 GB), same settings with `quantize_4bit: false`:

| Config | /step | /frame | Quality |
|---|---:|---:|---|
| bf16 + CPU offload | 3–8 s | 20–40 s | **reference** |

That's the target. See you over there.
