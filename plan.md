# Orbit Decay — Automated Generation Pipeline

## What This Does

Takes a single source photograph and repeatedly asks Qwen-Image-Edit-2509 (with dx8152's angle LoRA fused into Phr00t's 4-step rapid transformer) to rotate 15° around the subject. Each output becomes the next input. The image sequence is then assembled into a video with an accelerating frame rate — starting at 5 seconds per frame, ending at 25fps.

Two fully independent stages: **generate frames**, then **assemble video**. Either can be run, re-run, or tweaked without touching the other.

The loading recipe and inference parameters are chosen to faithfully reproduce the output of the HuggingFace Space `linoyts/Qwen-Image-Edit-Angles` — specifically the manual workflow of "upload a photo, generate, download the output, re-upload it, generate again, forever."

---

## Project Structure

```
orbit-decay/
├── plan.md
├── requirements.txt
├── generate.py                   # Stage 1: frame generation
├── assemble.py                   # Stage 2: video assembly
├── config.yaml                   # All parameters in one place
├── qwenimage/                    # Vendored from the HF Space (not in released diffusers yet)
│   ├── __init__.py
│   ├── pipeline_qwenimage_edit_plus.py
│   ├── transformer_qwenimage.py
│   └── qwen_fa3_processor.py     # used on Hopper only; we fall back to default attn
├── runs/
│   └── <run-name>/
│       ├── frames/               # frame_0000.png, frame_0001.png, ...
│       ├── state.json            # progress + per-frame seeds
│       ├── config.snapshot.yaml  # copy of config.yaml at run start
│       └── generate.log
└── output/                       # final video files (shared across runs)
```

Isolating each attempt under `runs/<run-name>/` keeps `portrait_v1`, `portrait_v2_seed99`, etc. from stomping on each other.

---

## Stage 1: Frame Generation (`generate.py`)

### Faithfully reproducing the HF Space

The Space uses a vendored diffusers variant (`QwenImageEditPlusPipeline`) and a specific LoRA-fusion recipe. Copy exactly:

```python
import torch
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

dtype = torch.bfloat16   # bf16, not fp16 — matches Space; 4070 handles it natively
device = "cuda"

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder="transformer",
        torch_dtype=dtype,
    ),
    torch_dtype=dtype,
).to(device)

# Fuse the angle LoRA at scale 1.25, then drop the adapter.
# The 1.25 boost above 1.0 is deliberate and part of the "look" the Space produces.
pipe.load_lora_weights(
    "dx8152/Qwen-Edit-2509-Multiple-angles",
    weight_name="镜头转换.safetensors",
    adapter_name="angles",
)
pipe.set_adapters(["angles"], adapter_weights=[1.0])
pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
pipe.unload_lora_weights()

# Skip spaces.aoti_blocks_load — Zero-GPU-only, and FA3 requires Hopper.
# We fall back to the default attention processor. Costs speed, nothing else.

# VRAM safety valve (enabled via config flag if --dry-run OOMs):
# pipe.enable_model_cpu_offload()
```

### Prompt format

Must be bilingual to match the Space's LoRA conditioning. For 15° right rotation:

```
"将镜头向右旋转15度 Rotate the camera 15 degrees to the right."
```

Chinese first, English second, terminal period. Left-rotation flips both halves. English-only gets a different LoRA response and a noticeably different look, so don't shorten.

### Core loop

```
load pipeline + fuse LoRA (one-time, ~60-90s)
compute output (width, height) from source (see "Resolution" below)

current_image = source_photo
for i in range(start_frame, total_frames):
    seed_i = random.randint(0, MAX_SEED)  # fresh seed per frame by default
    generator = torch.Generator(device="cuda").manual_seed(seed_i)

    output = pipe(
        image=[current_image],         # list, not single — Plus pipeline requirement
        prompt=bilingual_rotate_prompt(15),
        height=height,
        width=width,
        num_inference_steps=4,
        true_cfg_scale=1.0,            # only true_cfg_scale, no guidance_scale kwarg
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]

    save frame_NNNN.png                # lossless PNG
    append seed_i to state.json["frame_seeds"]
    update state.json (last_completed_frame, timestamp)
    current_image = output
    torch.cuda.empty_cache()           # cheap insurance over 300 iterations
```

**Why random seed per frame is the default.** The Space's default is `randomize_seed=True`. When you manually re-uploaded each frame you were implicitly getting a fresh random seed every call — that's the look you liked. Fixed seed is opt-in (`--fixed-seed K`), not the default.

### Resolution

Match the Space's aspect-preserving logic. On source load, once:

- Longest side → `longest_side` (default 1024)
- Other side scaled to preserve original aspect ratio
- Both dimensions snapped down to multiples of 8

All frames in a run share the same (width, height).

### Interrupt / Resume

`state.json`:
```json
{
  "source_image": "portrait.jpg",
  "run_name": "portrait_v1",
  "width": 768,
  "height": 1024,
  "total_frames": 300,
  "last_completed_frame": 147,
  "prompt": "将镜头向右旋转15度 Rotate the camera 15 degrees to the right.",
  "lora_fuse_scale": 1.25,
  "randomize_seed": true,
  "fixed_seed": null,
  "frame_seeds": [12345, 67890, "... one per completed frame ..."],
  "config_hash": "sha256:abc123..."
}
```

**Interrupt**: Ctrl+C at any time. The in-progress frame is discarded (never reaches disk). `state.json` always reflects the last fully written frame.

**Resume** (`--resume`): reads `state.json`, loads `frame_{last}.png` as input, continues from `last + 1`. If the current `config.yaml` hash differs from `config_hash` in state, **abort with a diff** unless `--force` is passed. Prevents silent half-one-config / half-another runs.

**Fork** (`--from-frame N`): loads `frame_NNNN.png` as input for step N+1; overwrites everything from N+1 onward. Frame N itself is NOT regenerated — it's the seed input for the new branch. `frame_seeds` is truncated to N entries and new seeds append from there.

**Fork with deterministic seed** (`--from-frame N --fixed-seed K`): same as above but switches that branch to a single fixed seed for reproducibility.

### CLI

```
python generate.py --source portrait.jpg --run-name portrait_v1 --frames 300   # fresh run
python generate.py --run-name portrait_v1 --resume                             # continue interrupted run
python generate.py --run-name portrait_v1 --from-frame 85                      # fork from frame 85
python generate.py --run-name portrait_v1 --from-frame 85 --fixed-seed 99      # fork deterministically
python generate.py --source portrait.jpg --run-name smoke --limit 1 --dry-run  # load pipeline + 1 frame, exit
python generate.py --source portrait.jpg --run-name burn30 --limit 30          # short drift-check run
```

`--limit N` caps a fresh run at N frames regardless of `total_frames` in config — the cheap way to scout compound-drift behavior before committing 2 hours.

### Logging and progress

- `tqdm` bar around the outer loop (ETA, per-frame elapsed).
- File logger at `runs/<run-name>/generate.log`: frame number, seed, elapsed, any warnings/tracebacks. Required — if the process dies at frame 247 at 1:43am, you want the traceback on disk.

---

## Stage 2: Video Assembly (`assemble.py`)

Completely independent of Stage 1. Reads from `runs/<run-name>/frames/`, writes to `output/`.

### Accelerating frame rate curve

Maps frame index to hold duration. Exponential default:

```
d(i) = start_hold * (end_hold / start_hold) ** (i / (N - 1))
```

where `end_hold = 1 / end_fps`. For defaults (start 5.0s, end 25fps → 0.04s, N=300):

- Frame 0 holds 5.00s
- Frame 150 holds ~0.45s
- Frame 299 holds 0.04s

### ffmpeg concat

Generate a concat demuxer file. **Quirk**: `duration` applies to the *next* file, so the last frame must be listed twice or it gets dropped:

```
file 'frame_0000.png'
duration 5.00
file 'frame_0001.png'
duration 4.72
...
file 'frame_0299.png'
duration 0.04
file 'frame_0299.png'
```

Then:
```
ffmpeg -f concat -safe 0 -i concat.txt -vf "fps=25" -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4
```

`fps=25` resamples the variable-duration stream to constant 25fps for clean playback on any player.

### Curve options

- `exponential` (default): slow start, rapid acceleration — the "gradual drift into fever dream" arc
- `linear`: even acceleration between start and end holds
- `custom`: read durations from a CSV for manual control

### CLI

```
python assemble.py --run-name portrait_v1                                     # default curve
python assemble.py --run-name portrait_v1 --start-hold 5.0 --end-fps 25
python assemble.py --run-name portrait_v1 --curve linear
python assemble.py --run-name portrait_v1 --range 0 150                       # subset (e.g. pre-collapse)
python assemble.py --run-name portrait_v1 --preview                           # 720p quick encode
```

---

## config.yaml

Single source of truth. `config_hash` in `state.json` is the SHA-256 of this file; see Resume behavior above.

```yaml
# Source
source_image: portrait.jpg
run_name: portrait_v1

# Generation
total_frames: 300
rotate_degrees: 15           # per-step rotation (positive = right)
lora_fuse_scale: 1.25        # Space's value; part of the "look"
inference_steps: 4
true_cfg_scale: 1.0
longest_side: 1024           # aspect preserved; other side snapped to /8
randomize_seed: true         # fresh seed per frame (Space default)
fixed_seed: null             # set to an int to force determinism
enable_cpu_offload: false    # flip to true if --dry-run OOMs

# Assembly
start_hold_seconds: 5.0
end_fps: 25
curve: exponential
output_format: mp4
output_codec: libx264
output_crf: 18
```

---

## requirements.txt (draft)

```
torch==2.5.1                # install with CUDA index URL (see below)
torchvision
diffusers>=0.34             # QwenImage classes may land upstream; vendored ./qwenimage is the fallback
transformers
accelerate
peft
safetensors
sentencepiece
huggingface_hub
pillow
pyyaml
tqdm
```

No `torchao`, no `spaces`, no `kernels` — those are Zero-GPU-specific. FA3 is unavailable on Ada; default attention processor is used.

On a 4070 with CUDA 12.1+, install torch with:
```
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

ffmpeg is a separate system install (`winget install ffmpeg` on Windows, brew/apt elsewhere).

---

## Time Estimates

### Stage 1: Frame Generation

Without FA3 + AOT-compiled blocks (both Zero-GPU-only):

| Scenario | Per frame | 300 frames |
|---|---|---|
| 4070, bf16, default attention | ~15-25s | ~75-125 min |
| 4070, bf16, CPU offload | ~25-40s | ~125-200 min |
| First run includes model download | +10-20 min | — |

Model download is ~15GB total. One-time, cached to `~/.cache/huggingface/`. Pipeline load + LoRA fuse: ~60-90s per run.

**Budget for a 300-frame run: ~1.5-2 hours on the 4070, plus first-time download.**

### Stage 2: Video Assembly

~10-30 seconds.

### Development / Setup

| Task | Estimate |
|---|---|
| Env + requirements install + model download | 45 min |
| Vendor `qwenimage/` from HF Space, verify imports | 15 min |
| Write `generate.py` with interrupt/resume/fork + logging | 2 hours |
| Write `assemble.py` with curve options + last-frame-dup | 30 min |
| `--dry-run` + `--limit 30` burn-in (look + drift check) | 45 min |
| Debug VRAM / attention fallback if needed | 0-2 hours |
| **Total dev time** | **~4-6 hours** |

---

## Open Questions

1. **VRAM fit (highest risk).** The 2509 base + rapid transformer + fused LoRA in bf16 may or may not fit in 12GB without CPU offload. `--dry-run` catches this before committing to a full run. If offload is needed, per-frame time roughly doubles — reflected in the estimate table above.

2. **Compound drift ceiling.** Each autoregressive step introduces VAE roundtrip loss + model noise. Practical experience with similar pipelines: recognizable subject preservation typically collapses somewhere between frames 40 and 100. That may be exactly the aesthetic you want, but validate with `--limit 30` / `--limit 60` / `--limit 100` before committing to 300. If drift is unpleasant sooner than expected, the easing curve can be re-tuned so that 25fps is reached *before* the collapse point, rather than after.

3. **Fork determinism default.** `--from-frame N` currently continues with random seeds (preserves the "each call is a fresh re-upload" feel). If reproducibility-on-fork matters more than feel-consistency, the default could flip to derive seeds from `(base_seed + frame_index)`. Current default prioritizes the look.
