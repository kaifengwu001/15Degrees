"""Pipeline loading: QwenImageEditPlus + rapid transformer + angle LoRA.

Mirrors `hf-space/app.py` exactly for visual fidelity, but adds a 4-bit
bitsandbytes (NF4) quantization path for the transformer so the whole thing
fits in 12 GB of VRAM with real CUDA kernels on Windows.

Two paths:

- `quantize_4bit=True` (default on 12 GB cards): transformer is loaded
  in bf16, then every `nn.Linear` is replaced in-place with a
  `bitsandbytes.nn.Linear4bit` (NF4). The LoRA is applied as an unfused
  adapter with weight == `lora_fuse_scale` (numerically equivalent to
  `pipe.fuse_lora(lora_scale=...)` during forward, since fuse cannot
  write into 4-bit storage).
- `quantize_4bit=False`: the original HF-Space bf16 path with LoRA
  fusion. Only use on cards with >=24 GB VRAM or via CPU offload.

The rapid transformer ships fp8 weights on disk. `torch_dtype=bfloat16`
at load time dequantizes to bf16 in memory so bnb can requantize to NF4.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

import torch

BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
RAPID_TRANSFORMER_ID = "linoyts/Qwen-Image-Edit-Rapid-AIO"
RAPID_TRANSFORMER_SUBFOLDER = "transformer"
LORA_REPO_ID = "dx8152/Qwen-Edit-2509-Multiple-angles"
LORA_WEIGHT_NAME = "\u955c\u5934\u8f6c\u6362.safetensors"  # 镜头转换.safetensors
LORA_ADAPTER_NAME = "angles"

DTYPE = torch.bfloat16


def load_pipeline(
    *,
    lora_fuse_scale: float,
    enable_cpu_offload: bool,
    quantize_4bit: bool,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Build the pipeline. See module docstring for the two supported paths."""
    log = logger or logging.getLogger("orbit")

    # Imports are local so --help / CLI errors don't require torch + diffusers.
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

    device = _resolve_device(log)

    log.info("Loading rapid transformer from %s (bf16 in RAM) ...", RAPID_TRANSFORMER_ID)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        RAPID_TRANSFORMER_ID,
        subfolder=RAPID_TRANSFORMER_SUBFOLDER,
        torch_dtype=DTYPE,
    )

    log.info("Loading base pipeline from %s ...", BASE_MODEL_ID)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL_ID,
        transformer=transformer,
        torch_dtype=DTYPE,
    )

    log.info(
        "Loading angle LoRA %s (weight=%s) and fusing at scale %.2f (bf16) ...",
        LORA_REPO_ID,
        LORA_WEIGHT_NAME,
        lora_fuse_scale,
    )
    pipe.load_lora_weights(
        LORA_REPO_ID,
        weight_name=LORA_WEIGHT_NAME,
        adapter_name=LORA_ADAPTER_NAME,
    )
    pipe.set_adapters([LORA_ADAPTER_NAME], adapter_weights=[1.0])
    pipe.fuse_lora(adapter_names=[LORA_ADAPTER_NAME], lora_scale=lora_fuse_scale)
    pipe.unload_lora_weights()
    log.info("LoRA fused at scale %.2f and unloaded.", lora_fuse_scale)

    if quantize_4bit:
        # Fuse-then-quantize: the LoRA 1.25x delta is already baked into the
        # bf16 base weights, so we can freely replace each `nn.Linear` with a
        # `bnb.Linear4bit`. Forward path is then pure 4-bit matmul with no
        # peft runtime overhead.
        _quantize_transformer_4bit_nf4(pipe.transformer, log)

    if enable_cpu_offload:
        log.info("enable_cpu_offload=True; enabling model CPU offload.")
        pipe.enable_model_cpu_offload()
    else:
        log.info("Moving pipeline to %s ...", device)
        pipe = pipe.to(device)

    log.info("Pipeline ready.")
    return pipe


def _quantize_transformer_4bit_nf4(transformer: torch.nn.Module, log: logging.Logger) -> None:
    """In-place swap every `nn.Linear` in `transformer` for `bnb.Linear4bit` (NF4).

    Config tuned for speed on Ada Lovelace + Windows:
      - NF4 quant type (best-fit for normal-distributed weights)
      - `compress_statistics=True` (nested double-quant) — needed to stay on
        bnb's fast fused kernel path. Turning it off drops us to a slow fallback.
      - `compute_dtype=bfloat16` (matches the HF Space pipeline dtype)
      - Only `proj_out` is kept in bf16; everything else is 4-bit.

    Must be called AFTER `fuse_lora + unload_lora_weights` so the 1.25x LoRA
    delta is already baked into the bf16 base weights.
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit, Params4bit
    except ImportError as exc:
        raise ImportError(
            "quantize_4bit=true requires `bitsandbytes`. Install with: "
            "pip install bitsandbytes"
        ) from exc

    # Keep high-signal / low-parameter-count layers in bf16. These are
    # the input+output projections, timestep conditioning, and the final
    # adaptive norm. Total extra VRAM is small (<100 MB) but quality gain
    # on a 4-step rapid model is large because per-layer error here cannot
    # be averaged out across additional denoising steps.
    skip_subtree_names = {
        "proj_out",
        "img_in",
        "txt_in",
        "time_text_embed",
        "norm_out",
    }

    replaced = 0
    total = 0

    def _walk(parent: torch.nn.Module, prefix: str = "") -> None:
        nonlocal replaced, total
        for name, child in list(parent.named_children()):
            if name in skip_subtree_names:
                continue

            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear):
                total += 1
                new_layer = Linear4bit(
                    input_features=child.in_features,
                    output_features=child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=DTYPE,
                    compress_statistics=True,   # keep fast kernel path
                    quant_type="nf4",
                    device="cpu",
                )
                new_layer.weight = Params4bit(
                    child.weight.data.to(DTYPE),
                    requires_grad=False,
                    quant_type="nf4",
                    compress_statistics=True,
                )
                if child.bias is not None:
                    new_layer.bias = torch.nn.Parameter(child.bias.data.to(DTYPE))

                new_layer = new_layer.to("cuda")
                setattr(parent, name, new_layer)
                replaced += 1
                del child
            elif isinstance(child, bnb.nn.Linear4bit):
                total += 1
            else:
                _walk(child, path)

    log.info("Quantizing transformer Linear -> bnb Linear4bit (NF4, compute=%s) ...", DTYPE)
    _walk(transformer)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Quantization complete: replaced %d / %d Linear layers.", replaced, total)


def _resolve_device(logger: logging.Logger) -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("CUDA available: %s", name)
        return "cuda"
    logger.warning("CUDA not available; falling back to CPU (will be extremely slow).")
    return "cpu"
