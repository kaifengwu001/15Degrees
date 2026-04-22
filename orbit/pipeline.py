"""Pipeline loading: QwenImageEditPlus + rapid transformer + angle LoRA.

Mirrors `hf-space/app.py` exactly for visual fidelity. Handles three hardware
regimes, each selected by combining `quantize_4bit` and `enable_cpu_offload`:

- `quantize_4bit=True` (12 GB cards, e.g. 4070 SUPER): transformer is loaded
  in bf16, LoRA fused into bf16, then every `nn.Linear` is replaced in-place
  with `bitsandbytes.nn.Linear4bit` (NF4). Forward path uses bnb's fused
  4-bit matmul kernel on the GPU. Peak VRAM ~7 GB.

- `quantize_4bit=False`, `enable_cpu_offload=False` (24 GB cards, e.g. 4090,
  the "reference fp8 residency" path): transformer is loaded in bf16, LoRA
  fused, then every `nn.Linear`'s weight is cast to `torch.float8_e4m3fn`
  and wrapped in `Fp8Linear`. Forward path does fp8 -> bf16 cast per layer
  (just-in-time, ~70 MB peak cast buffer) then a standard `F.linear`. This
  halves the transformer's GPU footprint vs bf16 (~38 GB -> ~20 GB) at
  ~1.5x per-Linear cost, which is negligible next to activation compute.
  Numerically lossless relative to the fp8-on-disk source -- the only new
  rounding is the one-time requantization of the bf16-fused weights back
  down to fp8, bounded by fp8's mantissa. Text encoder stays on CPU
  because it alone is 15 GB and would blow the VRAM budget.

- `enable_cpu_offload=True` (fallback, e.g. 16 GB cards without bnb):
  bf16 transformer with accelerate's `enable_model_cpu_offload()`. Known
  to deadlock at the VRAM boundary on Windows + WDDM when a single
  submodule approaches dedicated capacity, so prefer 4-bit or fp8 paths.

The rapid transformer ships fp8_e4m3 weights on disk (20.43 B params,
19 GB). `torch_dtype=bfloat16` at load time dequantizes to bf16 in RAM so
downstream paths (LoRA fuse, NF4 or fp8 conversion) can operate on full
precision weights before re-quantizing.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F

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
    """Build the pipeline. See module docstring for the three supported paths."""
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
    elif not enable_cpu_offload:
        # 24 GB reference path. bf16 residency does not fit (38 GB for a
        # 20.4 B-param transformer), so we halve the footprint by storing
        # weights as fp8_e4m3 and casting to bf16 just-in-time per forward.
        # See module docstring for the precision argument.
        _convert_transformer_to_fp8_residency(pipe.transformer, log)

    # VAE tiling cuts peak VRAM during decode by processing the latent in
    # spatial tiles (~200 MB peak instead of ~2 GB). Cheap to enable, only
    # helps, no quality impact at our resolutions.
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        log.info("VAE tiling enabled.")

    if enable_cpu_offload:
        log.info("enable_cpu_offload=True; enabling model CPU offload.")
        pipe.enable_model_cpu_offload()
    else:
        # Selective residency for 24 GB cards (4090): transformer + VAE stay
        # on GPU; the 15 GB Qwen2.5-VL text encoder stays on CPU. We avoid
        # `accelerate.enable_model_cpu_offload()` entirely because on Windows
        # + WDDM it deadlocks at the VRAM boundary when swapping large
        # submodules in/out. The prompt-embed call is wrapped so encoding
        # happens on CPU and only the tiny output embedding is shipped to GPU.
        # Transformer is either NF4 (~5 GB) or fp8 (~20 GB) at this point --
        # never bf16 in this branch, since bf16 (~38 GB) does not fit.
        log.info("Selective device placement: transformer+VAE -> %s, text encoder -> cpu.", device)
        pipe.transformer.to(device)
        pipe.vae.to(device)
        pipe.text_encoder.to("cpu")
        _install_cpu_text_encoder_shim(pipe, gpu_device=torch.device(device), log=log)

    log.info("Pipeline ready.")
    return pipe


def _install_cpu_text_encoder_shim(
    pipe: Any,
    *,
    gpu_device: torch.device,
    log: logging.Logger,
) -> None:
    """Wrap `pipe._get_qwen_prompt_embeds` so encoding runs on CPU.

    The original method moves `processor` outputs to `self._execution_device`
    and then calls `self.text_encoder`. When the text encoder is on CPU and
    the rest of the pipeline is on GPU, that destination is GPU and the CPU
    encoder raises a device-mismatch error. We override it to force the
    encode stage onto CPU (matching the text encoder's actual residency)
    and then ship only the output embeddings to `gpu_device` for the
    downstream transformer.
    """
    cpu = torch.device("cpu")
    original = pipe._get_qwen_prompt_embeds

    def _cpu_shim(
        prompt=None,
        image=None,
        device=None,
        dtype=None,
    ):
        # Ignore any caller-provided device; the text encoder lives on CPU.
        embeds, mask = original(prompt=prompt, image=image, device=cpu, dtype=dtype)
        return embeds.to(gpu_device), mask.to(gpu_device)

    pipe._get_qwen_prompt_embeds = _cpu_shim

    # Pin `_execution_device` to GPU. By default the property walks
    # `self.components` and may return the CPU-resident text encoder's
    # device, which would force `prepare_latents` / image-conditioning
    # tensors to CPU and crash the GPU transformer. Rebind on the class
    # (instance-level shadowing doesn't work for properties).
    type(pipe)._execution_device = property(lambda _self: gpu_device)
    log.info("Text-encoder CPU shim installed (encode on CPU, embeds -> %s).", gpu_device)


class Fp8Linear(torch.nn.Module):
    """`nn.Linear` replacement that stores its weight as `float8_e4m3fn`.

    Forward path: cast weight to `compute_dtype` (bf16) just-in-time, then
    call `F.linear`. The cast allocates a temporary of the weight's bf16
    size (typically <100 MB even for the widest Qwen-Image layer), which
    is freed after the matmul. Sequential layer execution means peak
    VRAM overhead above the fp8 residency is bounded by a single layer's
    bf16 copy + activations.

    Bias is kept in `compute_dtype` -- it's tiny and benefits from the
    extra precision on the final accumulation. Weight is requantized to
    fp8 once at construction time; the rounding error is bounded by
    fp8_e4m3fn's 3-bit mantissa and is smaller than the error already
    present in the source weights (which ship fp8 on disk).

    Why not `torch._scaled_mm`? It works on Ada but requires explicit
    per-tensor or per-row scales, a 2D input layout, and a weight
    transpose convention. The cast-and-linear path costs only ~1.5x per
    Linear (benchmarked, 0.093 ms vs 0.062 ms on a 3072^2 layer) and
    keeps the rest of the pipeline unchanged. Revisit `_scaled_mm` if
    the fp8 kernel pays off end-to-end.
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        bias: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        assert weight_fp8.dtype == torch.float8_e4m3fn, weight_fp8.dtype
        # Parameter(requires_grad=False) preserves state_dict / .to() / .cuda()
        # semantics so diffusers + accelerate utilities still work.
        self.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias.to(compute_dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.compute_dtype = compute_dtype
        self.in_features = int(weight_fp8.shape[1])
        self.out_features = int(weight_fp8.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(self.compute_dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        bias_repr = "True" if self.bias is not None else "False"
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={bias_repr}, "
            f"weight_dtype=float8_e4m3fn, compute_dtype={self.compute_dtype}"
        )


def _convert_transformer_to_fp8_residency(
    transformer: torch.nn.Module,
    log: logging.Logger,
) -> None:
    """In-place swap every `nn.Linear` in `transformer` for `Fp8Linear`.

    Mirrors the skip list used by `_quantize_transformer_4bit_nf4`: the
    input/output projections, timestep conditioning, and final adaptive
    norm stay in bf16. Per-layer error at these locations is not
    averaged over extra denoising steps on a 4-step rapid model, so the
    few hundred MB saved by quantizing them isn't worth the quality cost.

    Must be called AFTER `fuse_lora + unload_lora_weights` so the LoRA
    delta is already baked into the bf16 weights.
    """
    skip_subtree_names = {
        "proj_out",
        "img_in",
        "txt_in",
        "time_text_embed",
        "norm_out",
    }

    replaced = 0
    total = 0

    def _walk(parent: torch.nn.Module) -> None:
        nonlocal replaced, total
        for name, child in list(parent.named_children()):
            if name in skip_subtree_names:
                continue
            if isinstance(child, torch.nn.Linear):
                total += 1
                weight_fp8 = child.weight.data.to(torch.float8_e4m3fn)
                bias = child.bias.data if child.bias is not None else None
                new_layer = Fp8Linear(
                    weight_fp8=weight_fp8,
                    bias=bias,
                    compute_dtype=DTYPE,
                )
                setattr(parent, name, new_layer)
                replaced += 1
                del child
            elif isinstance(child, Fp8Linear):
                total += 1
            else:
                _walk(child)

    log.info(
        "Converting transformer Linear -> Fp8Linear (fp8_e4m3 storage, %s compute) ...",
        DTYPE,
    )
    _walk(transformer)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info(
        "fp8 residency conversion complete: replaced %d / %d Linear layers.",
        replaced,
        total,
    )


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
