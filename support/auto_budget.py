from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .gguf_layers import get_layer_count, read_metadata

GIB = 1024 ** 3
AUTO_N_CTX = -1
QWEN35_MIN_CTX = 16384
DEFAULT_AUTO_CTX = 8192
MODEL_GPU_MULTIPLIER = 1.08
MMPROJ_GPU_MULTIPLIER = 1.10
CTX_CANDIDATES = (
    262144,
    196608,
    131072,
    98304,
    65536,
    49152,
    32768,
    24576,
    16384,
    12288,
    8192,
    4096,
)


@dataclass(frozen=True)
class AutoBudgetAttempt:
    n_ctx: int
    n_gpu_layers: int
    estimated_layers: int
    model_gpu_gb: float
    kv_cache_gb: float


@dataclass(frozen=True)
class AutoBudgetPlan:
    n_ctx: int
    n_gpu_layers: int
    gguf_layers: int
    auto_n_ctx: bool
    auto_vram: bool
    free_gb: Optional[float]
    total_gb: Optional[float]
    model_full_gpu_gb: float
    mmproj_gpu_gb: float
    kv_cache_gb: float
    overhead_gb: float
    attempts: tuple[AutoBudgetAttempt, ...]


def is_auto_n_ctx(n_ctx: int) -> bool:
    try:
        return int(n_ctx) <= 0
    except Exception:
        return False


def normalize_n_ctx_for_chat_handler(chat_handler: str, n_ctx: int, qwen35_handlers: set[str]) -> int:
    if is_auto_n_ctx(n_ctx):
        return AUTO_N_CTX
    if chat_handler in qwen35_handlers and n_ctx < QWEN35_MIN_CTX:
        print(
            f"[llama-cpp_vlm] Qwen3.5 family needs a larger context. "
            f"Bumping n_ctx from {n_ctx} to {QWEN35_MIN_CTX} to avoid context-shift errors."
        )
        return QWEN35_MIN_CTX
    return n_ctx


def _metadata_value(meta: dict, suffix: str):
    suffix = suffix.lower()
    arch = meta.get("general.architecture")
    if arch:
        exact = f"{arch}.{suffix}"
        if exact in meta:
            return meta[exact]
    for key, value in meta.items():
        lowered = key.lower()
        if lowered == suffix or lowered.endswith(f".{suffix}"):
            return value
    return None


def _metadata_int(meta: dict, suffix: str, default: Optional[int] = None) -> Optional[int]:
    value = _metadata_value(meta, suffix)
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        return int(value)
    except Exception:
        return default


def _read_metadata_safely(model_path: str) -> dict:
    try:
        return read_metadata(model_path)
    except Exception:
        return {}


def _layer_count(model_path: str, meta: dict) -> int:
    layer_count = _metadata_int(meta, "block_count")
    if layer_count:
        return layer_count
    try:
        return get_layer_count(model_path) or 32
    except Exception:
        return 32


def _estimated_gpu_file_size_gb(path: Optional[str], multiplier: float) -> float:
    if not path:
        return 0.0
    try:
        return os.path.getsize(path) * multiplier / GIB
    except OSError:
        return 0.0


def estimate_kv_cache_gb(meta: dict, n_ctx: int, layer_count: int, safety: float = 1.20) -> float:
    head_count = _metadata_int(meta, "attention.head_count")
    head_count_kv = _metadata_int(meta, "attention.head_count_kv", head_count)
    embedding_length = _metadata_int(meta, "embedding_length")
    key_length = _metadata_int(meta, "attention.key_length")
    value_length = _metadata_int(meta, "attention.value_length")

    if head_count and head_count_kv and embedding_length:
        head_dim = max(1, int(round(embedding_length / head_count)))
        key_length = key_length or head_dim
        value_length = value_length or head_dim
        bytes_per_token = layer_count * head_count_kv * (key_length + value_length) * 2
    else:
        bytes_per_token = layer_count * 8192

    return bytes_per_token * max(1, n_ctx) * safety / GIB


def _runtime_overhead_gb(is_qwen35_family: bool, has_mmproj: bool, total_gb: Optional[float]) -> float:
    overhead = 0.75
    if is_qwen35_family:
        overhead += 0.25
    if has_mmproj:
        overhead += 0.25
    if total_gb and total_gb >= 23:
        overhead += 0.25
    return overhead


def _usable_free_gb(cuda_memory: Optional[tuple[float, float]]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not cuda_memory:
        return None, None, None
    free_gb, total_gb = cuda_memory
    if free_gb is None:
        return None, total_gb, None
    visible_free = min(free_gb, total_gb) if total_gb else free_gb
    return free_gb, total_gb, max(0.0, visible_free - 0.50)


def _auto_max_ctx(is_qwen35_family: bool, total_gb: Optional[float], auto_max_ctx: Optional[int]) -> int:
    default = 65536 if is_qwen35_family and (total_gb is None or total_gb >= 20) else 32768
    if auto_max_ctx is None:
        return default
    try:
        value = int(auto_max_ctx)
    except Exception:
        return default
    return max(4096, min(327680, value))


def _effective_manual_vram_limit(vram_limit: int, total_gb: Optional[float], usable_gb: Optional[float]) -> int:
    if vram_limit == -1 or total_gb is None:
        return vram_limit
    reserve_gb = 2.0 if total_gb >= 20 else max(1.0, total_gb * 0.10)
    physical_cap = max(1, int(total_gb - reserve_gb))
    if usable_gb is not None:
        physical_cap = min(physical_cap, max(1, int(usable_gb)))
    if vram_limit > physical_cap:
        print(
            f"[llama-cpp_vlm] Clamping vram_limit from {vram_limit} GB to {physical_cap} GB "
            f"based on current GPU memory."
        )
        return physical_cap
    return vram_limit


def _minimum_ctx(is_qwen35_family: bool) -> int:
    return QWEN35_MIN_CTX if is_qwen35_family else DEFAULT_AUTO_CTX


def _minimum_gpu_layers(layer_count: int, is_qwen35_family: bool, total_gb: Optional[float]) -> int:
    if not is_qwen35_family:
        return 1
    if total_gb and total_gb < 16:
        return min(layer_count, 4)
    return min(layer_count, 8)


def _context_candidates(min_ctx: int, max_ctx: int) -> tuple[int, ...]:
    values = {ctx for ctx in CTX_CANDIDATES if min_ctx <= ctx <= max_ctx}
    if max_ctx >= min_ctx:
        values.add(max_ctx)
    values.add(min_ctx)
    return tuple(sorted(values, reverse=True))


def _attempt_from_layers(
    n_ctx: int,
    n_gpu_layers: int,
    estimated_layers: int,
    layer_count: int,
    model_layer_gb: float,
    kv_layer_gb: float,
) -> AutoBudgetAttempt:
    offloaded_layers = layer_count if n_gpu_layers == -1 else max(1, min(layer_count, n_gpu_layers))
    return AutoBudgetAttempt(
        n_ctx=max(1, int(n_ctx)),
        n_gpu_layers=-1 if offloaded_layers >= layer_count else offloaded_layers,
        estimated_layers=max(0, int(estimated_layers)),
        model_gpu_gb=model_layer_gb * offloaded_layers,
        kv_cache_gb=kv_layer_gb * offloaded_layers,
    )


def _gpu_layers_from_total_budget(
    total_budget_gb: float,
    mmproj_gpu_gb: float,
    overhead_gb: float,
    layer_count: int,
    model_layer_gb: float,
    kv_layer_gb: float,
) -> tuple[int, int]:
    per_layer_gb = model_layer_gb + kv_layer_gb
    if per_layer_gb <= 0:
        return -1, layer_count
    layer_budget_gb = total_budget_gb - mmproj_gpu_gb - overhead_gb
    estimated_layers = int(layer_budget_gb / per_layer_gb)
    if estimated_layers >= layer_count:
        return -1, layer_count
    return max(1, estimated_layers), estimated_layers


def _fallback_attempts_for_context(
    base: AutoBudgetAttempt,
    layer_count: int,
    model_layer_gb: float,
    kv_layer_gb: float,
) -> list[AutoBudgetAttempt]:
    base_layers = layer_count if base.n_gpu_layers == -1 else base.n_gpu_layers
    fallbacks = []
    seen = {base_layers}
    for scale in (0.90, 0.80, 0.65):
        layers = max(1, int(base_layers * scale))
        if layers in seen:
            continue
        seen.add(layers)
        fallbacks.append(
            _attempt_from_layers(
                base.n_ctx,
                layers,
                layers,
                layer_count,
                model_layer_gb,
                kv_layer_gb,
            )
        )
    return fallbacks


def _attempt_for_context(
    n_ctx: int,
    meta: dict,
    layer_count: int,
    model_layer_gb: float,
    mmproj_gpu_gb: float,
    overhead_gb: float,
    usable_gb: Optional[float],
    vram_limit: int,
    auto_vram: bool,
) -> AutoBudgetAttempt:
    full_kv_cache_gb = estimate_kv_cache_gb(meta, n_ctx, layer_count)
    kv_layer_gb = full_kv_cache_gb / max(1, layer_count)
    if auto_vram and usable_gb is not None:
        total_budget_gb = usable_gb
    elif vram_limit != -1:
        total_budget_gb = vram_limit
    else:
        total_budget_gb = mmproj_gpu_gb + overhead_gb + (model_layer_gb + kv_layer_gb) * layer_count

    n_gpu_layers, estimated_layers = _gpu_layers_from_total_budget(
        total_budget_gb,
        mmproj_gpu_gb,
        overhead_gb,
        layer_count,
        model_layer_gb,
        kv_layer_gb,
    )
    return _attempt_from_layers(
        n_ctx,
        n_gpu_layers,
        estimated_layers,
        layer_count,
        model_layer_gb,
        kv_layer_gb,
    )


def resolve_auto_budget(
    model_path: str,
    mmproj_path: Optional[str],
    chat_handler: str,
    n_ctx: int,
    vram_limit: int,
    auto_max_ctx: Optional[int],
    is_qwen35_family: bool,
    cuda_memory: Optional[tuple[float, float]],
) -> AutoBudgetPlan:
    meta = _read_metadata_safely(model_path)
    layer_count = _layer_count(model_path, meta)
    model_full_gpu_gb = _estimated_gpu_file_size_gb(model_path, MODEL_GPU_MULTIPLIER)
    model_layer_gb = model_full_gpu_gb / max(1, layer_count)
    mmproj_gpu_gb = _estimated_gpu_file_size_gb(mmproj_path, MMPROJ_GPU_MULTIPLIER)
    has_mmproj = bool(mmproj_path)
    requested_auto_n_ctx = is_auto_n_ctx(n_ctx)
    free_gb, total_gb, usable_gb = _usable_free_gb(cuda_memory)
    auto_vram = vram_limit == -1 and is_qwen35_family and usable_gb is not None
    min_ctx = _minimum_ctx(is_qwen35_family)
    overhead_gb = _runtime_overhead_gb(is_qwen35_family, has_mmproj, total_gb)

    max_ctx = _auto_max_ctx(is_qwen35_family, total_gb, auto_max_ctx)
    effective_vram_limit = _effective_manual_vram_limit(vram_limit, total_gb, usable_gb)

    if requested_auto_n_ctx:
        if usable_gb is None and not auto_vram:
            max_ctx = min_ctx
        candidates = _context_candidates(min_ctx, max_ctx)
    else:
        if is_qwen35_family:
            fixed_ctx = max(min_ctx, int(n_ctx))
            if fixed_ctx > max_ctx:
                print(
                    f"[llama-cpp_vlm] Clamping n_ctx from {fixed_ctx} to auto_max_ctx={max_ctx} "
                    f"for Qwen3.5 family."
                )
                fixed_ctx = max_ctx
            candidates = _context_candidates(min_ctx, fixed_ctx)
        else:
            candidates = (max(1, int(n_ctx)),)

    min_gpu_layers = _minimum_gpu_layers(layer_count, is_qwen35_family, total_gb)
    attempts = []
    for ctx in candidates:
        base_attempt = _attempt_for_context(
            ctx,
            meta,
            layer_count,
            model_layer_gb,
            mmproj_gpu_gb,
            overhead_gb,
            usable_gb,
            effective_vram_limit,
            auto_vram,
        )
        attempts.append(base_attempt)
        if is_qwen35_family:
            full_kv_cache_gb = estimate_kv_cache_gb(meta, ctx, layer_count)
            attempts.extend(
                _fallback_attempts_for_context(
                    base_attempt,
                    layer_count,
                    model_layer_gb,
                    full_kv_cache_gb / max(1, layer_count),
                )
            )
        if not requested_auto_n_ctx and not is_qwen35_family:
            break
        if not auto_vram and requested_auto_n_ctx and (vram_limit == -1 or usable_gb is None):
            break

    if not attempts:
        attempts.append(
            _attempt_for_context(
                min_ctx,
                meta,
                layer_count,
                model_layer_gb,
                mmproj_gpu_gb,
                overhead_gb,
                usable_gb,
                effective_vram_limit,
                auto_vram,
            )
        )

    selected_index = 0
    if requested_auto_n_ctx and auto_vram:
        for idx, attempt in enumerate(attempts):
            if attempt.estimated_layers >= min_gpu_layers:
                selected_index = idx
                break
        else:
            selected_index = len(attempts) - 1
    elif requested_auto_n_ctx and effective_vram_limit != -1 and usable_gb is not None:
        for idx, attempt in enumerate(attempts):
            actual_gpu_gb = mmproj_gpu_gb + overhead_gb + attempt.model_gpu_gb + attempt.kv_cache_gb
            if actual_gpu_gb <= usable_gb:
                selected_index = idx
                break
        else:
            selected_index = len(attempts) - 1
    selected = attempts[selected_index]
    retry_attempts = tuple(attempts[selected_index:])

    return AutoBudgetPlan(
        n_ctx=selected.n_ctx,
        n_gpu_layers=selected.n_gpu_layers,
        gguf_layers=layer_count,
        auto_n_ctx=requested_auto_n_ctx,
        auto_vram=auto_vram,
        free_gb=free_gb,
        total_gb=total_gb,
        model_full_gpu_gb=model_full_gpu_gb,
        mmproj_gpu_gb=mmproj_gpu_gb,
        kv_cache_gb=selected.kv_cache_gb,
        overhead_gb=overhead_gb,
        attempts=retry_attempts or (selected,),
    )


def is_context_creation_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "failed to create llama context" in text
        or "out of memory" in text
        or "cuda out of memory" in text
        or ("cuda" in text and "memory" in text)
    )
