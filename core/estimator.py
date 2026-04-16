import os
import re

class ProfilingData:
    def __init__(self, model_path, ngl, ctx, total_layers, parallel_slots, threads,
                 kv_offload_on, flash_attn_on, batch_size,
                 get_model_info, get_gpu_info, get_cached_metadata, extract_quant,
                 kv_cache_k_type="q4_0", kv_cache_v_type="q4_0"):
        self.model_path = model_path
        self.ngl = ngl
        self.ctx = ctx
        self.total_layers = total_layers
        self.parallel_slots = parallel_slots
        self.threads = threads
        self.kv_offload_on = kv_offload_on
        self.flash_attn_on = flash_attn_on
        self.batch_size = batch_size
        self.kv_cache_k_type = kv_cache_k_type
        self.kv_cache_v_type = kv_cache_v_type
        
        self.get_model_info = get_model_info
        self.get_gpu_info = get_gpu_info
        self.get_cached_metadata = get_cached_metadata
        self.extract_quant = extract_quant

def calculate_max_ngl(vram, total_layers, model_gb):
    if vram == 0:
        return 0
    
    available_vram = vram / 1024
    
    # Baseline reserve: 0.5GB for OS/System, + 0.5GB for basic KV cache
    reserve_gb = 1.0
    usable_vram = max(available_vram - reserve_gb, 0.1)
    
    if model_gb is None or model_gb <= 0:
        # Fallback: assume roughly 200MB per layer for medium models
        estimated_per_layer = 0.2
    else:
        estimated_per_layer = model_gb / max(total_layers, 1)
    
    max_layers = int(usable_vram / estimated_per_layer) if estimated_per_layer > 0 else 0
    return max(0, min(max_layers, total_layers))

def _coerce_int_metadata(metadata, key):
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        # Игнорируем строки "true"/"false" (регистронезависимо)
        if value.lower() in ("true", "false"):
            return None
        match = re.search(r"-?\d+", value)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                return None
    return None

def estimate_memory_breakdown(p: ProfilingData):
    model_size_mb, __ = p.get_model_info(p.model_path)
    metadata = p.get_cached_metadata(p.model_path)

    arch = metadata.get("general.architecture")
    embedding = None
    head_count = None
    head_count_kv = None
    key_length = None
    if arch:
        embedding = _coerce_int_metadata(metadata, f"{arch}.embedding_length")
        head_count = _coerce_int_metadata(metadata, f"{arch}.attention.head_count")
        head_count_kv = _coerce_int_metadata(metadata, f"{arch}.attention.head_count_kv")
        key_length = _coerce_int_metadata(metadata, f"{arch}.attention.key_length")
    
    embedding = embedding or _coerce_int_metadata(metadata, "llama.embedding_length") or 4096
    head_count = head_count or _coerce_int_metadata(metadata, "llama.attention.head_count") or 32
    head_count_kv = head_count_kv or _coerce_int_metadata(metadata, "llama.attention.head_count_kv") or head_count
    
    actual_head_size = key_length or (embedding / max(head_count, 1))
    
    type_bytes = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}
    bytes_k = type_bytes.get(p.kv_cache_k_type, 0.5)
    bytes_v = type_bytes.get(p.kv_cache_v_type, 0.5)
    
    bytes_per_token_per_layer = head_count_kv * actual_head_size * (bytes_k + bytes_v)
    kv_multiplier = 0.7 if p.flash_attn_on else 1.0
    ctx_kv_mb = p.ctx * p.total_layers * bytes_per_token_per_layer / (1024 * 1024) * kv_multiplier * p.parallel_slots
    ctx_state_mb = max(128.0, p.ctx * embedding * 0.0000025)

    offload_pct = min(max(p.ngl / p.total_layers, 0), 1) if p.total_layers > 0 else 0
    gpu_weights_mb = model_size_mb * offload_pct if p.ngl > 0 else 0
    ram_weights_mb = max(model_size_mb - gpu_weights_mb, 0)

    gpu_kv_mb = ctx_kv_mb if (p.kv_offload_on and p.ngl > 0) else 0
    ram_kv_mb = 0 if (p.kv_offload_on and p.ngl > 0) else ctx_kv_mb

    gpu_total_mb = gpu_weights_mb + gpu_kv_mb
    ram_total_mb = ram_weights_mb + ram_kv_mb + ctx_state_mb

    return {
        "gpu_total_mb": gpu_total_mb,
        "ram_total_mb": ram_total_mb,
        "gpu_weights_mb": gpu_weights_mb,
        "ram_weights_mb": ram_weights_mb,
        "gpu_kv_mb": gpu_kv_mb,
        "ram_kv_mb": ram_kv_mb,
        "ctx_state_mb": ctx_state_mb,
        "parallel_slots": p.parallel_slots,
        "kv_offload_on": p.kv_offload_on,
        "flash_attn_on": p.flash_attn_on,
        "offload_pct": offload_pct,
    }

def estimate_model_params_b(metadata, model_path, model_size_gb, extract_quant):
    raw_param_count = metadata.get("general.parameter_count")
    if raw_param_count:
        try:
            return max(float(raw_param_count) / 1e9, 0.1)
        except Exception:
            pass

    size_label = metadata.get("general.size_label")
    if isinstance(size_label, str):
        text = size_label.upper()
        if "B" in text:
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                return max(float(match.group(1)), 0.1)

    model_name = os.path.basename(model_path).upper()
    moe_match = re.search(r"(\d+)X(\d+(?:\.\d+)?)B", model_name)
    if moe_match:
        experts = float(moe_match.group(1))
        expert_size = float(moe_match.group(2))
        return max(experts * expert_size, expert_size)

    param_match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    if param_match:
        return max(float(param_match.group(1)), 0.1)

    million_match = re.search(r"(\d+(?:\.\d+)?)M", model_name)
    if million_match:
        return max(float(million_match.group(1)) / 1000.0, 0.05)

    quant = extract_quant(model_path)
    bits_guess = {
        "IQ1_M": 1.75, "IQ1_S": 1.56, "Q1_0": 1.125, "Q1_K": 1.45,
        "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "TQ1_0": 1.35, "TQ2_0": 2.0,
        "IQ3_XXS": 3.06, "IQ3_S": 3.44, "IQ4_NL": 4.1, "IQ4_XS": 4.25,
        "Q2_K": 2.63, "Q3_K_S": 3.44, "Q3_K_M": 3.44, "Q3_K_L": 3.44,
        "Q4_0": 4.0, "Q4_1": 4.0, "Q4_K_S": 4.5, "Q4_K_M": 4.5,
        "Q5_0": 5.0, "Q5_1": 5.0, "Q5_K_S": 5.5, "Q5_K_M": 5.5, "MXFP4": 4.0,
        "Q6_K": 6.56, "Q8_0": 8.0, "Q8_K": 8.0, "F16": 16.0, "BF16": 16.0, "F32": 32.0
    }.get(quant, 4.5)
    approx_params = (model_size_gb * 8) / max(bits_guess, 1.0)
    return max(approx_params, 0.1)

def get_quant_speed_factor(model_path, extract_quant):
    quant = extract_quant(model_path)
    return {
        "IQ1_M": 1.25, "IQ1_S": 1.28, "Q1_0": 1.38, "Q1_K": 1.34,
        "IQ2_XXS": 1.22, "IQ2_XS": 1.18, "IQ2_S": 1.15, "TQ1_0": 1.32, "TQ2_0": 1.2,
        "IQ3_XXS": 1.12, "IQ3_S": 1.08, "IQ4_NL": 1.03, "IQ4_XS": 1.04,
        "Q2_K": 1.16, "Q3_K_S": 1.1, "Q3_K_M": 1.07, "Q3_K_L": 1.02,
        "Q4_0": 1.0, "Q4_1": 0.98, "Q4_K_S": 1.0, "Q4_K_M": 0.97,
        "Q5_0": 0.9, "Q5_1": 0.88, "Q5_K_S": 0.9, "Q5_K_M": 0.87, "MXFP4": 0.99,
        "Q6_K": 0.8, "Q8_0": 0.68, "Q8_K": 0.65, "F16": 0.55, "BF16": 0.58, "F32": 0.32
    }.get(quant, 1.0)

def _estimate_decode_tps(params_b, quant_factor, offload_pct, threads, ctx_factor,
                         flash_attn_on, kv_offload_on, vram_total_mb, ctx_load_factor):
    gpu_class = 0
    if vram_total_mb > 0:
        # Dynamic GPU class based on VRAM (roughly correlates with memory bandwidth)
        # 24GB (3090/4090) -> 175, 16GB (4080) -> 130, 12GB (3060/4070) -> 95, etc.
        if vram_total_mb >= 24000: gpu_class = 175
        elif vram_total_mb >= 20000: gpu_class = 150
        elif vram_total_mb >= 16000: gpu_class = 130
        elif vram_total_mb >= 12000: gpu_class = 95
        elif vram_total_mb >= 10000: gpu_class = 80
        elif vram_total_mb >= 8000: gpu_class = 70
        elif vram_total_mb >= 6000: gpu_class = 48
        else: gpu_class = 30

    small_model_boost = 1.0
    if params_b <= 0.75: small_model_boost = 2.2  # Повышено для соответствия реальности на 0.6B моделях
    elif params_b <= 1.5: small_model_boost = 1.32
    elif params_b <= 3.0: small_model_boost = 1.14

    fa_bonus = 1.12 if flash_attn_on else 1.0
    kv_bonus = 1.05 if kv_offload_on and offload_pct > 0 else 1.0
    ctx_decay = max(0.52, 1.02 - 0.42 * ctx_load_factor)

    gpu_tps = 0.0
    if gpu_class > 0 and offload_pct > 0:
        gpu_fit_penalty = min(max(offload_pct, 0.18), 1.0) ** 0.72
        gpu_tps = gpu_class * quant_factor * ctx_factor * ctx_decay * fa_bonus * kv_bonus
        gpu_tps *= small_model_boost / (params_b ** 0.63)
        gpu_tps *= gpu_fit_penalty

    effective_threads = min(max(threads, 1), 16)
    cpu_parallel = 0.72 + 0.28 * min(effective_threads / 8.0, 1.0)
    cpu_tps = (3.2 * effective_threads * cpu_parallel * quant_factor * ctx_factor * ctx_decay)
    cpu_tps *= small_model_boost / (params_b ** 0.88)

    if offload_pct >= 0.98 and gpu_tps > 0: return gpu_tps
    if offload_pct <= 0 or gpu_tps <= 0: return cpu_tps
    return 1.0 / ((offload_pct / gpu_tps) + ((1.0 - offload_pct) / cpu_tps))

def estimate_tokens_per_second(p: ProfilingData):
    __, model_size_gb = p.get_model_info(p.model_path)
    if model_size_gb is None: return None

    metadata = p.get_cached_metadata(p.model_path)
    offload_pct = min(max(p.ngl / p.total_layers, 0), 1) if p.total_layers > 0 else 0
    vram_total_mb = p.get_gpu_info()
    params_b = estimate_model_params_b(metadata, p.model_path, model_size_gb, p.extract_quant)
    quant_factor = get_quant_speed_factor(p.model_path, p.extract_quant)
    
    ctx_factor = min(1.18, (4096 / max(p.ctx, 512)) ** 0.14)
    ctx_factor = max(ctx_factor, 0.84)

    high_tps = _estimate_decode_tps(
        params_b, quant_factor, offload_pct, p.threads, ctx_factor,
        p.flash_attn_on, p.kv_offload_on, vram_total_mb, 0.08
    )
    low_tps = _estimate_decode_tps(
        params_b, quant_factor, offload_pct, p.threads, ctx_factor,
        p.flash_attn_on, p.kv_offload_on, vram_total_mb, 0.92
    )

    batch_factor = 1.0
    if p.batch_size < 256: batch_factor = 0.95
    elif p.batch_size > 4096: batch_factor = 0.97
    high_tps *= batch_factor
    low_tps *= batch_factor

    high_tps = max(0.1, min(high_tps, 600))
    low_tps = max(0.1, min(low_tps, high_tps))
    best_tps = max(high_tps * 1.08, high_tps)

    return {
        "low": round(low_tps, 1),
        "high": round(high_tps, 1),
        "best": round(min(best_tps, 650), 1),
        "params_b": round(params_b, 2),
        "quant_factor": quant_factor,
    }
