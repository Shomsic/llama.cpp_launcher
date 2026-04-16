import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from core.gguf_parser import GGUFMetadataParser
from core import hardware

@dataclass
class ServerConfig:
    model_path: str
    port: int = 8081
    ctx_size: int = 65536
    kv_cache_k_type: str = "q4_0"
    kv_cache_v_type: str = "q4_0"
    gpu_filter: Optional[str] = None
    ram_budget_mb: int = 0
    backend: str = "auto"  # auto, llama, ik_llama
    keep_alive: bool = False
    mmproj_path: Optional[str] = None
    
    # User-defined sampling and other params
    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    mirostat: int = 0
    n_predict: int = -1
    seed: int = -1
    parallel_slots: int = 1
    threads: Optional[int] = None
    batch_size: int = 512
    flash_attn: str = "auto"
    mmap: str = "on"
    mlock: str = "off"
    kv_offload: str = "on"
    cache_prompt: str = "on"
    reasoning: str = "off"
    rope_scale: str = "auto"
    rope_freq_base: str = "auto"
    draft_model_path: Optional[str] = None
    draft_ngl: int = 0
    custom_args: str = ""
    
    # Resulting flags
    flags: List[str] = field(default_factory=list)
    binary_path: Optional[str] = None
    is_ik_llama: bool = False

class FlagBuilder:
    SYSTEM_HEADROOM_MB = 5120
    COMPUTE_PER_GPU_MB = 512
    MIN_CRAM_MB = 512
    VRAM_OVERHEAD_PERCENT = 130
    SINGLE_GPU_HEADROOM_MB = 4096

    def __init__(self, hardware_info):
        self.hw = hardware_info

    def build(self, config: ServerConfig) -> ServerConfig:
        # 1. Binary Selection
        config.binary_path = self._find_binary(config.backend)
        if not config.binary_path:
            raise RuntimeError("llama-server binary not found")
        
        config.is_ik_llama = "ik_llama" in config.binary_path or self._check_ik_llama(config.binary_path)
        
        # 2. GGUF Metadata
        metadata = self._parse_gguf(config.model_path)
        
        # Define KV cache type key
        kv_type = f"{config.kv_cache_k_type}_{config.kv_cache_v_type}"
        
        # 3. Hardware & Model Analysis
        vram_free = hardware.get_total_vram_free()
        ram_avail = hardware.get_available_ram_mb()
        
        if config.ram_budget_mb > 0 and config.ram_budget_mb < ram_avail:
            ram_avail = config.ram_budget_mb
            
        model_size_mb = metadata['size_mb']
        total_mem_mb = vram_free + ram_avail
        
        # 4. Basic Flags
        flags = [
            "-m", config.model_path,
            "--port", str(config.port),
            "--ctx-size", str(config.ctx_size),
            "--jinja",
        ]
        
        # User-defined sampling and params
        if config.temp != 0.8: flags += ["--temp", str(config.temp)]
        if config.top_k != 40: flags += ["--top-k", str(config.top_k)]
        if config.top_p != 0.95: flags += ["--top-p", str(config.top_p)]
        if config.min_p != 0.05: flags += ["--min-p", str(config.min_p)]
        if config.repeat_penalty != 1.1: flags += ["--repeat-penalty", str(config.repeat_penalty)]
        if config.presence_penalty != 0.0: flags += ["--presence-penalty", str(config.presence_penalty)]
        if config.frequency_penalty != 0.0: flags += ["--frequency-penalty", str(config.frequency_penalty)]
        if config.mirostat != 0: flags += ["--mirostat", str(config.mirostat)]
        if config.n_predict != -1: flags += ["-n", str(config.n_predict)]
        if config.seed != -1: flags += ["--seed", str(config.seed)]
        if config.parallel_slots != 1: flags += ["--parallel", str(config.parallel_slots)]
        
        if config.flash_attn == "on": flags += ["--flash-attn", "on"]
        elif config.flash_attn == "off": flags += ["--no-flash-attn"]
        
        if config.mmap == "off": flags += ["--no-mmap"]
        if config.mlock == "on": flags += ["--mlock"]
        if config.kv_offload == "off": flags += ["--no-kv-offload"]
        if config.cache_prompt == "off": flags += ["--no-cache-prompt"]
        if config.reasoning == "on": flags += ["--reasoning"]
        if config.keep_alive: flags += ["--keep-alive"]
        
        if config.rope_scale != "auto": flags += ["--rope-scale", str(config.rope_scale)]
        if config.rope_freq_base != "auto": flags += ["--rope-freq-base", str(config.rope_freq_base)]
        
        if config.draft_model_path:
            flags += ["--draft", config.draft_model_path]
            if config.draft_ngl > 0:
                flags += ["--draft-ngl", str(config.draft_ngl)]
        
        if config.custom_args:
            flags += config.custom_args.split()
        
        if config.mmproj_path:
            flags += ["--mmproj", config.mmproj_path]

        
        # 5. Batch Sizes
        if config.batch_size == 512:
            fits_on_gpu = (model_size_mb * self.VRAM_OVERHEAD_PERCENT / 100) <= vram_free
            if fits_on_gpu and hardware.get_best_gpu_vram() > (model_size_mb + self.SINGLE_GPU_HEADROOM_MB):
                batch, ubatch = 8192, 1024
            elif fits_on_gpu:
                batch, ubatch = 4096, 512
            else:
                batch, ubatch = 2048, 512
            flags += ["-b", str(batch), "-ub", str(ubatch)]
        else:
            flags += ["-b", str(config.batch_size), "-ub", "512"]
        
        # 6. KV Cache
        flags += ["--cache-type-k", config.kv_cache_k_type, "--cache-type-v", config.kv_cache_v_type]
        
        # 7. Threads
        cores = config.threads if config.threads is not None else hardware.get_cpu_cores()
        flags += ["--threads", str(cores), "--threads-batch", str(cores)]
        
        # 8. ik_llama Specifics
        if config.is_ik_llama:
            flags += ["--run-time-repack", "--defrag-thold", "0.1"]
            if metadata['is_moe']:
                flags += ["-muge", "-ger"]
            if metadata.get('has_fused'):
                flags += ["-fused", "1"]
            if vram_free > 0:
                flags += ["-mqkv"]
                
            # Prompt cache / checkpoints
            if hardware.get_gpu_count() > 1:
                model_on_gpu_mb = min(model_size_mb * self.VRAM_OVERHEAD_PERCENT / 100, vram_free)
                vram_headroom = vram_free - model_on_gpu_mb - (metadata['kv_size_mb'][kv_type]) - (self.COMPUTE_PER_GPU_MB * hardware.get_gpu_count())
                vram_headroom = max(0, vram_headroom)
                cache_ram_mb = min(vram_headroom // 2, 4096)
                
                if cache_ram_mb < 256:
                    flags += ["-cram", "0", "--ctx-checkpoints", "0"]
                else:
                    checkpoints = max(2, min(16, cache_ram_mb // 200))
                    flags += ["-cram", str(cache_ram_mb), "--ctx-checkpoints", str(checkpoints)]
            else:
                cram = min(ram_avail // 10, 16384)
                if cram >= self.MIN_CRAM_MB:
                    flags += ["-cram", str(cram)]
        
        # 9. GPU Offloading & Tensor Splitting
        gpu_list = hardware.get_gpu_list()
        gpu_count = len(gpu_list)
        
        if gpu_count > 0:
            # Sort GPUs by bandwidth score (descending)
            sorted_indices = sorted(range(gpu_count), key=lambda i: gpu_list[i].get('bandwidth', 1), reverse=True)
            
            # Predict required VRAM
            # (model + overhead) + KV + compute workspace
            kv_size = metadata['kv_size_mb'].get(kv_type, 1024)
            model_with_overhead = model_size_mb * self.VRAM_OVERHEAD_PERCENT / 100
            needed_mb = model_with_overhead + kv_size + (self.COMPUTE_PER_GPU_MB * gpu_count)

            # Find smallest subset of fastest GPUs
            use_count = gpu_count
            for n in range(1, gpu_count + 1):
                subset_vram = sum(gpu_list[sorted_indices[i]]['vram_free'] for i in range(n))
                if needed_mb <= subset_vram:
                    use_count = n
                    break
            
            selected_gpu_indices = sorted_indices[:use_count]
            
            # Calculate tensor weights (vram_free * bandwidth)
            weights = []
            total_weighted = 0
            for i in range(gpu_count):
                if i in selected_gpu_indices:
                    w = gpu_list[i]['vram_free'] * gpu_list[i].get('bandwidth', 1)
                    weights.append(w)
                    total_weighted += w
                else:
                    weights.append(0)
            
            # Add flags
            flags += ["-ngl", "999"]
            
            best_gpu_index = gpu_list[sorted_indices[0]].get('index', 0)
            if config.gpu_filter and config.gpu_filter.isdigit():
                best_gpu_index = int(config.gpu_filter)
            
            flags += ["-mg", str(best_gpu_index)]
            
            if gpu_count > 1:
                # Built weighted split string
                if total_weighted > 0:
                    ts_string = ",".join([f"{w/total_weighted:.2f}" for w in weights])
                    flags += ["--tensor-split", ts_string]
                
                if config.is_ik_llama:
                    flags += ["--split-mode", "graph"]
            
        # 10. SSM/Mamba
        if metadata['has_ssm']:
            flags += ["--no-context-shift"]
        
        config.flags = flags
        return config

    def _find_binary(self, backend):
        # 1. Use AppState's method first
        if hasattr(self.hw, 'get_server_exe_path'):
            exe_path = self.hw.get_server_exe_path()
            if exe_path:
                norm_path = os.path.normpath(exe_path)
                if os.path.exists(norm_path):
                    return norm_path
                print(f"[FlagBuilder] AppState suggested {exe_path}, but it doesn't exist.")
            else:
                print(f"[FlagBuilder] AppState.get_server_exe_path() returned None.")
        
        # 2. Manual search as fallback
        llama_dir = getattr(self.hw, 'llama_dir_var', None)
        base_path = ""
        if llama_dir and hasattr(llama_dir, 'get'):
            base_path = os.path.normpath(llama_dir.get())

        bin_name = "llama-server.exe" if os.name == 'nt' else "llama-server"
        
        candidates = []
        if base_path:
            candidates.append(os.path.join(base_path, bin_name))
            candidates.append(os.path.join(base_path, "bin", bin_name))
            candidates.append(os.path.join(base_path, "build", "bin", bin_name))
        
        candidates.extend([
            os.path.expanduser(f"~/ik_llama.cpp/build/bin/{bin_name}"),
            os.path.expanduser(f"~/llama.cpp/build/bin/{bin_name}"),
        ])
        
        for bin_path in candidates:
            norm_path = os.path.normpath(bin_path)
            if os.path.exists(norm_path):
                return norm_path
            print(f"[FlagBuilder] Candidate {norm_path} not found.")
        
        try:
            import shutil
            path_bin = shutil.which(bin_name)
            if path_bin: return os.path.normpath(path_bin)
        except: pass
        
        return None

    def _check_ik_llama(self, bin_path):
        try:
            result = subprocess.run([bin_path, "--help"], capture_output=True, text=True, timeout=5)
            return "ikawrakow" in result.stdout or "split-mode-graph" in result.stdout
        except:
            return False

    def _parse_gguf(self, path):
        parser = GGUFMetadataParser()
        full_metadata = parser.get_cached_or_parse(path)
        
        size_mb = 0
        try:
            size_mb = os.path.getsize(path) // (1024 * 1024)
        except: pass

        # Check for fused up|gate tensors (ik_llama optimization)
        has_fused = False
        try:
            with open(path, 'rb') as f:
                # Fast scan: fused tensor names usually appear in the first 32KB
                sample = f.read(1024 * 32).decode('utf-8', errors='ignore')
                if 'ffn_up_gate' in sample or 'ffn_gate_up' in sample:
                    has_fused = True
        except: pass

        # Normalize metadata for the builder
        return {
            'size_mb': size_mb,
            'is_moe': full_metadata.get('general.architecture', '').lower().startswith('moe') or 'moe' in full_metadata.get('general.architecture', '').lower(),
            'has_ssm': 'mamba' in full_metadata.get('general.architecture', '').lower(),
            'has_fused': has_fused,
            'kv_size_mb': {
                'f16': full_metadata.get('kv_size_mb_f16', 1024), 
                'q8_0': full_metadata.get('kv_size_mb_q8', 512), 
                'q4_0': full_metadata.get('kv_size_mb_q4', 256)
            }
        }
