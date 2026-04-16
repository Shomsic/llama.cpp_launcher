import json
import os
import re
import time
import hashlib
import requests
import copy
import socket
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from core.hardware import get_gpu_list, get_total_ram_gb, get_gpu_count, get_total_vram_free, get_available_ram_mb
from core.flag_builder import ServerConfig, FlagBuilder
from core.server_manager import LlamaServerManager

SYSTEM_PROMPT = """You are an expert performance tuner for llama.cpp inference servers.

Your goal: maximize GENERATION tok/s while preserving output quality.
Speed is the primary metric. Quality must not be sacrificed for marginal speed gains.

# How this works
- I give you hardware info, model info, the full --help output, and the current config + benchmark results
- You propose ONE config per round
- I benchmark it and tell you the results (or report a crash)
- You learn from each result and propose a better config next round
- We do 8 rounds. Make each one count — never repeat a failed or worse config.

# Optimization priority (in order)
1. GPU split strategy: single GPU > graph split > row split > layer split
2. Flash attention: enable it — faster and unlocks KV quantization
3. KV cache quantization: q8_0 is the sweet spot (2x VRAM savings, negligible quality loss). q4_0 saves 4x VRAM but degrades long-context quality — only use if VRAM-starved.
4. Batch size tuning: 2048-4096 is usually optimal. Above 4096 = diminishing returns.
5. Thread count: generation threads = physical CPU cores. Hyperthreads don't help.
6. Memory flags: --no-mmap when model spans GPU + RAM. --mlock pins RAM (less swapping).

# Quality guardrails
- Prefer q8_0 KV over q4_0 — the speed difference is small but quality difference is real
- Quantized KV cache REQUIRES --flash-attn — crashes without it
- Don't reduce context size to gain speed — it cripples usability
- --parallel 1 unless multi-user is explicitly needed (extra slots waste VRAM)

# Crash recovery
- If a config crashed, your next proposal must be MORE conservative on the likely cause
- OOM = reduce batch, increase KV quantization, or spread across more GPUs
- Immediate crash = you used an unsupported or invalid flag — remove it
- Never repeat a config that crashed. Build on what worked instead.

# Rules
1. The baseline config flags are ALREADY applied. You only propose CHANGES.
2. Your "flags" JSON should contain ONLY flags you want to ADD or OVERRIDE.
   Example: {"--batch-size": "4096", "--flash-attn": "on"} — these replace the baseline values.
3. Boolean flags: {"--no-mmap": true, "--mlock": true}
4. To remove a baseline flag: {"--some-flag": false}
5. These flags are FIXED (never include them): {fixed_flags}
6. Your response MUST contain a JSON object with these keys:
   {"name": "short description", "flags": {...}, "reasoning": "why this should be faster"}
7. Think carefully about VRAM budget — total VRAM minus model size = room for KV + batch buffers
8. ONLY use flags from the --help output. Unknown flags crash the server.
"""

class AITuner:
    def __init__(self, state, app):
        self.state = state
        self.app = app
        # Use path from state or default
        self.history_file = Path(self.state.settings.get("tune_history_path", Path.home() / ".cache" / "llm-server" / "tune_history.jsonl"))
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Folder for detailed benchmark logs
        self.logs_dir = Path.home() / ".cache" / "llm-server" / "tuning_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.port = self.state.port_var.get() if hasattr(self.state, 'port_var') else 8081

    def _is_port_in_use(self, port=None) -> bool:
        if port is None:
            port = self.port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return False
            except OSError:
                return True

    def _get_hw_profile(self) -> str:
        gpus = get_gpu_list()
        gpu_list = []
        for i, g in enumerate(gpus):
            gpu_list.append({
                "index": i,
                "name": g.get("name", "unknown"),
                "vram_free_mb": g.get("vram_free", 0),
                "vram_total_mb": g.get("vram_total", 0)
            })
        
        profile = {
            "gpu_count": len(gpus),
            "gpus": gpu_list,
            "ram_available_mb": get_total_ram_gb() * 1024,
            "physical_cores": self.state.threads_var.get() if hasattr(self.state, 'threads_var') else 8
        }
        return json.dumps(profile, indent=2)

    def _get_model_profile(self, model_path: str) -> str:
        from core.gguf_parser import GGUFMetadataParser
        parser = GGUFMetadataParser()
        meta = parser.get_cached_or_parse(model_path)
        
        size_mb = 0
        try:
            size_mb = os.path.getsize(model_path) // (1024 * 1024)
        except: pass

        profile = {
            "name": os.path.basename(model_path),
            "architecture": meta.get("general.architecture", "unknown"),
            "layers": meta.get("llm.block_count", 0),
            "experts": meta.get("llm.expert_count", 0),
            "size_mb": size_mb,
            "is_moe": meta.get("llm.expert_count", 0) > 1,
            "backend": "ik_llama.cpp" if "ik_llama" in self.state.get_server_exe_path() else "llama.cpp"
        }
        return json.dumps(profile, indent=2)

    def _get_hw_hash(self) -> str:
        gpus = get_gpu_list()
        gpu_str = "".join([f"{g.get('name')}{g.get('vram_total')}" for g in gpus])
        return hashlib.md5(gpu_str.encode()).hexdigest()[:8]

    def _load_tune_history(self, hw_hash: str) -> str:
        if not self.history_file.exists():
            return "(No previous tuning data)"
        
        history = []
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        history.append(json.loads(line))
                    except: continue
        except: pass

        if not history:
            return "(No previous tuning data)"

        same_hw = [e for e in history if e.get("hw_hash") == hw_hash]
        lines = []
        if same_hw:
            lines.append("## Results on THIS hardware:")
            # Group by model
            models = {}
            for e in same_hw:
                m = e.get("model", "unknown")
                if m not in models: models[m] = []
                models[m].append(e)
            
            for m, entries in models.items():
                ok = [e for e in entries if e.get("status") == "ok"]
                if ok:
                    best = max(ok, key=lambda x: x.get("gen_tps", 0))
                    lines.append(f"  {m}: best={best['gen_tps']} tok/s ({best['name']})")
        
        return "\n".join(lines[:50]) if lines else "(No previous tuning data)"

    def _append_tune_history(self, entry: Dict[str, Any]):
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + "\n")
        except: pass

    def _get_tune_cache_key(self, model_path: str, hw_hash: str) -> str:
        import os
        model_name = os.path.basename(model_path)
        total_size_bytes = os.path.getsize(model_path)
        return f"{model_name}_{total_size_bytes}_hw{hw_hash}"

    def _get_tune_cache_path(self, model_path: str, hw_hash: str) -> Path:
        cache_dir = Path.home() / ".cache" / "llm-server"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._get_tune_cache_key(model_path, hw_hash)
        return cache_dir / f"tune_{key}.json"

    def _load_tune_cache(self, model_path: str, hw_hash: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_tune_cache_path(model_path, hw_hash)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def _save_tune_cache(self, model_path: str, hw_hash: str, data: Dict[str, Any]):
        cache_path = self._get_tune_cache_path(model_path, hw_hash)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except:
            pass

    def _benchmark(self, config: ServerConfig) -> Tuple[float, float, str]:
        """
        Returns (gen_tps, pp_tps, status).
        Status can be 'ok', 'crashed', or 'timeout'.
        """
        manager = LlamaServerManager()
        logs = []
        
        def collect_logs(line):
            logs.append(line)

        cmd = [config.binary_path] + config.flags
        print(f"[AITuner] Executing benchmark command: {' '.join(cmd)}")
        
        success, err = manager.start(cmd, on_log=collect_logs)
        if not success:
            return 0.0, 0.0, f"crashed: {err}"

        log_filename = f"bench_{int(time.time())}_{config.port}.log"
        log_path = self.logs_dir / log_filename

        # Dynamic timeout based on model size (120s base + 30s per GB)
        model_size_mb = self._get_model_size_mb(config.model_path)
        health_timeout = 120 + (model_size_mb // 1024) * 30
        
        start_time = time.time()
        healthy = False
        health_err = "No response"
        while time.time() - start_time < health_timeout:
            # Check if process is still alive
            if manager.process and manager.process.poll() is not None:
                health_err = f"Process died with return code {manager.process.returncode}"
                if logs:
                    health_err += f", last log: {logs[-1]}"
                break
            try:
                for addr in ["127.0.0.1", "localhost"]:
                    try:
                        resp = requests.get(f"http://{addr}:{self.port}/health", timeout=2)
                        if resp.status_code == 200:
                            healthy = True
                            break
                    except:
                        pass
                if healthy: break
            except Exception as e:
                health_err = str(e)
            if not healthy and logs:
                health_err = f"Last log: {logs[-1]}"
            time.sleep(1)

        if not healthy:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(logs))
            manager.stop()
            last_log = logs[-1] if logs else "No logs available"
            return 0.0, 0.0, f"timeout/health_fail ({health_err}): {last_log}"

        try:
            # Try multiple prompts to avoid "empty response" issues with small models
            test_prompts = [
                "Hello, who are you?",
                "Explain the theory of relativity in simple terms.",
                "Write a short poem about AI."
            ]
            
            best_gen = 0.0
            best_pp = 0.0
            
            for prompt in test_prompts:
                payload = {
                    "model": "test",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.1
                }
                
                try:
                    start_req = time.time()
                    resp = requests.post(
                        f"http://127.0.0.1:{self.port}/v1/chat/completions",
                        json=payload,
                        timeout=60
                    )
                    end_req = time.time()
                    
                    if resp.status_code != 200:
                        print(f"[AITuner] Server error {resp.status_code}: {resp.text[:100]}")
                        continue
                    
                    res_json = resp.json()
                    content = res_json['choices'][0]['message'].get('content', '')
                    
                    # Extract usage and timings from the response
                    usage = res_json.get('usage', {})
                    n_gen = usage.get('completion_tokens', 0)
                    n_pp = usage.get('prompt_tokens', 0)
                    
                    timings = res_json.get('timings', {})
                    gen_tps = timings.get('predicted_per_second', 0)
                    pp_tps = timings.get('prompt_per_second', 0)
                    
                    # Debug: print timings and keys
                    print(f"[AITuner] Response keys: {list(res_json.keys())}")
                    print(f"[AITuner] Usage: {usage}")
                    print(f"[AITuner] Timings: {timings}")
                    
                    if gen_tps > 0:
                        # We have timings directly
                        print(f"[AITuner] Timings: gen={gen_tps:.2f} tok/s, pp={pp_tps:.2f} tok/s")
                    elif n_gen > 0:
                        # Estimate from total duration
                        duration = end_req - start_req
                        if duration > 0:
                            gen_tps = n_gen / duration
                            pp_tps = n_pp / duration
                            print(f"[AITuner] Estimated from duration: gen={gen_tps:.2f} tok/s (n={n_gen}, t={duration:.2f}s)")
                        else:
                            gen_tps = 0.0
                            pp_tps = 0.0
                    else:
                        gen_tps = 0.0
                        pp_tps = 0.0
                    
                    # Keep slot debug for now
                    try:
                        slots_resp = requests.get(f"http://127.0.0.1:{self.port}/slots", timeout=2)
                        slots_data = slots_resp.json()
                        slot = slots_data[0] if isinstance(slots_data, list) else slots_data
                        if isinstance(slot, dict):
                            print(f"[AITuner] Slot keys: {list(slot.keys())}")
                    except Exception as e:
                        pass

                    if gen_tps > best_gen:
                        best_gen = gen_tps
                        best_pp = pp_tps
                        print(f"[AITuner] Valid generation: {gen_tps:.2f} tok/s (n={n_gen})")
                        # Return immediately after first successful measurement to save time
                        break
                except Exception as e:
                    print(f"[AITuner] Prompt error: {e}")
                    continue
            
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(logs))

            manager.stop()
            if best_gen == 0.0:
                # Server is OK but generation failed or was too fast/small to measure
                return 0.0, 0.0, "ok_but_no_generation"
                
            return best_gen, best_pp, "ok"
            
        except Exception as e:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(logs))
            manager.stop()
            last_log = logs[-1] if logs else str(e)
            return 0.0, 0.0, f"request_fail ({e}): {last_log}"

    def _apply_flag_overrides(self, baseline_flags: List[str], overrides: Dict[str, Any]) -> List[str]:
        """
        Apply LLM flag overrides to baseline flags.
        Follows llm-server logic: protected flags are not changed.
        Returns new flag list.
        """
        protected = {"-m", "--host", "--port", "--ctx-size", "-ot"}
        # Convert baseline flags to a mutable list
        new_flags = baseline_flags.copy()
        
        # Process each override
        for flag, value in overrides.items():
            # Skip protected flags
            if flag in protected:
                continue
            # If value is False, remove flag from list (if present)
            if value is False:
                # Remove flag and its value if flag is a key-value flag
                # We'll just remove all occurrences of flag (simple)
                i = 0
                while i < len(new_flags):
                    if new_flags[i] == flag:
                        # Remove flag and possibly following value if it's not another flag
                        if i+1 < len(new_flags) and not new_flags[i+1].startswith('-'):
                            del new_flags[i+1]
                        del new_flags[i]
                    else:
                        i += 1
                continue
            
            # Determine if flag expects a value (boolean flags have no value)
            # In llama.cpp, boolean flags start with --no- or --, but we can't assume.
            # We'll treat flag as boolean if value is True, else flag + value.
            if value is True:
                # Add boolean flag if not already present
                if flag not in new_flags:
                    new_flags.append(flag)
            else:
                # Flag with value: replace existing or append
                # Find index of flag
                try:
                    idx = new_flags.index(flag)
                    # Replace next element with value
                    if idx+1 < len(new_flags) and not new_flags[idx+1].startswith('-'):
                        new_flags[idx+1] = str(value)
                    else:
                        new_flags.insert(idx+1, str(value))
                except ValueError:
                    # Flag not present, append flag and value
                    new_flags.extend([flag, str(value)])
        return new_flags

    def _compute_kv_size_mb(self, model_path, ctx_size, kv_type="q4_0"):
        """
        Estimate KV cache size in MB for given ctx_size and kv_type.
        kv_type: "q4_0", "q8_0", "f16"
        Returns approximate size in MB.
        """
        from core.gguf_parser import GGUFMetadataParser
        parser = GGUFMetadataParser()
        meta = parser.get_cached_or_parse(model_path)
        # Try to extract architecture-specific keys
        # First look for generic llm.block_count, else try architecture-specific
        layers = meta.get("llm.block_count", 0)
        if layers == 0:
            # Try to find any *.block_count
            for key in meta:
                if key.endswith(".block_count"):
                    layers = meta[key]
                    break
        head_count_kv = meta.get("llm.head_count_kv", 0)
        if head_count_kv == 0:
            for key in meta:
                if "head_count_kv" in key:
                    head_count_kv = meta[key]
                    break
        key_length = meta.get("llm.key_length", 0)
        if key_length == 0:
            for key in meta:
                if "key_length" in key and "swa" not in key:
                    key_length = meta[key]
                    break
        value_length = meta.get("llm.value_length", 0)
        if value_length == 0:
            for key in meta:
                if "value_length" in key and "swa" not in key:
                    value_length = meta[key]
                    break
        
        # If missing, use defaults
        if layers == 0: layers = 48
        if head_count_kv == 0: head_count_kv = 8
        if key_length == 0: key_length = 128
        if value_length == 0: value_length = 128
        
        bytes_per_element = {"q4_0": 0.5, "q8_0": 1.0, "f16": 2.0}.get(kv_type, 0.5)
        kv_bytes = ctx_size * layers * head_count_kv * (key_length + value_length) * bytes_per_element
        return kv_bytes / (1024 * 1024)

    def _get_model_size_mb(self, model_path):
        """Return model size in MB."""
        import os
        try:
            return os.path.getsize(model_path) // (1024 * 1024)
        except:
            return 0

    def _check_oom_safety(self, overrides: Dict[str, Any], model_path: str, baseline_flags: List[str]) -> bool:
        """
        Estimate memory usage of proposed config and check if it fits within system memory.
        Returns True if safe, False if likely OOM.
        """
        # Get model size
        model_size_mb = self._get_model_size_mb(model_path)
        if model_size_mb == 0:
            # fallback: assume 2GB
            model_size_mb = 2048
        
        # Determine ctx_size from baseline flags (default 4096)
        ctx_size = 4096
        try:
            idx = baseline_flags.index("--ctx-size")
            ctx_size = int(baseline_flags[idx + 1])
        except (ValueError, IndexError):
            pass
        
        # Determine KV type from overrides or baseline
        kv_type = "q4_0"
        # parse baseline flags for --cache-type-k
        try:
            idx = baseline_flags.index("--cache-type-k")
            kv_type = baseline_flags[idx + 1]
        except (ValueError, IndexError):
            pass
        # Override if specified in overrides
        if "--cache-type-k" in overrides:
            kv_type = overrides["--cache-type-k"]
        if "--cache-type-v" in overrides:
            kv_type = overrides["--cache-type-v"]  # prefer v
        
        # Compute KV size
        kv_mb = self._compute_kv_size_mb(model_path, ctx_size, kv_type)
        
        # Estimate batch buffer overhead: 2048 MB as in llm-server
        batch_overhead_mb = 2048
        
        total_estimated_mb = model_size_mb + kv_mb + batch_overhead_mb
        
        # Get system memory
        total_vram_mb = get_total_vram_free()
        total_ram_mb = get_available_ram_mb()
        # Use a small safety reserve (1GB) instead of 5GB, 
        # and since Baseline already works, we trust the system more.
        system_total_mb = total_vram_mb + total_ram_mb - 1024
        
        if total_estimated_mb > system_total_mb:
            print(f"[OOM Protection] Estimated {total_estimated_mb:.0f}MB > system capacity {system_total_mb}MB")
            return False
        return True

    def _query_llm(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Queries the currently running baseline server for a config proposal."""
        try:
            payload = {
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.3
            }
            resp = requests.post(
                f"http://127.0.0.1:{self.port}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            content = resp.json()['choices'][0]['message'].get('content', '')
            
            # Extract JSON block
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except:
            pass
        return None



    def apply_cached_tune(self, config: ServerConfig) -> bool:
        """
        Apply cached tuned flag overrides to the given ServerConfig.
        Returns True if cache existed and overrides were applied.
        """
        hw_hash = self._get_hw_hash()
        cached = self._load_tune_cache(config.model_path, hw_hash)
        if cached is None:
            return False
        best_config = cached.get('best_config', {})
        overrides = best_config.get('flags', {})
        if not overrides:
            # baseline wins or empty overrides
            return True  # cache exists but no overrides to apply
        # Apply overrides using the same logic as tuning
        config.flags = self._apply_flag_overrides(config.flags, overrides)
        return True

    def tune(self, model_path: str, rounds: int = 8, on_progress=None, retune: bool = False):
        # Check for cached result first
        hw_hash = self._get_hw_hash()
        if not retune:
            cached = self._load_tune_cache(model_path, hw_hash)
            if cached is not None:
                if on_progress: on_progress(f"✓ Using cached tuning result from {cached.get('tuned_at', 'unknown time')}")
                best_config_data = cached.get('best_config', {})
                gen_tps = best_config_data.get('gen_tps', 0.0)
                pp_tps = best_config_data.get('pp_tps', 0.0)
                
                # Reconstruct ServerConfig from baseline + cached overrides
                config = ServerConfig(
                    model_path=model_path,
                    port=self.port,
                    ctx_size=4096,
                    flash_attn="on",
                    custom_args="--perf --metrics"
                )
                builder = FlagBuilder(self.state)
                config = builder.build(config)
                
                overrides = best_config_data.get('flags', {})
                config.flags = self._apply_flag_overrides(config.flags, overrides)
                
                return config, gen_tps
        
        if on_progress: on_progress("🚀 Starting AI-Tuning...")
        
        # Check if port is already in use
        if self._is_port_in_use():
            if on_progress: on_progress(f"❌ Port {self.port} is already in use. Please stop any existing server.")
            return None, 0.0
        
        # Warn if model is very large relative to system memory (>70%)
        model_size_mb = self._get_model_size_mb(model_path)
        if model_size_mb > 0:
            from core.hardware import get_total_ram_gb, get_gpu_list
            total_ram_mb = get_total_ram_gb() * 1024
            total_vram_mb = 0
            gpus = get_gpu_list()
            for g in gpus:
                total_vram_mb += g.get("vram_total", 0)
            system_mem_mb = total_vram_mb + total_ram_mb
            if system_mem_mb > 0:
                model_pct = model_size_mb * 100 // system_mem_mb
                if model_pct > 70:
                    if on_progress: on_progress(f"⚠️  WARNING: Model uses {model_pct}% of system memory ({model_size_mb}MB / {system_mem_mb}MB).")
                    if on_progress: on_progress("   AI Tune reloads the model each round — this will be very slow and may trigger OOM.")
        
        hw_profile = self._get_hw_profile()
        model_profile = self._get_model_profile(model_path)
        
        # Baseline config
        config = ServerConfig(
            model_path=model_path,
            port=self.port,
            ctx_size=4096, # Use smaller ctx for benchmarking to ensure stability
            flash_attn="on", # Force flash-attn for KV quantization stability
            custom_args="--perf --metrics"
        )
        builder = FlagBuilder(self.state) 
        try:
            config = builder.build(config)
        except Exception as e:
            if on_progress: on_progress(f"❌ Configuration error: {e}")
            return None, 0.0
        
        # 1. Baseline Benchmark
        if on_progress: on_progress("Round 0: Benchmarking baseline...")
        try:
            gen_tps, pp_tps, status = self._benchmark(config)
        except Exception as e:
            if on_progress: on_progress(f"❌ Baseline error: {e}")
            return None, 0.0
        
        if status == "ok":
            if gen_tps == 0.0:
                if on_progress: on_progress("⚠️ Baseline returned 0.00 tok/s. Check if model can generate text.")
            pass
        else:
            if on_progress: on_progress(f"❌ Baseline failed: {status}")
            return None, 0.0

        best_gen = gen_tps
        best_pp = pp_tps
        best_config = copy.deepcopy(config)
        best_name = "baseline"
        
        self._append_tune_history({
            "timestamp": time.time(),
            "model": os.path.basename(model_path),
            "hw_hash": hw_hash,
            "round": 0,
            "name": "baseline",
            "gen_tps": gen_tps,
            "pp_tps": pp_tps,
            "status": "ok",
            "flags": {}
        })

        if on_progress: on_progress(f"Baseline: {gen_tps:.2f} tok/s")

        # 2. Setup for LLM queries
        # We need a server running the baseline to query it
        manager = LlamaServerManager()
        try:
            manager.start([config.binary_path] + config.flags)
        except Exception as e:
            if on_progress: on_progress(f"❌ Failed to start baseline server for queries: {e}")
            return None, 0.0
        
        # Wait for baseline to be healthy
        start_time = time.time()
        while time.time() - start_time < 60:
            try:
                if requests.get(f"http://127.0.0.1:{self.port}/health").status_code == 200:
                    break
            except: pass
            time.sleep(1)

        # Help text
        help_text = ""
        try:
            help_text = subprocess.check_output([config.binary_path, "--help"], text=True, stderr=subprocess.STDOUT)
        except: pass

        fixed_flags = " ".join(config.flags)
        system_prompt = SYSTEM_PROMPT.replace("{fixed_flags}", fixed_flags)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"# Hardware\n{hw_profile}\n\n# Model\n{model_profile}\n\n# Server --help\n{help_text}\n\n# Baseline\nFlags: {fixed_flags}\nBenchmark: gen={gen_tps} tok/s, pp={pp_tps} tok/s\n\n# History\n{self._load_tune_history(hw_hash)}\n\nPropose your first config."}
        ]

        # 3. Tuning Loop
        round = 1
        crash_retries = 0
        max_crashes = 4
        all_results = []
        best_overrides = {}
        while round <= rounds:
            if on_progress: on_progress(f"Round {round}/{rounds}: Querying AI...")
            
            proposal = self._query_llm(messages)
            if not proposal:
                if on_progress: on_progress("⚠️ No valid proposal from AI, skipping round.")
                round += 1
                continue
            
            name = proposal.get("name", f"config_{round}")
            overrides = proposal.get("flags", {})
            reasoning = proposal.get("reasoning", "")
            
            # OOM protection check
            if not self._check_oom_safety(overrides, model_path, config.flags):
                if on_progress: on_progress(f"⏭️ Skipping {name} — estimated memory exceeds system capacity.")
                # Provide feedback to LLM
                feedback = f"Result for '{name}': SKIPPED — estimated memory exceeds system capacity. Do NOT increase KV cache or batch size. Try other flags."
                messages.append({"role": "assistant", "content": json.dumps(proposal)})
                messages.append({"role": "user", "content": feedback})
                round += 1
                continue
            
            if on_progress: 
                # Show exactly what the AI proposed
                flags_str = ", ".join([f"{k}={v}" for k, v in overrides.items()])
                on_progress(f"Testing: {name}\nProposed Flags: {flags_str}\nReason: {reasoning}")
            
            # Build test config
            test_config = copy.deepcopy(config)
            # Apply overrides using robust parser
            test_config.flags = self._apply_flag_overrides(config.flags, overrides)

            # Stop baseline, test new config, restart baseline
            manager.stop()
            cur_gen, cur_pp, cur_status = self._benchmark(test_config)
            
            # Crash handling
            if "crashed" in cur_status.lower():
                crash_retries += 1
                if crash_retries <= max_crashes:
                    # Free retry, round does not advance
                    if on_progress: on_progress(f"🔄 Crash retry {crash_retries}/{max_crashes} (round not counted)")
                    # Provide feedback to LLM
                    feedback = f"Result for '{name}': CRASHED. The config failed to start or benchmark. Avoid similar flags. You have {max_crashes - crash_retries} free retries left."
                    messages.append({"role": "assistant", "content": json.dumps(proposal)})
                    messages.append({"role": "user", "content": feedback})
                    # Restart baseline for next query
                    manager.start([config.binary_path] + config.flags)
                    time.sleep(5)
                    continue
                else:
                    # No more free retries, count as a round
                    if on_progress: on_progress(f"❌ Config crashed (no more free retries).")
            else:
                crash_retries = 0  # reset on successful benchmark
            
            # Restart baseline for next query
            manager.start([config.binary_path] + config.flags)
            # Wait for baseline health
            time.sleep(5)

            if cur_status == "ok":
                if cur_gen > best_gen:
                    best_gen = cur_gen
                    best_pp = cur_pp
                    best_config = copy.deepcopy(test_config)
                    best_name = name
                    best_overrides = overrides
                    if on_progress: on_progress(f"★ New Best: {cur_gen:.2f} tok/s!")
                else:
                    if on_progress: on_progress(f"Result: {cur_gen:.2f} tok/s (No improvement)")
            else:
                if on_progress: on_progress(f"❌ Config {cur_status}.")
            
            # Update conversation
            feedback = f"Result for '{name}': {cur_status}. Gen: {cur_gen:.2f} tok/s, PP: {cur_pp:.2f} tok/s. Current best: {best_gen:.2f} tok/s. Propose next."
            messages.append({"role": "assistant", "content": json.dumps(proposal)})
            messages.append({"role": "user", "content": feedback})
            
            self._append_tune_history({
                "timestamp": time.time(),
                "model": os.path.basename(model_path),
                "hw_hash": hw_hash,
                "round": round,
                "name": name,
                "gen_tps": cur_gen,
                "pp_tps": cur_pp,
                "status": cur_status,
                "flags": overrides
            })
            
            # Add to in-memory results for cache
            all_results.append({
                "round": round,
                "name": name,
                "gen_tps": cur_gen,
                "pp_tps": cur_pp,
                "flags": overrides,
                "status": cur_status
            })
            
            round += 1

        manager.stop()
        
        # Save results to cache
        cache_data = {
            "model": os.path.basename(model_path),
            "tuned_at": datetime.datetime.utcnow().isoformat() + "Z",
            "provider": "self",
            "baseline_gen_tps": gen_tps,
            "baseline_pp_tps": pp_tps,
            "rounds": rounds,
            "all_results": all_results
        }
        if best_name == "baseline":
            cache_data["baseline_wins"] = True
            cache_data["best_config"] = {
                "name": "baseline",
                "flags": {},
                "gen_tps": best_gen,
                "pp_tps": best_pp
            }
        else:
            cache_data["best_config"] = {
                "name": best_name,
                "flags": best_overrides,
                "gen_tps": best_gen,
                "pp_tps": best_pp
            }
        self._save_tune_cache(model_path, hw_hash, cache_data)
        
        if on_progress: on_progress(f"AI-Tuning complete. Winner: {best_name} ({best_gen:.2f} tok/s)")
        
        return best_config, best_gen
