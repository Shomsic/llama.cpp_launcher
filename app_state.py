"""
Центральное состояние приложения.
Все общие данные, кэши, tk-переменные и хелперы — здесь.
Вкладки и компоненты получают ссылку на AppState и работают через него.
"""
import os
import re
import json
import threading
import subprocess
import tkinter as tk
from pathlib import Path

from core.config import (
    DEFAULT_SETTINGS, PRESETS, SETTINGS_FILE, LOG_DIR, GGUF_VALUE_TYPES
)
from core.hardware import get_gpu_info as _hw_get_gpu_info, get_gpu_name as _hw_get_gpu_name, get_total_ram_gb, get_cpu_cores
from core.gguf_parser import (
    GGUFMetadataParser, validate_gguf_file,
    extract_quant_from_filename, get_quant_description
)
from core.estimator import (
    ProfilingData, estimate_memory_breakdown, estimate_tokens_per_second as _est_tps,
    calculate_max_ngl as _est_max_ngl, _coerce_int_metadata
)
from core.i18n import _


class AppState:
    """Единый объект состояния, доступный всем частям приложения."""

    def __init__(self):
        # --- Настройки ---
        self.settings = DEFAULT_SETTINGS.copy()
        self._load_settings_from_file()

        # --- Ядро ---
        self.gguf_parser = GGUFMetadataParser()

        # --- Состояние сервера ---
        self.server_manager = None  # LlamaServerManager (предпочтительный способ)
        self.server_process = None  # Popen (fallback, для обратной совместимости)
        self.running = False
        self.stopping = False
        self.running_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.current_log_file = None
        self.log_thread = None

        # --- Кэши ---
        self._metadata_cache: dict = {}
        self._metadata_loading: set = set()
        self._cached_vram_mb: int | None = None
        self._cached_gpu_name: str | None = None
        self._model_total_layers: int = 32
        self.max_ngl: int = 128
        self.max_ctx: int = 32768
        self._scale_update_job = None

        # --- Tk-переменные (создаются позже через init_tk_vars) ---
        self.root: tk.Tk | None = None

        # --- Callbacks для UI (устанавливаются компонентами) ---
        self._update_listeners: list = []
        self.log_callback = None          # (message: str) -> None
        self.set_status_callback = None   # (message: str) -> None

    # ──────────────────────── Tk Variables ────────────────────────

    def init_tk_vars(self, root: tk.Tk):
        """Создаёт все tk-переменные. Вызывать ПОСЛЕ создания корневого окна."""
        self.root = root
        s = self.settings

        # --- Путь к llama.cpp ---
        self.llama_dir_var = tk.StringVar(value=s.get("llama_dir", ""))

        # --- Активная модель ---
        self.active_model_var = tk.StringVar(value=s.get("active_model", ""))
        self.active_model_name_var = tk.StringVar(value=_("no_model"))
        self.active_model_path_var = tk.StringVar(value=_("select_model_hint"))
        self.active_model_meta_var = tk.StringVar(value="")

        # --- Черновая модель (Speculative Decoding) ---
        self.draft_model_var = tk.StringVar(value=s.get("draft_model", ""))
        self.draft_model_name_var = tk.StringVar(value=_("no_model"))
        self.draft_ngl_var = tk.IntVar(value=s.get("draft_ngl", 0))

        # --- Основные параметры ---
        self.host_var = tk.StringVar(value=s.get("host", "127.0.0.1"))
        self.port_var = tk.IntVar(value=s.get("port", 8080))
        self.ngl_var = tk.IntVar(value=s.get("ngl", 0))
        cpu_cores = get_cpu_cores()
        self.ctx_var = tk.IntVar(value=s.get("ctx", 4096))
        self.threads_var = tk.IntVar(value=s.get("threads", min(4, cpu_cores)))
        self.gpu_offload_pct_var = tk.IntVar(value=s.get("gpu_offload_pct", 100))

        # --- Параметры генерации ---
        self.temp_var = tk.DoubleVar(value=s.get("temp", 0.8))
        self.top_k_var = tk.IntVar(value=s.get("top_k", 40))
        self.top_p_var = tk.DoubleVar(value=s.get("top_p", 0.95))
        self.min_p_var = tk.DoubleVar(value=s.get("min_p", 0.05))
        self.repeat_penalty_var = tk.DoubleVar(value=s.get("repeat_penalty", 1.0))
        self.presence_penalty_var = tk.DoubleVar(value=s.get("presence_penalty", 0.0))
        self.frequency_penalty_var = tk.DoubleVar(value=s.get("frequency_penalty", 0.0))
        self.mirostat_var = tk.IntVar(value=s.get("mirostat", 0))
        self.n_predict_var = tk.IntVar(value=s.get("n_predict", -1))
        self.reasoning_var = tk.StringVar(value=s.get("reasoning", "auto"))

        # --- Расширенные ---
        self.parallel_slots_var = tk.IntVar(value=s.get("parallel_slots", 1))
        self.batch_size_var = tk.IntVar(value=s.get("batch_size", 2048))
        self.flash_attn_var = tk.StringVar(value=s.get("flash_attn", "auto"))
        self.seed_var = tk.IntVar(value=s.get("seed", -1))
        self.mmap_var = tk.StringVar(value=s.get("mmap", "on"))
        self.mlock_var = tk.StringVar(value=s.get("mlock", "off"))
        self.kv_offload_var = tk.StringVar(value=s.get("kv_offload", "on"))
        self.cache_prompt_var = tk.StringVar(value=s.get("cache_prompt", "on"))
        self.rope_scale_var = tk.StringVar(value=s.get("rope_scale", "auto"))
        self.rope_freq_base_var = tk.StringVar(value=s.get("rope_freq_base", "auto"))
        self.api_key_var = tk.StringVar(value=s.get("api_key", ""))
        self.webui_var = tk.StringVar(value=s.get("webui", "on"))
        self.custom_args_var = tk.StringVar(value=s.get("custom_args", ""))
        self.language_var = tk.StringVar(value=s.get("language", "ru"))

        # --- Пресеты ---
        self.preset_var = tk.StringVar(value="— Пользовательские —")

        # --- Benchmark ---
        self.bench_prompt_var = tk.IntVar(value=s.get("bench_prompt", 512))
        self.bench_predict_var = tk.IntVar(value=s.get("bench_predict", 128))
        self.bench_threads_var = tk.IntVar(value=s.get("bench_threads", 8))
        self.bench_ngl_var = tk.IntVar(value=s.get("bench_ngl", 999))

    # ──────────────────────── Listeners ────────────────────────

    def add_update_listener(self, callback):
        self._update_listeners.append(callback)

    def notify_update(self):
        for cb in self._update_listeners:
            try:
                cb()
            except Exception:
                pass

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def set_status(self, message: str):
        if self.set_status_callback:
            self.set_status_callback(message)

    # ──────────────────────── Settings I/O ────────────────────────

    def _load_settings_from_file(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k in self.settings:
                            self.settings[k] = v
                    if "custom_presets" in self.settings:
                        for name, preset in self.settings["custom_presets"].items():
                            PRESETS[name] = preset
        except Exception as e:
            print(f"Ошибка загрузки настроек: {e}")

    def save_settings(self):
        """Синхронизирует tk-переменные → dict и пишет JSON."""
        if self.root:
            self.settings["llama_dir"] = self.llama_dir_var.get()
            self.settings["host"] = self.host_var.get()
            self.settings["port"] = self.port_var.get()
            self.settings["api_key"] = self.api_key_var.get()
            self.settings["webui"] = self.webui_var.get()
            self.settings["reasoning"] = self.reasoning_var.get()
            self.settings["cache_prompt"] = self.cache_prompt_var.get()
            self.settings["ngl"] = self.ngl_var.get()
            self.settings["ctx"] = self.ctx_var.get()
            self.settings["threads"] = self.threads_var.get()
            self.settings["parallel_slots"] = self.parallel_slots_var.get()
            self.settings["gpu_offload_pct"] = self.gpu_offload_pct_var.get()
            self.settings["temp"] = self.temp_var.get()
            self.settings["top_k"] = self.top_k_var.get()
            self.settings["top_p"] = self.top_p_var.get()
            self.settings["min_p"] = self.min_p_var.get()
            self.settings["repeat_penalty"] = self.repeat_penalty_var.get()
            self.settings["presence_penalty"] = self.presence_penalty_var.get()
            self.settings["frequency_penalty"] = self.frequency_penalty_var.get()
            self.settings["mirostat"] = self.mirostat_var.get()
            self.settings["n_predict"] = self.n_predict_var.get()
            self.settings["batch_size"] = self.batch_size_var.get()
            self.settings["flash_attn"] = self.flash_attn_var.get()
            self.settings["seed"] = self.seed_var.get()
            self.settings["mmap"] = self.mmap_var.get()
            self.settings["mlock"] = self.mlock_var.get()
            self.settings["kv_offload"] = self.kv_offload_var.get()
            self.settings["rope_scale"] = self.rope_scale_var.get()
            self.settings["rope_freq_base"] = self.rope_freq_base_var.get()
            self.settings["custom_args"] = self.custom_args_var.get()
            self.settings["active_model"] = self.active_model_var.get()
            self.settings["draft_model"] = self.draft_model_var.get()
            self.settings["draft_ngl"] = self.draft_ngl_var.get()
            self.settings["language"] = self.language_var.get()
            self.settings["bench_prompt"] = self.bench_prompt_var.get()
            self.settings["bench_predict"] = self.bench_predict_var.get()
            self.settings["bench_threads"] = self.bench_threads_var.get()
            self.settings["bench_ngl"] = self.bench_ngl_var.get()
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения настроек: {e}")

    # ──────────────────────── Hardware (с кэшем) ────────────────────────

    def get_gpu_info(self) -> int:
        """VRAM в МБ (кэшируется на весь сеанс)."""
        if self._cached_vram_mb is not None:
            return self._cached_vram_mb
        self._cached_vram_mb = _hw_get_gpu_info()
        return self._cached_vram_mb

    def get_gpu_name(self) -> str:
        if self._cached_gpu_name is not None:
            return self._cached_gpu_name
        self._cached_gpu_name = _hw_get_gpu_name()
        return self._cached_gpu_name

    @staticmethod
    def get_total_ram_gb() -> float:
        return get_total_ram_gb()

    @staticmethod
    def get_cpu_cores() -> int:
        return get_cpu_cores()

    # ──────────────────────── Model helpers ────────────────────────

    @staticmethod
    def get_model_info(model_path: str):
        """Возвращает (size_mb, size_gb) или (None, None)."""
        if not model_path or not os.path.isfile(model_path):
            return None, None
        try:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            return size_mb, size_mb / 1024
        except OSError:
            return None, None

    def get_model_total_layers(self, model_path: str) -> int:
        if not model_path or not os.path.isfile(model_path):
            return 32
        metadata = self.gguf_parser.get_cached_or_parse(model_path)
        if metadata and "__error__" not in metadata:
            arch = metadata.get("general.architecture")
            candidates = []
            if arch:
                candidates.extend([f"{arch}.block_count", f"{arch}.decoder.block_count", f"{arch}.n_layer"])
            candidates.extend(["llama.block_count", "llama.n_layer", "block_count"])
            for key in candidates:
                value = self.coerce_int_metadata(metadata, key)
                if value and 5 <= value <= 512:
                    return value
        # Fallback по имени файла
        name = os.path.basename(model_path).lower()
        for pattern, layers in [('70b', 80), ('34b', 60), ('13b', 40), ('mixtral', 40),
                                 ('3b', 32), ('1b', 24), ('0.5b', 12)]:
            if pattern in name:
                return layers
        return 32

    def get_model_max_context(self, model_path: str) -> int:
        metadata = self.gguf_parser.get_cached_or_parse(model_path)
        if metadata and "__error__" not in metadata:
            arch = metadata.get("general.architecture")
            candidates = []
            if arch:
                candidates.extend([f"{arch}.context_length", f"{arch}.max_context_length", f"{arch}.n_ctx_train"])
            candidates.extend(["llama.context_length", "general.context_length"])
            for key in candidates:
                value = self.coerce_int_metadata(metadata, key)
                if value and value >= 512:
                    return value
        return 32768

    def calculate_max_ngl(self, model_path: str) -> int:
        vram = self.get_gpu_info()
        total_layers = self.get_model_total_layers(model_path)
        __, model_gb = self.get_model_info(model_path)
        return _est_max_ngl(vram, total_layers, model_gb)

    def recalculate_limits(self, model_path: str):
        """Пересчитывает max_ngl / max_ctx / total_layers для модели."""
        if not model_path or not os.path.isfile(model_path):
            return
        self._model_total_layers = self.get_model_total_layers(model_path)
        self.max_ngl = self.calculate_max_ngl(model_path)
        self.max_ctx = self.get_model_max_context(model_path)

    # ──────────────────────── Memory / TPS ────────────────────────

    def estimate_memory(self, model_path: str) -> dict:
        """Оценка GPU/RAM с делегированием в estimator."""
        model_size_mb, __ = self.get_model_info(model_path)
        if model_size_mb is None:
            return {"gpu_total_mb": 0, "ram_total_mb": 0, "gpu_weights_mb": 0,
                    "ram_weights_mb": 0, "gpu_kv_mb": 0, "ram_kv_mb": 0,
                    "ctx_state_mb": 0, "parallel_slots": 1, "kv_offload_on": True,
                    "flash_attn_on": True, "offload_pct": 0}
        p = self._make_profiling_data(model_path)
        return estimate_memory_breakdown(p)

    def estimate_tps(self) -> dict | None:
        """Оценка tok/s. Возвращает dict или None."""
        model_path = self.active_model_var.get()
        if not model_path or not os.path.isfile(model_path):
            return None
        p = self._make_profiling_data(model_path)
        return _est_tps(p)

    def _make_profiling_data(self, model_path: str) -> ProfilingData:
        return ProfilingData(
            model_path=model_path,
            ngl=self.ngl_var.get(),
            ctx=self.ctx_var.get(),
            total_layers=max(self._model_total_layers, 1),
            parallel_slots=max(self.parallel_slots_var.get(), 1),
            threads=self.threads_var.get(),
            kv_offload_on=self.kv_offload_var.get() == "on",
            flash_attn_on=self.flash_attn_var.get() in ("on", "auto"),
            batch_size=self.batch_size_var.get(),
            get_model_info=self.get_model_info,
            get_gpu_info=self.get_gpu_info,
            get_cached_metadata=self.gguf_parser.get_cached_or_parse,
            extract_quant=extract_quant_from_filename,
        )

    def calculate_memory_text(self) -> str:
        model_path = self.active_model_var.get()
        if not model_path:
            return "GPU: 0 GB | RAM: 0 GB | Total: 0 GB"
        __, model_size_gb = self.get_model_info(model_path)
        if model_size_gb is None:
            return "GPU: ? | RAM: ? | Total: ?"
        mem = self.estimate_memory(model_path)
        gpu_gb = mem["gpu_total_mb"] / 1024
        ram_gb = mem["ram_total_mb"] / 1024
        total_gb = gpu_gb + ram_gb
        kv_target = "GPU" if mem["kv_offload_on"] and mem["gpu_kv_mb"] > 0 else "RAM"
        kv_amount = mem["gpu_kv_mb"] if kv_target == "GPU" else mem["ram_kv_mb"]
        return (f"{_('vram')} {gpu_gb:.2f} GB | {_('ram')} {ram_gb:.2f} GB | "
                f"{_('total')} {total_gb:.2f} GB | KV→{kv_target}: {kv_amount / 1024:.2f} GB | "
                f"slots: {mem['parallel_slots']}")

    # ──────────────────────── Model path helpers ────────────────────────

    def get_server_exe_path(self) -> str | None:
        path = self.llama_dir_var.get()
        if not path:
            return None
        for exe in ("llama-server.exe", "llama-cli.exe"):
            full = os.path.join(path, exe)
            if os.path.exists(full):
                return full
        return None

    def get_benchmark_exe_path(self) -> str | None:
        path = self.llama_dir_var.get()
        if not path:
            return None
        full = os.path.join(path, "llama-bench.exe")
        if os.path.exists(full):
            return full
        return None

    def validate_llama_dir(self) -> tuple[bool, str]:
        path = self.llama_dir_var.get()
        if not path:
            return False, _("path_not_selected")
        if not os.path.isdir(path):
            return False, _("path_not_exists")
        if os.path.exists(os.path.join(path, "llama-server.exe")):
            status = _("server_found")
        elif os.path.exists(os.path.join(path, "llama-cli.exe")):
            status = _("cli_found")
        else:
            return False, _("binaries_not_found")
        
        if os.path.exists(os.path.join(path, "llama-bench.exe")):
            status += " + llama-bench"
            
        return True, status

    # ──────────────────────── Models list ────────────────────────

    def get_all_models(self, max_depth: int = 5) -> list[str]:
        models = set()
        skip_patterns = ['mmproj', 'mmproj-model']
        embed_patterns = ['qwen3-embed', 'mxbai-embed', 'bge-embed']

        def _should_skip(fname_lower):
            if any(x in fname_lower for x in skip_patterns):
                return True
            if 'embedding' in fname_lower and 'instruct' not in fname_lower:
                if any(x in fname_lower for x in embed_patterns):
                    return True
            return False

        for d in self.settings.get("model_dirs", []):
            if not os.path.isdir(d):
                continue
            base_depth = len(Path(d).parts)
            for root_dir, dirs, files in os.walk(d):
                if len(Path(root_dir).parts) - base_depth >= max_depth:
                    dirs.clear()
                    continue
                for f in files:
                    if f.lower().endswith(".gguf") and not _should_skip(f.lower()):
                        models.add(os.path.join(root_dir, f))

        for f in self.settings.get("model_files", []):
            if os.path.isfile(f) and f.lower().endswith(".gguf"):
                if not _should_skip(os.path.basename(f).lower()):
                    models.add(f)

        return sorted(models)

    # ──────────────────────── Metadata helpers ────────────────────────

    @staticmethod
    def coerce_int_metadata(metadata: dict, key: str):
        return _coerce_int_metadata(metadata, key)

    @staticmethod
    def get_model_param_label(metadata: dict, model_path: str):
        """Возвращает (label_str, source) или (None, None)."""
        raw = metadata.get("general.parameter_count")
        if raw:
            try:
                v = int(raw)
                if v >= 1e9:
                    return f"{v / 1e9:.1f}B", "metadata"
                if v >= 1e6:
                    return f"{v / 1e6:.1f}M", "metadata"
                return f"{v:,}", "metadata"
            except Exception:
                pass
        size_label = metadata.get("general.size_label")
        if isinstance(size_label, str) and size_label.strip():
            return size_label.strip(), "metadata"
        name = os.path.basename(model_path).upper()
        m = re.search(r"(\d+X\d+(?:\.\d+)?)B", name)
        if m:
            return m.group(1) + "B", "filename"
        m = re.search(r"(\d+(?:\.\d+)?)B", name)
        if m:
            return m.group(1) + "B", "filename"
        m = re.search(r"(\d+(?:\.\d+)?)M", name)
        if m:
            return m.group(1) + "M", "filename"
        return None, None

    def sync_active_model_ui(self):
        """Обновляет краткую информацию об активной модели (name/path/meta vars)."""
        model_path = self.active_model_var.get().strip()
        if not model_path:
            self.active_model_name_var.set(_("no_model"))
            self.active_model_path_var.set(_("select_model_hint"))
            self.active_model_meta_var.set("")
            return

        model_name = os.path.basename(model_path)
        self.active_model_name_var.set(model_name)
        self.active_model_path_var.set(model_path)

        __, size_gb = self.get_model_info(model_path)
        metadata = self.gguf_parser.get_cached_or_parse(model_path)
        arch = metadata.get("general.architecture") if metadata else None
        param_label, __ = self.get_model_param_label(metadata, model_path)
        quant_label = extract_quant_from_filename(model_path)
        ctx = self.coerce_int_metadata(metadata, f"{arch}.context_length") if arch else None
        if ctx is None:
            ctx = self.coerce_int_metadata(metadata, "llama.context_length")
        layers = self.coerce_int_metadata(metadata, f"{arch}.block_count") if arch else None
        if layers is None:
            layers = self.coerce_int_metadata(metadata, "llama.block_count")

        summary = []
        if size_gb:
            summary.append(f"{size_gb:.2f} GB")
        if arch:
            summary.append(arch.upper())
        if param_label:
            summary.append(param_label)
        if quant_label and quant_label != "Unknown":
            summary.append(quant_label)
        if ctx:
            summary.append(f"ctx {ctx:,}")
        if layers:
            summary.append(_("layers_summary").format(layers))
        self.active_model_meta_var.set(" | ".join(summary))

    def sync_draft_model_ui(self):
        """Обновляет информацию о черновой модели."""
        model_path = self.draft_model_var.get().strip()
        if not model_path:
            self.draft_model_name_var.set(_("no_model"))
            return

        model_name = os.path.basename(model_path)
        self.draft_model_name_var.set(f"Draft: {model_name}")
