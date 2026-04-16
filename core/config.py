"""
Константы, пресеты и пути приложения.
Не содержит бизнес-логику — только данные конфигурации.
"""
import sys
from pathlib import Path

# Корень проекта. 
# Если запущено как EXE — это папка с .exe файлом.
# Если запущен скрипт — это папка проекта.
if getattr(sys, 'frozen', False):
    _PROJECT_ROOT = Path(sys.executable).parent
else:
    _PROJECT_ROOT = Path(__file__).parent.parent

SETTINGS_FILE = str(_PROJECT_ROOT / "settings.json")
LOG_DIR = _PROJECT_ROOT / "logs"


def ensure_log_dir():
    """Создаёт директорию логов при первом использовании."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS = {
    "llama_dir": "",
    "model_dirs": [],
    "model_files": [],
    "active_model": "",
    "draft_model": "",
    "draft_ngl": 0,
    "host": "127.0.0.1",
    "port": 8080,
    "api_key": "",
    "webui": "on",
    "backend": "auto",
    "reasoning": "auto",
    "cache_prompt": "on",
    "ngl": 0,
    "ctx": 4096,
    "threads": 4,
    "parallel_slots": 1,
    "gpu_offload_pct": 100,
    "temp": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
    "n_predict": -1,
    "batch_size": 2048,
    "flash_attn": "auto",
    "seed": -1,
    "mmap": "on",
    "mlock": "off",
    "kv_offload": "on",
    "rope_scale": "auto",
    "rope_freq_base": "auto",
    "custom_args": "",
    "window_geometry": "1000x700",
    "tray_minimize": False,
    "language": "ru",
    "custom_presets": {},
    "bench_prompt": 512,
    "bench_predict": 128,
    "bench_threads": 8,
    "bench_ngl": 999
}

GPU_MEMORY_TIERS = [
    (16384, 43),
    (12288, 35),
    (8192, 28),
    (4096, 20),
    (0, 10)
]

GGUF_VALUE_TYPES = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}

PRESETS = {
    "🚀 Быстрый ответ": {
        "ctx": 2048, "temp": 0.3, "top_k": 40, "top_p": 0.9, "min_p": 0.05,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "batch_size": 2048, "flash_attn": "auto", "mirostat": 0
    },
    "💬 Чат / Диалог": {
        "ctx": 4096, "temp": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.05,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "batch_size": 2048, "flash_attn": "auto", "mirostat": 0
    },
    "📝 Креатив": {
        "ctx": 8192, "temp": 1.2, "top_k": 50, "top_p": 0.95, "min_p": 0.1,
        "repeat_penalty": 1.0, "presence_penalty": 0.15, "frequency_penalty": 0.05,
        "batch_size": 1024, "flash_attn": "auto", "mirostat": 0
    },
    "💻 Генерация кода": {
        "ctx": 8192, "temp": 0.1, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "batch_size": 2048, "flash_attn": "auto", "mirostat": 0
    },
    "🔬 Анализ данных": {
        "ctx": 16384, "temp": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "batch_size": 1024, "flash_attn": "auto", "mirostat": 0
    },
    "📖 Суммаризация": {
        "ctx": 8192, "temp": 0.2, "top_k": 40, "top_p": 0.9, "min_p": 0.05,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "batch_size": 1536, "flash_attn": "auto", "mirostat": 0
    },
    "🎯 Точный ответ": {
        "ctx": 4096, "temp": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "batch_size": 2048, "flash_attn": "auto", "mirostat": 0
    },
}
