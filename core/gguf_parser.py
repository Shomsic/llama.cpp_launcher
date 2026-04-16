import os
import re
import struct

try:
    import gguf
    from gguf import GGUFReader
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

from .config import GGUF_VALUE_TYPES

class GGUFMetadataParser:
    def __init__(self):
        self._cache = {}

    def get_cached_or_parse(self, model_path):
        if not model_path or not os.path.isfile(model_path):
            return {}
        if model_path not in self._cache:
            self._cache[model_path] = self.parse_metadata(model_path)
        return self._cache.get(model_path, {})

    def parse_metadata(self, model_path):
        """Парсит метаданные GGUF файла.

        Приоритет: быстрый бинарный парсер (читает только заголовок, не грузит тензоры)
        → fallback: gguf-пакет (медленнее, mmapит весь файл).
        """
        metadata = {}
        if not model_path or not os.path.isfile(model_path):
            return metadata

        # Быстрый путь: собственный бинарный парсер — читает только метаданные,
        # пропускает тензорные данные, работает за ~0.1-0.5 с вместо 10-15 с.
        try:
            return self._parse_gguf_metadata_fallback(model_path)
        except Exception as e:
            # Логируем ошибку для отладки, но не прерываем выполнение
            print(f"[GGUF] Бинарный парсер не смог прочитать {model_path}: {e}")

        # Медленный fallback: gguf-пакет (mmapит весь файл)
        if HAS_GGUF:
            try:
                return self._parse_gguf_metadata_with_package(model_path)
            except Exception as e:
                metadata["__error__"] = f"Ошибка (gguf-пакет): {e}"
                return metadata

        metadata["__error__"] = "Не удалось разобрать GGUF файл (установите пакет gguf)"
        return metadata

    def _parse_gguf_metadata_with_package(self, model_path):
        metadata = {}
        reader = GGUFReader(model_path, "c")

        metadata["__gguf_version__"] = getattr(reader, "version", "?")
        metadata["__tensor_count__"] = len(getattr(reader, "tensors", []))
        metadata["__metadata_count__"] = len(getattr(reader, "fields", {}))

        for key, field in reader.fields.items():
            metadata[key] = self._decode_gguf_field(field)
        return metadata

    def _parse_gguf_metadata_fallback(self, model_path):
        metadata = {}
        with open(model_path, "rb") as handle:
            magic = handle.read(4)
            if magic != b"GGUF":
                raise ValueError("неверная сигнатура GGUF")

            version = self._read_struct(handle, "<I")
            if version < 1 or version > 3:
                raise ValueError(f"неподдерживаемая версия GGUF: {version}")

            tensor_count = self._read_gguf_count(handle, version)
            metadata_count = self._read_gguf_count(handle, version)

            metadata["__gguf_version__"] = version
            metadata["__tensor_count__"] = tensor_count
            metadata["__metadata_count__"] = metadata_count

            for _ in range(metadata_count):
                key = self._read_gguf_string(handle, version)
                value_type = self._read_struct(handle, "<I")
                metadata[key] = self._read_gguf_value(handle, version, value_type)

        return metadata

    def _decode_gguf_field(self, field):
        if field is None:
            return None

        # Сначала пробуем использовать встроенный метод contents (новые версии gguf)
        if hasattr(field, "contents") and callable(field.contents):
            try:
                content = field.contents()
                return self._normalize_gguf_value(content)
            except Exception:
                pass

        # Для старых версий или fallback (через data индексы в parts)
        if hasattr(field, "parts") and hasattr(field, "data"):
            data_indices = field.data
            parts = field.parts
            
            if data_indices and len(data_indices) == 1:
                idx = data_indices[0]
                if idx < len(parts):
                    return self._normalize_gguf_value(parts[idx])
            elif data_indices:
                return [self._normalize_gguf_value(parts[idx]) for idx in data_indices if idx < len(parts)]

        # Самый старый fallback (поиск value)
        raw_value = None
        for attr in ("value", "data"):
            if hasattr(field, attr):
                raw_value = getattr(field, attr)
                if raw_value is not None:
                    break

        return self._normalize_gguf_value(raw_value)

    def _normalize_gguf_value(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        
        # Обработка массивов NumPy (gguf может возвращать ndarray)
        if hasattr(value, "tobytes") and hasattr(value, "dtype"):
            # Если это массив символов (uint8 для строк)
            if value.dtype.name == "uint8":
                return value.tobytes().decode("utf-8", errors="replace")
        
        if hasattr(value, "tolist"):
            # Оптимизация: не загружаем гигантские массивы (токены и т.д.) целиком в память
            try:
                length = len(value)
                if length > 32:
                    # Берём срез только для отображения в UI
                    head = value[:5]
                    # Для NumPy массивов строк/байтов
                    if hasattr(head, "dtype") and head.dtype.name == "uint8" and head.ndim > 1:
                         # Если это массив строк в gguf (список байтовых массивов)
                         str_items = [self._normalize_gguf_value(x) for x in head]
                    else:
                         str_items = [str(self._normalize_gguf_value(x)) for x in head]
                    return f"[{', '.join(str_items)}, ... ({length} элементов)]"
            except Exception:
                pass

            lst = value.tolist()
            if isinstance(lst, list):
                if len(lst) == 1:
                    return self._normalize_gguf_value(lst[0])
                return [self._normalize_gguf_value(item) for item in lst]
            return lst

        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            normalized = [self._normalize_gguf_value(item) for item in value]
            if len(normalized) > 32:
                str_items = [str(x) for x in normalized[:5]]
                return f"[{', '.join(str_items)}, ... ({len(normalized)} элементов)]"
            return normalized

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _read_struct(self, handle, fmt):
        size = struct.calcsize(fmt)
        data = handle.read(size)
        if len(data) != size:
            raise EOFError("неожиданный конец GGUF файла")
        return struct.unpack(fmt, data)[0]

    def _read_gguf_count(self, handle, version):
        return self._read_struct(handle, "<Q" if version >= 2 else "<I")

    def _read_gguf_string(self, handle, version):
        length = self._read_gguf_count(handle, version)
        if length < 0 or length > 64 * 1024 * 1024:
            raise ValueError(f"некорректная длина строки: {length}")
        data = handle.read(length)
        if len(data) != length:
            raise EOFError("строка GGUF обрезана")
        return data.decode("utf-8", errors="replace")

    def _read_gguf_value(self, handle, version, value_type):
        if value_type == 8:
            return self._read_gguf_string(handle, version)
        if value_type == 9:
            item_type = self._read_struct(handle, "<I")
            item_count = self._read_gguf_count(handle, version)
            # Оптимизация: для гигантских массивов (токены, merges и т.п.)
            # читаем только первые N элементов, остальные пропускаем побайтово.
            MAX_ITEMS_TO_LOAD = 32
            if item_count > MAX_ITEMS_TO_LOAD and item_type not in (8, 9):
                # Скалярный тип — можно вычислить размер и seek
                if item_type in GGUF_VALUE_TYPES:
                    fmt, __ = GGUF_VALUE_TYPES[item_type]
                    item_size = struct.calcsize(fmt)
                    head_items = [self._read_gguf_value(handle, version, item_type)
                                  for _ in range(MAX_ITEMS_TO_LOAD)]
                    skip_bytes = (item_count - MAX_ITEMS_TO_LOAD) * item_size
                    handle.seek(skip_bytes, 1)  # 1 = SEEK_CUR
                    return f"[{', '.join(map(str, head_items[:5]))}, ... ({item_count} элементов)]"
            items = [self._read_gguf_value(handle, version, item_type) for _ in range(item_count)]
            if len(items) > 32:
                return f"[{', '.join(map(str, items[:5]))}, ... ({len(items)} элементов)]"
            return items
        if value_type not in GGUF_VALUE_TYPES:
            raise ValueError(f"неподдерживаемый тип GGUF: {value_type}")

        fmt, __ = GGUF_VALUE_TYPES[value_type]
        return self._read_struct(handle, fmt)

def validate_gguf_file(model_path):
    if not model_path or not os.path.isfile(model_path):
        return False, "Файл не существует"

    try:
        size = os.path.getsize(model_path)
        if size < 1024 * 1024:
            return False, "Файл слишком маленький (< 1 МБ)"
        with open(model_path, 'rb') as f:
            magic = f.read(4)
            if magic not in (b'GGUF', b'GGML', b'GGJT', b'GGHB'):
                return False, "Файл не имеет сигнатуру GGUF (возможно повреждён)"
        return True, "OK"
    except OSError as e:
        return False, f"Ошибка чтения файла: {e}"
    except Exception as e:
        return False, f"Неожиданная ошибка: {e}"

def get_quant_from_metadata(metadata):
    """Определяет тип квантизации по метаданным (general.file_type)."""
    ftype = metadata.get("general.file_type")
    
    # Справочник типов GGUF (из llama.cpp/gguf.py)
    FTYPES = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 
        7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
        10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
        14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
        18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "IQ3_XXS",
        22: "IQ1_S", 23: "IQ4_NL", 24: "IQ3_S", 25: "IQ2_S",
        26: "IQ4_XS", 27: "I8", 28: "I16", 29: "I32", 30: "I64",
        31: "F64", 32: "IQ1_M", 33: "BF16", 34: "Q4_0_4_4", 35: "Q4_0_4_8",
        36: "Q4_0_8_8", 37: "TQ1_0", 38: "TQ2_0", 39: "IQ4_XS", 40: "MXFP4",
        41: "Q1_0"  # Bonsai 1-bit
    }
    
    if isinstance(ftype, (int, float)):
        return FTYPES.get(int(ftype), "Unknown")
    return "Unknown"

def extract_quant_from_filename(path):
    name = os.path.basename(path).upper()
    quant_types = [
        "IQ1_M", "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
        "Q8_0", "Q8_K", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
        "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q3_K_M", "Q3_K_L", "Q3_K_S",
        "Q2_K", "Q1_0", "Q1_K", "F16", "F32", "F64", "BF16", "I16", "I32", "I64",
        "TQ1_0", "TQ2_0", "MXFP4"
    ]
    for q in quant_types:
        if q in name:
            return q
    return "Unknown"

def get_quant_description(quant_type):
    """Возвращает описание типа квантизации.
    
    IQ1 бывает двух видов:
      - IQ1_S / IQ1_M: веса {0, 1}  — 1-бит беззнаковый (только 0 и 1)
      - TQ1_0:          веса {-1, 0, 1} — 1-бит троичный (ternary)
    """
    descriptions = {
        # --- Float (не квантованные) ---
        "F64":     "64-bit IEEE 754 (не квантованная, максимальная точность)",
        "F32":     "32-bit IEEE 754 (не квантованная, стандарт обучения)",
        "F16":     "16-bit IEEE 754 half-precision (половинная точность)",
        "BF16":    "16-bit Bfloat16 (сокращённый FP32, лучше для DL)",
        # --- Integer raw ---
        "I8":      "8-bit целое (сырые активации, не для инференса)",
        "I16":     "16-bit целое (сырые активации)",
        "I32":     "32-bit целое (сырые активации)",
        "I64":     "64-bit целое (сырые активации)",
        # --- Q8 ---
        "Q8_0":    "8-bit квантизация, блок 32 (высокое качество, большой размер)",
        "Q8_K":    "8-bit квантизация K-тип, блок 256 (смешанный)",
        "Q8_1":    "8-bit квантизация, блок 32 (вариант 1)",
        # --- Q6 ---
        "Q6_K":    "6-bit квантизация K-тип, блок 256 (хорошее качество)",
        # --- Q5 ---
        "Q5_0":    "5-bit квантизация, блок 32, веса {0..31}",
        "Q5_1":    "5-bit квантизация, блок 32, вариант 1",
        "Q5_K_M":  "5-bit K-тип medium (рекомендуется для большинства задач)",
        "Q5_K_S":  "5-bit K-тип small (компактнее Q5_K_M, чуть хуже качество)",
        # --- Q4 ---
        "Q4_0":    "4-bit квантизация, блок 32, базовый метод",
        "Q4_1":    "4-bit квантизация, блок 32, вариант 1 (лучше Q4_0)",
        "Q4_K_M":  "4-bit K-тип medium (лучший баланс качество/размер в 4-bit)",
        "Q4_K_S":  "4-bit K-тип small (меньше Q4_K_M, чуть ниже точность)",
        # --- Q3 ---
        "Q3_K_M":  "3-bit K-тип medium (агрессивная квантизация, ощутимая потеря)",
        "Q3_K_L":  "3-bit K-тип large (лучше Q3_K_M, крупнее)",
        "Q3_K_S":  "3-bit K-тип small (самый компактный из 3-bit)",
        # --- Q2 ---
        "Q2_K":    "2-bit K-тип, блок 256 (очень агрессивная, заметная потеря)",
        # --- IQ (i-quant) серия — улучшенные методы от llama.cpp ---
        "IQ4_XS":  "4-bit i-quant XS (чуть меньше Q4_K_S, схожее качество)",
        "IQ4_NL":  "4-bit i-quant нелинейный (нелинейные шаги квантизации)",
        "IQ3_XXS": "3-bit i-quant XXS (очень малый, лучше Q2_K по качеству)",
        "IQ3_S":   "3-bit i-quant S (компактнее Q3_K_S при близком качестве)",
        "IQ2_XXS": "2-bit i-quant XXS (экстремально малый, применим для больших моделей)",
        "IQ2_XS":  "2-bit i-quant XS (чуть лучше IQ2_XXS)",
        "IQ2_S":   "2-bit i-quant S (лучший из 2-bit i-quant)",
        # --- IQ1 — 1-bit квантизация, два принципиально разных подхода ---
        # Беззнаковая: каждый вес принимает значения {0, 1} — 1 бит/вес
        "IQ1_S":   "1-bit i-quant S — веса {0, 1} (беззнаковый 1-bit, макс. сжатие)",
        "IQ1_M":   "1-bit i-quant M — веса {0, 1} (беззнаковый 1-bit, чуть крупнее IQ1_S)",
        # Троичная: каждый вес принимает значения {-1, 0, +1} — ~1.585 бит/вес
        "TQ1_0":   "1-bit троичная (Ternary) — веса {-1, 0, +1} (~1.6 бит/вес, лучше IQ1)",
        "TQ2_0":   "2-bit троичная (Ternary) — веса {-1, 0, +1} с 2-бит упаковкой",
        # --- MXFP ---
        "MXFP4":   "4-bit MX Float (Microsoft MX формат, аппаратное ускорение)",
        # --- Q1 (старый / редкий формат) ---
        "Q1_0":    "1-bit квантизация, веса {0, 1} (устаревший, редко используется)",
        "Q1_K":    "1-bit K-тип, блок 256 (экспериментальный)",
    }
    return descriptions.get(quant_type, f"{quant_type} (описание отсутствует)")
