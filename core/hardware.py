import os
import subprocess
import re
import ctypes
from ctypes import wintypes


def _run_wmic(args, timeout=5):
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        result = subprocess.run(
            args,
            capture_output=True, text=True, timeout=timeout,
            startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        # wmic отсутствует
        return None
    except Exception:
        pass
    return None


def _get_gpu_info_via_setupapi():
    """Fallback: получает VRAM через SetupAPI + реестр (работает без wmic)."""
    try:
        # Загружаем библиотеки
        cfgmgr32 = ctypes.WinDLL('cfgmgr32')
        setupapi = ctypes.WinDLL('setupapi')
        advapi32 = ctypes.WinDLL('advapi32')

        # CM_Locate_DevNodeA, CM_Get_Parent, CM_Get_Device_ID
        # Используем простой подход: перечисляем GPU через EnumDisplayDevices
        return _get_gpu_info_via_enum_display_devices()
    except Exception:
        return 0


def _get_gpu_info_via_enum_displayDevices():
    """Получает VRAM через EnumDisplayDevices — работает на любом Windows."""
    try:
        import winreg
        vram_total = 0

        # Перечисляем видеоадаптеры через реестр
        # Видеокарты находятся в: HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}
        gpu_class_guid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
        gpu_key_path = r"SYSTEM\CurrentControlSet\Control\Class" + "\\" + gpu_class_guid

        try:
            gpu_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, gpu_key_path, 0, winreg.KEY_READ)
        except OSError:
            return 0

        idx = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(gpu_key, idx)
                subkey = winreg.OpenKey(gpu_key, subkey_name, 0, winreg.KEY_READ)
                idx += 1

                # Пропускаем Microsoft Basic Display Adapter
                try:
                    driver_desc, __ = winreg.QueryValueEx(subkey, "DriverDesc")
                    if "Microsoft Basic" in str(driver_desc):
                        winreg.CloseKey(subkey)
                        continue
                except OSError:
                    pass

                # Читаем HardwareInformation.MemorySize (REG_QWORD или REG_SZ)
                try:
                    mem_size, reg_type = winreg.QueryValueEx(subkey, "HardwareInformation.MemorySize")
                    if reg_type == winreg.REG_QWORD:
                        vram_mb = mem_size // (1024 * 1024)
                    elif reg_type == winreg.REG_SZ:
                        vram_mb = int(str(mem_size)) // (1024 * 1024)
                    elif reg_type == winreg.REG_DWORD:
                        vram_mb = mem_size // (1024 * 1024)
                    else:
                        vram_mb = 0

                    if vram_mb > 0 and vram_mb < 65536:  # Разумный上限: < 64 GB
                        if vram_mb > vram_total:
                            vram_total = vram_mb
                except OSError:
                    pass

                winreg.CloseKey(subkey)
            except OSError:
                break

        winreg.CloseKey(gpu_key)
        return vram_total
    except Exception:
        return 0


def _get_gpu_name_via_enumDisplayDevices():
    """Fallback: получает имя GPU через EnumDisplayDevices."""
    try:
        import winreg
        gpu_class_guid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
        gpu_key_path = r"SYSTEM\CurrentControlSet\Control\Class" + "\\" + gpu_class_guid

        try:
            gpu_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, gpu_key_path, 0, winreg.KEY_READ)
        except OSError:
            return "Не определена"

        idx = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(gpu_key, idx)
                subkey = winreg.OpenKey(gpu_key, subkey_name, 0, winreg.KEY_READ)
                idx += 1

                try:
                    driver_desc, __ = winreg.QueryValueEx(subkey, "DriverDesc")
                    desc_str = str(driver_desc).strip()
                    if desc_str and "Microsoft Basic" not in desc_str:
                        winreg.CloseKey(subkey)
                        winreg.CloseKey(gpu_key)
                        return desc_str
                except OSError:
                    pass

                winreg.CloseKey(subkey)
            except OSError:
                break

        winreg.CloseKey(gpu_key)
        return "Не определена"
    except Exception:
        return "Не определена"


def get_gpu_info():
    """Возвращает VRAM в МБ. Пробует NVIDIA → AMD/Intel (wmic) → Reестр fallback."""
    vram_mb = 0

    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    # 1. NVIDIA (nvidia-smi)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if lines:
                mem_str = lines[0].strip()
                mem_str = re.sub(r'[^\d.]', '', mem_str)
                if mem_str:
                    return int(float(mem_str))
    except Exception:
        pass

    # 2. AMD/Intel — пробуем wmic
    wmic_output = _run_wmic(["wmic", "path", "Win32_VideoController", "get", "AdapterRAM"])
    if wmic_output:
        lines = wmic_output.split("\n")[1:]  # Пропускаем заголовок
        for line in lines:
            line = line.strip()
            if line and line.isdigit():
                vram_mb = int(line) // (1024 * 1024)
                if vram_mb > 0:
                    return vram_mb

    # 3. Пробуем wmic с форматом list
    wmic_output = _run_wmic(["wmic", "path", "Win32_VideoController", "get", "AdapterRAM,Name", "/format:list"])
    if wmic_output:
        for line in wmic_output.split("\n"):
            if "AdapterRAM=" in line:
                val = line.split("=", 1)[1].strip()
                if val and val.isdigit():
                    vram_mb = int(val) // (1024 * 1024)
                    if vram_mb > 0:
                        return vram_mb

    # 4. Fallback: реестр Windows (работает без wmic)
    vram_mb = _get_gpu_info_via_setupapi()
    if vram_mb > 0:
        return vram_mb

    return vram_mb


def get_gpu_name():
    """Возвращает имя GPU. Пробует NVIDIA → AMD/Intel (wmic) → Реестр fallback."""
    gpu_name = "Не определена"

    # 1. NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0].strip()
            return gpu_name
    except Exception:
        pass

    # 2. AMD/Intel через wmic
    wmic_output = _run_wmic(["wmic", "path", "Win32_VideoController", "get", "Name", "/format:list"])
    if wmic_output:
        for line in wmic_output.split("\n"):
            if "Name=" in line:
                name = line.split("=", 1)[1].strip()
                if name and "Microsoft" not in name:
                    return name

    # 3. Fallback: реестр Windows
    gpu_name = _get_gpu_name_via_enumDisplayDevices()
    return gpu_name

def get_total_ram_gb():
    """Определяет общий объём RAM системы"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Для Windows fallback
        try:
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]
            memoryStatus = MEMORYSTATUSEX()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
            return memoryStatus.ullTotalPhys / (1024 ** 3)
        except:
            pass
        return 16  # Дефолт если не удалось определить

def get_cpu_cores():
    return os.cpu_count() or 4
