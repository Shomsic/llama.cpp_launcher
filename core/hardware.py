import os
import subprocess
import re
import ctypes
from ctypes import wintypes
import hashlib

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
        return None
    except Exception:
        pass
    return None

def _get_gpu_info_via_setupapi():
    """Fallback: получает VRAM через SetupAPI + реестр (работает без wmic)."""
    try:
        return _get_gpu_info_via_enum_displayDevices()
    except Exception:
        return 0

def _get_gpu_info_via_enum_displayDevices():
    """Получает VRAM через EnumDisplayDevices — работает на любом Windows."""
    try:
        import winreg
        vram_total = 0
        gpu_class_guid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
        gpu_key_path = r"SYSTEM\CurrentControlSet\Control\Class\\" + gpu_class_guid
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
                try:
                    driver_desc, __ = winreg.QueryValueEx(subkey, "DriverDesc")
                    if "Microsoft Basic" in str(driver_desc):
                        winreg.CloseKey(subkey)
                        continue
                except OSError: pass
                try:
                    mem_size, reg_type = winreg.QueryValueEx(subkey, "HardwareInformation.MemorySize")
                    if reg_type == winreg.REG_QWORD: vram_mb = mem_size // (1024 * 1024)
                    elif reg_type == winreg.REG_SZ: vram_mb = int(str(mem_size)) // (1024 * 1024)
                    elif reg_type == winreg.REG_DWORD: vram_mb = mem_size // (1024 * 1024)
                    else: vram_mb = 0
                    if vram_mb > 0 and vram_mb < 65536:
                        if vram_mb > vram_total: vram_total = vram_mb
                except OSError: pass
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
        gpu_key_path = r"SYSTEM\CurrentControlSet\Control\Class\\" + gpu_class_guid
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
                except OSError: pass
                winreg.CloseKey(subkey)
            except OSError:
                break
        winreg.CloseKey(gpu_key)
        return "Не определена"
    except Exception:
        return "Не определена"

def get_gpu_info():
    """Возвращает суммарный VRAM всех GPU в МБ."""
    vram_mb = 0
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            vram_mb = sum(int(re.sub(r'[^\d]', '', line)) for line in lines if line.strip())
            return vram_mb
    except Exception: pass
    wmic_output = _run_wmic(["wmic", "path", "Win32_VideoController", "get", "AdapterRAM"])
    if wmic_output:
        lines = wmic_output.split("\n")[1:]
        for line in lines:
            line = line.strip()
            if line and line.isdigit(): vram_mb += int(line) // (1024 * 1024)
        if vram_mb > 0: return vram_mb
    wmic_output = _run_wmic(["wmic", "path", "Win32_VideoController", "get", "AdapterRAM,Name", "/format:list"])
    if wmic_output:
        for line in wmic_output.split("\n"):
            if "AdapterRAM=" in line:
                val = line.split("=", 1)[1].strip()
                if val and val.isdigit(): vram_mb += int(val) // (1024 * 1024)
        if vram_mb > 0: return vram_mb
    vram_mb = _get_gpu_info_via_setupapi()
    return vram_mb

def get_gpu_list():
    """Возвращает список словарей с информацией о GPU."""
    gpus = []
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        res_vram = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        res_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        res_free = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        res_pcie = subprocess.run(
            ["nvidia-smi", "--query-gpu=pcie.link.width.current,pcie.link.gen.current", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        if res_vram.returncode == 0 and res_vram.stdout.strip():
            vrams = res_vram.stdout.strip().split("\n")
            names = res_name.stdout.strip().split("\n") if res_name.returncode == 0 else ["Unknown"] * len(vrams)
            frees = res_free.stdout.strip().split("\n") if res_free.returncode == 0 else ["0"] * len(vrams)
            pcies = res_pcie.stdout.strip().split("\n") if res_pcie.returncode == 0 else ["1,1"] * len(vrams)
            
            for i in range(len(vrams)):
                p_width, p_gen = 1, 1
                try:
                    if i < len(pcies):
                        p_parts = pcies[i].split(",")
                        p_width = int(p_parts[0].strip())
                        p_gen = int(p_parts[1].strip())
                except: pass

                gpus.append({
                    "index": i,
                    "name": names[i] if i < len(names) else "Unknown", 
                    "vram_total": int(vrams[i].strip()), 
                    "vram_free": int(frees[i].strip()) if i < len(frees) else 0,
                    "pcie_width": p_width,
                    "pcie_gen": p_gen,
                    "bandwidth": p_width * p_gen
                })
            return gpus
    except Exception: pass
    try:
        import winreg
        gpu_class_guid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
        gpu_key_path = r"SYSTEM\CurrentControlSet\Control\Class\\" + gpu_class_guid
        gpu_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, gpu_key_path, 0, winreg.KEY_READ)
        idx = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(gpu_key, idx)
                subkey = winreg.OpenKey(gpu_key, subkey_name, 0, winreg.KEY_READ)
                idx += 1
                try:
                    name, __ = winreg.QueryValueEx(subkey, "DriverDesc")
                    name = str(name)
                except: name = "Unknown"
                vram_mb = 0
                try:
                    mem_size, reg_type = winreg.QueryValueEx(subkey, "HardwareInformation.MemorySize")
                    if reg_type == winreg.REG_QWORD: vram_mb = mem_size // (1024 * 1024)
                    elif reg_type == winreg.REG_SZ: vram_mb = int(str(mem_size)) // (1024 * 1024)
                    elif reg_type == winreg.REG_DWORD: vram_mb = mem_size // (1024 * 1024)
                except: pass
                if vram_mb > 0: gpus.append({
                    "index": idx-1,
                    "name": name, 
                    "vram_total": vram_mb, 
                    "vram_free": 0,
                    "pcie_width": 1,
                    "pcie_gen": 1,
                    "bandwidth": 1
                })
                winreg.CloseKey(subkey)
            except OSError: break
        winreg.CloseKey(gpu_key)
    except Exception: pass
    return gpus

def get_gpu_name():
    gpu_name = "Не определена"
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0].strip()
            return gpu_name
    except Exception: pass
    wmic_output = _run_wmic(["wmic", "path", "Win32_VideoController", "get", "Name", "/format:list"])
    if wmic_output:
        for line in wmic_output.split("\n"):
            if "Name=" in line:
                name = line.split("=", 1)[1].strip()
                if name and "Microsoft" not in name: return name
    gpu_name = _get_gpu_name_via_enumDisplayDevices()
    return gpu_name

def get_total_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        try:
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            memoryStatus = MEMORYSTATUSEX()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
            return memoryStatus.ullTotalPhys / (1024 ** 3)
        except: pass
        return 16

def get_cpu_cores():
    return os.cpu_count() or 4

def get_gpu_count():
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().split("\n"))
    except: pass
    return 0

def get_total_vram_free():
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            return sum(int(x) for x in result.stdout.strip().split("\n"))
    except: pass
    return 0

def get_best_gpu_index():
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            best_idx = 0
            max_free = -1
            for line in result.stdout.strip().split("\n"):
                idx, free = map(int, line.split(","))
                if free > max_free:
                    max_free = free
                    best_idx = idx
            return best_idx
    except: pass
    return 0

def get_best_gpu_vram():
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, startupinfo=startupinfo
        )
        if result.returncode == 0 and result.stdout.strip():
            return max(int(x) for x in result.stdout.strip().split("\n"))
    except: pass
    return 0

def get_available_ram_mb():
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
        return 8192
