# Llama.cpp Launcher

A powerful desktop application for launching and managing **llama.cpp** language model servers with an intuitive graphical interface. Built with Python and Tkinter, featuring GPU acceleration support, intelligent parameter estimation, and comprehensive model management.

## Features

### 🚀 Core Functionality
- **Easy Server Launch**: Simple GUI for starting llama.cpp inference servers
- **GPU Support**: Automatic GPU detection and layer offloading optimization (ngl)
- **Model Management**: Browse, configure, and manage GGUF/Ollama format models
- **Parameter Auto-Estimation**: Intelligent memory breakdown and token-per-second prediction

### 🎛️ Advanced Features
- **Multi-GPU Support**: Configure GPU offloading percentage and parallel slots
- **Preset Management**: Save and load custom server configurations
- **Benchmark Tool**: Performance testing with customizable prompt parameters
- **Draft Model Support**: Speculative decoding with draft models
- **Web UI Integration**: Built-in web interface toggle
- **Real-time Monitoring**: Live server logs with color-coded output
- **Tray Integration**: Minimize to system tray for background operation

### ⚙️ Parameter Control
- **Sampling**: Temperature, Top-K, Top-P, Min-P, Mirostat
- **Penalties**: Repeat, presence, and frequency penalties
- **Memory**: Context size, batch size, KV cache offloading
- **Optimization**: Flash attention, rope scaling, mmap/mlock
- **Custom Arguments**: Fallback support for custom llama.cpp parameters

### 🌍 Multi-Language Support
- Russian (РУ)
- English (EN)
- Extensible i18n system for additional languages

## Project Structure

```
llama.cpp_launcher/
├── main.py                 # Application entry point
├── app_state.py           # Central state management
├── LlamaLauncher.spec     # PyInstaller configuration
├── start.bat              # Windows batch launcher
│
├── core/                  # Core business logic
│   ├── config.py          # Constants and presets
│   ├── hardware.py        # GPU/CPU information detection
│   ├── gguf_parser.py     # GGUF metadata parsing
│   ├── estimator.py       # Memory and performance estimation
│   ├── benchmark_manager.py # Benchmarking utilities
│   ├── server_manager.py  # Llama.cpp server process management
│   └── i18n.py            # Internationalization system
│
├── ui/                    # User interface layer
│   ├── app.py             # Main UI application class
│   ├── main_window.py     # Window layout and tabs management
│   ├── styles.py          # Styling utilities
│   │
│   ├── components/        # Reusable UI components
│   │   ├── gpu_card.py    # GPU information widget
│   │   ├── parameters_panel.py  # Parameter configuration panels
│   │   ├── toast.py       # Toast notification system
│   │   └── tooltip.py     # Tooltip helper
│   │
│   └── tabs/              # Application tabs
│       ├── launch_tab.py  # Server launch interface
│       ├── models_tab.py  # Model management
│       └── benchmark_tab.py # Performance benchmarking
│
└── build/                 # PyInstaller build artifacts
```

## System Requirements

- **OS**: Windows 7+ (tested on Windows 11)
- **Python**: 3.10+
- **RAM**: 8GB minimum (16GB+ for quantization inference)
- **GPU**: Optional but recommended (NVIDIA/AMD with appropriate drivers)
- **Dependencies**: See requirements

## Installation

### From Source

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/YourUsername/llama.cpp_launcher.git
   cd llama.cpp_launcher
   pip install -r requirements.txt
   ```

2. **Download llama.cpp binary:**
   - Download from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
   - Extract to a known location (e.g., `C:\llama.cpp`)

3. **Launch the application:**
   ```bash
   python main.py
   ```
   Or use the provided `start.bat` script

### From Executable

1. Download the latest release `.exe` from GitHub releases
2. Run the executable directly - no installation required
3. Configure llama.cpp path and model directories in the first launch

## Quick Start

### First Run Setup
1. **Configure llama.cpp Path**: Point to your llama.cpp installation directory
2. **Add Model Directories**: Select folders containing GGUF model files
3. **Select Active Model**: Choose your default model for inference

### Launching a Server
1. **Go to Launch Tab**: Select your desired model and parameters
2. **Adjust Settings**: Modify GPU offloading, context size, sampling parameters
3. **Click Start Server**: The application will display live logs
4. **Access Web UI**: Open `http://localhost:8080` in your browser (default)

### Using Presets
1. **Adjust Parameters**: Configure all settings as desired
2. **Save Preset**: Click "Save as Preset" and name your configuration
3. **Load Preset**: Quickly apply saved configurations via dropdown menu

### Benchmarking
1. **Go to Benchmark Tab**: Configure test parameters
2. **Run Benchmark**: Test tokens-per-second performance
3. **View Results**: Analyze memory usage and throughput metrics

## Key Components

### AppState (app_state.py)
Central state management hub providing:
- Settings persistence (JSON-based)
- GGUF model metadata caching
- GPU information management
- Server process lifecycle
- Thread-safe operations

### GGUFMetadataParser (core/gguf_parser.py)
- Parses GGUF file format metadata
- Extracts model architecture information
- Determines required memory allocation
- Quantization type detection

### LlamaServerManager (core/server_manager.py)
- Launches llama.cpp server processes
- Manages stdout/stderr streaming
- Handles graceful shutdown
- Port availability detection

### Parameter Estimator (core/estimator.py)
- Calculates memory breakdown (model, KV cache, computation)
- Estimates tokens-per-second performance
- Determines optimal GPU layer offloading
- Profiling data collection

## Configuration

### Settings File
Settings are automatically saved to `settings.json` in the application directory:
```json
{
  "llama_dir": "C:\\llama.cpp",
  "model_dirs": ["C:\\models"],
  "host": "127.0.0.1",
  "port": 8080,
  "ngl": 50,
  "ctx": 4096,
  "threads": 8,
  "temperature": 0.8,
  "language": "en"
}
```

### Environment Variables
- `LLAMA_LAUNCHER_HOME`: Override default settings directory
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are used (llama.cpp)

## Known Issues & Fixes

### Bugs Found and Fixed (Code Review)

#### 1. **Downloader.py - Early Return in Loop** ✅ FIXED
**Issue**: The `download()` method had an early `return` statement inside the loop processing multiple files, causing only the first file to be downloaded.
```python
# BEFORE (Line 111)
for filename in files_to_download:
    try:
        filepath = hf_hub_download(...)
        downloaded.append((filename, filepath))
    except Exception as e:
        failed.append((filename, str(e)))
    
    return downloaded, failed  # ❌ Returns after first file!
```
**Fix**: Moved the return statement outside the loop to process all files.
```python
# AFTER
for filename in files_to_download:
    # ... same code ...
return downloaded, failed  # ✅ Returns after all files processed
```

#### 2. **FlagBuilder.py - Undefined Variable `kv_type`** ✅ FIXED
**Issue**: Variables `kv_type` were used on lines 163 and 187 but never defined, causing `NameError` when accessing `metadata['kv_size_mb'][kv_type]`.
**Fix**: Added variable definition from config at the beginning of `build()` method:
```python
kv_type = f"{config.kv_cache_k_type}_{config.kv_cache_v_type}"
```

#### 3. **Estimator.py - Duplicate Dictionary Key** ✅ FIXED
**Issue**: In `get_quant_speed_factor()`, the key `"Q1_0"` appeared twice in the dictionary with different values (1.38 and 2.8), causing the second value to override the first.
```python
# BEFORE
{
    "Q1_0": 1.38,  # Line 168
    ...
    "Q1_0": 2.8,   # Line 174 - ❌ Duplicate key!
}
```
**Fix**: Removed the duplicate `"Q1_0": 2.8` entry, keeping the correct value.

#### 4. **ServerManager.py - Missing State Update in `stop()`** ✅ FIXED
**Issue**: The `stop()` method did not set `self.running = False`, causing potential race conditions and state inconsistencies after server stop is called.
```python
# BEFORE
def stop(self):
    with self.lock:
        if not self.running or not self.process:
            return False
        self.stop_event.set()
        # ... terminate process ...
        return True  # ❌ self.running is still True!
```
**Fix**: Added `self.running = False` after setting stop_event:
```python
# AFTER
self.stop_event.set()
self.running = False  # ✅ Correctly update state
```

### Additional Issues Identified

#### 5. **Hardware Detection - Unicode Strings**
⚠️ **Note**: Various functions in `hardware.py` use Russian Cyrillic strings ("Не определена" = "Not determined") for fallback values. While this works on Windows 11 with UTF-8, it may cause issues on older systems or when logging to non-UTF-8 files. Consider using ASCII-only fallback messages for better compatibility.

#### 6. **GGUF Metadata Missing Fields**
⚠️ **Potential Issue**: The `flag_builder.py` accesses metadata keys like `metadata['is_moe']`, `metadata['kv_size_mb']`, `metadata['has_fused']`, and `metadata['has_ssm']` without checking if they exist. If the `_parse_gguf()` method doesn't return these keys, KeyError will occur. Recommend adding defensive checks:
```python
if metadata.get('is_moe'):  # Instead of metadata['is_moe']
```

## Troubleshooting

### GPU Not Detected
- Verify NVIDIA drivers are installed (`nvidia-smi` should work)
- Check CUDA installation for NVIDIA GPUs
- Try disabling GPU offload and running on CPU

### Server Won't Start
- Verify llama.cpp path in settings
- Check port availability (ensure port 8080 is free)
- Review logs for specific errors
- Try with reduced context size or batch size

### Memory Issues
- Lower context size (`ctx` parameter)
- Reduce batch size
- Decrease GPU offloading percentage
- Use larger quantization (Q3, Q4 instead of Q6/Q8)

### Model Not Found
- Ensure GGUF files have `.gguf` extension
- Check model directory paths in settings
- Verify file permissions for read access

## Building Executable

Using PyInstaller (configured in `LlamaLauncher.spec`):

```bash
pip install pyinstaller
pyinstaller LlamaLauncher.spec
```

The compiled `.exe` will be in the `dist/` directory.

## API Integration

The application manages llama.cpp server via:
- **Default Endpoint**: `http://localhost:8080/api/chat/completions`
- **Web UI**: `http://localhost:8080/` (OpenAI-compatible interface)

Compatible with any OpenAI-compatible client library:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)
```

## Performance Tips

1. **GPU Offloading**: Use highest `ngl` value your VRAM supports
2. **Batch Size**: Increase for throughput, decrease for latency
3. **Flash Attention**: Enable when available (GPU memory saver)
4. **Context Caching**: Use `cache_prompt=on` for repeated context
5. **Thread Count**: Match CPU core count for optimal performance

## Contributing

Contributions are welcome! Areas of interest:
- Additional language translations
- New GPU backend support (AMD, Intel)
- Enhanced performance estimation algorithms
- UI/UX improvements

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The underlying inference engine
- [Tkinter](https://docs.python.org/3/library/tkinter.html) - GUI framework
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) - Modern styling

## Changelog

### v1.0.0 (Initial Release)
- Core server launch functionality
- GPU detection and optimization
- Model management interface
- Parameter estimation system
- Benchmarking tools
- Multi-language support
- Windows executable distribution

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Include system information (OS, Python version, GPU model)
- Attach relevant logs from the `logs/` directory

---

## Code Quality & Maintenance

### Recent Improvements
- ✅ Fixed critical bug in batch file downloads (downloader.py)
- ✅ Resolved undefined variable issues (flag_builder.py)
- ✅ Removed duplicate dictionary keys (estimator.py)
- ✅ Fixed thread state management (server_manager.py)

### Recommended Improvements for Future Development

1. **Type Safety**
   - Add type hints to all function parameters
   - Use `typing.TypedDict` for complex dictionary structures
   - Consider using Pydantic for data validation

2. **Error Handling**
   - Add `.get()` method with defaults when accessing metadata dictionaries
   - Implement proper logging with Python's `logging` module
   - Add validation for config values before use

3. **Testing**
   - Add unit tests for `estimator.py` calculations
   - Test GPU detection on various hardware configurations
   - Mock `subprocess` calls in server_manager tests

4. **Code Organization**
   - Extract metadata dictionary access patterns into helper methods
   - Consider using a configuration class instead of dictionaries
   - Separate UI logic from business logic more cleanly

5. **Unicode Compatibility**
   - Replace Cyrillic fallback strings with ASCII equivalents
   - Ensure all file operations explicitly use UTF-8 encoding
   - Test on Windows 7/8 systems

### Code Statistics
- **Total Files**: ~25 Python modules
- **Core Modules**: ~8 (hardware, gguf_parser, estimator, server_manager, etc.)
- **UI Modules**: ~8 (app, components, tabs)
- **Lines of Code**: ~4000+ (excluding comments)
- **Languages**: Python 3.10+, Tkinter (UI)

## Contributing

Found a bug or want to improve this project? Consider:
1. Checking existing issues first
2. Writing a clear bug report with reproduction steps
3. Testing fixes on your hardware configuration
4. Following the existing code style

## License

This project is provided as-is for educational and personal use.

**Created with ❤️ for the open-source LLM community**
