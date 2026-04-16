"""
Основной класс UI приложения (LlamaLauncherApp).
"""
import os
import sys
import threading
import subprocess
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter import BOTH, LEFT, RIGHT, TOP, BOTTOM, HORIZONTAL, VERTICAL, W, E, N, S, X, Y, END, NORMAL, DISABLED, SUNKEN, SOLID, EXTENDED
from pathlib import Path
import ctypes
import re
import time
import webbrowser
import shlex
import socket
import copy
from queue import Queue

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
except ImportError:
    import tkinter.ttk as ttkb
    ttkb = None

from core.i18n import _, I18n
from core.config import PRESETS, LOG_DIR, SETTINGS_FILE, DEFAULT_SETTINGS, ensure_log_dir
from core.hardware import get_gpu_info, get_gpu_name
from core.gguf_parser import GGUFMetadataParser, extract_quant_from_filename, get_quant_description, get_quant_from_metadata
from core.server_manager import LlamaServerManager
from core.flag_builder import FlagBuilder, ServerConfig
from core.estimator import ProfilingData, estimate_memory_breakdown, estimate_tokens_per_second as _est_tps
from core.ai_tuner import AITuner
from ui.components.tooltip import TooltipManager

from ui.components.toast import ToastManager
from ui.components.gpu_card import GpuCard
from ui.components.selectable_label import SelectableLabel
from ui.components.parameters_panel import PresetPanel, BasicParamsPanel, GenerationParamsPanel, AdvancedParamsPanel
from ui.tabs.benchmark_tab import BenchmarkTab


class LlamaLauncherApp:
    """Главный класс UI — использует tabs из ui/tabs/."""

    def __init__(self, root, state):
        self.root = root
        self.state = state
        state.init_tk_vars(root)
        self.settings = state.settings
        
        self.log_queue = Queue()
        self._schedule_log_flush()

        self.root.title(_("app_title"))
        geom = self.settings.get("window_geometry", "1000x700")
        self.root.geometry(geom)
        self.root.minsize(800, 600)

        # Кэш метаданных (используем кеш из state)
        self._metadata_cache = state.gguf_parser._cache
        self._metadata_loading = set()

        # Hotkeys
        self.root.bind("<Control-s>", lambda e: self.start_server() if not self.state.running else None)
        self.root.bind("<Control-q>", lambda e: self.stop_server() if self.state.running else None)
        self.root.bind("<Control-r>", lambda e: self.refresh_models())
        self.root.bind("<F5>", lambda e: self.refresh_models())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind_all("<Control-c>", self._global_copy)

        # Стили и UI
        self._setup_styles()
        self.toast = ToastManager(root)
        self.tuner = AITuner(state, self)
        self._create_ui()

        # Подписка на обновления state
        state.add_update_listener(self._update_memory_estimate)

        # Загрузка состояния (откладываем на 100мс для стабильности mainloop)
        self.root.after(100, self._refresh_startup_state)

    def _setup_styles(self):
        if ttkb:
            self.style = ttkb.Style("darkly")
        else:
            self.style = ttk.Style()
            self.style.theme_use("clam")

        self.root.configure(bg="#252a31")

        self.style.configure(".", font=("Segoe UI", 10))
        self.style.configure("TNotebook", background="#252a31", borderwidth=0, tabmargins=(0, 0, 0, 0))
        self.style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"),
                             background="#39424c", foreground="#d6dde5")
        self.style.map("TNotebook.Tab",
                       background=[("selected", "#586574"), ("active", "#46515c")],
                       foreground=[("selected", "#f1f4f7")])
        self.style.configure("TLabelframe", background="#2f363f", borderwidth=1, relief="solid")
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"),
                             foreground="#d9e1e8", background="#2f363f")
        self.style.configure("TFrame", background="#252a31")
        self.style.configure("TLabel", background="#252a31", foreground="#d8e0e8")
        self.style.configure("TEntry", fieldbackground="#1f242a", foreground="#e5ebf0")
        self.style.configure("TSpinbox", fieldbackground="#1f242a", foreground="#e5ebf0")
        self.style.configure("TCombobox", fieldbackground="#1f242a", foreground="#e5ebf0")
        self.style.configure("Card.TFrame", background="#2f363f", relief="solid", borderwidth=1)
        self.style.configure("Header.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        self.style.configure("Success.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 10))
        self.style.configure("Danger.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 10))
        self.style.configure("LaunchTitle.TLabel", font=("Segoe UI", 17, "bold"),
                             foreground="#edf2f7", background="#252a31")
        self.style.configure("Subtle.TLabel", font=("Segoe UI", 9), foreground="#97a8b8", background="#252a31")

        for name, color in [("green", "#6f9c84"), ("warning", "#c59a66"), ("red", "#b86c70")]:
            self.style.configure(f"{name}.Horizontal.TProgressbar",
                                foreground=color, background=color,
                                troughcolor="#1f242a", bordercolor="#1f242a",
                                lightcolor=color, darkcolor=color)

    def _create_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Вкладка запуска
        launch_frame = ttk.Frame(self.notebook)
        self.notebook.add(launch_frame, text=_("tab_launch"))
        self._create_launch_tab(launch_frame)

        # Вкладка моделей
        models_frame = ttk.Frame(self.notebook)
        self.notebook.add(models_frame, text=_("tab_models"))
        self._create_models_tab(models_frame)

        # Вкладка настроек
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text=_("settings"))
        self._create_settings_tab(settings_frame)

        # Вкладка теста
        bench_frame = ttk.Frame(self.notebook)
        self.notebook.add(bench_frame, text=_("tab_benchmark"))
        self.benchmark_tab = BenchmarkTab(bench_frame, self.state, self.toast, self)

        # Статус-бар
        self._create_status_bar()

    # ───────────── Вкладка запуска ─────────────

    def _create_launch_tab(self, frame):
        frame.pack_propagate(False)

        # Header
        self._create_header(frame)

        main_container = ttk.PanedWindow(frame, orient=HORIZONTAL)
        main_container.pack(fill=BOTH, expand=True, padx=15, pady=(8, 10))

        left_outer = ttk.Frame(main_container)
        right_frame = ttk.Frame(main_container)
        main_container.add(left_outer, weight=3)
        main_container.add(right_frame, weight=2)

        left_frame = self._create_scrollable_panel(left_outer)

        # GPU Card — справа сверху
        self.gpu_card = GpuCard(right_frame, self.state)

        # Левая панель — секции
        self._create_model_selection_section(left_frame)
        self.preset_panel = PresetPanel(left_frame, self.state, self.toast)
        self.tooltip = TooltipManager(self.root)
        self.basic_panel = BasicParamsPanel(left_frame, self.state, self.tooltip)
        self.generation_panel = GenerationParamsPanel(left_frame, self.state, self.tooltip)
        self.advanced_panel = AdvancedParamsPanel(left_frame, self.state, self.tooltip)
        self._create_llama_dir_section(left_frame)

        # Лог — справа снизу
        self._create_log_section(right_frame)

        # Кнопки управления внизу
        self._create_control_buttons(frame)

    def _create_scrollable_panel(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0, bg="#252a31")
        scrollbar = ttk.Scrollbar(container, orient=VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scroll_frame.bind("<MouseWheel>", _on_mousewheel)

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        return scroll_frame

    def _create_header(self, parent):
        header = ttk.Frame(parent)
        header.pack(fill=X, padx=15, pady=(10, 0))

        title_frame = ttk.Frame(header)
        title_frame.pack(side=LEFT, fill=X, expand=True)

        ttk.Label(title_frame, text="\U0001f999 Llama.cpp Launcher",
                  style="LaunchTitle.TLabel").pack(side=LEFT)
        ttk.Label(title_frame,
                  text=_("app_subtitle"),
                  style="Subtle.TLabel").pack(side=LEFT, padx=(12, 0), pady=(4, 0))

        actions_frame = ttk.Frame(header)
        actions_frame.pack(side=RIGHT)

        self.open_web_btn = ttk.Button(actions_frame, text=_("btn_web_ui"),
                                       command=self.open_web_ui, style="Header.TButton")
        self.open_web_btn.pack(side=RIGHT, padx=(5, 0))

        self.copy_cmd_btn = ttk.Button(actions_frame, text=_("btn_copy_cmd"),
                                       command=self.copy_command, style="Header.TButton")
        self.copy_cmd_btn.pack(side=RIGHT, padx=(5, 0))

        self.open_logs_btn = ttk.Button(actions_frame, text=_("btn_logs"),
                                        command=self.open_logs_folder, style="Header.TButton")
        self.open_logs_btn.pack(side=RIGHT, padx=(5, 0))

    def _create_llama_dir_section(self, parent):
        section = ttk.LabelFrame(parent, text=" " + _("path_llama") + " ", padding=10)
        section.pack(fill=X, pady=(0, 10))

        row = ttk.Frame(section)
        row.pack(fill=X)

        ttk.Label(row, text=_("path_folder")).pack(side=LEFT, padx=(0, 5))

        self.llama_dir_var = self.state.llama_dir_var
        self.llama_dir_entry = ttk.Entry(row, textvariable=self.llama_dir_var, width=40)
        self.llama_dir_entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))

        ttk.Button(row, text=_("btn_browse"), command=self.browse_llama_dir, width=12).pack(side=LEFT)

        self.llama_status_label = SelectableLabel(section, text="", font=("Segoe UI", 9))
        self.llama_status_label.pack(fill=X, pady=(5, 0))

        if self.settings.get("llama_dir"):
            self.validate_llama_dir()

    def _create_model_selection_section(self, parent):
        section = ttk.LabelFrame(parent, text=_("active_model_section"), padding=10)
        section.pack(fill=X, pady=(0, 10))

        self.active_model_var = self.state.active_model_var
        self.active_model_name_var = self.state.active_model_name_var
        self.active_model_path_var = self.state.active_model_path_var
        self.active_model_meta_var = self.state.active_model_meta_var

        self.active_model_label = SelectableLabel(section, text=self.active_model_name_var.get(),
                                                      font=("Segoe UI", 10, "bold"))
        self.active_model_label.pack(anchor=W)
        
        # Связываем переменную с текстом SelectableLabel
        self.state.active_model_name_var.trace_add("write", 
            lambda *args: self.active_model_label.config(text=self.state.active_model_name_var.get()))

        self.active_model_path_label = SelectableLabel(
            section, text=self.active_model_path_var.get(),
            font=("Segoe UI", 8), foreground="#90caf9", wraplength=520, justify=LEFT
        )
        self.active_model_path_label.pack(fill=X, pady=(3, 0))
        self.state.active_model_path_var.trace_add("write", 
            lambda *args: self.active_model_path_label.config(text=self.state.active_model_path_var.get()))

        self.active_model_meta_label = SelectableLabel(
            section, text=self.active_model_meta_var.get(),
            font=("Segoe UI", 8), foreground="#b0bec5", wraplength=520, justify=LEFT
        )
        self.active_model_meta_label.pack(fill=X, pady=(4, 0))
        self.state.active_model_meta_var.trace_add("write", 
            lambda *args: self.active_model_meta_label.config(text=self.state.active_model_meta_var.get()))

        # Черновая модель
        self._create_draft_model_section(section)

    def _create_draft_model_section(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=X, pady=(10, 0))

        ttk.Label(row, text="Draft:", font=("Segoe UI", 8, "bold"), foreground="#ffa726").pack(side=LEFT)
        self.draft_model_label = ttk.Label(row, textvariable=self.state.draft_model_name_var,
                                            font=("Segoe UI", 8), foreground="#ffb74d")
        self.draft_model_label.pack(side=LEFT, padx=(5, 0))

        # ngld
        ttk.Label(row, text=" ngld:", font=("Segoe UI", 8)).pack(side=LEFT, padx=(10, 0))
        self.draft_ngl_spin = ttk.Spinbox(row, from_=0, to=999, textvariable=self.state.draft_ngl_var, width=5)
        self.draft_ngl_spin.pack(side=LEFT, padx=(2, 0))

        ttk.Button(row, text="✕", width=2, command=self.clear_draft_model).pack(side=RIGHT)

    def clear_draft_model(self):
        self.state.draft_model_var.set("")
        self.state.sync_draft_model_ui()
        self.state.save_settings()

    def _create_log_section(self, parent):
        section = ttk.LabelFrame(parent, text=_("log_section"), padding=10)
        section.pack(fill=BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(section, width=50, height=30,
                                                    font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill=BOTH, expand=True)
        self.log_text.config(state=DISABLED)
        self.log_text.bind("<Button-3>", self.show_context_menu)

        btn_frame = ttk.Frame(section)
        btn_frame.pack(fill=X, pady=(5, 0))

        ttk.Button(btn_frame, text=_("btn_clear_log"), command=self.clear_log).pack(side=LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text=_("btn_copy_all_log"), command=self.copy_all_log).pack(side=LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text=_("btn_save_log"), command=self.save_log).pack(side=LEFT, padx=(0, 5))

    def _create_control_buttons(self, parent):
        section = ttk.Frame(parent, style="LaunchBar.TFrame", padding=(18, 12))
        section.pack(fill=X, side=BOTTOM, padx=15, pady=(0, 15))

        self.start_btn = ttk.Button(section, text=_("btn_start"),
                                    command=self.start_server, style="Success.TButton")
        self.start_btn.pack(side=LEFT, padx=(0, 8))

        self.stop_btn = ttk.Button(section, text=_("btn_stop"),
                                   command=self.stop_server, state=DISABLED, style="Danger.TButton")
        self.stop_btn.pack(side=LEFT, padx=(0, 12))

        ttk.Label(section,
                  text="Ctrl+S запуск • Ctrl+Q остановка • F5 обновить модели",
                  style="Subtle.TLabel").pack(side=LEFT)

        ttk.Label(section, text="").pack(side=LEFT, fill=X, expand=True)

        self.status_indicator = tk.Canvas(section, width=16, height=16, highlightthickness=0, bg="#dfe9f3")
        self.status_indicator.pack(side=RIGHT, padx=(10, 0))
        self.status_dot = self.status_indicator.create_oval(2, 2, 14, 14, fill="#666666", outline="")

    def run_ai_tune_ui(self):
        """UI wrapper for AI Tuning process."""
        # We need the current active model
        model_path = self.state.active_model_var.get().strip()
        if not model_path:
            self.toast.show("⚠️ Пожалуйста, выберите модель для тюнинга")
            return
        if not os.path.isfile(model_path):
            self.toast.show("⚠️ Файл модели не найден")
            return
        
        # We need the BenchmarkTab to show progress
        if not hasattr(self, 'benchmark_tab'):
            self.toast.show("⚠️ Ошибка: вкладка теста не инициализирована")
            return

        # Disable buttons during tune
        self.start_btn.config(state=DISABLED)
        if hasattr(self, 'ai_tune_btn'):
            self.ai_tune_btn.config(state=DISABLED)

        def worker():
            try:
                # Tuning loop
                # Note: tuner.tune is a blocking call
                best_config, best_tps = self.tuner.tune(
                    model_path,
                    on_progress=lambda msg: self.root.after(0, lambda: self.benchmark_tab._append_log(msg))
                )
                
                if best_config:
                    # Save as a preset
                    model_name = os.path.basename(model_path).split(".gguf")[0]
                    preset_name = f"✨ {model_name} (AI-Optimized) {best_tps:.2f} tok/s"
                    
                    preset_data = {
                        "ctx": best_config.ctx_size,
                        "batch_size": best_config.batch_size,
                        "flash_attn": best_config.flash_attn,
                        "temp": best_config.temp,
                        "top_k": best_config.top_k,
                        "top_p": best_config.top_p,
                        "min_p": best_config.min_p,
                        "repeat_penalty": best_config.repeat_penalty,
                        "presence_penalty": best_config.presence_penalty,
                        "frequency_penalty": best_config.frequency_penalty,
                        "mirostat": best_config.mirostat,
                        "n_predict": best_config.n_predict,
                    }
                    
                    custom_presets = self.settings.setdefault("custom_presets", {})
                    custom_presets[preset_name] = preset_data
                    self.state.save_settings()
                    
                    self.root.after(0, lambda: self.toast.show(f"✅ Пресет {preset_name} сохранен"))
                else:
                    self.root.after(0, lambda: self.toast.show("⚠️ Тюнинг завершен без улучшений"))
                    
            except Exception as e:
                traceback.print_exc()
                self.root.after(0, lambda: self.toast.show(f"❌ Ошибка тюнинга: {e}"))
            finally:
                # Re-enable buttons
                self.root.after(0, self._enable_tune_buttons)

        threading.Thread(target=worker, daemon=True).start()

    def _enable_tune_buttons(self):
        self.start_btn.config(state=NORMAL)
        if hasattr(self, 'ai_tune_btn'):
            self.ai_tune_btn.config(state=NORMAL)


    # ───────────── Вкладка моделей ─────────────

    def _create_models_tab(self, frame):
        from ui.tabs.models_tab import ModelsTab
        self.models_tab = ModelsTab(self.root, self.state, self.notebook, self)
        self.models_tab.create(frame)


    # ───────────── Вкладка настроек ─────────────

    def _create_settings_tab(self, frame):
        from core.i18n import LANGUAGES
        container = ttk.Frame(frame, padding=30)
        container.pack(fill=BOTH, expand=True)

        section = ttk.LabelFrame(container, text=" " + _("settings") + " ", padding=20)
        section.pack(fill=X)

        # Язык
        row = ttk.Frame(section)
        row.pack(fill=X, pady=10)

        self.lang_label = ttk.Label(row, text=_("language"), font=("Segoe UI", 10, "bold"))
        self.lang_label.pack(side=LEFT, padx=(0, 20))

        # Инвертируем карту для удобства
        display_to_code = {v: k for k, v in LANGUAGES.items()}
        code_to_display = LANGUAGES

        current_code = self.state.language_var.get()
        self.lang_display_var = tk.StringVar(value=code_to_display.get(current_code, "Русский"))

        def on_lang_ui_change(*args):
             new_display = self.lang_display_var.get()
             new_code = display_to_code.get(new_display)
             if new_code and new_code != self.state.language_var.get():
                 self.state.language_var.set(new_code)
                 from core.i18n import I18n
                 I18n().set_language(new_code)
                 self.state.save_settings()

        self.lang_combo = ttk.Combobox(row, textvariable=self.lang_display_var,
                                  values=list(code_to_display.values()), state="readonly", width=15)
        self.lang_combo.pack(side=LEFT)
        self.lang_display_var.trace_add("write", on_lang_ui_change)

        self.lang_hint = ttk.Label(section, text="Изменения применяются мгновенно для основных элементов.",
                  style="Subtle.TLabel")
        self.lang_hint.pack(fill=X, pady=(20, 0))

    # ───────────── Методы моделей ─────────────

    def refresh_models(self):
        """Обновляет список моделей через вкладку моделей."""
        if hasattr(self, 'models_tab'):
            self.models_tab.refresh_models()

    def _on_model_selected(self, event=None):
        selection = self.models_tree.selection()
        if not selection:
            self.selected_model_hint_var.set(_("model_path_hint"))
            return

        item = selection[0]
        values = self.models_tree.item(item, "values")
        model_path = values[0] if values else self.models_tree.item(item, "text")
        if model_path and hasattr(self, '_load_gguf_metadata'):
            self._load_gguf_metadata(model_path)

    def select_model_from_list(self):
        selection = self.models_tree.selection()
        if not selection:
            return
        item = selection[0]
        values = self.models_tree.item(item, "values")
        model_path = values[0] if values else self.models_tree.item(item, "text")
        if model_path:
            self._set_active_model(model_path)

    def add_model_dir(self):
        folder = filedialog.askdirectory(title=_("add_folder_title"))
        if folder:
            dirs = self.settings.setdefault("model_dirs", [])
            if folder not in dirs:
                dirs.append(folder)
                self.state.save_settings()
                self.refresh_models()

    def add_model_files(self):
        files = filedialog.askopenfilenames(
            title=_("add_files_title"),
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if files:
            file_list = self.settings.setdefault("model_files", [])
            for f in files:
                if f not in file_list:
                    file_list.append(f)
            self.state.save_settings()
            self.refresh_models()

    def remove_models(self):
        selection = self.models_tree.selection()
        if not selection:
            return
        
        if messagebox.askyesno(_("remove_confirm_title"), _("remove_confirm_msg")):
            to_remove = []
            for item in selection:
                values = self.models_tree.item(item, "values")
                path = values[0] if values else self.models_tree.item(item, "text")
                to_remove.append(path)
                
            file_list = self.settings.get("model_files", [])
            self.settings["model_files"] = [f for f in file_list if f not in to_remove]
            self.state.save_settings()
            self.refresh_models()

    def remove_model_dir(self):
        dirs = self.settings.get("model_dirs", [])
        if not dirs:
            messagebox.showinfo(_("btn_remove_cat"), "No registered model directories.")
            return

        # Показываем диалог с списком каталогов для выбора
        dialog = tk.Toplevel(self.root)
        dialog.title("Удалить каталог моделей")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Выберите каталог(и) для удаления:", font=("Segoe UI", 10)).pack(fill=X, padx=10, pady=10)

        listbox = tk.Listbox(dialog, selectmode=tk.EXTENDED, font=("Segoe UI", 9))
        listbox.pack(fill=BOTH, expand=True, padx=10, pady=5)
        for d in dirs:
            listbox.insert(tk.END, d)

        def on_delete():
            selection = listbox.curselection()
            if not selection:
                return
            if not messagebox.askyesno("Подтверждение", "Удалить выбранные каталоги из списка?"):
                return
            to_remove = [dirs[i] for i in selection]
            self.settings["model_dirs"] = [d for d in dirs if d not in to_remove]
            self.state.save_settings()
            self.refresh_models()
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=X, padx=10, pady=5)

        ttk.Button(btn_frame, text="Удалить выбранные", command=on_delete).pack(side=LEFT)
        ttk.Button(btn_frame, text="Отмена", command=dialog.destroy).pack(side=RIGHT)

    def set_active_model(self):
        selection = self.models_listbox.curselection()
        if not selection:
            return
        models = getattr(self, '_models_paths', [])
        idx = selection[0]
        if idx < len(models):
            self._set_active_model(models[idx])

    def set_draft_model(self):
        selection = self.models_listbox.curselection()
        if not selection:
            return
        models = getattr(self, '_models_paths', [])
        idx = selection[0]
        if idx < len(models):
            self._set_draft_model(models[idx])

    def _set_active_model(self, model_path):
        self.state.active_model_var.set(model_path)
        if model_path and os.path.isfile(model_path):
            self.state.recalculate_limits(model_path)
            if hasattr(self, 'models_tab'):
                self.models_tab._load_gguf_metadata(model_path)
        self._sync_active_model_ui()
        self._update_memory_estimate()
        self._update_gpu_info_card()
        self.state.save_settings()
        self.set_status(f"Выбрана модель: {os.path.basename(model_path)}")

    def _set_draft_model(self, model_path):
        self.state.draft_model_var.set(model_path)
        self.state.sync_draft_model_ui()
        self.state.save_settings()
        self.set_status(f"Выбрана черновая модель: {os.path.basename(model_path)}")

    def _sync_active_model_ui(self):
        """Обновляет краткую карточку активной модели."""
        self.state.sync_active_model_ui()
        model_path = self.state.active_model_var.get().strip()
        if not model_path:
            if hasattr(self, 'model_info_label'):
                self.model_info_label.config(text=_("no_model"))
            return

        name = os.path.basename(model_path)
        __, size_gb = self.state.get_model_info(model_path)
        metadata = self.state.gguf_parser.get_cached_or_parse(model_path)
        arch = metadata.get("general.architecture", "")
        quant = get_quant_from_metadata(metadata)
        if quant == "Unknown":
            quant = extract_quant_from_filename(model_path)

        info = f"{name}"
        if size_gb:
            info += f" ({size_gb:.1f} GB)"
        if arch:
            info += f" | {arch.upper()}"
        info += f" | {quant}"

        if hasattr(self, 'model_info_label'):
            self.model_info_label.config(text=info)

    # ───────────── GGUF Metadata ─────────────

    def _load_gguf_metadata(self, model_path):
        if not model_path or not os.path.isfile(model_path):
            self._clear_metadata()
            return
        if not hasattr(self, 'meta_tree'):
            return

        if model_path in self._metadata_cache:
            self._display_metadata(self._metadata_cache[model_path], model_path)
            return

        if model_path in self._metadata_loading:
            return

        self._metadata_loading.add(model_path)
        self.meta_header.config(text=_("meta_loading"))
        self._clear_metadata()
        self.meta_tree.insert("", "end", text=_("loading"),
                              values=(_("meta_loading"),), tags=("warning",))

        thread = threading.Thread(target=self._load_metadata_thread,
                                  args=(model_path,), daemon=True)
        thread.start()

    def _load_metadata_thread(self, model_path):
        try:
            metadata = self.state.gguf_parser.get_cached_or_parse(model_path)
            self._metadata_cache[model_path] = metadata
            if self.root.winfo_exists():
                self.root.after(0, self._display_metadata, metadata, model_path)
        except Exception as e:
            if self.root.winfo_exists():
                self.root.after(0, self._metadata_load_error, model_path, str(e))
        finally:
            self._metadata_loading.discard(model_path)

    def _display_metadata(self, metadata, model_path):
        if not hasattr(self, 'meta_tree'):
            return

        try:
            for item in self.meta_tree.get_children():
                self.meta_tree.delete(item)

            if "__error__" in metadata:
                self.meta_header.config(text=f"⚠️ Ошибка: {os.path.basename(model_path)}")
                self._clear_metadata()
                return

            model_name = os.path.basename(model_path)
            __, size_gb = self.state.get_model_info(model_path)

            if size_gb:
                gguf_version = metadata.get("__gguf_version__", "?")
                tensor_count = metadata.get("__tensor_count__", 0)
                self.meta_header.config(text=f"{model_name} ({size_gb:.2f} GB) | GGUF v{gguf_version} | {tensor_count:,} тензоров")
            else:
                self.meta_header.config(text=model_name)

            if len(metadata) <= 3:
                self._show_metadata_warning(_("meta_not_found"))
                return

            arch_raw = metadata.get("general.architecture", "unknown")
            if isinstance(arch_raw, list) and arch_raw:
                arch = str(arch_raw[0])
            elif not isinstance(arch_raw, str):
                arch = str(arch_raw)
            else:
                arch = arch_raw

            self.meta_arch_label.config(
                text=f"{_('arch')} {arch.upper()}" if arch != "unknown" else f"{_('arch')} unknown")

            param_str, param_source = self.state.get_model_param_label(metadata, model_path)
            if param_str:
                suffix = "" if param_source == "metadata" else " (filename)"
                self.meta_params_label.config(text=f"{_('params')} {param_str}{suffix}")
            else:
                self.meta_params_label.config(text=f"{_('params')} none")

            # Пытаемся взять квантизацию из метаданных, если ее нет в имени
            quant_type = get_quant_from_metadata(metadata)
            if quant_type == "Unknown":
                quant_type = extract_quant_from_filename(model_path)
            
            quant_desc = get_quant_description(quant_type)
            self.meta_quant_label.config(text=f"{_('quant')} {quant_type}")
            if hasattr(self, 'meta_quant_desc'):
                self.meta_quant_desc.config(text=quant_desc)

            ctx_length = metadata.get(f"{arch}.context_length") or metadata.get("llama.context_length")
            if ctx_length:
                try:
                    self.meta_context_label.config(text=f"{_('ctx')} {int(ctx_length):,}")
                except (ValueError, TypeError):
                    self.meta_context_label.config(text=f"{_('ctx')} unknown")
            else:
                self.meta_context_label.config(text=f"{_('ctx')} unknown")

            block_count = metadata.get(f"{arch}.block_count") or metadata.get("llama.block_count")
            if block_count:
                try:
                    self.meta_layers_label.config(text=f"🧱 Слои: {int(block_count)}")
                except (ValueError, TypeError):
                    self.meta_layers_label.config(text="🧱 Слои: неизвестно")
            else:
                self.meta_layers_label.config(text="🧱 Слои: неизвестно")

            embedding_length = metadata.get(f"{arch}.embedding_length") or metadata.get("llama.embedding_length")
            if embedding_length:
                try:
                    self.meta_embedding_label.config(text=f"📍 Embedding: {int(embedding_length)}")
                except (ValueError, TypeError):
                    self.meta_embedding_label.config(text="")
            else:
                self.meta_embedding_label.config(text="")

            self._populate_metadata_tree(metadata, arch)

            if model_path == self.state.active_model_var.get():
                self._sync_active_model_ui()

        except Exception as e:
            self.meta_header.config(text="⚠️ Ошибка загрузки метаданных")
            self._show_metadata_warning(f"Не удалось загрузить метаданные: {e}")

    def _metadata_load_error(self, model_path, error):
        if hasattr(self, 'meta_header'):
            self.meta_header.config(text=f"⚠️ Ошибка: {os.path.basename(model_path)}")
        self._show_metadata_warning(f"Не удалось загрузить метаданные: {error}")

    def _clear_metadata(self):
        for attr in ('meta_arch_label', 'meta_params_label', 'meta_quant_label',
                      'meta_quant_desc', 'meta_context_label', 'meta_layers_label',
                      'meta_embedding_label'):
            if hasattr(self, attr):
                getattr(self, attr).config(text="")
        if hasattr(self, 'meta_tree'):
            for item in self.meta_tree.get_children():
                self.meta_tree.delete(item)

    def _show_metadata_warning(self, message):
        if hasattr(self, 'meta_tree'):
            self.meta_tree.insert("", "end", text="⚠️ Предупреждение",
                                   values=(message,), tags=("warning",))

    def _populate_metadata_tree(self, metadata, arch):
        if not hasattr(self, 'meta_tree'):
            return

        important_keys = [
            "general.architecture", "general.name", "general.description",
            "general.size_label", "general.parameter_count",
            f"{arch}.context_length", f"{arch}.embedding_length",
            f"{arch}.block_count", f"{arch}.feed_forward_length",
            f"{arch}.attention.head_count", f"{arch}.attention.head_count_kv",
            f"{arch}.attention.key_length", f"{arch}.attention.value_length",
        ]

        important_items = []
        other_items = []

        for key, value in sorted(metadata.items()):
            if key.startswith("__"):
                continue
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            if key in important_keys:
                important_items.append((key, value))
            else:
                other_items.append((key, value))

        for key, value in important_items:
            self.meta_tree.insert("", "end", text=str(key), values=(str(value),), tags=("important",))

        if other_items:
            other_node = self.meta_tree.insert("", "end", text="Другие параметры", values=("",), open=False)
            for key, value in other_items:
                self.meta_tree.insert(other_node, "end", text=str(key), values=(str(value),))

    # ───────────── Действия ─────────────

    def browse_llama_dir(self):
        folder = filedialog.askdirectory(title="Выберите папку с llama.cpp")
        if folder:
            self.state.llama_dir_var.set(folder)
            self.validate_llama_dir()

    def validate_llama_dir(self):
        path = self.state.llama_dir_var.get()
        if not path:
            self.llama_status_label.config(text="Папка не выбрана")
            return False, "Папка не выбрана"
        if not os.path.isdir(path):
            self.llama_status_label.config(text="⚠️ Папка не существует")
            return False, "Папка не существует"
        if os.path.exists(os.path.join(path, "llama-server.exe")):
            self.llama_status_label.config(text="✓ llama-server.exe найден")
            return True, "OK"
        if os.path.exists(os.path.join(path, "llama-cli.exe")):
            self.llama_status_label.config(text="✓ llama-cli.exe найден")
            return True, "OK"
        self.llama_status_label.config(text="⚠️ llama-server.exe не найден")
        return False, "llama-server.exe не найден"

    def open_web_ui(self):
        server_exe = self.state.get_server_exe_path()
        if not server_exe or os.path.basename(server_exe).lower() != "llama-server.exe":
            self.toast.show("⚠️ Web UI доступен только для llama-server.exe")
            return
        port = 8080
        try:
            port = self.state.port_var.get()
        except Exception:
            pass
        url = f"http://127.0.0.1:{port}"
        webbrowser.open(url)
        self.toast.show(f"🌐 Открыт {url}")

    def copy_command(self):
        server_exe = self.state.get_server_exe_path()
        if not server_exe:
            self.toast.show("⚠️ llama.cpp не настроена")
            return
        model_path = self.state.active_model_var.get()
        if not model_path:
            self.toast.show("⚠️ Модель не выбрана")
            return

        try:
            cmd = self._build_command()
        except ValueError as e:
            self.toast.show(f"⚠️ {e}")
            return
        cmd_str = subprocess.list2cmdline(cmd) if sys.platform == "win32" else shlex.join(cmd)
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd_str)
        self.toast.show("📋 Команда скопирована!")

    def _build_command(self, backend=None):
        """Строит полную команду запуска с использованием FlagBuilder.
        backend: forced backend ('llama' or 'ik_llama'), otherwise uses state.
        """
        server_exe = self.state.get_server_exe_path()
        if not server_exe:
            return []
        model_path = self.state.active_model_var.get()
        if not model_path:
            return []
        
        # Создаем конфиг из состояния UI
        s = self.state
        config = ServerConfig(
            model_path=model_path,
            port=s.port_var.get(),
            ctx_size=s.ctx_var.get(),
            kv_quality="high" if s.kv_quality_var.get() == "high" else "mid" if s.kv_quality_var.get() == "mid" else "low",
            ram_budget_mb=0,
            backend=backend if backend else (s.backend_var.get() if hasattr(s, 'backend_var') else "auto"),
            keep_alive=s.keep_alive_var.get() if hasattr(s, 'keep_alive_var') else False,
            mmproj_path=s.mmproj_var.get(),
            temp=s.temp_var.get(),
            top_k=s.top_k_var.get(),
            top_p=s.top_p_var.get(),
            min_p=s.min_p_var.get(),
            repeat_penalty=s.repeat_penalty_var.get(),
            presence_penalty=s.presence_penalty_var.get(),
            frequency_penalty=s.frequency_penalty_var.get(),
            mirostat=s.mirostat_var.get(),
            n_predict=s.n_predict_var.get(),
            seed=s.seed_var.get(),
            parallel_slots=s.parallel_slots_var.get(),
            threads=s.threads_var.get(),
            batch_size=s.batch_size_var.get(),
            flash_attn=s.flash_attn_var.get(),
            mmap=s.mmap_var.get(),
            mlock=s.mlock_var.get(),
            kv_offload=s.kv_offload_var.get(),
            cache_prompt=s.cache_prompt_var.get(),
            reasoning=s.reasoning_var.get(),
            rope_scale=s.rope_scale_var.get(),
            rope_freq_base=s.rope_freq_base_var.get(),
            draft_model_path=s.draft_model_var.get(),
            draft_ngl=s.draft_ngl_var.get(),
            custom_args=s.custom_args_var.get()
        )
        
        builder = FlagBuilder(self.state)
        config = builder.build(config)
        
        # Apply cached AI‑tuned flag overrides if available
        if hasattr(self, 'tuner'):
            applied = self.tuner.apply_cached_tune(config)
            if applied:
                # Optional: log that cached tuning is being used
                pass
        
        cmd = [config.binary_path] + config.flags
        
        if os.path.basename(config.binary_path).lower() == "llama-server.exe":
            host_val = s.host_var.get().strip() or "127.0.0.1"
            cmd.extend(["--host", host_val])
            api_key = s.api_key_var.get().strip()
            if api_key:
                cmd.extend(["--api-key", api_key])
            if s.webui_var.get() == "off":
                cmd.append("--no-webui")
        else:
            cmd.append("-i")
        return cmd


    def open_logs_folder(self):
        ensure_log_dir()
        os.startfile(str(LOG_DIR))
        self.toast.show("📂 Папка логов открыта")

    def run_ai_tune(self, bench_tab, rounds=None, retune=None):
        model_path = bench_tab.bench_model_var.get()
        if not model_path:
            self.toast.show("⚠️ Модель не выбрана")
            return
        
        # Determine rounds and retune values
        if rounds is None and hasattr(bench_tab, 'ai_tune_rounds_var'):
            rounds = bench_tab.ai_tune_rounds_var.get()
        if retune is None and hasattr(bench_tab, 'ai_tune_retune_var'):
            retune = bench_tab.ai_tune_retune_var.get()
        rounds = rounds or 8
        retune = retune or False
        
        bench_tab.results_text.delete(1.0, END)
        bench_tab.results_text.insert(END, f"🪄 Запуск AI-Tuning (rounds={rounds}, retune={retune})...\n\n")
        self.toast.show("🪄 AI-Tuning запущен. Результаты во вкладке Тест")
        
        def worker():
            try:
                best_config, best_tps = self.tuner.tune(
                    model_path, 
                    rounds=rounds,
                    retune=retune,
                    on_progress=lambda msg: self.root.after(0, lambda: bench_tab._append_log(msg))
                )
                
                if best_config is None:
                    self.root.after(0, lambda: bench_tab._append_log("\n❌ Тюнинг не удалось завершить (не найден подходящий конфиг)."))
                    return
                
                # Сохраняем лучший результат в пресеты
                model_name = os.path.basename(model_path).split(".gguf")[0]
                preset_name = f"✨ {model_name} (AI-Optimized) {best_tps:.2f} tok/s"
                
                # Создаем словарь настроек из config
                preset_data = {
                    "ctx": best_config.ctx_size,
                    "batch_size": best_config.batch_size,
                    "flash_attn": best_config.flash_attn,
                    "temp": best_config.temp,
                    "top_k": best_config.top_k,
                    "top_p": best_config.top_p,
                    "min_p": best_config.min_p,
                    "repeat_penalty": best_config.repeat_penalty,
                    "presence_penalty": best_config.presence_penalty,
                    "frequency_penalty": best_config.frequency_penalty,
                    "mirostat": best_config.mirostat,
                    "n_predict": best_config.n_predict,
                }
                
                custom_presets = self.settings.setdefault("custom_presets", {})
                custom_presets[preset_name] = preset_data
                self.state.save_settings()
                
                self.root.after(0, lambda: bench_tab._append_log(f"\n\n✅ Тюнинг завершен!\nЛучший результат: {best_tps:.2f} tok/s\nПресет сохранен как: {preset_name}"))
                self.root.after(0, lambda: self.toast.show(f"✅ Пресет {preset_name} сохранен"))
            except Exception as e:
                traceback.print_exc()
                self.root.after(0, lambda e=e: bench_tab._append_log(f"\n❌ Ошибка тюнинга: {e}"))
                self.root.after(0, lambda e=e: self.toast.show(f"❌ Ошибка тюнинга: {e}"))
        
        threading.Thread(target=worker, daemon=True).start()

    # ───────────── Сервер ─────────────

    def start_server(self):
        valid, msg = self.validate_llama_dir()
        if not valid:
            messagebox.showerror("Ошибка", msg)
            return
        
        model_path = self.state.active_model_var.get()
        if not model_path:
            messagebox.showerror("Ошибка", "Модель не выбрана")
            return
        if not os.path.isfile(model_path):
            messagebox.showerror("Ошибка", "Файл модели не найден")
            return
        
        # Основная команда
        cmd = self._build_command()
        if not cmd:
            messagebox.showerror("Ошибка", "Не удалось собрать команду")
            return
        
        # Fallback команда (стандартный llama-server)
        fallback_cmd = self._build_command(backend="llama")
        
        server_mgr = LlamaServerManager()
        
        def on_log(line):
            self.root.after(0, self._append_log, line)
        
        def on_stop():
            self.root.after(0, self._on_server_stopped)
        
        ok, err = server_mgr.start(cmd, cwd=self.state.llama_dir_var.get(),
                                     on_log=on_log, on_stop=on_stop,
                                     fallback_args=fallback_cmd if fallback_cmd != cmd else None)
        
        if ok:
            self.state.running = True
            self.state.server_manager = server_mgr  # сохраняем менеджер
            self.state.server_process = server_mgr.process
            self.start_btn.config(state=DISABLED)
            self.stop_btn.config(state=NORMAL)
            self.set_status(_("status_running"))
            host = self.state.host_var.get()
            port = self.state.port_var.get()
            self.url_label.config(text=f"http://{host}:{port}")
            self.status_indicator.itemconfig(self.status_dot, fill="#4caf50")
        else:
            messagebox.showerror("Ошибка", err)


    def stop_server(self):
        if not self.state.running:
            return
        self.state.stopping = True
        
        # Используем серверный менеджер для корректной остановки
        server_mgr = getattr(self.state, 'server_manager', None)
        if server_mgr:
            server_mgr.stop()
        elif self.state.server_process:
            # Fallback: прямая остановка
            try:
                self.state.server_process.terminate()
                try:
                    self.state.server_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.state.server_process.kill()
                    self.state.server_process.wait(timeout=1)
            except Exception as e:
                self._append_log(f"[Ошибка остановки: {e}]")
            
            # Закрываем потоки чтения, если они остались
            if hasattr(self.state, 'server_manager'):
                self.state.server_manager._cleanup_process()
            
            self._on_server_stopped()

    def _schedule_log_flush(self):
        """Периодически сбрасывает очередь логов в UI, чтобы не перегружать event loop."""
        try:
            while not self.log_queue.empty():
                line = self.log_queue.get_nowait()
                self._append_log_threadsafe(line)
        except Exception:
            pass
        self.root.after(100, self._schedule_log_flush)

    def _append_log(self, line):
        self.log_queue.put(line)

    def _append_log_threadsafe(self, line):
        if self.log_text and self.root.winfo_exists():
            self.log_text.config(state=NORMAL)
            self.log_text.insert(END, line + "\n")
            self.log_text.see(END)
            self.log_text.config(state=DISABLED)

    def _on_server_stopped(self):
        if self.root.winfo_exists():
            self.root.after(0, self._on_server_stopped_threadsafe)

    def _on_server_stopped_threadsafe(self):
        self.state.running = False
        self.state.stopping = False
        if hasattr(self, 'start_btn') and self.start_btn.winfo_exists():
            self.start_btn.config(state=NORMAL)
            self.stop_btn.config(state=DISABLED)
            self.set_status(_("status_ready"))
            self.url_label.config(text="")
            self.status_indicator.itemconfig(self.status_dot, fill="#666666")

    def set_status(self, text):
        self.status_label.config(text=text)

    # ───────────── Лог ─────────────

    def clear_log(self):
        if self.log_text:
            self.log_text.config(state=NORMAL)
            self.log_text.delete("1.0", END)

    def copy_all_log(self):
        if self.log_text:
            text = self.log_text.get("1.0", "end-1c")  # Убираем trailing newline
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def save_log(self):
        if not self.log_text:
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_text.get("1.0", END))

    def show_context_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Копировать всё", command=self.copy_all_log)
        menu.add_command(label="Очистить", command=self.clear_log)
        menu.post(event.x_root, event.y_root)

    # ───────────── Обновления ─────────────

    def _create_status_bar(self):
        status = ttk.Frame(self.root, relief=SUNKEN, borderwidth=1)
        status.pack(fill=X, side=BOTTOM)
        
        self.status_label = SelectableLabel(status, text=_("status_ready"), relief=SUNKEN)
        self.status_label.pack(side=LEFT, padx=5, pady=2)
        
        self.url_label = SelectableLabel(status, text="", foreground="#90caf9")
        self.url_label.pack(side=RIGHT, padx=5, pady=2)

    def _refresh_startup_state(self):
        self.refresh_models()
        self._sync_active_model_ui()
        self.state.sync_draft_model_ui()
        self._update_gpu_info_card()
        self._update_memory_estimate()

    def _update_gpu_info_card(self):
        if hasattr(self, 'gpu_card'):
            self.gpu_card._update_gpu_name()
            self.gpu_card.update()

    def _update_memory_estimate(self):
        """Обновляет все оценки памяти и лимиты слайдеров."""
        # Защита от рекурсивных вызовов
        if getattr(self, '_updating_memory', False):
            return
        self._updating_memory = True
        try:
            self._do_update_memory_estimate()
        finally:
            self._updating_memory = False

    def _do_update_memory_estimate(self):
        """Фактическая логика обновления оценок памяти."""
        ngl = self.state.ngl_var.get()
        ctx = self.state.ctx_var.get()

        # Обновляем лейблы
        if hasattr(self, 'basic_panel'):
            self.basic_panel.ngl_label.config(text=str(ngl))
            self.basic_panel.ctx_label.config(text=str(ctx))
            self.basic_panel.threads_label.config(text=str(self.state.threads_var.get()))
            self.basic_panel.memory_label.config(text=f"🧠 Оценка памяти: {self.state.calculate_memory_text()}")

        model_path = self.state.active_model_var.get()
        if model_path and os.path.isfile(model_path):
            self.state.recalculate_limits(model_path)

            if hasattr(self, 'basic_panel'):
                self.basic_panel.sync_from_state()

        # Обновляем GPU card
        if hasattr(self, 'gpu_card'):
            self.gpu_card.update()

    # ───────────── Toast ─────────────

    def show_toast(self, message, duration=3000):
        self.toast.show(message, duration)

    # ───────────── Прочее ─────────────

    def update_texts(self):
        self.root.title(_("app_title"))
        if self.notebook:
            self.notebook.tab(0, text=_("tab_launch"))
            self.notebook.tab(1, text=_("tab_models"))
            self.notebook.tab(2, text=_("settings"))
        
        if hasattr(self, 'lang_label'):
            self.lang_label.config(text=_("language"))
        if hasattr(self, 'lang_hint'):
            self.lang_hint.config(text=_("lang_applied"))

    def apply_geometry(self):
        geom = self.settings.get("window_geometry", "1000x700")
        try:
            self.root.geometry(geom)
        except Exception:
            pass

    def save_geometry(self):
        self.settings["window_geometry"] = self.root.geometry()

    def on_close(self):
        # Сообщения об ошибках при выходе
        if self.state.running:
            if messagebox.askyesno(_("exit_title"), _("exit_msg")):
                self.stop_server()
            else:
                return
        self.save_geometry()
        self.state.save_settings()
        self.root.destroy()

    def _global_copy(self, event):
        """Позволяет копировать текст любого виджета под курсором через Ctrl+C."""
        focused = self.root.focus_get()
        # Если текущий фокус на поле ввода и там есть выделение — позволяем стандартное копирование
        if focused:
            try:
                if hasattr(focused, 'selection_get') and focused.selection_get():
                    return
            except:
                pass

        # Иначе пытаемся скопировать текст виджета, который находится под мышью
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget:
            text = None
            try:
                # Пробуем достать текст через cget (для Label)
                text = widget.cget('text')
            except:
                try:
                    # Пробуем через get (для Entry/Text)
                    text = widget.get()
                except:
                    pass
            
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(str(text))
                # Показываем уведомление, если текст не слишком длинный
                display_text = (text[:50] + '...') if len(str(text)) > 50 else text
                self.show_toast(f"Скопировано: {display_text}")
                return "break"
