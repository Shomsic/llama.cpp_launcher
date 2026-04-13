"""
Вкладка запуска (Launch Tab)
"""
import tkinter as tk
from tkinter import ttk, HORIZONTAL, VERTICAL, LEFT, RIGHT, BOTH, X, Y, END, SUNKEN, DISABLED, NORMAL
from ui.components.tooltip import TooltipManager as Tooltip


class LaunchTab:
    """Вкладка запуска с левой и правой панелями."""

    def __init__(self, parent, state, notebook, app):
        self.parent = parent
        self.state = state
        self.notebook = notebook
        self.app = app  # Ссылка на главный app для колбэков
        self.tooltip = Tooltip(parent)

    def create(self, frame):
        """Создает UI вкладки запуска."""
        self._create_header(frame)
        
        main_container = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        main_container.pack(fill=BOTH, expand=True, padx=15, pady=(8, 10))
        
        left_outer = ttk.Frame(main_container)
        right_frame = ttk.Frame(main_container)
        main_container.add(left_outer, weight=3)
        main_container.add(right_frame, weight=2)
        
        left_frame = self._create_scrollable_panel(left_outer)
        
        # GPU Card - справа сверху
        from ui.components import GpuCard
        self.gpu_card = GpuCard(right_frame, self.state)
        
        # Левая панель -sections
        self._create_llama_dir_section(left_frame)
        self._create_preset_section(left_frame)
        self._create_basic_params_section(left_frame)
        self._create_generation_params_section(left_frame)
        self._create_advanced_params_section(left_frame)
        self._create_model_selection_section(left_frame)
        
        # Лог - справа снизу
        self._create_log_section(right_frame)
        
        # Кнопки управления внизу
        self._create_control_buttons(frame)
        
        return frame

    def _create_scrollable_panel(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0, bg="#252a31")
        scrollbar = ttk.Scrollbar(container, orient=VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        # Сначала создаём окно, затем привязываем обработчик
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        return scroll_frame

    def _create_header(self, parent):
        header = ttk.Frame(parent)
        header.pack(fill=X, padx=15, pady=(10, 0))
        
        title_frame = ttk.Frame(header)
        title_frame.pack(side=LEFT, fill=X, expand=True)
        
        ttk.Label(title_frame, text="🚀 Llama.cpp Launcher",
                 style="LaunchTitle.TLabel").pack(side=LEFT)
        ttk.Label(title_frame, text="Быстрый запуск llama.cpp с понятными лимитами",
                 style="Subtle.TLabel").pack(side=LEFT, padx=(12, 0), pady=(4, 0))
        
        actions_frame = ttk.Frame(header)
        actions_frame.pack(side=RIGHT)
        
        self.app.open_web_btn = ttk.Button(actions_frame, text="🌐 Web UI",
                                        command=self.app.open_web_ui, style="Header.TButton")
        self.app.open_web_btn.pack(side=RIGHT, padx=(5, 0))
        
        self.app.copy_cmd_btn = ttk.Button(actions_frame, text="📋 Копировать",
                                     command=self.app.copy_command, style="Header.TButton")
        self.app.copy_cmd_btn.pack(side=RIGHT, padx=(5, 0))
        
        self.app.open_logs_btn = ttk.Button(actions_frame, text="📂 Логи",
                                       command=self.app.open_logs_folder, style="Header.TButton")
        self.app.open_logs_btn.pack(side=RIGHT, padx=(5, 0))

    def _create_llama_dir_section(self, parent):
        section = ttk.LabelFrame(parent, text=" Путь к llama.cpp ", padding=10)
        section.pack(fill=X, pady=(0, 10))

        row = ttk.Frame(section)
        row.pack(fill=X)

        ttk.Label(row, text="Папка:").pack(side=LEFT, padx=(0, 5))

        # Используем state var вместо создания отдельного
        self.app.llama_dir_var = self.state.llama_dir_var
        self.app.llama_dir_entry = ttk.Entry(row, textvariable=self.state.llama_dir_var, width=40)
        self.app.llama_dir_entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))

        btn = ttk.Button(row, text="Обзор...", command=self.app.browse_llama_dir, width=12)
        btn.pack(side=LEFT)

        self.app.llama_status_label = ttk.Label(section, text="", font=("Segoe UI", 9))
        self.app.llama_status_label.pack(fill=X, pady=(5, 0))

        if self.state.settings.get("llama_dir"):
            self.app.validate_llama_dir()

    def _create_preset_section(self, parent):
        from ui.components import PresetPanel
        panel = PresetPanel(parent, self.state, self.app.toast if hasattr(self.app, 'toast') else None)
        self.preset_panel = panel

    def _create_basic_params_section(self, parent):
        from ui.components import BasicParamsPanel
        panel = BasicParamsPanel(parent, self.state, self.tooltip)
        self.basic_params_panel = panel

    def _create_generation_params_section(self, parent):
        from ui.components import GenerationParamsPanel
        panel = GenerationParamsPanel(parent, self.state, self.tooltip)
        self.generation_params_panel = panel

    def _create_advanced_params_section(self, parent):
        from ui.components import AdvancedParamsPanel
        panel = AdvancedParamsPanel(parent, self.state, self.tooltip)
        self.advanced_params_panel = panel

    def _create_model_selection_section(self, parent):
        section = ttk.LabelFrame(parent, text=" Выбор модели ", padding=10)
        section.pack(fill=X, pady=(0, 10))

        row = ttk.Frame(section)
        row.pack(fill=X)

        ttk.Label(row, text="Модель:").pack(side=LEFT, padx=(0, 5))

        # Используем state vars для синхронизации
        self.app.active_model_var = self.state.active_model_var

        self.app.model_combo = ttk.Combobox(row, textvariable=self.state.active_model_var,
                                       state="readonly", width=40)
        self.app.model_combo.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        self.app.model_combo.bind("<<ComboboxSelected>>", lambda e: self.app._sync_active_model_ui())

        browse_cmd = self.app.browse_model if hasattr(self.app, 'browse_model') else self._browse_model_fallback
        self.app.btn_browse_model = ttk.Button(row, text="Обзор...",
                                         command=browse_cmd, width=12)
        self.app.btn_browse_model.pack(side=LEFT)

        self.app.model_info_label = ttk.Label(section, text="", font=("Segoe UI", 9))
        self.app.model_info_label.pack(fill=X, pady=(5, 0))

    def _browse_model_fallback(self):
        """Fallback для выбора модели через диалог."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Выберите GGUF модель",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if path:
            self.state.active_model_var.set(path)
            if hasattr(self.app, '_sync_active_model_ui'):
                self.app._sync_active_model_ui()

    def _create_log_section(self, parent):
        section = ttk.LabelFrame(parent, text=" Лог ", padding=10)
        section.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        self.app.log_text = tk.Text(section, height=15, width=40, font=("Consolas", 9),
                            background="#1f242a", foreground="#d8e0e8",
                            relief=SUNKEN, borderwidth=1)
        self.app.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(section, orient=VERTICAL, command=self.app.log_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.app.log_text.configure(yscrollcommand=scrollbar.set)

    def _create_control_buttons(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=X, padx=15, pady=(10, 15))
        
        self.app.start_btn = ttk.Button(frame, text="▶ ЗАПУСТИТЬ",
                                    command=self.app.start_server,
                                    style="Success.TButton")
        self.app.start_btn.pack(side=LEFT, fill=X, expand=True)
        
        self.app.stop_btn = ttk.Button(frame, text="⏹ ОСТАНОВИТЬ",
                                   command=self.app.stop_server,
                                   style="Danger.TButton", state=DISABLED)
        self.app.stop_btn.pack(side=LEFT, fill=X, expand=True, padx=(10, 0))


__all__ = ["LaunchTab"]