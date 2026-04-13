"""
Вкладка моделей (Models Tab)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, HORIZONTAL, VERTICAL, LEFT, RIGHT, BOTH, X, Y, END
import threading
import os
from core.i18n import _


class ModelsTab:
    """Вкладка со списком GGUF файлов и метаданными."""

    def __init__(self, parent, state, notebook, app):
        self.parent = parent
        self.state = state
        self.notebook = notebook
        self.app = app  # Ссылка на главный app

    def create(self, frame):
        """Создает UI вкладки моделей."""
        main_container = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - список файлов
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)
        
        # Правая панель - метаданные
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)
        
        # Создаем секции
        self._create_toolbar(left_panel)
        self._create_models_list(left_panel)
        self._create_metadata_panel(right_panel)
        
        # Загружаем модели при старте
        self.app.refresh_models()
        
        return frame

    def _create_toolbar(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=X, pady=(0, 5))
        
        ttk.Button(toolbar, text="📁 " + _("btn_add_dir"),
                  command=self._add_folder).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(toolbar, text="📄 " + _("btn_add_file"),
                  command=self._add_files).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(toolbar, text="🗑 " + _("btn_remove"),
                  command=self._remove_selected,
                  style="Danger.TButton").pack(side=LEFT, padx=(0, 5))
        
        ttk.Label(toolbar, text=_("search_label")).pack(side=LEFT, padx=(10, 5))
        
        self.app.search_var = tk.StringVar()
        search_entry = ttk.Entry(toolbar, textvariable=self.app.search_var, width=20)
        search_entry.pack(side=LEFT)
        search_entry.bind("<KeyRelease>", self._on_search)

    def _create_models_list(self, parent):
        # Treeview для списка моделей
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=BOTH, expand=True)
        
        columns = ("size", "arch", "params", "quant")
        self.app.models_tree = ttk.Treeview(tree_frame, columns=columns,
                                        show="tree headings", height=20)
        
        self.app.models_tree.heading("#0", text=_("tree_file"))
        self.app.models_tree.heading("size", text=_("tree_size"))
        self.app.models_tree.heading("arch", text=_("tree_arch"))
        self.app.models_tree.heading("params", text=_("tree_params"))
        self.app.models_tree.heading("quant", text=_("tree_quant"))
        
        self.app.models_tree.column("#0", width=300)
        self.app.models_tree.column("size", width=80)
        self.app.models_tree.column("arch", width=100)
        self.app.models_tree.column("params", width=80)
        self.app.models_tree.column("quant", width=60)
        
        # Scrollbar
        vsb = ttk.Scrollbar(tree_frame, orient=VERTICAL,
                          command=self.app.models_tree.yview)
        vsb.pack(side=RIGHT, fill=Y)
        self.app.models_tree.configure(yscrollcommand=vsb.set)
        
        self.app.models_tree.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Double-click для выбора модели
        self.app.models_tree.bind("<Double-Button-1>", self._on_model_double_click)

    def _create_metadata_panel(self, parent):
        section = ttk.LabelFrame(parent, text=_("meta_panel"), padding=10)
        section.pack(fill=BOTH, expand=True)
        
        # Заголовок
        self.app.meta_header = ttk.Label(section, text=_("select_model_list"),
                                      font=("Segoe UI", 10, "bold"))
        self.app.meta_header.pack(fill=X, pady=(0, 5))
        
        # Основная инфо
        self.app.meta_arch_label = ttk.Label(section, text="")
        self.app.meta_arch_label.pack(fill=X)
        
        self.app.meta_params_label = ttk.Label(section, text="")
        self.app.meta_params_label.pack(fill=X)
        
        self.app.meta_quant_label = ttk.Label(section, text="")
        self.app.meta_quant_label.pack(fill=X)
        
        self.app.meta_context_label = ttk.Label(section, text="")
        self.app.meta_context_label.pack(fill=X)
        
        self.app.meta_layers_label = ttk.Label(section, text="")
        self.app.meta_layers_label.pack(fill=X)
        
        # Treeview для всех метаданных
        meta_frame = ttk.Frame(section)
        meta_frame.pack(fill=BOTH, expand=True, pady=(10, 0))
        
        self.app.meta_tree = ttk.Treeview(meta_frame, height=15, columns=("value",), show="tree")
        self.app.meta_tree.heading("#0", text=_("meta_key_value"))
        self.app.meta_tree.heading("value", text="")
        self.app.meta_tree.column("#0", width=200)
        self.app.meta_tree.column("value", width=300)
        
        vsb = ttk.Scrollbar(meta_frame, orient=VERTICAL,
                  command=self.app.meta_tree.yview)
        vsb.pack(side=RIGHT, fill=Y)
        self.app.meta_tree.configure(yscrollcommand=vsb.set)
        
        self.app.meta_tree.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Загрузка метаданных при выборе
        self.app.models_tree.bind("<<TreeviewSelect>>", self._on_model_select)

    def _add_folder(self):
        folder = filedialog.askdirectory(title=_("add_folder_title"))
        if folder:
            dirs = self.state.settings.setdefault("model_dirs", [])
            if folder not in dirs:
                dirs.append(folder)
                self.state.save_settings()
                self.app.refresh_models()

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title=_("add_files_title"),
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if files:
            file_list = self.state.settings.setdefault("model_files", [])
            for f in files:
                if f not in file_list:
                    file_list.append(f)
            self.state.save_settings()
            self.app.refresh_models()

    def _remove_selected(self):
        selection = self.app.models_tree.selection()
        if not selection:
            return
        if messagebox.askyesno(_("remove_confirm_title"), _("remove_confirm_msg")):
            to_remove = []
            for item in selection:
                path = self.app.models_tree.item(item, "values")[0] if self.app.models_tree.item(item, "values") else self.app.models_tree.item(item, "text")
                to_remove.append(path)

            # Удаляем из model_files (если там есть)
            file_list = self.state.settings.get("model_files", [])
            self.state.settings["model_files"] = [f for f in file_list if f not in to_remove]
            # Удаляем из model_dirs (если там есть)
            dir_list = self.state.settings.get("model_dirs", [])
            self.state.settings["model_dirs"] = [d for d in dir_list if d not in to_remove]

            self.state.save_settings()
            self.app.refresh_models()

    def _on_search(self, event=None):
        query = self.app.search_var.get().lower()
        if not query:
            # Если поиск очищен — перезагружаем модели
            self.app.refresh_models()
            return

        # Treeview не поддерживает hide=True, поэтому фильтруем через пересоздание
        all_models = getattr(self.app, '_all_models_cache', [])
        if not all_models:
            return

        for item in self.app.models_tree.get_children():
            self.app.models_tree.delete(item)

        for model_path, display_name in all_models:
            if query in display_name.lower():
                self.app.models_tree.insert("", "end", text=display_name, values=(model_path,))

    def _on_model_select(self, event=None):
        selection = self.app.models_tree.selection()
        if not selection:
            return
        item = selection[0]
        # Путь модели может быть в values[0] или в text
        values = self.app.models_tree.item(item, "values")
        model_path = values[0] if values else self.app.models_tree.item(item, "text")
        if model_path and hasattr(self.app, '_load_gguf_metadata'):
            self.app._load_gguf_metadata(model_path)

    def _on_model_double_click(self, event=None):
        selection = self.app.models_tree.selection()
        if not selection:
            return
        item = selection[0]
        values = self.app.models_tree.item(item, "values")
        model_path = values[0] if values else self.app.models_tree.item(item, "text")
        if model_path:
            self.state.active_model_var.set(model_path)
            if hasattr(self.app, '_sync_active_model_ui'):
                self.app._sync_active_model_ui()
            if hasattr(self.app, 'notebook'):
                self.app.notebook.select(0)  # Переход на вкладку запуска


__all__ = ["ModelsTab"]