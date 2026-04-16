"""
Вкладка моделей (Models Tab)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, HORIZONTAL, VERTICAL, LEFT, RIGHT, BOTH, X, Y, END
import threading
import os
from core.i18n import _
from core.downloader import GGUFDownloader
from core.hardware import get_gpu_info, get_total_ram_gb
from core.gguf_parser import get_quant_from_metadata, extract_quant_from_filename, get_quant_description

class ModelsTab:
    """Вкладка со списком GGUF файлов и метаданными."""

    def __init__(self, parent, state, notebook, app):
        self.parent = parent
        self.state = state
        self.notebook = notebook
        self.app = app  # Ссылка на главный app
        self._metadata_loading = set()

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
        
        ttk.Button(toolbar, text="☁️ " + _("btn_download"),
                      command=self._download_model).pack(side=LEFT, padx=(0, 5))
        
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
        
        self.app.models_tree.heading("#0", text=_("tree_file"), command=lambda: self._sort_tree("#0", False))
        self.app.models_tree.heading("size", text=_("tree_size"), command=lambda: self._sort_tree("size", False))
        self.app.models_tree.heading("arch", text=_("tree_arch"), command=lambda: self._sort_tree("arch", False))
        self.app.models_tree.heading("params", text=_("tree_params"), command=lambda: self._sort_tree("params", False))
        self.app.models_tree.heading("quant", text=_("tree_quant"), command=lambda: self._sort_tree("quant", False))
        
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
        
        # Инфо-лейбл (переносим из app.py)
        self.app.models_info_label = ttk.Label(parent, text="", font=("Segoe UI", 8),
                                                foreground="#90caf9")
        self.app.models_info_label.pack(fill=X, pady=(6, 0))
        

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
        
        self.app.meta_embedding_label = ttk.Label(section, text="")
        self.app.meta_embedding_label.pack(fill=X)
        
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
        
    def _download_model(self):
        """Открывает диалог загрузки модели из HF."""
        vram = get_gpu_info() # MB
        ram = int(get_total_ram_gb() * 1024) # MB
        
        dialog = ModelsTab.HFDownloadDialog(self.parent, self.state, vram, ram)
        self.parent.wait_window(dialog)
        
        # Обновляем список моделей после закрытия диалога
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
        if model_path:
            self._load_gguf_metadata(model_path)
        
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

    def _sort_tree(self, col, reverse):
        l = [(self.app.models_tree.set(k, col), k) for k in self.app.models_tree.get_children('')]
        
        try:
            l.sort(key=lambda t: float(t[0].replace(' GB', '').replace(' MB', '').replace(',', '')), reverse=reverse)
        except ValueError:
            l.sort(key=lambda t: t[0], reverse=reverse)
            
        for index, (val, k) in enumerate(l):
            self.app.models_tree.move(k, '', index)
            
        self.app.models_tree.heading(col, command=lambda: self._sort_tree(col, not reverse))

        
    def refresh_models(self):
        """Асинхронно загружает список моделей."""
        self.app.models_tree.delete(*self.app.models_tree.get_children())
        self.app.models_tree.insert("", "end", text=_("loading"),
                                     values=(_("loading"),), tags=("warning",))
        self.app.models_info_label.config(text=_("loading"))
        
        def worker():
            try:
                models = self.state.get_all_models()
                try:
                    self.app.root.after(0, self._refresh_models_ui, models)
                except (RuntimeError, tk.TclError):
                    pass # Приложение закрывается
            except Exception as e:
                try:
                    self.app.root.after(0, lambda: self.app.models_info_label.config(text=f"Error: {e}"))
                except (RuntimeError, tk.TclError):
                    pass
        
        threading.Thread(target=worker, daemon=True).start()
        
    def _refresh_models_ui(self, models):
        """Обновляет UI со списком моделей."""
        if not self.app.root.winfo_exists():
            return
        self.app.models_tree.delete(*self.app.models_tree.get_children())
        for model_path in models:
            name = os.path.basename(model_path)
            self.app.models_tree.insert("", "end", text=name, values=(model_path,))
        self.app._models_paths = models
        
        count = len(models)
        self.app.models_info_label.config(text=_("found_models").format(count))
        
        
    def _load_gguf_metadata(self, model_path):
        if not model_path or not os.path.isfile(model_path):
            self._clear_metadata()
            return
        if not hasattr(self.app, 'meta_tree'):
            return
        
        cache = self.state.gguf_parser._cache
        if model_path in cache:
            self._display_metadata(cache[model_path], model_path)
            return
        
        if model_path in self._metadata_loading:
            return
        
        self._metadata_loading.add(model_path)
        self.app.meta_header.config(text=_("meta_loading"))
        self._clear_metadata()
        self.app.meta_tree.insert("", "end", text=_("loading"),
                                    values=(_("meta_loading"),), tags=("warning",))
        
        thread = threading.Thread(target=self._load_metadata_thread,
                                    args=(model_path,), daemon=True)
        thread.start()
        
    def _load_metadata_thread(self, model_path):
        try:
            metadata = self.state.gguf_parser.get_cached_or_parse(model_path)
            self.state.gguf_parser._cache[model_path] = metadata
            if self.app.root.winfo_exists():
                self.app.root.after(0, self._display_metadata, metadata, model_path)
        except Exception as e:
            if self.app.root.winfo_exists():
                self.app.root.after(0, self._metadata_load_error, model_path, str(e))
        finally:
            self._metadata_loading.discard(model_path)
        
    def _display_metadata(self, metadata, model_path):
        if not hasattr(self.app, 'meta_tree'):
            return
        
        try:
            for item in self.app.meta_tree.get_children():
                self.app.meta_tree.delete(item)
        
            if "__error__" in metadata:
                self.app.meta_header.config(text=f"⚠️ Ошибка: {os.path.basename(model_path)}")
                self._clear_metadata()
                return
        
            model_name = os.path.basename(model_path)
            __, size_gb = self.state.get_model_info(model_path)
        
            if size_gb:
                gguf_version = metadata.get("__gguf_version__", "?")
                tensor_count = metadata.get("__tensor_count__", 0)
                self.app.meta_header.config(text=f"{model_name} ({size_gb:.2f} GB) | GGUF v{gguf_version} | {tensor_count:,} тензоров")
            else:
                self.app.meta_header.config(text=model_name)
        
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
        
            self.app.meta_arch_label.config(
                text=f"{_('arch')} {arch.upper()}" if arch != "unknown" else f"{_('arch')} unknown")
        
            param_str, param_source = self.state.get_model_param_label(metadata, model_path)
            if param_str:
                suffix = "" if param_source == "metadata" else " (filename)"
                self.app.meta_params_label.config(text=f"{_('params')} {param_str}{suffix}")
            else:
                self.app.meta_params_label.config(text=f"{_('params')} none")
        
            quant_type = get_quant_from_metadata(metadata)
            if quant_type == "Unknown":
                quant_type = extract_quant_from_filename(model_path)
            
            quant_desc = get_quant_description(quant_type)
            self.app.meta_quant_label.config(text=f"{_('quant')} {quant_type}")
            if hasattr(self.app, 'meta_quant_desc'):
                self.app.meta_quant_desc.config(text=quant_desc)
        
            ctx_length = metadata.get(f"{arch}.context_length") or metadata.get("llama.context_length")
            if ctx_length:
                try:
                    self.app.meta_context_label.config(text=f"{_('ctx')} {int(ctx_length):,}")
                except (ValueError, TypeError):
                    self.app.meta_context_label.config(text=f"{_('ctx')} unknown")
            else:
                self.app.meta_context_label.config(text=f"{_('ctx')} unknown")
        
            block_count = metadata.get(f"{arch}.block_count") or metadata.get("llama.block_count")
            if block_count:
                try:
                    self.app.meta_layers_label.config(text=f"🧱 Слои: {int(block_count)}")
                except (ValueError, TypeError):
                    self.app.meta_layers_label.config(text="🧱 Слои: неизвестно")
            else:
                self.app.meta_layers_label.config(text="🧱 Слои: неизвестно")
        
            embedding_length = metadata.get(f"{arch}.embedding_length") or metadata.get("llama.embedding_length")
            if embedding_length:
                try:
                    self.app.meta_embedding_label.config(text=f"📍 Embedding: {int(embedding_length)}")
                except (ValueError, TypeError):
                    self.app.meta_embedding_label.config(text="")
            else:
                self.app.meta_embedding_label.config(text="")
        
            self._populate_metadata_tree(metadata, arch)
        
            if model_path == self.state.active_model_var.get():
                self.app._sync_active_model_ui()
        
        except Exception as e:
            self.app.meta_header.config(text="⚠️ Ошибка загрузки метаданных")
            self._show_metadata_warning(f"Не удалось загрузить метаданные: {e}")
        
    def _metadata_load_error(self, model_path, error):
        if hasattr(self.app, 'meta_header'):
            self.app.meta_header.config(text=f"⚠️ Ошибка: {os.path.basename(model_path)}")
        self._show_metadata_warning(f"Не удалось загрузить метаданные: {error}")
        
    def _clear_metadata(self):
        for attr in ('meta_arch_label', 'meta_params_label', 'meta_quant_label',
                      'meta_quant_desc', 'meta_context_label', 'meta_layers_label',
                      'meta_embedding_label'):
            if hasattr(self.app, attr):
                getattr(self.app, attr).config(text="")
        if hasattr(self.app, 'meta_tree'):
            for item in self.app.meta_tree.get_children():
                self.app.meta_tree.delete(item)
        
    def _show_metadata_warning(self, message):
        if hasattr(self.app, 'meta_tree'):
            self.app.meta_tree.insert("", "end", text="⚠️ Предупреждение",
                                        values=(message,), tags=("warning",))
        
    def _populate_metadata_tree(self, metadata, arch):
        if not hasattr(self.app, 'meta_tree'):
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
            self.app.meta_tree.insert("", "end", text=str(key), values=(str(value),), tags=("important",))
        
        if other_items:
            other_node = self.app.meta_tree.insert("", "end", text="Другие параметры", values=("",), open=False)
            for key, value in other_items:
                self.app.meta_tree.insert(other_node, "end", text=str(key), values=(str(value),))
        
        
    __all__ = ["ModelsTab"]
    
    class HFDownloadDialog(tk.Toplevel):
        def __init__(self, parent, state, vram, ram):
            super().__init__(parent)
            self.state = state
            self.vram = vram
            self.ram = ram
            self.downloader = GGUFDownloader(state)
            
            self.title("Download GGUF from HuggingFace")
            self.geometry("600x500")
            self.resizable(False, False)
            self.transient(parent)
            self.grab_set()
            
            self._create_ui()
        
        def _create_ui(self):
            main_frame = ttk.Frame(self, padding=20)
            main_frame.pack(fill=BOTH, expand=True)
            
            # Repository input
            repo_frame = ttk.Frame(main_frame)
            repo_frame.pack(fill=X, pady=(0, 10))
            ttk.Label(repo_frame, text="HF Repository (username/model):").pack(side=LEFT)
            self.repo_var = tk.StringVar()
            self.repo_entry = ttk.Entry(repo_frame, textvariable=self.repo_var, width=40)
            self.repo_entry.pack(side=LEFT, padx=10, fill=X, expand=True)
            
            self.scan_btn = ttk.Button(repo_frame, text="Scan", command=self._scan_repo)
            self.scan_btn.pack(side=LEFT)
            
            # Quantization list
            list_frame = ttk.LabelFrame(main_frame, text="Available Quantizations", padding=10)
            list_frame.pack(fill=BOTH, expand=True, pady=10)
            
            self.quant_var = tk.StringVar()
            self.quant_list = ttk.Combobox(list_frame, textvariable=self.quant_var, state="readonly")
            self.quant_list.pack(fill=X, pady=(0, 5))
            
            self.rec_label = ttk.Label(list_frame, text="", font=("Segoe UI", 9, "italic"))
            self.rec_label.pack(fill=X)
            
            # Download directory
            dir_frame = ttk.Frame(main_frame)
            dir_frame.pack(fill=X, pady=10)
            self.dir_var = tk.StringVar(value=self.state.settings.get("model_dirs", [os.path.expanduser("~/ai_models")])[0])
            ttk.Label(dir_frame, text="Save to:").pack(side=LEFT)
            ttk.Entry(dir_frame, textvariable=self.dir_var).pack(side=LEFT, padx=10, fill=X, expand=True)
            ttk.Button(dir_frame, text="Browse", command=self._browse_dir).pack(side=LEFT)
            
            # Actions
            btn_frame = ttk.Frame(main_frame)
            btn_frame.pack(fill=X, pady=(10, 0))
            
            self.download_btn = ttk.Button(btn_frame, text="Download", command=self._start_download, state="disabled")
            self.download_btn.pack(side=RIGHT, padx=5)
            ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=RIGHT)
            
        def _browse_dir(self):
            folder = filedialog.askdirectory()
            if folder:
                self.dir_var.set(folder)
            
        def _scan_repo(self):
            repo = self.repo_var.get().strip()
            if not repo:
                messagebox.showerror("Error", "Please enter a repository name")
                return
            
            self.scan_btn.config(state="disabled")
            self.rec_label.config(text="Scanning...")
            
            def worker():
                try:
                    quants = self.downloader.list_available_quantizations(repo)
                    if not quants:
                        self.after(0, lambda: messagebox.showwarning("Warning", "No GGUF files found in this repo"))
                        self.after(0, lambda: self.scan_btn.config(state="normal"))
                        return
                    
                    # Update combobox
                    quant_names = [q[0] for q in quants]
                    self.after(0, lambda: self.quant_list.config(values=quant_names))
                    
                    # Recommendation
                    ctx = self.state.settings.get("ctx", 4096)
                    rec = self.downloader.recommend_quant(quants, self.vram, self.ram, ctx)
                    if rec:
                        self.after(0, lambda: self.rec_label.config(text=f"🌟 Recommended: {rec[0]} - {rec[1]}"))
                        self.after(0, lambda: self.quant_var.set(rec[0]))
                    
                    self.after(0, lambda: self.download_btn.config(state="normal"))
                    self.after(0, lambda: self.scan_btn.config(state="normal"))
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Error", f"Scan failed: {e}"))
                    self.after(0, lambda: self.scan_btn.config(state="normal"))
            
            threading.Thread(target=worker, daemon=True).start()
            
        def _start_download(self):
            repo = self.repo_var.get().strip()
            quant = self.quant_var.get()
            dest = self.dir_var.get().strip()
            
            if not repo or not dest:
                messagebox.showerror("Error", "Repository and destination are required")
                return
            
            files = self.downloader.get_model_files(repo, quant)
            if not files:
                messagebox.showerror("Error", "No files matched the selected quantization")
                return
            
            self.download_btn.config(state="disabled")
            self.scan_btn.config(state="disabled")
            
            def worker():
                try:
                    downloaded, failed = self.downloader.download(repo, files, dest)
                    if downloaded:
                        self.after(0, lambda: messagebox.showinfo("Success", f"Downloaded {len(downloaded)} file(s)"))
                        self.after(0, self.destroy)
                    elif failed:
                        errs = "\n".join([f"{f}: {e}" for f, e in failed])
                        self.after(0, lambda: messagebox.showerror("Error", f"Downloads failed:\n{errs}"))
                        self.after(0, lambda: self.download_btn.config(state="normal"))
                        self.after(0, lambda: self.scan_btn.config(state="normal"))
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Error", f"Unexpected error: {e}"))
                    self.after(0, lambda: self.download_btn.config(state="normal"))
                    self.after(0, lambda: self.scan_btn.config(state="normal"))
            
            threading.Thread(target=worker, daemon=True).start()
