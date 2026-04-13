import tkinter as tk
from tkinter import ttk, scrolledtext, END, NORMAL, DISABLED, W, E, X, Y, BOTH, LEFT, RIGHT
from core.i18n import _
from core.benchmark_manager import LlamaBenchManager
import os

class BenchmarkTab:
    def __init__(self, parent, state, toast):
        self.parent = parent
        self.state = state
        self.toast = toast
        self.bench_manager = LlamaBenchManager()
        
        self._create_ui()

    def _create_ui(self):
        main_container = ttk.Frame(self.parent)
        main_container.pack(fill=BOTH, expand=True, padx=15, pady=15)

        # Левая панель параметров
        left_panel = ttk.Frame(main_container, width=300)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 15))
        left_panel.pack_propagate(False)

        params_section = ttk.LabelFrame(left_panel, text=_("bench_params_section"), padding=10)
        params_section.pack(fill=X)

        # Prompt
        ttk.Label(params_section, text=_("bench_prompt_size")).grid(row=0, column=0, sticky=W, pady=5)
        self.p_spin = ttk.Spinbox(params_section, from_=1, to=32768, textvariable=self.state.bench_prompt_var, width=10)
        self.p_spin.grid(row=0, column=1, sticky=E, pady=5)

        # Predict
        ttk.Label(params_section, text=_("bench_predict_size")).grid(row=1, column=0, sticky=W, pady=5)
        self.n_spin = ttk.Spinbox(params_section, from_=1, to=8192, textvariable=self.state.bench_predict_var, width=10)
        self.n_spin.grid(row=1, column=1, sticky=E, pady=5)

        # Threads
        ttk.Label(params_section, text=_("bench_threads")).grid(row=2, column=0, sticky=W, pady=5)
        self.t_spin = ttk.Spinbox(params_section, from_=1, to=128, textvariable=self.state.bench_threads_var, width=10)
        self.t_spin.grid(row=2, column=1, sticky=E, pady=5)

        # NGL
        ttk.Label(params_section, text=_("bench_gpu_layers")).grid(row=3, column=0, sticky=W, pady=5)
        self.ngl_spin = ttk.Spinbox(params_section, from_=0, to=999, textvariable=self.state.bench_ngl_var, width=10)
        self.ngl_spin.grid(row=3, column=1, sticky=E, pady=5)

        self.run_btn = ttk.Button(left_panel, text=_("btn_run_benchmark"), style="Success.TButton", command=self.run_benchmark)
        self.run_btn.pack(fill=X, pady=(15, 0))

        self.stop_btn = ttk.Button(left_panel, text=_("btn_stop"), style="Danger.TButton", command=self.stop_benchmark, state=DISABLED)
        self.stop_btn.pack(fill=X, pady=(5, 0))

        ttk.Button(left_panel, text=_("btn_copy_all_log"), command=self.copy_all_results).pack(fill=X, pady=(15, 0))

        # Правая панель результатов
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)

        results_section = ttk.LabelFrame(right_panel, text=_("bench_results_section"), padding=10)
        results_section.pack(fill=BOTH, expand=True)

        self.results_text = scrolledtext.ScrolledText(results_section, font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4")
        self.results_text.pack(fill=BOTH, expand=True)
        self.results_text.bind("<Button-3>", self.show_context_menu)
        
        # Инфо о выбранной модели
        self.model_info_label = ttk.Label(left_panel, text="", font=("Segoe UI", 9), wraplength=280, justify=LEFT)
        self.model_info_label.pack(fill=X, pady=(20, 0))
        self._update_model_info()
        
        # Подписка на обновление модели
        self.state.add_update_listener(self._update_model_info)

    def _update_model_info(self):
        model_path = self.state.active_model_var.get()
        if not model_path:
            self.model_info_label.config(text=_("no_model"))
            return
        
        name = os.path.basename(model_path)
        self.model_info_label.config(text=f"Модель: {name}")

    def run_benchmark(self):
        model_path = self.state.active_model_var.get()
        if not model_path or not os.path.exists(model_path):
            self.toast.show(_("no_model"), "warning")
            return

        exe_path = self.state.get_benchmark_exe_path()
        if not exe_path:
            self.toast.show("llama-bench.exe не найден", "danger")
            return

        self.results_text.delete(1.0, END)
        self.results_text.insert(END, f"{_('bench_running')}\n\n")
        
        self.run_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)

        success, msg = self.bench_manager.run(
            exe_path,
            model_path,
            prompt=self.state.bench_prompt_var.get(),
            predict=self.state.bench_predict_var.get(),
            threads=self.state.bench_threads_var.get(),
            ngl=self.state.bench_ngl_var.get(),
            on_output=self._on_bench_output,
            on_finished=self._on_bench_finished
        )

        if not success:
            self.results_text.insert(END, f"\n{_('bench_error')}: {msg}\n")
            self._on_bench_finished()

    def _on_bench_output(self, line):
        self.parent.after(0, lambda: self._append_log(line))

    def _append_log(self, line):
        self.results_text.insert(END, line + "\n")
        self.results_text.see(END)

    def _on_bench_finished(self):
        self.parent.after(0, self._ui_on_finished)

    def _ui_on_finished(self):
        self.results_text.insert(END, f"\n{_('bench_finished')}\n")
        self.run_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
        self.state.save_settings()

    def stop_benchmark(self):
        if self.bench_manager.stop():
            self.results_text.insert(END, "\n[Stopped by user]\n")

    def copy_all_results(self):
        text = self.results_text.get("1.0", "end-1c")
        self.parent.clipboard_clear()
        self.parent.clipboard_append(text)
        self.toast.show(_("btn_copy_all_log"))

    def show_context_menu(self, event):
        menu = tk.Menu(self.parent, tearoff=0)
        menu.add_command(label=_("btn_copy_all_log"), command=self.copy_all_results)
        menu.add_command(label=_("btn_clear_log"), command=lambda: self.results_text.delete(1.0, END))
        menu.post(event.x_root, event.y_root)
