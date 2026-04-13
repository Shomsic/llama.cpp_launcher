"""Панели параметров: Basic, Generation, Advanced и Presets."""
import tkinter as tk
from tkinter import ttk, simpledialog, W, E, X, LEFT

from core.config import PRESETS
from core.i18n import _


class PresetPanel:
    """Выбор и сохранение пресетов."""

    def __init__(self, parent, state, toast):
        self.state = state
        self.toast = toast

        section = ttk.LabelFrame(parent, text=" " + _("presets") + " ", padding=10)
        section.pack(fill=X, pady=(0, 10))

        row = ttk.Frame(section)
        row.pack(fill=X)

        ttk.Label(row, text=_("select_preset")).pack(side=LEFT, padx=(0, 5))

        preset_names = list(PRESETS.keys())
        self.combo = ttk.Combobox(
            row, textvariable=state.preset_var,
            values=["— Пользовательские —"] + preset_names,
            state="readonly", width=25
        )
        self.combo.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        self.combo.bind("<<ComboboxSelected>>", self._apply_preset)

        ttk.Button(row, text=_("btn_save"), command=self._save_custom, width=12).pack(side=LEFT)

        self.desc_label = ttk.Label(section, text="", font=("Segoe UI", 8),
                                    foreground="#90caf9", wraplength=400)
        self.desc_label.pack(fill=X, pady=(5, 0))

    def _apply_preset(self, _event=None):
        selected = self.state.preset_var.get()
        if selected not in PRESETS:
            self.desc_label.config(text="")
            return
        preset = PRESETS[selected]
        var_map = {
            'ctx': 'ctx_var', 'temp': 'temp_var', 'top_k': 'top_k_var',
            'top_p': 'top_p_var', 'min_p': 'min_p_var',
            'repeat_penalty': 'repeat_penalty_var', 'presence_penalty': 'presence_penalty_var',
            'frequency_penalty': 'frequency_penalty_var', 'batch_size': 'batch_size_var',
            'flash_attn': 'flash_attn_var', 'mirostat': 'mirostat_var'
        }
        desc = []
        for key, val in preset.items():
            attr = var_map.get(key)
            if attr and hasattr(self.state, attr):
                getattr(self.state, attr).set(val)
            desc.append(f"{key}: {val}")
        self.desc_label.config(text=" | ".join(desc))
        self.state.notify_update()

    def _save_custom(self):
        name = simpledialog.askstring(_("btn_save"), _("select_preset")) # Reusing keys if appropriate or I should add specific ones. Let's add specific if needed.
        # Wait, i18n has "btn_save" as "💾 Сохранить". 
        # Actually I'll use direct strings for now and fix them in i18n if I missed some.
        name = simpledialog.askstring(_("btn_save"), _("btn_save"))
        if not name:
            return
        s = self.state
        preset = {
            'ctx': s.ctx_var.get(), 'temp': s.temp_var.get(),
            'top_k': s.top_k_var.get(), 'top_p': s.top_p_var.get(),
            'min_p': s.min_p_var.get(), 'repeat_penalty': s.repeat_penalty_var.get(),
            'presence_penalty': s.presence_penalty_var.get(),
            'frequency_penalty': s.frequency_penalty_var.get(),
            'batch_size': s.batch_size_var.get(), 'flash_attn': s.flash_attn_var.get(),
            'mirostat': s.mirostat_var.get()
        }
        s.settings.setdefault('custom_presets', {})[name] = preset
        PRESETS[name] = preset
        self.combo.config(values=["— Пользовательские —"] + list(PRESETS.keys()))
        s.preset_var.set(name)
        s.save_settings()
        self._apply_preset()
        self.toast.show(f"💾 Пресет '{name}' сохранён!")


class BasicParamsPanel:
    """Секция основных параметров: host, port, NGL, ctx, threads, GPU %."""

    def __init__(self, parent, state, tooltip):
        self.state = state
        s = state

        section = ttk.LabelFrame(parent, text=" " + _("basic_params") + " ", padding=10)
        section.pack(fill=X, pady=(0, 10))
        grid = ttk.Frame(section)
        grid.pack(fill=X)

        # Model info
        self.model_info_label = ttk.Label(grid, text=_("no_model"), font=("Segoe UI", 9))
        self.model_info_label.grid(row=0, column=0, columnspan=4, sticky=W, pady=(0, 5))

        # Host
        lbl = ttk.Label(grid, text=_("host_ip"))
        lbl.grid(row=1, column=0, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "IP сервера (127.0.0.1 — локально, 0.0.0.0 — из локальной сети)")
        ttk.Entry(grid, textvariable=s.host_var, width=12).grid(row=1, column=1, sticky=W, padx=(0, 15), pady=3)

        # Port
        lbl = ttk.Label(grid, text=_("port"))
        lbl.grid(row=1, column=2, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Порт HTTP сервера (1024-65535)")
        ttk.Spinbox(grid, from_=1024, to=65535, textvariable=s.port_var, width=10).grid(
            row=1, column=3, sticky=W, padx=(0, 15), pady=3)

        # NGL
        lbl = ttk.Label(grid, text=_("gpu_layers"))
        lbl.grid(row=2, column=0, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Количество слоёв модели для загрузки в GPU")
        self.ngl_scale = ttk.Scale(grid, from_=0, to=s.max_ngl, variable=s.ngl_var,
                                   command=lambda _: s.notify_update())
        self.ngl_scale.grid(row=2, column=1, columnspan=3, sticky=E + W, padx=(0, 5), pady=3)
        self.ngl_label = ttk.Label(grid, text="0", width=4)
        self.ngl_label.grid(row=2, column=4, sticky=W, pady=3)

        # Context
        lbl = ttk.Label(grid, text=_("ctx_size"))
        lbl.grid(row=3, column=0, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Размер контекста в токенах")
        self.ctx_scale = ttk.Scale(grid, from_=512, to=s.max_ctx, variable=s.ctx_var,
                                   command=lambda _: s.notify_update())
        self.ctx_scale.grid(row=3, column=1, columnspan=3, sticky=E + W, padx=(0, 5), pady=3)
        self.ctx_label = ttk.Label(grid, text="4096", width=6)
        self.ctx_label.grid(row=3, column=4, sticky=W, pady=3)

        # Threads
        cpu_cores = s.get_cpu_cores()
        lbl = ttk.Label(grid, text=_("threads"))
        lbl.grid(row=4, column=0, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Количество CPU потоков")
        self.threads_scale = ttk.Scale(grid, from_=1, to=cpu_cores, variable=s.threads_var,
                                       command=lambda _: s.notify_update())
        self.threads_scale.grid(row=4, column=1, columnspan=3, sticky=E + W, padx=(0, 5), pady=3)
        self.threads_label = ttk.Label(grid, text=str(s.threads_var.get()), width=4)
        self.threads_label.grid(row=4, column=4, sticky=W, pady=3)

        # Memory estimate
        self.memory_label = ttk.Label(grid, text="Оценка памяти: 0 MB",
                                      font=("Segoe UI", 9, "bold"), foreground="#4caf50")
        self.memory_label.grid(row=5, column=0, columnspan=5, sticky=W, pady=(10, 3))

        # GPU offload %
        lbl = ttk.Label(grid, text=_("gpu_ram_pct"))
        lbl.grid(row=6, column=0, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Процент слоёв в GPU (0=CPU, 100=GPU)")
        self.gpu_pct_scale = ttk.Scale(grid, from_=0, to=100, variable=s.gpu_offload_pct_var,
                                       command=self._on_gpu_pct_changed)
        self.gpu_pct_scale.grid(row=6, column=1, columnspan=3, sticky=E + W, padx=(0, 5), pady=3)
        self.gpu_pct_label = ttk.Label(grid, text="100%", width=5)
        self.gpu_pct_label.grid(row=6, column=4, sticky=W, pady=3)

        for i in range(4):
            grid.columnconfigure(i, weight=1)

    def _on_gpu_pct_changed(self, _=None):
        s = self.state
        pct = s.gpu_offload_pct_var.get()
        self.gpu_pct_label.config(text=f"{pct}%")
        model_path = s.active_model_var.get()
        if model_path:
            calculated_ngl = int(s.max_ngl * pct / 100)
            s.ngl_var.set(calculated_ngl)
            self.ngl_label.config(text=str(calculated_ngl))
        s.notify_update()

    def sync_from_state(self):
        """Синхронизирует виджеты из state (лимиты слайдеров, labels)."""
        s = self.state
        self.ngl_scale.config(to=s.max_ngl)
        if s.ngl_var.get() > s.max_ngl:
            s.ngl_var.set(s.max_ngl)
        self.ctx_scale.config(to=s.max_ctx)
        if s.ctx_var.get() > s.max_ctx:
            s.ctx_var.set(s.max_ctx)

        # Обновляем лимит threads
        cpu_cores = s.get_cpu_cores()
        self.threads_scale.config(to=cpu_cores)
        if s.threads_var.get() > cpu_cores:
            s.threads_var.set(cpu_cores)

        self.ngl_label.config(text=str(s.ngl_var.get()))
        self.ctx_label.config(text=str(s.ctx_var.get()))
        self.threads_label.config(text=str(s.threads_var.get()))
        self.memory_label.config(text=f"🧠 {_('mem_estimate')} {s.calculate_memory_text()}")

        # Sync GPU %
        if s.max_ngl > 0:
            pct = int(round(s.ngl_var.get() / max(s.max_ngl, 1) * 100))
            s.gpu_offload_pct_var.set(max(0, min(pct, 100)))
            self.gpu_pct_label.config(text=f"{s.gpu_offload_pct_var.get()}%")

    def update_model_info(self):
        s = self.state
        model_path = s.active_model_var.get()
        if not model_path:
            self.model_info_label.config(text=_("no_model"))
            return
        import os
        name = os.path.basename(model_path)
        __, gb = s.get_model_info(model_path)
        if gb:
            self.model_info_label.config(text=f"Модель: {name} ({gb:.1f} GB)")
        else:
            self.model_info_label.config(text=f"Модель: {name}")


class GenerationParamsPanel:
    """Параметры генерации: temp, top_k/p, min_p, penalties, mirostat, etc."""

    def __init__(self, parent, state, tooltip):
        section = ttk.LabelFrame(parent, text=" " + _("generation_params") + " ", padding=10)
        section.pack(fill=X, pady=(0, 10))
        grid = ttk.Frame(section)
        grid.pack(fill=X)
        s = state

        params = [
            (0, 0, "Температура:", s.temp_var, {"from_": 0.0, "to": 2.0, "increment": 0.1, "format": "%.1f", "width": 10},
             "Температура генерации (0=детерминировано, 2=креативно)"),
            (0, 2, "Top-k:", s.top_k_var, {"from_": 0, "to": 100, "width": 8},
             "Ограничение выборки k лучших токенов (0=отключено)"),
            (1, 0, "Top-p:", s.top_p_var, {"from_": 0.0, "to": 1.0, "increment": 0.05, "format": "%.2f", "width": 10},
             "Nucleus sampling (0.9=90% вероятных токенов)"),
            (1, 2, "Min-p:", s.min_p_var, {"from_": 0.0, "to": 1.0, "increment": 0.05, "format": "%.2f", "width": 8},
             "Минимальная вероятность относительно top токена"),
            (2, 0, "repeat_penalty:", s.repeat_penalty_var, {"from_": 0.0, "to": 2.0, "increment": 0.1, "format": "%.1f", "width": 10},
             "Штраф за повторяющиеся токены (1.0=без штрафа)"),
            (2, 2, "presence_penalty:", s.presence_penalty_var, {"from_": -2.0, "to": 2.0, "increment": 0.1, "format": "%.1f", "width": 8},
             "Штраф за присутствие токена (-2.0 до 2.0)"),
            (3, 0, "frequency_penalty:", s.frequency_penalty_var, {"from_": -2.0, "to": 2.0, "increment": 0.1, "format": "%.1f", "width": 10},
             "Штраф за частоту токена (-2.0 до 2.0)"),
            (3, 2, "mirostat:", s.mirostat_var, {"from_": 0, "to": 2, "width": 8},
             "Mirostat: 0=отключен, 1=Mirostat, 2=Mirostat 2.0"),
            (4, 0, "max_tokens:", s.n_predict_var, {"from_": -1, "to": 8192, "width": 10},
             "Максимум токенов для генерации (-1=без ограничений)"),
        ]
        for row, col, label_text, var, spin_opts, tip_text in params:
            lbl = ttk.Label(grid, text=label_text)
            lbl.grid(row=row, column=col, sticky=W, padx=(0, 5), pady=3)
            tooltip.bind(lbl, tip_text)
            ttk.Spinbox(grid, textvariable=var, **spin_opts).grid(
                row=row, column=col + 1, sticky=W, padx=(0, 15) if col == 0 else 0, pady=3)

        # Reasoning combo
        lbl = ttk.Label(grid, text="reasoning (think):")
        lbl.grid(row=4, column=2, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Генерация рассуждений (<think>) для Deepseek и подобных моделей")
        ttk.Combobox(grid, textvariable=s.reasoning_var, values=["auto", "on", "off"],
                     width=8, state="readonly").grid(row=4, column=3, sticky=W, pady=3)

        for i in range(3):
            grid.columnconfigure(i, weight=1)


class AdvancedParamsPanel:
    """Расширенные настройки: slots, batch, flash_attn, seed, mmap, mlock, kv_offload, etc."""

    def __init__(self, parent, state, tooltip):
        self.state = state
        section = ttk.LabelFrame(parent, text=" " + _("advanced_params") + " ", padding=10)
        section.pack(fill=X, pady=(0, 10))
        grid = ttk.Frame(section)
        grid.pack(fill=X)
        s = state

        # Row 0
        lbl = ttk.Label(grid, text="Слоты сервера:")
        lbl.grid(row=0, column=0, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Количество параллельных слотов сервера (-np)")
        spin = ttk.Spinbox(grid, from_=1, to=32, textvariable=s.parallel_slots_var, width=10)
        spin.grid(row=0, column=1, sticky=W, padx=(0, 15), pady=3)
        spin.bind("<KeyRelease>", lambda _: s.notify_update())

        lbl = ttk.Label(grid, text="batch_size:")
        lbl.grid(row=0, column=2, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Размер логического batch (по умолчанию 2048)")
        ttk.Spinbox(grid, from_=1, to=8192, textvariable=s.batch_size_var, width=8).grid(
            row=0, column=3, sticky=W, pady=3)

        # Row 1
        combos = [
            (1, 0, "flash_attn:", s.flash_attn_var, ["on", "off", "auto"],
             "Flash Attention (ускоряет длинный контекст)"),
            (2, 0, "mmap:", s.mmap_var, ["on", "off"], "Memory mapping"),
            (2, 2, "mlock:", s.mlock_var, ["on", "off"], "Удерживать модель в RAM"),
            (3, 0, "kv_offload:", s.kv_offload_var, ["on", "off"], "Выгрузка KV кэша в GPU"),
            (3, 2, "cache_prompt:", s.cache_prompt_var, ["on", "off"], "Кэширование промптов"),
            (5, 2, "Web UI:", s.webui_var, ["on", "off"], "Встроенный чат Web UI"),
        ]
        for row, col, label_text, var, values, tip in combos:
            lbl = ttk.Label(grid, text=label_text)
            lbl.grid(row=row, column=col, sticky=W, padx=(0, 5), pady=3)
            tooltip.bind(lbl, tip)
            if var and values:
                cb = ttk.Combobox(grid, textvariable=var, values=values, width=8, state="readonly")
                cb.grid(row=row, column=col + 1, sticky=W, padx=(0, 15) if col == 0 else 0, pady=3)
                if var is s.kv_offload_var:
                    cb.bind("<<ComboboxSelected>>", lambda _: s.notify_update())

        # Seed spinbox
        lbl = ttk.Label(grid, text="seed:")
        lbl.grid(row=1, column=2, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "Зерно RNG (-1=случайное)")
        ttk.Spinbox(grid, from_=-1, to=2147483647, textvariable=s.seed_var, width=8).grid(
            row=1, column=3, sticky=W, pady=3)

        # Row 4 — rope
        lbl = ttk.Label(grid, text="rope_scale:")
        lbl.grid(row=4, column=0, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "RoPE масштабирование (для экстенсивного контекста)")
        ttk.Entry(grid, textvariable=s.rope_scale_var, width=10).grid(
            row=4, column=1, sticky=W, padx=(0, 15), pady=3)

        lbl = ttk.Label(grid, text="rope_freq_base:")
        lbl.grid(row=4, column=2, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "RoPE базовая частота")
        ttk.Entry(grid, textvariable=s.rope_freq_base_var, width=10).grid(
            row=4, column=3, sticky=W, pady=3)

        # Row 5 — API key, webui
        lbl = ttk.Label(grid, text="API Key:")
        lbl.grid(row=5, column=0, sticky=W, padx=(0, 5), pady=3)
        tooltip.bind(lbl, "API ключ для авторизации (пусто=отключён)")
        ttk.Entry(grid, textvariable=s.api_key_var, width=15).grid(
            row=5, column=1, sticky=W, padx=(0, 15), pady=3)

        # Row 6 — custom args
        lbl = ttk.Label(grid, text=_("custom_args"))
        lbl.grid(row=6, column=0, columnspan=2, sticky=W, pady=(10, 3))
        tooltip.bind(lbl, "Свободные аргументы через пробел (напр. --metrics -cb)")
        ttk.Entry(grid, textvariable=s.custom_args_var, width=50).grid(
            row=6, column=2, columnspan=2, sticky=W, pady=(10, 3))
