"""Карточка GPU / Память — прогресс-бары VRAM/RAM и оценка tok/s."""
import os
import tkinter as tk
from tkinter import ttk, HORIZONTAL, LEFT, RIGHT, X

from core.gguf_parser import extract_quant_from_filename
from core.i18n import _


class GpuCard:
    """Виджет-карточка для блока GPU / Память."""

    def __init__(self, parent, state):
        self.state = state

        card = ttk.LabelFrame(parent, text=" " + _("gpu_memory") + " ", padding=10)
        card.pack(fill=X, pady=(0, 10))
        self.frame = card

        # GPU Name
        self.gpu_name_label = ttk.Label(card, text=_("gpu_detecting"), font=("Segoe UI", 9))
        self.gpu_name_label.pack(fill=X, pady=(0, 5))

        # VRAM
        vram_frame = ttk.Frame(card)
        vram_frame.pack(fill=X, pady=(0, 5))
        ttk.Label(vram_frame, text=_("vram"), font=("Segoe UI", 9, "bold"), width=6).pack(side=LEFT)
        self.vram_bar = ttk.Progressbar(vram_frame, orient=HORIZONTAL, mode="determinate")
        self.vram_bar.pack(side=LEFT, fill=X, expand=True, padx=(5, 5))
        self.vram_text = ttk.Label(vram_frame, text="0 / 0 GB", font=("Segoe UI", 9), width=14)
        self.vram_text.pack(side=LEFT)

        # RAM
        ram_frame = ttk.Frame(card)
        ram_frame.pack(fill=X, pady=(0, 5))
        ttk.Label(ram_frame, text=_("ram"), font=("Segoe UI", 9, "bold"), width=6).pack(side=LEFT)
        self.ram_bar = ttk.Progressbar(ram_frame, orient=HORIZONTAL, mode="determinate")
        self.ram_bar.pack(side=LEFT, fill=X, expand=True, padx=(5, 5))
        self.ram_text = ttk.Label(ram_frame, text="0 / 32 GB", font=("Segoe UI", 9), width=14)
        self.ram_text.pack(side=LEFT)

        # Total
        self.total_label = ttk.Label(card, text=_("total") + " 0 GB",
                                     font=("Segoe UI", 9, "bold"), foreground="#4caf50")
        self.total_label.pack(fill=X, pady=(5, 0))

        # Warning
        self.warning_label = ttk.Label(card, text="", font=("Segoe UI", 9), foreground="#c59a66")
        self.warning_label.pack(fill=X, pady=(3, 0))

        # TPS
        tps_frame = ttk.Frame(card)
        tps_frame.pack(fill=X, pady=(8, 0))
        ttk.Label(tps_frame, text=_("forecast"), font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        self.tps_label = ttk.Label(tps_frame, text="— tok/s",
                                   font=("Segoe UI", 11, "bold"), foreground="#8cb39d")
        self.tps_label.pack(side=RIGHT)
        self.tps_detail = ttk.Label(tps_frame, text="", font=("Segoe UI", 8), foreground="#9aabb9")
        self.tps_detail.pack(side=RIGHT, padx=(0, 10))
        self.tps_hint = ttk.Label(card, text="", font=("Segoe UI", 8), foreground="#8e9ba8")
        self.tps_hint.pack(fill=X, pady=(4, 0))

        # Первичное обновление
        self._update_gpu_name()
        state.add_update_listener(self.update)

    # ─── GPU Name ───

    def _update_gpu_name(self):
        vram = self.state.get_gpu_info()
        if vram > 0:
            name = self.state.get_gpu_name()
            self.gpu_name_label.config(text=_("gpu_detected").format(name, vram))
        else:
            self.gpu_name_label.config(text=_("no_gpu"))

    # ─── Public update ───

    def update(self):
        if not hasattr(self, 'vram_bar'):
            return
        self._update_memory_bars()
        self._update_tps()

    # ─── Memory bars ───

    def _update_memory_bars(self):
        s = self.state
        model_path = s.active_model_var.get()
        if not model_path or not os.path.isfile(model_path):
            self.vram_bar.config(value=0)
            self.vram_text.config(text=_("no_model"))
            self.ram_bar.config(value=0)
            self.ram_text.config(text="—")
            self.total_label.config(text=_("total") + " —")
            self.warning_label.config(text="")
            return

        __, model_gb = s.get_model_info(model_path)
        if model_gb is None:
            return

        mem = s.estimate_memory(model_path)
        gpu_gb = mem["gpu_total_mb"] / 1024
        ram_gb = mem["ram_total_mb"] / 1024
        total_gb = gpu_gb + ram_gb

        vram_mb = s.get_gpu_info()
        vram_gb = vram_mb / 1024 if vram_mb > 0 else 0

        # VRAM bar
        if vram_gb > 0:
            pct = min(gpu_gb / vram_gb * 100, 100)
            self.vram_bar.config(value=pct)
            self.vram_text.config(text=f"{gpu_gb:.2f} / {vram_gb:.1f} GB")
            if pct > 90:
                self.vram_bar.config(style="red.Horizontal.TProgressbar")
            elif pct > 75:
                self.vram_bar.config(style="warning.Horizontal.TProgressbar")
            else:
                self.vram_bar.config(style="green.Horizontal.TProgressbar")
        else:
            self.vram_bar.config(value=0)
            self.vram_text.config(text=_("no_gpu_text"))

        # RAM bar
        ram_total_gb = s.get_total_ram_gb()
        ram_pct = min(ram_gb / ram_total_gb * 100, 100)
        self.ram_bar.config(value=ram_pct)
        self.ram_text.config(text=f"{ram_gb:.2f} / {ram_total_gb:.1f} GB")

        self.total_label.config(text=f"{_('total')} {total_gb:.2f} GB")

        # Warnings
        if vram_gb > 0 and gpu_gb > vram_gb * 0.95:
            self.warning_label.config(text=_("vram_full"), foreground="#f44336")
        elif vram_gb > 0 and gpu_gb > vram_gb * 0.8:
            self.warning_label.config(text=_("vram_high"), foreground="#ff9800")
        elif mem["parallel_slots"] > 1:
            self.warning_label.config(
                text=_("kv_slots_warn").format(mem['parallel_slots']),
                foreground="#c59a66")
        elif mem["kv_offload_on"] and mem["ram_kv_mb"] <= 1:
            self.warning_label.config(
                text=_("kv_gpu_info"), foreground="#546e7a")
        else:
            self.warning_label.config(text="")

    # ─── TPS ───

    def _update_tps(self):
        s = self.state
        profile = s.estimate_tps()
        if profile is None:
            self.tps_label.config(text="— tok/s")
            self.tps_detail.config(text="")
            self.tps_hint.config(text="")
            return

        low, high, best = profile["low"], profile["high"], profile["best"]

        if high >= 45:
            color, label = "#8cb39d", _("tps_excellent")
        elif high >= 22:
            color, label = "#a9bf8a", _("tps_good")
        elif high >= 8:
            color, label = "#c59a66", _("tps_fair")
        else:
            color, label = "#b86c70", _("tps_slow")

        self.tps_label.config(text=f"⚡ {low:.0f}-{high:.0f} tok/s", foreground=color)

        ngl = s.ngl_var.get()
        vram = s.get_gpu_info()
        mode = "GPU" if ngl > 0 and vram > 0 else ("CPU" if ngl == 0 else "Mix")
        ctx = s.ctx_var.get()
        self.tps_detail.config(text=f"~ {label} | {mode} | best {best:.0f}")
        self.tps_hint.config(text=_("tps_hint").format(ctx))
