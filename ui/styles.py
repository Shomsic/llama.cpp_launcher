"""Конфигурация стилей и тем оформления для ttk / ttkbootstrap."""
import tkinter as tk
from tkinter import ttk

try:
    import ttkbootstrap as ttkb
    HAS_TTKB = True
except ImportError:
    ttkb = None
    HAS_TTKB = False


def configure_styles(root: tk.Tk):
    """Настраивает все ttk-стили. Возвращает объект Style."""
    if HAS_TTKB:
        style = ttkb.Style("darkly")
    else:
        style = ttk.Style()
        style.theme_use("clam")

    root.configure(bg="#252a31")

    style.configure(".", font=("Segoe UI", 10))
    style.configure("TNotebook", background="#252a31", borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"),
                     background="#39424c", foreground="#d6dde5")
    style.map("TNotebook.Tab",
              background=[("selected", "#586574"), ("active", "#46515c")],
              foreground=[("selected", "#f1f4f7")])

    style.configure("TLabelframe", background="#2f363f", borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"),
                     foreground="#d9e1e8", background="#2f363f")
    style.configure("TFrame", background="#252a31")
    style.configure("TLabel", background="#252a31", foreground="#d8e0e8")
    style.configure("TEntry", fieldbackground="#1f242a", foreground="#e5ebf0")
    style.configure("TSpinbox", fieldbackground="#1f242a", foreground="#e5ebf0")
    style.configure("TCombobox", fieldbackground="#1f242a", foreground="#e5ebf0")

    style.configure("Card.TFrame", background="#2f363f", relief="solid", borderwidth=1)
    style.configure("Header.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
    style.configure("Success.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 10))
    style.configure("Danger.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 10))
    style.configure("Info.TLabel", font=("Segoe UI", 9), foreground="#9aabb9", background="#252a31")
    style.configure("Value.TLabel", font=("Segoe UI", 10, "bold"), background="#252a31", foreground="#e5ebf0")
    style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), background="#252a31", foreground="#dbe3ea")
    style.configure("LaunchBar.TFrame", background="#303841")
    style.configure("LaunchTitle.TLabel", font=("Segoe UI", 17, "bold"),
                     foreground="#edf2f7", background="#252a31")
    style.configure("Subtle.TLabel", font=("Segoe UI", 9), foreground="#97a8b8", background="#252a31")

    # Прогресс-бары
    for name, color in [("green", "#6f9c84"), ("warning", "#c59a66"), ("red", "#b86c70")]:
        style.configure(f"{name}.Horizontal.TProgressbar",
                        foreground=color, background=color,
                        troughcolor="#1f242a", bordercolor="#1f242a",
                        lightcolor=color, darkcolor=color)

    return style
