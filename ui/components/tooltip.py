"""Менеджер тултипов — всплывающие подсказки при наведении."""
import tkinter as tk
from tkinter import SOLID


class TooltipManager:
    """Создаёт один label-тултип и показывает/прячет его по запросу."""

    def __init__(self, root: tk.Tk):
        self._root = root
        self._label = tk.Label(
            root, text="", foreground="#ccc", background="#2a2f36",
            font=("Segoe UI", 8), relief=SOLID, padx=8, pady=4,
            wraplength=320, justify="left"
        )
        self._label.place_forget()

    def show(self, event, text: str):
        x = event.widget.winfo_rootx() - self._root.winfo_rootx()
        y = event.widget.winfo_rooty() - self._root.winfo_rooty() + event.widget.winfo_height()
        self._label.config(text=text)
        self._label.place(x=x, y=y)
        self._label.lift()

    def hide(self, _event=None):
        self._label.place_forget()

    def bind(self, widget, text: str):
        """Удобный метод: привязывает тултип к виджету."""
        widget.bind("<Enter>", lambda e: self.show(e, text))
        widget.bind("<Leave>", self.hide)
