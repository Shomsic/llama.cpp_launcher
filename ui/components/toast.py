"""Toast-уведомления — кратковременные всплывающие сообщения."""
import tkinter as tk


class ToastManager:
    def __init__(self, root: tk.Tk):
        self._root = root
        self._label: tk.Label | None = None

    def show(self, message: str, duration: int = 3000):
        self._hide()
        self._label = tk.Label(
            self._root, text=message, font=("Segoe UI", 10, "bold"),
            bg="#323232", fg="#ffffff", padx=15, pady=10, relief="raised", bd=3
        )
        self._label.place(relx=0.5, rely=0.95, anchor="s")
        self._root.after(duration, self._hide)

    def _hide(self):
        if self._label:
            self._label.place_forget()
            self._label = None
