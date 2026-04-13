"""Точка входа в программу."""
import sys
import tkinter as tk
from tkinter import messagebox
from core.i18n import I18n
from app_state import AppState


def main():
    root = tk.Tk()
    
    # Инициализация состояния (без tk-переменных - их создаст MainWindow)
    state = AppState()
    
    # Инициализация языка
    lang = state.settings.get("language", "ru")
    I18n().set_language(lang)
    
    try:
        from ui.main_window import MainWindow
        window = MainWindow(root, state)
        window.run()
    except Exception as e:
        from core.i18n import _
        messagebox.showerror(_("error"), _("boot_error").format(e))
        sys.exit(1)

    root.mainloop()


if __name__ == "__main__":
    main()
