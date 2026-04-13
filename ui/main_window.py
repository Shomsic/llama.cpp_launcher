"""
Главное окно приложения.
"""
import tkinter as tk

from core.i18n import I18n


class MainWindow:
    """Класс главного окна."""
    
    def __init__(self, root: tk.Tk, state):
        self.root = root
        self.state = state
        self._app = None
    
    def run(self):
        """Запускает UI приложения."""
        from ui.app import LlamaLauncherApp
        self._app = LlamaLauncherApp(self.root, self.state)
        self._app.update_texts()
        
        I18n().add_listener(self._on_language_change)
        
        return self._app
    
    def _on_language_change(self):
        if self._app:
            self._app.update_texts()


__all__ = ["MainWindow"]