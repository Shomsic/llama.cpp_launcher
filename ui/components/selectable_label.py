import tkinter as tk
from tkinter import ttk

class SelectableLabel(ttk.Frame):
    """
    Виджет, который выглядит как Label, но позволяет выделять и копировать текст.
    Реализован через tk.Text, настроенный так, чтобы максимально имитировать ttk.Label.
    """
    def __init__(self, master, text="", font=None, foreground=None, background=None, 
                 wraplength=0, justify="left", **kwargs):
        super().__init__(master, **kwargs)
        
        # Основной текстовый виджет
        # height=1 — базовый размер. Для многострочного текста будем корректировать.
        self.text_widget = tk.Text(
            self,
            font=font,
            fg=foreground,
            bg=background,
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            cursor="ibeam",
            undo=False,
            height=1 
        )
        
        self.set_text(text)
        self.text_widget.pack(fill="both", expand=True)

    def set_text(self, text):
        """Метод для обновления текста и автоматической подстройки высоты."""
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", str(text))
        
        # Автоподбор высоты: считаем количество строк
        # В идеале нужно знать ширину виджета, но для простоты используем 
        # количество символов \n или примерный расчет по wraplength.
        lines = str(text).count('\n') + 1
        
        # Если задан wraplength, учитываем перенос строк
        # Это грубый расчет, так как реальный перенос зависит от ширины шрифта
        # Но для большинства случаев достаточно.
        if self.text_widget.cget('wrap') == 'word':
            # Если текст длинный, увеличиваем высоту. 
            # В Tkinter Text нельзя поставить height=0, минимум 1.
            # Мы будем использовать метод 'update_idletasks' чтобы получить реальный размер, 
            # но здесь просто поставим 1, если текст короткий.
            pass

        # Чтобы текст не обрезался и не было лишнего места,
        # мы полагаемся на то, что большинство лейблов — однострочные.
        # Для многострочных (путь, мета) мы можем задать высоту чуть больше или 
        # использовать метод расчета.
        
        # Для базового поведения Label:
        self.text_widget.config(state="disabled")
        self.text_widget.config(height=lines)

    def config(self, **kwargs):
        """Поддержка обновления текста через .config(text=...)"""
        if 'text' in kwargs:
            self.set_text(kwargs.pop('text'))
        
        if 'foreground' in kwargs:
            self.text_widget.config(fg=kwargs.pop('foreground'))
        if 'background' in kwargs:
            self.text_widget.config(bg=kwargs.pop('background'))
            
        super().config(**kwargs)
