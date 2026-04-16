import tkinter as tk
from tkinter import ttk

class CollapsibleFrame(ttk.Frame):
    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.is_open = True
        
        # Header
        self.header = ttk.Frame(self)
        self.header.pack(fill="x", expand=True)
        
        self.toggle_btn = ttk.Button(self.header, text="▼", width=3, command=self.toggle)
        self.toggle_btn.pack(side="left")
        
        self.title_label = ttk.Label(self.header, text=title, font=("Segoe UI", 10, "bold"))
        self.title_label.pack(side="left", padx=5)
        
        # Content area
        self.content = ttk.Frame(self)
        self.content.pack(fill="both", expand=True)

    def toggle(self):
        if self.is_open:
            self.content.pack_forget()
            self.toggle_btn.config(text="►")
        else:
            self.content.pack(fill="both", expand=True)
            self.toggle_btn.config(text="▼")
        self.is_open = not self.is_open
