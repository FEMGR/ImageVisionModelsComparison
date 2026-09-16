"""
handle user interaction, allowing the user to
"point and pick" files natively using their OS file explorer.
"""

# core/ui_utils.py

import tkinter as tk
from tkinter import filedialog


def prompt_for_custom_model() -> str:
    """Opens a file dialog for the user to select their custom model weights."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select your custom model weights (.pth, .pt, .onnx)",
        filetypes=[("Model Files", "*.pth *.pt *.onnx *.h5"), ("All Files", "*.*")],
    )
    return file_path


def prompt_for_images() -> list:
    """Opens a dialog for the user to select one or multiple images."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select plant images to evaluate",
        filetypes=[("Images", "*.jpg *.jpeg *.png")],
    )
    return list(file_paths)
