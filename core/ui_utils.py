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


def prompt_for_images():
    """Opens a file dialog allowing the selection of multiple image files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter root window

    # Note the 's' at the end of askopenfilenames
    file_paths = filedialog.askopenfilenames(
        title="Select Plant Image(s)",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All Files", "*.*"),
        ],
    )

    # askopenfilenames returns a tuple of strings, which works natively with len() and loops
    return file_paths
