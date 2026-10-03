"""
handle user interaction, allowing the user to
"point and pick" files natively using their OS file explorer.
"""

# core/ui_utils.py
import os
import tkinter as tk
from tkinter import filedialog
from typing import List

PICTURE_DIR = "/home/graubo/Pictures"


def prompt_for_custom_model() -> str:
    """Open a dialog for the user to select a local Hugging Face model folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    folder_path = filedialog.askdirectory(
        title="Select your custom Hugging Face model folder",
    )
    return folder_path


def prompt_for_images(initial_dir: str = PICTURE_DIR) -> List[str]:
    """Opens a file dialog to select multiple images from a specific directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the main root window
    root.attributes("-topmost", True)  # Force dialog to pop up in FRONT of PyCharm

    # Fall back to current working directory if initial_dir is invalid
    if not initial_dir or not os.path.exists(initial_dir):
        initial_dir = os.getcwd()

    selected_files = filedialog.askopenfilenames(
        title="Select Plant Image(s)",
        initialdir=initial_dir,
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All Files", "*.*"),
        ],
    )

    # Clean up Tkinter instance completely so it won't block main loop
    root.destroy()

    # Always return a explicit Python list
    return list(selected_files) if selected_files else []
