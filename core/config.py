
#core/config.py

import os

# Centralized directory for all downloaded model weights
# This defaults to a folder named "weights" in your main project directory.
WEIGHTS_DIR = "./weights"

# Ensure the base directory exists
os.makedirs(WEIGHTS_DIR, exist_ok=True)