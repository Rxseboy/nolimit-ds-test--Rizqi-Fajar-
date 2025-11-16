"""
utils.py
---------
General utility helpers used across the project:
- Logging setup
- Seed control
- Device selection
- Safe directory creation
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Ensure reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    """Create a folder if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def print_header(title: str):
    """Pretty console header output."""
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60 + "\n")
