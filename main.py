"""
Transformer Chatbot — Main Entry Point

Usage:
    python main.py              # runs training with config.yaml
"""

import sys
from pathlib import Path

# Make `scripts/` importable from project root
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from scripts.train_model import main

if __name__ == "__main__":
    main()
