# app.py (di root directory)
import sys
from pathlib import Path

# Memastikan root directory terdaftar di sys.path agar impor modul 'src' berfungsi dengan baik
root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from app.gradio_app import main

if __name__ == "__main__":
    main()
