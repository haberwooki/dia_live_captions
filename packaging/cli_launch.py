"""PyInstaller entry point for the console EXE (livecaptions.exe).

Thin wrapper so the frozen app goes through the same main() as `python -m
livecaptions` — which means configure_runtime() (HF cache + logging) runs first.
"""
from livecaptions.__main__ import main

if __name__ == "__main__":
    main()
