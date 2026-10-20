"""Compatibility entry point for the app, now using local NumPy embeddings.

Run with: streamlit run streamlit_chroma.py
The existing video_data files are shared with streamlit.py.
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("streamlit.py")), run_name="__main__")
