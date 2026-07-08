"""Streamlit entry point — multi-page controller.

Wires up the Dashboard and About pages via st.navigation. The dashboard talks to the FastAPI
backend, so start that first:  uvicorn src.api.main:app --reload
Run the UI with:  streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a standalone script, so the project root isn't on sys.path.
# Add it so `import src...` resolves in this file and the page modules it imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.ui.views import about, dashboard  # noqa: E402

st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="wide")

pages = [
    st.Page(dashboard.render, title="Dashboard", icon="⚽", default=True, url_path="dashboard"),
    st.Page(about.render, title="About", icon="ℹ️", url_path="about"),
]
st.navigation(pages).run()
