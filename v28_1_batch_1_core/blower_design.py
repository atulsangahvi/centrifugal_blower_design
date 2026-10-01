"""Backward-compatible Streamlit entry point.

Use this file as the Streamlit main file if your existing Streamlit Cloud app is
already configured to run blower_design.py. New deployments may use app.py.
"""
from blower_toolkit.ui.streamlit_app import main

if __name__ == "__main__":
    main()
else:
    main()
