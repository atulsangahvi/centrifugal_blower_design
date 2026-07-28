"""Streamlit UI wrapper.

The heavy engineering engine is kept in blower_toolkit.engine so future UI,
API, batch-design, and test modules can reuse the same calculations.
"""
from blower_toolkit.engine import run_app

def main() -> None:
    run_app()
