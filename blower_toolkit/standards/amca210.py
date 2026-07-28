"""AMCA 210 / ASHRAE 51 placeholder checks.

This module is intentionally separated so formal laboratory-test correction and
rating worksheets can be expanded without touching the main design engine.
"""

def rating_note() -> str:
    return "Preliminary selection only. Validate airflow, pressure and power by AMCA 210 / ASHRAE 51 or ISO 5801 testing."
