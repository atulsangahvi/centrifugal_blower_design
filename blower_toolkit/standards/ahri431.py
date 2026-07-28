"""AHRI 431 SI / AHRI 430 I-P AHU supply fan rating placeholders."""

def ahri431_checklist() -> list[str]:
    return [
        "Identify fan arrangement: housed, plenum, fan array, ducted/unducted discharge.",
        "Declare airflow, fan static pressure, speed, shaft power and efficiency in SI units.",
        "Use AMCA 210 / ASHRAE 51 laboratory data as rating basis.",
        "Mark output as preliminary unless tested/certified under the required program.",
    ]
