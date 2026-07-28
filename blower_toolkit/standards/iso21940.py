"""ISO 21940 balancing guidance placeholders."""

def balance_guidance(high_speed: bool = False) -> str:
    return "Balance to ISO 21940 G2.5" if high_speed else "Balance to ISO 21940 G6.3 minimum; use G2.5 for high speed/low vibration products."
