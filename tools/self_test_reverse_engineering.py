"""Quick engineering smoke test for the v25 reverse-engineering module.
Run: python tools/self_test_reverse_engineering.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blower_toolkit.reverse_engineering import (
    provisional_geometry, OperatingInput, raw_meanline_curve,
    normalize_curve_to_benchmark, benchmark_similarity_point,
    operating_point, area_law_volute_stations,
)
from blower_toolkit.engine import wheel_performance


def main():
    g, _ = provisional_geometry(800, 60)
    g.scroll_internal_width_mm = 1007
    g.discharge_width_mm = 1007
    g.discharge_height_mm = 1007
    op = OperatingInput(rpm=600, temp_c=35, altitude_m=0, rh_pct=50)
    w, raw, meta = raw_meanline_curve(g, op, "Area-law rectangular")
    normalized, _ = normalize_curve_to_benchmark(raw, g, op)
    q0 = meta["q_reference_m3h"]
    p = operating_point(normalized, q0)
    b = benchmark_similarity_point(g.d2_mm, op.rpm, meta["density_kg_m3"], (1.007)**2)
    assert abs(p["Total_Pa"] - b["total_pressure_pa"]) < 1.0
    assert abs(p["Shaft_kW"] - b["shaft_power_kw"]) < 0.1
    perf = wheel_performance(w, meta["density_kg_m3"], op.rpm, q0/3600)
    vol = area_law_volute_stations(w, q0/3600, perf)
    assert len(raw) >= 40 and len(vol) >= 30
    assert 50000 < q0 < 70000
    assert 600 < p["Static_Pa"] < 1000
    print("PASS — v25 reverse-engineering smoke test")
    print(f"600 rpm / 800 mm sanity anchor: {q0:,.0f} m3/h, {p['Static_Pa']:.0f} Pa static, {p['Shaft_kW']:.1f} kW shaft")


if __name__ == "__main__":
    main()
