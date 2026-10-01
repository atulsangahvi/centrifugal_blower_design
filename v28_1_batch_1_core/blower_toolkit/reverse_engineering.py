from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from blower_toolkit.engine import (
    G,
    FAMILIES,
    MATERIALS,
    Duty,
    Wheel,
    PerfPoint,
    air_density,
    air_viscosity,
    blade_arc_geometry,
    selected_motor,
    wheel_performance,
)

# -----------------------------------------------------------------------------
# Reverse-engineering reference data
# -----------------------------------------------------------------------------

REFERENCE_PRESETS: Dict[str, dict] = {
    "KQ800 2026 V4.0 — uploaded drawing": {
        "reference_name": "KQ800 135-degree centrifugal fan — V4.0",
        "asset": "assets/KQ800_2026_V4_reference.jpeg",
        "drawing_code": "3DW.903.Q080.7",
        "erp_code": "01111096",
        "drawing_date": "2026-04-18",
        "fan_standard": "JB/T 9068-2017",
        "fan_standard_note": "Forward-curved multi-blade centrifugal fan standard stated on drawing.",
        "balance_grade": "G4.0",
        "base_side_length_mm": 1330.0,
        "frame_inner_width_mm": 970.0,
        "overall_height_mm": 1672.0,
        "shaft_center_height_mm": 658.0,
        "lower_frame_height_mm": 190.0,
        "overall_axial_width_mm": 1390.0,
        "body_axial_width_mm": 1111.0,
        "inner_body_axial_width_mm": 1057.0,
        "discharge_clear_width_mm": 1007.0,
        "discharge_clear_height_mm": 1007.0,
        "discharge_flange_outer_width_mm": 1087.0,
        "discharge_flange_outer_height_mm": 1087.0,
        "discharge_flange_holes": 24,
        "discharge_flange_hole_diameter_mm": 6.4,
        "base_slots": 4,
        "base_slot_mm": "17 x 22",
        "shaft_diameter_mm": 60.0,
        "declared_fan_weight_kg": 270.0,
        "discharge_angle_deg": 135.0,
        "notes": (
            "No motor included. Drawing also calls for impeller dynamic balance G4.0. "
            "Model designation suggests an 800-size family but the impeller OD is not "
            "explicitly dimensioned in the supplied outline drawing."
        ),
    },
    "WDL-800 2020 V1.0 — uploaded drawing": {
        "reference_name": "WDL-800 135-degree centrifugal fan — V1.0",
        "asset": "assets/WDL800_2020_V1_reference.jpeg",
        "drawing_code": "0110736",
        "erp_code": "",
        "drawing_date": "2020-06-24",
        "fan_standard": "JB/T 9068-2017 / drawing family reference",
        "fan_standard_note": "Outline drawing for a forward-curved multi-blade blower family.",
        "balance_grade": "Not legible / verify",
        "base_side_length_mm": 1330.0,
        "frame_inner_width_mm": 970.0,
        "overall_height_mm": 1672.5,
        "shaft_center_height_mm": 0.0,
        "lower_frame_height_mm": 0.0,
        "overall_axial_width_mm": 1390.0,
        "body_axial_width_mm": 1107.0,
        "inner_body_axial_width_mm": 1057.0,
        "discharge_clear_width_mm": 1007.0,
        "discharge_clear_height_mm": 1007.0,
        "discharge_flange_outer_width_mm": 1087.0,
        "discharge_flange_outer_height_mm": 1087.0,
        "discharge_flange_holes": 0,
        "discharge_flange_hole_diameter_mm": 0.0,
        "base_slots": 4,
        "base_slot_mm": "17 x 22",
        "shaft_diameter_mm": 55.0,
        "declared_fan_weight_kg": 276.0,
        "discharge_angle_deg": 135.0,
        "notes": (
            "Older uploaded outline. Shaft end is marked approximately 55 mm. "
            "Use the newer KQ800 drawing for the G4.0 balance requirement unless the "
            "actual production revision specifies otherwise."
        ),
    },
}

# Independent commercial benchmark used ONLY as a sanity scale, not as WDL data.
# Kruger FDA 500 official catalogue selection example:
# D=500 mm (catalogue states fan size = impeller diameter), Q=20,000 m3/h,
# Pt=737 Pa, N=828 rpm, shaft power=6.5 kW, total eta=62%, rho=1.22 kg/m3.
FDA500_BENCHMARK = {
    "manufacturer": "Kruger",
    "series": "FDA double-inlet forward-curved",
    "d2_m": 0.500,
    "q_m3h": 20000.0,
    "pt_pa": 737.0,
    "rpm": 828.0,
    "shaft_kw": 6.5,
    "eta_total": 0.62,
    "rho_kg_m3": 1.22,
    "outlet_velocity_ms": 13.6,
    "dynamic_pressure_pa": 114.0,
    "test_arrangement": "AMCA 210, Installation Type B (free inlet, ducted outlet)",
}


@dataclass
class ReferenceEnvelope:
    reference_name: str
    drawing_code: str = ""
    drawing_date: str = ""
    fan_standard: str = ""
    balance_grade: str = "G4.0"
    base_side_length_mm: float = 0.0
    frame_inner_width_mm: float = 0.0
    overall_height_mm: float = 0.0
    shaft_center_height_mm: float = 0.0
    lower_frame_height_mm: float = 0.0
    overall_axial_width_mm: float = 0.0
    body_axial_width_mm: float = 0.0
    inner_body_axial_width_mm: float = 0.0
    discharge_clear_width_mm: float = 0.0
    discharge_clear_height_mm: float = 0.0
    discharge_flange_outer_width_mm: float = 0.0
    discharge_flange_outer_height_mm: float = 0.0
    discharge_flange_holes: int = 0
    discharge_flange_hole_diameter_mm: float = 0.0
    base_slots: int = 0
    base_slot_mm: str = ""
    shaft_diameter_mm: float = 0.0
    declared_fan_weight_kg: float = 0.0
    discharge_angle_deg: float = 135.0
    notes: str = ""


@dataclass
class GeometryInput:
    family: str = "Multi-Blade Sirocco (Cage)"
    arrangement: str = "DWDI (double inlet)"
    d2_mm: float = 800.0
    d1_mm: float = 696.0
    b2_total_mm: float = 400.0
    b1_total_mm: float = 420.0
    blade_count: int = 48
    beta1_deg: float = 75.0
    beta2_deg: float = 152.0
    blade_thickness_mm: float = 2.0
    plate_thickness_mm: float = 3.0
    hub_diameter_mm: float = 120.0
    shaft_diameter_mm: float = 60.0
    cutoff_clearance_mm: float = 40.0
    scroll_internal_width_mm: float = 1007.0
    discharge_width_mm: float = 1007.0
    discharge_height_mm: float = 1007.0
    material: str = "Galvanized Steel"


@dataclass
class OperatingInput:
    rpm: float = 600.0
    temp_c: float = 35.0
    altitude_m: float = 0.0
    rh_pct: float = 50.0
    density_kg_m3: float = 0.0
    min_flow_fraction: float = 0.25
    max_flow_fraction: float = 1.40
    curve_points: int = 49


@dataclass
class TestCalibration:
    enabled: bool = False
    test_rpm: float = 0.0
    test_airflow_m3h: float = 0.0
    test_pressure_pa: float = 0.0
    pressure_kind: str = "Static"
    test_shaft_power_kw: float = 0.0
    test_motor_input_kw: float = 0.0
    motor_efficiency_pct: float = 90.0
    drive_efficiency_pct: float = 95.0


@dataclass
class DriveMechanicalInput:
    drive_type: str = "Belt drive"
    bearing_span_mm: float = 0.0
    pulley_pitch_diameter_mm: float = 0.0
    pulley_overhang_mm: float = 0.0
    belt_pull_factor: float = 2.5
    impeller_mass_kg: float = 0.0
    shaft_diameter_mm: float = 60.0
    shaft_yield_mpa: float = 350.0
    allowable_vm_mpa: float = 90.0
    motor_efficiency_pct: float = 92.0
    drive_efficiency_pct: float = 95.0
    motor_margin_pct: float = 15.0
    balance_grade: str = "G4.0"


# -----------------------------------------------------------------------------
# Basic utilities and provenance
# -----------------------------------------------------------------------------

def preset_to_envelope(name: str) -> ReferenceEnvelope:
    p = REFERENCE_PRESETS[name]
    keys = ReferenceEnvelope.__dataclass_fields__.keys()
    return ReferenceEnvelope(**{k: p.get(k) for k in keys})


def envelope_frame(env: ReferenceEnvelope) -> pd.DataFrame:
    rows = []
    units = {
        "discharge_flange_holes": "count",
        "base_slots": "count",
        "base_slot_mm": "mm",
        "reference_name": "",
        "drawing_code": "",
        "drawing_date": "",
        "fan_standard": "",
        "balance_grade": "",
        "notes": "",
    }
    for k, v in asdict(env).items():
        unit = units.get(k, "mm" if k.endswith("_mm") else ("kg" if k.endswith("_kg") else ("deg" if k.endswith("_deg") else "")))
        rows.append({"Parameter": k, "Value": v, "Unit": unit, "Source": "Supplier outline drawing / user entry"})
    return pd.DataFrame(rows)


def provisional_geometry(d2_mm: float = 800.0, shaft_mm: float = 60.0) -> Tuple[GeometryInput, pd.DataFrame]:
    fam = FAMILIES["Multi-Blade Sirocco (Cage)"]
    d1 = fam["d1d2"] * d2_mm
    b2 = fam["b2d2"] * d2_mm
    g = GeometryInput(
        d2_mm=d2_mm,
        d1_mm=d1,
        b2_total_mm=b2,
        b1_total_mm=1.05 * b2,
        blade_count=int(fam["z"]),
        beta1_deg=float(fam["beta1"]),
        beta2_deg=float(fam["beta2"]),
        blade_thickness_mm=2.0,
        plate_thickness_mm=3.0,
        hub_diameter_mm=max(2.0 * shaft_mm, 0.15 * d1),
        shaft_diameter_mm=shaft_mm,
        cutoff_clearance_mm=0.05 * d2_mm,
        scroll_internal_width_mm=1007.0,
        discharge_width_mm=1007.0,
        discharge_height_mm=1007.0,
        material="Galvanized Steel",
    )
    rows = [
        ("D2", g.d2_mm, "Nominal model-size hypothesis — VERIFY physically", "LOW"),
        ("D1", g.d1_mm, "Assumed family ratio D1/D2", "LOW"),
        ("b2 total", g.b2_total_mm, "Assumed family ratio b2/D2", "LOW"),
        ("b1 total", g.b1_total_mm, "Assumed from b2", "LOW"),
        ("Blade count", g.blade_count, "Assumed family default", "LOW"),
        ("beta1", g.beta1_deg, "Assumed family default; angle from tangent", "LOW"),
        ("beta2", g.beta2_deg, "Assumed family default; angle from tangent", "LOW"),
        ("Blade thickness", g.blade_thickness_mm, "Assumed fabrication value", "LOW"),
        ("Plate thickness", g.plate_thickness_mm, "Assumed fabrication value", "LOW"),
        ("Shaft diameter", g.shaft_diameter_mm, "Drawing value for KQ800 V4", "HIGH"),
        ("Scroll internal width", g.scroll_internal_width_mm, "Inferred from 1007 mm clear discharge width — VERIFY", "MEDIUM"),
        ("Cutoff clearance", g.cutoff_clearance_mm, "Assumed 5% D2", "LOW"),
    ]
    return g, pd.DataFrame(rows, columns=["Parameter", "Value", "Source", "Confidence"])


def readiness_score(provenance: pd.DataFrame, calibrated: bool = False) -> dict:
    critical = ["D2", "D1", "b2 total", "Blade count", "beta1", "beta2", "RPM"]
    if provenance is None or provenance.empty:
        return {"score": 0, "grade": "D", "message": "No geometry provenance recorded."}
    score = 0.0
    max_score = 0.0
    for param in critical:
        max_score += 10.0
        row = provenance[provenance["Parameter"].astype(str).str.lower() == param.lower()]
        if row.empty:
            continue
        src = str(row.iloc[0].get("Source", "")).lower()
        conf = str(row.iloc[0].get("Confidence", "")).upper()
        if "physical" in src or "manufacturer" in src or conf == "HIGH":
            score += 10
        elif "drawing" in src or conf == "MEDIUM":
            score += 7
        elif "assum" in src or conf == "LOW":
            score += 2
        else:
            score += 4
    if calibrated:
        score += 25
        max_score += 25
    pct = round(100 * score / max(max_score, 1), 0)
    grade = "A" if pct >= 85 else "B" if pct >= 70 else "C" if pct >= 50 else "D"
    msg = {
        "A": "Good reverse-engineering basis; still requires prototype test validation.",
        "B": "Useful engineering model; several dimensions or test anchors still need confirmation.",
        "C": "Preliminary only; do not manufacture from this geometry without measurements/calibration.",
        "D": "Concept/sanity-check level only; too many critical aerodynamic inputs are assumed.",
    }[grade]
    return {"score": pct, "grade": grade, "message": msg}


# -----------------------------------------------------------------------------
# Drawing measurement helpers
# -----------------------------------------------------------------------------

def line_length_px(p1: Sequence[float], p2: Sequence[float]) -> float:
    return float(math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])))


def mm_per_pixel(p1: Sequence[float], p2: Sequence[float], known_length_mm: float) -> float:
    px = line_length_px(p1, p2)
    if px <= 0 or known_length_mm <= 0:
        raise ValueError("Calibration line and known length must be positive.")
    return known_length_mm / px


def measure_line_mm(p1: Sequence[float], p2: Sequence[float], scale_mm_per_px: float) -> float:
    return line_length_px(p1, p2) * scale_mm_per_px


def fit_circle_three_points(points: Sequence[Sequence[float]]) -> Tuple[Tuple[float, float], float]:
    if len(points) != 3:
        raise ValueError("Exactly three points are required to fit a circle.")
    (x1, y1), (x2, y2), (x3, y3) = [(float(p[0]), float(p[1])) for p in points]
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2.0
    cd = (temp - x3 * x3 - y3 * y3) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-9:
        raise ValueError("The three points are nearly collinear; choose three points around the circle.")
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    radius = math.hypot(cx - x1, cy - y1)
    return (cx, cy), radius


# -----------------------------------------------------------------------------
# Commercial benchmark / similarity sanity check
# -----------------------------------------------------------------------------

def benchmark_coefficients() -> dict:
    b = FDA500_BENCHMARK
    d = b["d2_m"]
    n = b["rpm"]
    q = b["q_m3h"] / 3600.0
    rho = b["rho_kg_m3"]
    u = math.pi * d * n / 60.0
    a = math.pi * d * d / 4.0
    phi = q / (a * u)
    psi_t = b["pt_pa"] / (0.5 * rho * u * u)
    lam = b["shaft_kw"] * 1000.0 / (rho * a * u ** 3)
    return {"phi": phi, "psi_total": psi_t, "lambda_power": lam}


def benchmark_similarity_point(
    d2_mm: float,
    rpm: float,
    density_kg_m3: float,
    outlet_area_m2: float,
) -> dict:
    """Geometric/speed scaling of one official FDA500 selection example.

    This is deliberately labelled a benchmark, not a prediction for the WDL/KQ fan.
    It is used to catch grossly implausible reverse-engineering results.
    """
    b = FDA500_BENCHMARK
    rd = (d2_mm / 1000.0) / b["d2_m"]
    rn = rpm / b["rpm"]
    rrho = density_kg_m3 / b["rho_kg_m3"]
    q_m3h = b["q_m3h"] * rn * rd ** 3
    pt = b["pt_pa"] * rrho * rn ** 2 * rd ** 2
    shaft_kw = b["shaft_kw"] * rrho * rn ** 3 * rd ** 5
    q_m3s = q_m3h / 3600.0
    v_out = q_m3s / max(outlet_area_m2, 1e-6)
    vp = 0.5 * density_kg_m3 * v_out ** 2
    sp = pt - vp
    return {
        "airflow_m3h": q_m3h,
        "total_pressure_pa": pt,
        "velocity_pressure_pa": vp,
        "static_pressure_pa": sp,
        "shaft_power_kw": shaft_kw,
        "outlet_velocity_ms": v_out,
        "benchmark_eta_total_pct": b["eta_total"] * 100.0,
    }


def benchmark_speed_sweep(
    d2_mm: float,
    density_kg_m3: float,
    outlet_area_m2: float,
    rpm_min: float = 300.0,
    rpm_max: float = 900.0,
    rpm_step: float = 50.0,
) -> pd.DataFrame:
    rows = []
    n = rpm_min
    while n <= rpm_max + 1e-9:
        p = benchmark_similarity_point(d2_mm, n, density_kg_m3, outlet_area_m2)
        p["rpm"] = n
        rows.append(p)
        n += rpm_step
    return pd.DataFrame(rows)[[
        "rpm", "airflow_m3h", "static_pressure_pa", "total_pressure_pa",
        "shaft_power_kw", "outlet_velocity_ms", "velocity_pressure_pa",
        "benchmark_eta_total_pct"
    ]]


# -----------------------------------------------------------------------------
# Geometry / casing / raw mean-line curve
# -----------------------------------------------------------------------------

def validate_geometry(g: GeometryInput) -> List[str]:
    errors = []
    if g.d2_mm <= 0:
        errors.append("D2 must be positive.")
    if g.d1_mm <= 0 or g.d1_mm >= g.d2_mm:
        errors.append("D1 must be positive and smaller than D2.")
    if g.b2_total_mm <= 0 or g.b1_total_mm <= 0:
        errors.append("b1 and b2 must be positive.")
    if g.blade_count < 3:
        errors.append("Blade count must be at least 3.")
    if not 5 <= g.beta1_deg <= 175 or not 5 <= g.beta2_deg <= 175:
        errors.append("Blade angles must be between 5 and 175 degrees from tangent.")
    fam = FAMILIES.get(g.family, {})
    if not fam.get("plenum"):
        if g.discharge_width_mm <= 0 or g.discharge_height_mm <= 0:
            errors.append("Discharge clear width and height are required for static-pressure prediction.")
        if g.scroll_internal_width_mm <= 0:
            errors.append("Scroll internal axial width is required.")
    return errors


def build_wheel(
    g: GeometryInput,
    q_reference_m3h: float,
    density: float,
    rpm: float,
    casing_method: str = "Area-law rectangular",
    scroll_velocity_fraction_cu2: float = 0.75,
) -> Tuple[Wheel, dict]:
    errors = validate_geometry(g)
    if errors:
        raise ValueError("; ".join(errors))
    d2 = g.d2_mm / 1000.0
    d1 = g.d1_mm / 1000.0
    b2 = g.b2_total_mm / 1000.0
    b1 = g.b1_total_mm / 1000.0
    outlet_w = g.discharge_width_mm / 1000.0
    outlet_h = g.discharge_height_mm / 1000.0
    B = g.scroll_internal_width_mm / 1000.0
    cutoff = g.cutoff_clearance_mm / 1000.0
    r3 = d2 / 2.0 + cutoff

    base = Wheel(
        family=g.family,
        arrangement=g.arrangement,
        d2=d2,
        d1=d1,
        b2=b2,
        b1=b1,
        z=int(g.blade_count),
        beta1=float(g.beta1_deg),
        beta2=float(g.beta2_deg),
        t_blade=g.blade_thickness_mm / 1000.0,
        t_plate=g.plate_thickness_mm / 1000.0,
        r3=r3,
        volute_width=B,
        throat_area=0.0,
        outlet_w=outlet_w,
        outlet_h=outlet_h,
        cutoff_clr=cutoff,
        hub_d=g.hub_diameter_mm / 1000.0,
        shaft_d=g.shaft_diameter_mm / 1000.0,
        material=g.material,
    )

    # First pass without a throat area to estimate the exit tangential component.
    perf0 = wheel_performance(base, density, rpm, q_reference_m3h / 3600.0)
    throat_velocity = max(8.0, scroll_velocity_fraction_cu2 * max(perf0.cu2, 1.0))
    throat_area = max((q_reference_m3h / 3600.0) / throat_velocity, 1e-4)

    if casing_method == "Reference discharge only":
        # Keep a neutral throat estimate based on discharge-to-throat contraction.
        throat_area = min(throat_area, 0.85 * outlet_w * outlet_h)
    elif casing_method == "Log spiral / free-vortex":
        # Preserve the same throat area for the performance engine; station generation
        # will use the free-vortex log-spiral relation.
        pass
    elif casing_method == "Area-law rectangular":
        pass
    else:
        raise ValueError(f"Unknown casing method: {casing_method}")

    w = replace(base, throat_area=throat_area)
    meta = {
        "throat_velocity_ms": throat_velocity,
        "throat_area_m2": throat_area,
        "r3_mm": r3 * 1000.0,
        "outlet_area_m2": outlet_w * outlet_h,
        "casing_method": casing_method,
    }
    return w, meta


def _curve_reference_flow(g: GeometryInput, op: OperatingInput) -> float:
    rho = op.density_kg_m3 if op.density_kg_m3 > 0 else air_density(op.temp_c, op.altitude_m, op.rh_pct)
    area = (g.discharge_width_mm / 1000.0) * (g.discharge_height_mm / 1000.0)
    return benchmark_similarity_point(g.d2_mm, op.rpm, rho, area)["airflow_m3h"]


def raw_meanline_curve(
    g: GeometryInput,
    op: OperatingInput,
    casing_method: str = "Area-law rectangular",
) -> Tuple[Wheel, pd.DataFrame, dict]:
    rho = op.density_kg_m3 if op.density_kg_m3 > 0 else air_density(op.temp_c, op.altitude_m, op.rh_pct)
    q_ref = _curve_reference_flow(g, op)
    w, meta = build_wheel(g, q_ref, rho, op.rpm, casing_method)
    visc = air_viscosity(op.temp_c)
    rows = []
    for frac in np.linspace(op.min_flow_fraction, op.max_flow_fraction, int(op.curve_points)):
        q_m3h = q_ref * float(frac)
        p = wheel_performance(w, rho, op.rpm, q_m3h / 3600.0, visc=visc)
        rows.append({
            "FlowFrac": frac,
            "Flow_m3h": q_m3h,
            "Flow_m3s": q_m3h / 3600.0,
            "Total_Pa": p.dp_total,
            "Static_Pa": p.dp_static,
            "Shaft_kW": p.p_shaft / 1000.0,
            "EtaTotal": p.eta_total,
            "EtaStatic": p.eta_static,
            "OutletVelocity_ms": p.v_out,
            "TipSpeed_ms": p.u2,
            "Cu2_ms": p.cu2,
            "Cm2_ms": p.cm2,
            "Slip": p.slip,
        })
    meta.update({"density_kg_m3": rho, "q_reference_m3h": q_ref})
    return w, pd.DataFrame(rows), meta


def interpolate_curve(curve: pd.DataFrame, q_m3h: float, col: str) -> Optional[float]:
    x = curve.dropna(subset=["Flow_m3h", col]).sort_values("Flow_m3h")
    if len(x) < 2:
        return None
    if q_m3h < x.Flow_m3h.min() or q_m3h > x.Flow_m3h.max():
        return None
    return float(np.interp(q_m3h, x.Flow_m3h, x[col]))


def normalize_curve_to_benchmark(
    raw: pd.DataFrame,
    g: GeometryInput,
    op: OperatingInput,
) -> Tuple[pd.DataFrame, dict]:
    rho = op.density_kg_m3 if op.density_kg_m3 > 0 else air_density(op.temp_c, op.altitude_m, op.rh_pct)
    area = (g.discharge_width_mm / 1000.0) * (g.discharge_height_mm / 1000.0)
    bench = benchmark_similarity_point(g.d2_mm, op.rpm, rho, area)
    q0 = bench["airflow_m3h"]
    pt_raw = interpolate_curve(raw, q0, "Total_Pa")
    pwr_raw = interpolate_curve(raw, q0, "Shaft_kW")
    if pt_raw is None or pwr_raw is None or pt_raw <= 1 or pwr_raw <= 0:
        raise ValueError("Raw curve does not span the benchmark reference flow; extend the curve range.")
    f_p = bench["total_pressure_pa"] / pt_raw
    f_w = bench["shaft_power_kw"] / pwr_raw
    out = raw.copy()
    out["Total_Pa"] = out["Total_Pa"] * f_p
    out["Shaft_kW"] = out["Shaft_kW"] * f_w
    q = out["Flow_m3h"] / 3600.0
    v = q / max(area, 1e-9)
    vp = 0.5 * rho * v ** 2
    out["Static_Pa"] = out["Total_Pa"] - vp
    out["OutletVelocity_ms"] = v
    out["EtaTotal"] = np.clip(q * out["Total_Pa"] / np.maximum(out["Shaft_kW"] * 1000.0, 1.0), 0, 0.90)
    out["EtaStatic"] = np.clip(q * out["Static_Pa"] / np.maximum(out["Shaft_kW"] * 1000.0, 1.0), 0, 0.88)
    return out, {
        "pressure_factor": f_p,
        "power_factor": f_w,
        "anchor_flow_m3h": q0,
        "anchor_total_pressure_pa": bench["total_pressure_pa"],
        "anchor_shaft_kw": bench["shaft_power_kw"],
        "note": "Comparison-normalized to an independent Kruger FDA500 similarity benchmark; NOT a WDL/KQ manufacturer rating.",
    }


def calibrate_curve_to_test(
    raw: pd.DataFrame,
    g: GeometryInput,
    op: OperatingInput,
    test: TestCalibration,
) -> Tuple[pd.DataFrame, dict]:
    if not test.enabled:
        return raw.copy(), {"enabled": False}
    if test.test_airflow_m3h <= 0 or test.test_pressure_pa <= 0 or test.test_rpm <= 0:
        raise ValueError("Test RPM, airflow and pressure are required for calibration.")

    # Scale the raw model from the modelling speed to the test speed before calibrating.
    scaled = scale_curve_speed(raw, op.rpm, test.test_rpm, density_ratio=1.0)
    raw_pt = interpolate_curve(scaled, test.test_airflow_m3h, "Total_Pa")
    raw_kw = interpolate_curve(scaled, test.test_airflow_m3h, "Shaft_kW")
    if raw_pt is None or raw_kw is None:
        raise ValueError("Test airflow lies outside the current raw curve range.")

    rho = op.density_kg_m3 if op.density_kg_m3 > 0 else air_density(op.temp_c, op.altitude_m, op.rh_pct)
    area = (g.discharge_width_mm / 1000.0) * (g.discharge_height_mm / 1000.0)
    q = test.test_airflow_m3h / 3600.0
    vp = 0.5 * rho * (q / max(area, 1e-9)) ** 2
    test_pt = test.test_pressure_pa + vp if test.pressure_kind.lower().startswith("static") else test.test_pressure_pa

    if test.test_shaft_power_kw > 0:
        test_shaft = test.test_shaft_power_kw
    elif test.test_motor_input_kw > 0:
        test_shaft = test.test_motor_input_kw * (test.motor_efficiency_pct / 100.0) * (test.drive_efficiency_pct / 100.0)
    else:
        test_shaft = 0.0

    f_p = test_pt / max(raw_pt, 1.0)
    f_w = test_shaft / max(raw_kw, 1e-6) if test_shaft > 0 else 1.0

    # Apply calibration at the model speed. Pressure and power factors are assumed
    # constant for this preliminary product-line calibration.
    out = raw.copy()
    out["Total_Pa"] *= f_p
    out["Shaft_kW"] *= f_w
    qv = out["Flow_m3h"] / 3600.0
    vv = qv / max(area, 1e-9)
    vpv = 0.5 * rho * vv ** 2
    out["Static_Pa"] = out["Total_Pa"] - vpv
    out["OutletVelocity_ms"] = vv
    out["EtaTotal"] = np.clip(qv * out["Total_Pa"] / np.maximum(out["Shaft_kW"] * 1000.0, 1.0), 0, 0.92)
    out["EtaStatic"] = np.clip(qv * out["Static_Pa"] / np.maximum(out["Shaft_kW"] * 1000.0, 1.0), 0, 0.90)
    return out, {
        "enabled": True,
        "pressure_factor": f_p,
        "power_factor": f_w,
        "test_total_pressure_pa": test_pt,
        "test_shaft_power_kw": test_shaft,
        "note": "Calibrated to user-entered test data; verify test setup, air density and instrumentation.",
    }


def scale_curve_speed(curve: pd.DataFrame, old_rpm: float, new_rpm: float, density_ratio: float = 1.0) -> pd.DataFrame:
    if old_rpm <= 0 or new_rpm <= 0:
        raise ValueError("RPM values must be positive.")
    r = new_rpm / old_rpm
    out = curve.copy()
    out["Flow_m3h"] *= r
    out["Flow_m3s"] *= r
    out["Total_Pa"] *= density_ratio * r ** 2
    out["Static_Pa"] *= density_ratio * r ** 2
    out["Shaft_kW"] *= density_ratio * r ** 3
    if "OutletVelocity_ms" in out:
        out["OutletVelocity_ms"] *= r
    if "TipSpeed_ms" in out:
        out["TipSpeed_ms"] *= r
    if "Cu2_ms" in out:
        out["Cu2_ms"] *= r
    if "Cm2_ms" in out:
        out["Cm2_ms"] *= r
    return out


def solve_speed_for_duty(
    base_curve: pd.DataFrame,
    base_rpm: float,
    target_airflow_m3h: float,
    target_static_pa: float,
    rpm_min: float,
    rpm_max: float,
    rpm_step: float = 2.0,
) -> dict:
    best = None
    rpm = rpm_min
    while rpm <= rpm_max + 1e-9:
        c = scale_curve_speed(base_curve, base_rpm, rpm)
        ps = interpolate_curve(c, target_airflow_m3h, "Static_Pa")
        kw = interpolate_curve(c, target_airflow_m3h, "Shaft_kW")
        pt = interpolate_curve(c, target_airflow_m3h, "Total_Pa")
        eta = interpolate_curve(c, target_airflow_m3h, "EtaStatic")
        if ps is not None and kw is not None:
            err = abs(ps - target_static_pa)
            row = {
                "rpm": rpm,
                "airflow_m3h": target_airflow_m3h,
                "static_pressure_pa": ps,
                "total_pressure_pa": pt,
                "shaft_power_kw": kw,
                "eta_static": eta,
                "pressure_error_pa": ps - target_static_pa,
                "abs_pressure_error_pa": err,
            }
            if best is None or err < best["abs_pressure_error_pa"]:
                best = row
        rpm += rpm_step
    if best is None:
        raise ValueError("Target airflow does not fall within the scaled curve over the selected RPM range.")
    return best


def operating_point(curve: pd.DataFrame, q_m3h: float) -> dict:
    fields = ["Static_Pa", "Total_Pa", "Shaft_kW", "EtaTotal", "EtaStatic", "OutletVelocity_ms", "TipSpeed_ms"]
    out = {"Flow_m3h": q_m3h}
    for f in fields:
        out[f] = interpolate_curve(curve, q_m3h, f)
    return out


def comparison_check(raw_curve: pd.DataFrame, g: GeometryInput, op: OperatingInput) -> pd.DataFrame:
    rho = op.density_kg_m3 if op.density_kg_m3 > 0 else air_density(op.temp_c, op.altitude_m, op.rh_pct)
    area = (g.discharge_width_mm / 1000.0) * (g.discharge_height_mm / 1000.0)
    b = benchmark_similarity_point(g.d2_mm, op.rpm, rho, area)
    q = b["airflow_m3h"]
    raw_pt = interpolate_curve(raw_curve, q, "Total_Pa")
    raw_kw = interpolate_curve(raw_curve, q, "Shaft_kW")
    raw_sp = interpolate_curve(raw_curve, q, "Static_Pa")
    rows = []
    for name, rawv, refv, unit in [
        ("Airflow anchor", q, q, "m3/h"),
        ("Total pressure", raw_pt, b["total_pressure_pa"], "Pa"),
        ("Static pressure", raw_sp, b["static_pressure_pa"], "Pa"),
        ("Shaft power", raw_kw, b["shaft_power_kw"], "kW"),
    ]:
        ratio = rawv / refv if rawv is not None and refv else None
        status = "PASS" if ratio is not None and 0.60 <= ratio <= 1.60 else "WARN"
        rows.append({"Check": name, "Raw model": rawv, "Independent benchmark": refv, "Unit": unit, "Ratio raw/benchmark": ratio, "Status": status})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Volute / scroll alternatives
# -----------------------------------------------------------------------------

def area_law_volute_stations(
    w: Wheel,
    q_m3s: float,
    perf: PerfPoint,
    step_deg: int = 10,
    velocity_fraction_cu2: float = 0.75,
) -> pd.DataFrame:
    B = max(w.volute_width, 1e-4)
    r_base = max(w.r3, w.d2 / 2 + w.cutoff_clr)
    v_target = max(8.0, velocity_fraction_cu2 * max(perf.cu2, 1.0))
    # Small non-zero start area avoids a singular tongue at theta=0.
    a0 = max(B * max(w.cutoff_clr, 0.01), 0.01 * q_m3s / v_target)
    rows = []
    for th in range(0, 361, step_deg):
        frac = th / 360.0
        q_local = q_m3s * frac
        area = a0 + q_local / v_target
        h = area / B
        R = r_base + h
        v_local = q_local / area if area > 0 else 0.0
        rows.append({
            "theta_deg": th,
            "cumulative_flow_m3s": q_local,
            "area_m2": area,
            "radial_height_mm": h * 1000.0,
            "R_mm": R * 1000.0,
            "local_velocity_ms": v_local,
            "target_scroll_velocity_ms": v_target,
        })
    return pd.DataFrame(rows)


def log_spiral_volute_stations(w: Wheel, q_m3s: float, perf: PerfPoint, step_deg: int = 10) -> pd.DataFrame:
    if w.r3 <= 0 or w.volute_width <= 0:
        return pd.DataFrame()
    r2 = w.d2 / 2.0
    cu2 = max(perf.cu2, 1.0)
    k = q_m3s / (2.0 * math.pi * w.volute_width * r2 * cu2)
    rows = []
    for th in range(0, 361, step_deg):
        R = w.r3 * math.exp(k * math.radians(th))
        area = w.volute_width * max(R - w.r3, 0.0)
        qlocal = q_m3s * th / 360.0
        vlocal = qlocal / area if area > 1e-9 else 0.0
        rows.append({
            "theta_deg": th,
            "cumulative_flow_m3s": qlocal,
            "area_m2": area,
            "radial_height_mm": max(R - w.r3, 0.0) * 1000.0,
            "R_mm": R * 1000.0,
            "local_velocity_ms": vlocal,
            "target_scroll_velocity_ms": None,
        })
    return pd.DataFrame(rows)


def reference_trace_to_stations(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df is None or trace_df.empty:
        return pd.DataFrame()
    cols = {c.lower(): c for c in trace_df.columns}
    if "theta_deg" not in cols or "r_mm" not in cols:
        raise ValueError("Reference trace CSV must contain theta_deg and R_mm columns.")
    out = trace_df[[cols["theta_deg"], cols["r_mm"]]].copy()
    out.columns = ["theta_deg", "R_mm"]
    out = out.dropna().sort_values("theta_deg")
    out["cumulative_flow_m3s"] = None
    out["area_m2"] = None
    out["radial_height_mm"] = None
    out["local_velocity_ms"] = None
    out["target_scroll_velocity_ms"] = None
    return out


# -----------------------------------------------------------------------------
# Mechanical / drive / balance for a centre-hung DIDW fan
# -----------------------------------------------------------------------------

def estimate_impeller_mass(g: GeometryInput) -> float:
    mat = MATERIALS.get(g.material, MATERIALS["Galvanized Steel"])
    rho = mat["density"]
    r2 = g.d2_mm / 2000.0
    r1 = g.d1_mm / 2000.0
    b = g.b2_total_mm / 1000.0
    t_blade = g.blade_thickness_mm / 1000.0
    t_plate = g.plate_thickness_mm / 1000.0
    hub_r = max(g.hub_diameter_mm / 2000.0, g.shaft_diameter_mm / 1000.0)
    # DWDI centre plate plus two side rings/shrouds.
    centre_plate = math.pi * max(r2 ** 2 - hub_r ** 2, 0.0) * t_plate * rho
    shrouds = 2.0 * math.pi * max(r2 ** 2 - r1 ** 2, 0.0) * t_plate * rho * 0.85
    # Approximate circular-arc chord from the existing geometry routine when possible.
    try:
        temp = GeometryInput(**asdict(g))
        w, _ = build_wheel(temp, 10000.0, 1.2, 600.0)
        chord = blade_arc_geometry(w).chord_mm / 1000.0
    except Exception:
        chord = 1.2 * max(r2 - r1, 0.01)
    blades = g.blade_count * chord * b * t_blade * rho
    hub_len = max(0.12, 1.5 * g.hub_diameter_mm / 1000.0)
    hub = math.pi / 4.0 * (g.hub_diameter_mm / 1000.0) ** 2 * hub_len * rho * 0.55
    return 1.10 * (centre_plate + shrouds + blades + hub)


def _von_mises_shaft_stress_mpa(d_m: float, M_nm: float, T_nm: float, kb: float = 1.5, kt: float = 1.3) -> float:
    sigma_b = 32.0 * kb * M_nm / (math.pi * d_m ** 3)
    tau = 16.0 * kt * T_nm / (math.pi * d_m ** 3)
    return math.sqrt(sigma_b ** 2 + 3.0 * tau ** 2) / 1e6


def required_shaft_diameter_mm(M_nm: float, T_nm: float, allowable_vm_mpa: float) -> float:
    for d_mm in np.arange(20.0, 201.0, 1.0):
        if _von_mises_shaft_stress_mpa(d_mm / 1000.0, M_nm, T_nm) <= allowable_vm_mpa:
            return float(math.ceil(d_mm / 5.0) * 5.0)
    return 200.0


def center_hung_mechanical_check(
    g: GeometryInput,
    rpm: float,
    shaft_power_kw: float,
    inp: DriveMechanicalInput,
) -> dict:
    omega = 2 * math.pi * rpm / 60.0
    torque = shaft_power_kw * 1000.0 / max(omega, 1e-9)
    wheel_mass = inp.impeller_mass_kg if inp.impeller_mass_kg > 0 else estimate_impeller_mass(g)
    W = wheel_mass * G
    L = inp.bearing_span_mm / 1000.0
    a = inp.pulley_overhang_mm / 1000.0
    pulley_d = inp.pulley_pitch_diameter_mm / 1000.0

    belt_tangential = 0.0
    belt_radial = 0.0
    if inp.drive_type == "Belt drive" and pulley_d > 0:
        belt_tangential = 2.0 * torque / pulley_d
        belt_radial = inp.belt_pull_factor * belt_tangential

    if L > 0:
        reaction_drive = W / 2.0 + belt_radial * (1.0 + a / L)
        reaction_nondrive = W / 2.0 - belt_radial * a / L
        m_wheel = W * L / 4.0
    else:
        reaction_drive = reaction_nondrive = None
        m_wheel = 0.0

    m_belt = belt_radial * a
    # Conservative combination: wheel central bending plus the external pulley moment.
    m_design = m_wheel + m_belt
    req_d = required_shaft_diameter_mm(m_design, torque, inp.allowable_vm_mpa)
    d_used = inp.shaft_diameter_mm / 1000.0
    vm = _von_mises_shaft_stress_mpa(d_used, m_design, torque) if d_used > 0 else None
    sf_yield = inp.shaft_yield_mpa / vm if vm and vm > 0 else None

    # Rayleigh-style critical speed from central wheel static deflection; belt load
    # is deliberately excluded from this simple rotor critical-speed estimate.
    if L > 0 and d_used > 0:
        E = 205e9
        I = math.pi * d_used ** 4 / 64.0
        delta = W * L ** 3 / (48.0 * E * I)
        ncrit = 60.0 / (2.0 * math.pi) * math.sqrt(G / max(delta, 1e-12))
        ratio = rpm / ncrit
    else:
        delta = None
        ncrit = None
        ratio = None

    return {
        "shaft_power_kw": shaft_power_kw,
        "rpm": rpm,
        "torque_nm": torque,
        "impeller_mass_kg": wheel_mass,
        "bearing_span_mm": inp.bearing_span_mm,
        "belt_tangential_force_n": belt_tangential,
        "estimated_total_belt_radial_load_n": belt_radial,
        "drive_bearing_reaction_n": reaction_drive,
        "non_drive_bearing_reaction_n": reaction_nondrive,
        "wheel_bending_moment_nm": m_wheel,
        "pulley_overhang_moment_nm": m_belt,
        "conservative_design_bending_moment_nm": m_design,
        "shaft_diameter_used_mm": inp.shaft_diameter_mm,
        "shaft_diameter_required_mm": req_d,
        "von_mises_stress_mpa": vm,
        "yield_safety_factor": sf_yield,
        "static_deflection_at_wheel_mm": delta * 1000.0 if delta is not None else None,
        "first_critical_rpm_est": ncrit,
        "operating_to_critical_ratio": ratio,
        "note": (
            "Centre-hung DIDW preliminary shaft check. Belt pull uses user-entered factor; "
            "final shaft/bearing design needs actual pulley location, belt tensions, stepped shaft geometry and bearing data."
        ),
    }


def balance_residual_unbalance(balance_grade: str, rpm: float, rotor_mass_kg: float, planes: int = 2) -> dict:
    try:
        Ggrade = float(str(balance_grade).upper().replace("G", "").replace(" ", ""))
    except Exception:
        Ggrade = 4.0
    if rpm <= 0 or rotor_mass_kg <= 0:
        return {"grade": balance_grade, "e_per_mm": None, "U_total_gmm": None, "U_per_plane_gmm": None}
    e_mm = 9.549 * Ggrade / rpm
    U_gmm = 9549.0 * Ggrade * rotor_mass_kg / rpm
    return {
        "grade": f"G{Ggrade:g}",
        "e_per_mm": e_mm,
        "U_total_gmm": U_gmm,
        "U_per_plane_gmm": U_gmm / max(int(planes), 1),
        "planes": planes,
        "note": "Permissible residual unbalance calculated from ISO balance-grade relationship; verify applicable balancing standard and correction-plane locations.",
    }


def motor_selection_for_operating_range(
    curve: pd.DataFrame,
    target_q_m3h: float,
    max_expected_q_m3h: float,
    mech: DriveMechanicalInput,
) -> dict:
    qmax = max(max_expected_q_m3h, target_q_m3h)
    window = curve[(curve.Flow_m3h >= max(curve.Flow_m3h.min(), 0.5 * target_q_m3h)) & (curve.Flow_m3h <= qmax)]
    if window.empty:
        window = curve
    max_shaft = float(window.Shaft_kW.max())
    duty_shaft = interpolate_curve(curve, target_q_m3h, "Shaft_kW")
    motor_input_required = max_shaft / max((mech.drive_efficiency_pct / 100.0) * (mech.motor_efficiency_pct / 100.0), 0.50)
    motor_with_margin = motor_input_required * (1.0 + mech.motor_margin_pct / 100.0)
    motor = selected_motor(motor_with_margin)
    return {
        "duty_shaft_kw": duty_shaft,
        "max_shaft_kw_in_expected_range": max_shaft,
        "motor_input_required_kw": motor_input_required,
        "motor_with_margin_kw": motor_with_margin,
        "selected_standard_motor_kw": motor,
        "expected_max_airflow_m3h": qmax,
        "note": "Forward-curved wheels can overload as airflow rises; motor is sized from maximum shaft power in the declared operating-flow range, not only the duty point.",
    }


# -----------------------------------------------------------------------------
# Manufacturing geometry and readiness tables
# -----------------------------------------------------------------------------

def geometry_provenance_table(g: GeometryInput, source_map: Optional[dict] = None) -> pd.DataFrame:
    source_map = source_map or {}
    mapping = [
        ("D2", g.d2_mm, "mm"),
        ("D1", g.d1_mm, "mm"),
        ("b2 total", g.b2_total_mm, "mm"),
        ("b1 total", g.b1_total_mm, "mm"),
        ("Blade count", g.blade_count, "count"),
        ("beta1", g.beta1_deg, "deg from tangent"),
        ("beta2", g.beta2_deg, "deg from tangent"),
        ("Blade thickness", g.blade_thickness_mm, "mm"),
        ("Plate thickness", g.plate_thickness_mm, "mm"),
        ("Hub diameter", g.hub_diameter_mm, "mm"),
        ("Shaft diameter", g.shaft_diameter_mm, "mm"),
        ("Cutoff clearance", g.cutoff_clearance_mm, "mm"),
        ("Scroll internal width", g.scroll_internal_width_mm, "mm"),
        ("Discharge width", g.discharge_width_mm, "mm"),
        ("Discharge height", g.discharge_height_mm, "mm"),
    ]
    rows = []
    for p, v, u in mapping:
        meta = source_map.get(p, {})
        rows.append({
            "Parameter": p,
            "Value": v,
            "Unit": u,
            "Source": meta.get("source", "User entry / not classified"),
            "Confidence": meta.get("confidence", "UNSET"),
        })
    return pd.DataFrame(rows)


def measurement_checklist(prov: pd.DataFrame, drive: Optional[DriveMechanicalInput] = None) -> pd.DataFrame:
    critical = [
        ("Impeller outside diameter D2", "D2", "Aerodynamic scale and tip speed"),
        ("Impeller eye diameter D1", "D1", "Inlet area and velocity triangle"),
        ("Total impeller width b2", "b2 total", "Flow capacity"),
        ("Blade count", "Blade count", "Slip, noise and loading"),
        ("Blade inlet angle beta1", "beta1", "Incidence loss"),
        ("Blade outlet angle beta2", "beta2", "Euler work and pressure"),
        ("Blade thickness", "Blade thickness", "Blockage and manufacturing"),
        ("Tongue/cutoff clearance", "Cutoff clearance", "Scroll losses/noise"),
        ("Scroll internal width", "Scroll internal width", "Volute area progression"),
        ("Shaft diameter/steps/keyway", "Shaft diameter", "Mechanical design"),
    ]
    rows = []
    for item, p, why in critical:
        r = prov[prov.Parameter == p]
        src = str(r.iloc[0].Source) if not r.empty else "Missing"
        conf = str(r.iloc[0].Confidence) if not r.empty else "MISSING"
        ready = conf.upper() in ("HIGH", "MEASURED", "VERIFIED") or "physical" in src.lower() or "manufacturer" in src.lower()
        rows.append({"Measurement / verification": item, "Current source": src, "Confidence": conf, "Ready for release?": "YES" if ready else "NO", "Why needed": why})
    if drive is not None:
        for name, value, why in [
            ("Bearing centre distance", drive.bearing_span_mm, "Shaft bending/critical speed"),
            ("Pulley pitch diameter", drive.pulley_pitch_diameter_mm, "Belt force and motor speed ratio"),
            ("Pulley overhang from drive bearing", drive.pulley_overhang_mm, "Shaft bending and bearing load"),
            ("Actual impeller mass", drive.impeller_mass_kg, "Rotor stress/critical speed/balance"),
        ]:
            rows.append({"Measurement / verification": name, "Current source": "User entry", "Confidence": "HIGH" if value > 0 else "MISSING", "Ready for release?": "YES" if value > 0 else "NO", "Why needed": why})
    return pd.DataFrame(rows)


def envelope_fit_checks(env: ReferenceEnvelope, g: GeometryInput, volute: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(name, required, available, note):
        if required is None or available is None or available <= 0:
            status, margin = "UNKNOWN", None
        else:
            margin = available - required
            status = "PASS" if margin >= 0 else "FAIL"
        rows.append({"Check": name, "Required_mm": required, "Available_mm": available if available and available > 0 else None, "Margin_mm": margin, "Status": status, "Note": note})

    add("Impeller OD vs overall side height", g.d2_mm, env.overall_height_mm, "Coarse packaging check only.")
    add("Wheel/scroll axial width", max(g.b2_total_mm, g.scroll_internal_width_mm), env.overall_axial_width_mm, "Does not include bearing/pulley clearances.")
    add("Discharge clear width", g.discharge_width_mm, env.discharge_clear_width_mm, "Compare clear opening, not flange outside size.")
    add("Discharge clear height", g.discharge_height_mm, env.discharge_clear_height_mm, "Compare clear opening, not flange outside size.")
    if volute is not None and not volute.empty:
        rmax = float(volute.R_mm.max())
        # Available radius is conservatively limited by shaft centre to top if known.
        if env.shaft_center_height_mm > 0 and env.overall_height_mm > 0:
            avail_top = env.overall_height_mm - env.shaft_center_height_mm
        else:
            avail_top = env.overall_height_mm / 2.0 if env.overall_height_mm > 0 else None
        add("Calculated maximum scroll radius", rmax, avail_top, "Reference casing may be asymmetric; trace actual scroll before release.")
    return pd.DataFrame(rows)


def blade_profile_table(w: Wheel) -> pd.DataFrame:
    arc = blade_arc_geometry(w)
    df = pd.DataFrame(arc.points, columns=["x_mm", "y_mm"])
    df["roll_radius_mm"] = arc.R_mm
    df["chord_mm"] = arc.chord_mm
    df["camber_deg"] = arc.camber_deg
    df["wrap_deg"] = arc.wrap_deg
    return df


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def impeller_figure(w: Wheel):
    import matplotlib.pyplot as plt
    arc = blade_arc_geometry(w)
    fig, ax = plt.subplots(figsize=(7, 7))
    r2 = w.d2 * 500.0
    r1 = w.d1 * 500.0
    ax.add_patch(plt.Circle((0, 0), r2, fill=False, linewidth=1.8))
    ax.add_patch(plt.Circle((0, 0), r1, fill=False, linewidth=1.2))
    if w.hub_d > 0:
        ax.add_patch(plt.Circle((0, 0), w.hub_d * 500.0, fill=False, linestyle="--"))
    pts0 = np.array(arc.points)
    for k in range(w.z):
        ang = 2 * math.pi * k / w.z
        c, s = math.cos(ang), math.sin(ang)
        pts = np.column_stack((pts0[:, 0] * c - pts0[:, 1] * s, pts0[:, 0] * s + pts0[:, 1] * c))
        ax.plot(pts[:, 0], pts[:, 1], linewidth=0.7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Impeller reconstruction — D2 {w.d2*1000:.0f} mm / D1 {w.d1*1000:.0f} mm / {w.z} blades")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    return fig


def blade_figure(w: Wheel):
    import matplotlib.pyplot as plt
    arc = blade_arc_geometry(w)
    p = np.array(arc.points)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p[:, 0], p[:, 1], linewidth=2)
    ax.plot([p[0,0], p[-1,0]], [p[0,1], p[-1,1]], linestyle="--", linewidth=1)
    ax.scatter([p[0,0], p[-1,0]], [p[0,1], p[-1,1]])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Circular-arc blade centreline — beta1={w.beta1:.1f}°, beta2={w.beta2:.1f}°, R={arc.R_mm:.1f} mm")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    return fig


def volute_figure(volute: pd.DataFrame, w: Wheel, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    if volute is not None and not volute.empty:
        th = np.radians(volute.theta_deg.to_numpy(float))
        r = volute.R_mm.to_numpy(float)
        ax.plot(r * np.cos(th), r * np.sin(th), linewidth=2)
    ax.add_patch(plt.Circle((0, 0), w.d2 * 500.0, fill=False, linestyle="--"))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    return fig


def curve_figure(curves: Dict[str, pd.DataFrame], selected_q: Optional[float] = None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for label, c in curves.items():
        axes[0].plot(c.Flow_m3h, c.Static_Pa, label=label)
        axes[1].plot(c.Flow_m3h, c.Shaft_kW, label=label)
        axes[2].plot(c.Flow_m3h, c.EtaTotal * 100, label=label)
    if selected_q:
        for ax in axes:
            ax.axvline(selected_q, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Static pressure Pa")
    axes[1].set_ylabel("Shaft power kW")
    axes[2].set_ylabel("Total efficiency %")
    for ax in axes:
        ax.set_xlabel("Airflow m³/h")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_png(fig) -> bytes:
    import matplotlib.pyplot as plt
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return bio.getvalue()


# -----------------------------------------------------------------------------
# Excel/PDF/DXF export
# -----------------------------------------------------------------------------

def create_reverse_excel(
    env: ReferenceEnvelope,
    g: GeometryInput,
    provenance: pd.DataFrame,
    raw_curve: pd.DataFrame,
    selected_curve: pd.DataFrame,
    benchmark_sweep_df: pd.DataFrame,
    volute: pd.DataFrame,
    mechanical: dict,
    balance: dict,
    motor: dict,
    checklist: pd.DataFrame,
    calibration_meta: dict,
) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    summary = [
        ("REVERSE ENGINEERING TOOLKIT v25 — PRELIMINARY", ""),
        ("Reference", env.reference_name),
        ("Drawing code", env.drawing_code),
        ("Fan standard stated", env.fan_standard),
        ("Balance grade", env.balance_grade),
        ("Fan family", g.family),
        ("Arrangement", g.arrangement),
        ("D2 mm", g.d2_mm),
        ("D1 mm", g.d1_mm),
        ("b2 total mm", g.b2_total_mm),
        ("Blade count", g.blade_count),
        ("beta1 deg from tangent", g.beta1_deg),
        ("beta2 deg from tangent", g.beta2_deg),
        ("IMPORTANT", "Do not manufacture until assumed dimensions are measured and performance is validated by prototype test."),
    ]
    for r, (a, b) in enumerate(summary, 1):
        ws.cell(r, 1, a)
        ws.cell(r, 2, b)
    ws["A1"].font = Font(bold=True, size=14)
    ws["A1"].fill = PatternFill("solid", fgColor="FFF2CC")
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 90

    def add_df(name: str, df: pd.DataFrame):
        sh = wb.create_sheet(name[:31])
        if df is None:
            return
        for c, col in enumerate(df.columns, 1):
            cell = sh.cell(1, c, str(col))
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="D9EAF7")
        for r, row in enumerate(df.itertuples(index=False), 2):
            for c, val in enumerate(row, 1):
                sh.cell(r, c, None if pd.isna(val) else val)
        sh.freeze_panes = "A2"
        for col in sh.columns:
            letter = col[0].column_letter
            sh.column_dimensions[letter].width = min(40, max(12, max(len(str(x.value)) if x.value is not None else 0 for x in col) + 2))

    add_df("Reference Envelope", envelope_frame(env))
    add_df("Geometry Provenance", provenance)
    add_df("Raw Meanline", raw_curve)
    add_df("Selected Curve", selected_curve)
    add_df("Benchmark Speed Sweep", benchmark_sweep_df)
    add_df("Volute Stations", volute)
    add_df("Mechanical", pd.DataFrame([mechanical]))
    add_df("Balance", pd.DataFrame([balance]))
    add_df("Motor Sizing", pd.DataFrame([motor]))
    add_df("Measurement Checklist", checklist)
    add_df("Calibration", pd.DataFrame([calibration_meta]))

    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def create_reverse_pdf(
    env: ReferenceEnvelope,
    g: GeometryInput,
    selected_point: dict,
    readiness: dict,
    mechanical: dict,
    balance: dict,
    motor: dict,
    provenance: pd.DataFrame,
    checklist: pd.DataFrame,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Reverse-Engineered Centrifugal Blower — Preliminary Engineering Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph("NOT FOR PRODUCTION RELEASE until critical geometry is physically measured and the prototype is tested.", styles["Heading3"]),
        Spacer(1, 8),
    ]
    summary_rows = [
        ["Reference", env.reference_name],
        ["Drawing", env.drawing_code],
        ["Standard stated", env.fan_standard],
        ["Balance grade", env.balance_grade],
        ["Family", g.family],
        ["Arrangement", g.arrangement],
        ["D2 / D1", f"{g.d2_mm:.0f} / {g.d1_mm:.0f} mm"],
        ["b2 total", f"{g.b2_total_mm:.0f} mm"],
        ["Blades", f"{g.blade_count}, beta1={g.beta1_deg:.1f}°, beta2={g.beta2_deg:.1f}°"],
        ["Readiness", f"Grade {readiness['grade']} — {readiness['score']:.0f}%"],
    ]
    if selected_point:
        summary_rows += [
            ["Airflow", f"{selected_point.get('Flow_m3h', 0):,.0f} m3/h"],
            ["Static pressure", f"{selected_point.get('Static_Pa', 0):,.0f} Pa"],
            ["Total pressure", f"{selected_point.get('Total_Pa', 0):,.0f} Pa"],
            ["Shaft power", f"{selected_point.get('Shaft_kW', 0):.2f} kW"],
        ]
    t = Table(summary_rows, colWidths=[130, 380])
    t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.3, colors.grey), ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story += [t, Spacer(1, 12), Paragraph("Motor / Mechanical", styles["Heading2"])]
    mm = [
        ["Selected standard motor", f"{motor.get('selected_standard_motor_kw', 0):.1f} kW"],
        ["Max shaft power in declared range", f"{motor.get('max_shaft_kw_in_expected_range', 0):.2f} kW"],
        ["Shaft diameter used / required", f"{mechanical.get('shaft_diameter_used_mm')} / {mechanical.get('shaft_diameter_required_mm')} mm"],
        ["Estimated first critical speed", f"{mechanical.get('first_critical_rpm_est') or 0:.0f} rpm"],
        ["Balance residual U total", f"{balance.get('U_total_gmm') or 0:.1f} g·mm"],
    ]
    tt = Table(mm, colWidths=[220, 290])
    tt.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.3, colors.grey), ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke)]))
    story += [tt, PageBreak(), Paragraph("Geometry Provenance", styles["Heading2"])]

    ptab = [[str(c) for c in provenance.columns]] + [[str(v) for v in row] for row in provenance.itertuples(index=False)]
    pt = Table(ptab, repeatRows=1, colWidths=[90, 70, 70, 220, 55])
    pt.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("FONTSIZE", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story += [pt, Spacer(1, 12), Paragraph("Measurements Still Needed", styles["Heading2"])]
    ctab = [["Measurement", "Confidence", "Release?", "Why"]] + [[str(r[0]), str(r[2]), str(r[3]), str(r[4])] for r in checklist.itertuples(index=False)]
    ct = Table(ctab, repeatRows=1, colWidths=[175, 65, 50, 220])
    ct.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("FONTSIZE", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(ct)
    doc.build(story)
    return bio.getvalue()


def create_reverse_dxf(
    env: ReferenceEnvelope,
    g: GeometryInput,
    w: Wheel,
    volute: pd.DataFrame,
    balance_grade: str,
    project: str = "Reverse engineered blower",
) -> bytes:
    try:
        import ezdxf
    except Exception:
        return b""

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for name in ["OUTLINE", "CENTER", "DIM", "TABLE", "REFERENCE"]:
        if name not in doc.layers:
            doc.layers.new(name)
    txt = max(g.d2_mm / 55.0, 6.0)

    def T(s, p, h=txt, layer="TABLE"):
        msp.add_text(str(s), dxfattribs={"height": h, "layer": layer}).set_placement(p)

    # Impeller front view at origin.
    r2 = g.d2_mm / 2.0
    r1 = g.d1_mm / 2.0
    msp.add_circle((0,0), r2, dxfattribs={"layer":"OUTLINE"})
    msp.add_circle((0,0), r1, dxfattribs={"layer":"OUTLINE"})
    msp.add_circle((0,0), g.hub_diameter_mm/2.0, dxfattribs={"layer":"CENTER"})
    arc = blade_arc_geometry(w)
    pts0 = np.array(arc.points)
    for k in range(g.blade_count):
        a = 2*math.pi*k/g.blade_count
        c,s=math.cos(a),math.sin(a)
        pts=[(float(x*c-y*s), float(x*s+y*c)) for x,y in pts0]
        msp.add_lwpolyline(pts, dxfattribs={"layer":"OUTLINE"})
    T("VIEW A - IMPELLER FRONT", (-r2, r2+2.5*txt))
    T(f"D2={g.d2_mm:.1f}  D1={g.d1_mm:.1f}  Z={g.blade_count}", (-r2, r2+1.2*txt))

    # DWDI side section.
    ox = 1.8*r2
    half = g.b2_total_mm/2.0
    msp.add_line((ox,-r2),(ox,r2),dxfattribs={"layer":"CENTER"})
    for side in (-1,1):
        x = ox + side*half
        msp.add_line((x,-r2),(x,r2),dxfattribs={"layer":"OUTLINE"})
        msp.add_line((ox,-r2),(x,-r2),dxfattribs={"layer":"OUTLINE"})
        msp.add_line((ox,r2),(x,r2),dxfattribs={"layer":"OUTLINE"})
    T("VIEW B - DWDI SIDE / CENTRE PLATE", (ox-half, r2+2.5*txt))
    T(f"b2 total={g.b2_total_mm:.1f}  shaft dia={g.shaft_diameter_mm:.1f}", (ox-half, r2+1.2*txt))

    # Single blade.
    oy = -1.65*r2
    bpts=[(x, y+oy) for x,y in arc.points]
    msp.add_lwpolyline(bpts, dxfattribs={"layer":"OUTLINE"})
    T("VIEW C - SINGLE BLADE CENTRELINE", (-r2, oy-r2*0.15))
    T(f"beta1={g.beta1_deg:.1f} deg  beta2={g.beta2_deg:.1f} deg FROM TANGENT", (-r2, oy-r2*0.15-1.4*txt))
    T(f"Roll R={arc.R_mm:.1f} mm  chord={arc.chord_mm:.1f} mm  wrap={arc.wrap_deg:.1f} deg", (-r2, oy-r2*0.15-2.8*txt))

    # Volute profile offset right.
    vx0 = 4.0*r2
    if volute is not None and not volute.empty:
        vpts=[]
        for row in volute.itertuples():
            th=math.radians(float(row.theta_deg)); rr=float(row.R_mm)
            vpts.append((vx0+rr*math.cos(th), rr*math.sin(th)))
        msp.add_lwpolyline(vpts, dxfattribs={"layer":"OUTLINE"})
        msp.add_circle((vx0,0),r2,dxfattribs={"layer":"CENTER"})
        T("VIEW D - CALCULATED VOLUTE / SCROLL", (vx0-r2, max(float(volute.R_mm.max()),r2)+2.5*txt))
        T("NOT AN EXACT TRACE OF SUPPLIER CASING UNLESS REFERENCE-STATION DATA IS USED", (vx0-r2, max(float(volute.R_mm.max()),r2)+1.1*txt))

    # Discharge flange detail from drawing.
    fx = vx0 + (float(volute.R_mm.max()) if volute is not None and not volute.empty else r2) + 2.0*r2
    outer_w = env.discharge_flange_outer_width_mm
    outer_h = env.discharge_flange_outer_height_mm
    clear_w = env.discharge_clear_width_mm
    clear_h = env.discharge_clear_height_mm
    if outer_w > 0 and outer_h > 0:
        msp.add_lwpolyline([(fx,0),(fx+outer_w,0),(fx+outer_w,outer_h),(fx,outer_h),(fx,0)],dxfattribs={"layer":"REFERENCE"})
        dx=(outer_w-clear_w)/2.0; dy=(outer_h-clear_h)/2.0
        msp.add_lwpolyline([(fx+dx,dy),(fx+dx+clear_w,dy),(fx+dx+clear_w,dy+clear_h),(fx+dx,dy+clear_h),(fx+dx,dy)],dxfattribs={"layer":"REFERENCE"})
        # Approximate evenly distributed 24-hole pattern (6 per side) for reference only.
        if env.discharge_flange_holes == 24 and env.discharge_flange_hole_diameter_mm > 0:
            offset=max(dx/2.0,10.0)
            xs=np.linspace(fx+offset,fx+outer_w-offset,6)
            ys=np.linspace(offset,outer_h-offset,6)
            for x in xs:
                msp.add_circle((float(x),offset),env.discharge_flange_hole_diameter_mm/2.0,dxfattribs={"layer":"REFERENCE"})
                msp.add_circle((float(x),outer_h-offset),env.discharge_flange_hole_diameter_mm/2.0,dxfattribs={"layer":"REFERENCE"})
            for y in ys:
                msp.add_circle((fx+offset,float(y)),env.discharge_flange_hole_diameter_mm/2.0,dxfattribs={"layer":"REFERENCE"})
                msp.add_circle((fx+outer_w-offset,float(y)),env.discharge_flange_hole_diameter_mm/2.0,dxfattribs={"layer":"REFERENCE"})
        T("VIEW E - DISCHARGE FLANGE REFERENCE", (fx, outer_h+2.5*txt))
        T(f"Outer {outer_w:.0f}x{outer_h:.0f}; clear {clear_w:.0f}x{clear_h:.0f}; drawing holes {env.discharge_flange_holes}-dia{env.discharge_flange_hole_diameter_mm:g}", (fx, outer_h+1.1*txt))

    # Title notes.
    ty = oy - r2*0.55
    for i,line in enumerate([
        f"PROJECT: {project}",
        f"REFERENCE: {env.reference_name} / {env.drawing_code}",
        f"FAMILY: {g.family}  ARRANGEMENT: {g.arrangement}",
        f"BALANCE GRADE: {balance_grade}",
        "PRELIMINARY REVERSE-ENGINEERING DRAWING - ASSUMED DIMENSIONS MUST BE VERIFIED",
        "VALIDATE AERODYNAMICS BY AMCA 210 / ISO 5801 TEST BEFORE SERIES PRODUCTION",
    ]):
        T(line, (-r2, ty-i*1.5*txt), layer="TABLE")

    stream=io.StringIO(); doc.write(stream)
    return stream.getvalue().encode("utf-8")


def make_manufacturing_package(
    env: ReferenceEnvelope,
    g: GeometryInput,
    provenance: pd.DataFrame,
    w: Wheel,
    raw_curve: pd.DataFrame,
    selected_curve: pd.DataFrame,
    benchmark_sweep_df: pd.DataFrame,
    selected_point: dict,
    volute: pd.DataFrame,
    mechanical: dict,
    balance: dict,
    motor: dict,
    checklist: pd.DataFrame,
    readiness: dict,
    calibration_meta: dict,
    project_root: Optional[Path] = None,
) -> bytes:
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("01_reverse_engineering_summary.pdf", create_reverse_pdf(env,g,selected_point,readiness,mechanical,balance,motor,provenance,checklist))
        z.writestr("02_engineering_workbook.xlsx", create_reverse_excel(env,g,provenance,raw_curve,selected_curve,benchmark_sweep_df,volute,mechanical,balance,motor,checklist,calibration_meta))
        dxf=create_reverse_dxf(env,g,w,volute,env.balance_grade,env.reference_name)
        if dxf:
            z.writestr("03_manufacturing_reference.dxf",dxf)
        z.writestr("04_geometry_provenance.csv",provenance.to_csv(index=False))
        z.writestr("05_raw_meanline_curve.csv",raw_curve.to_csv(index=False))
        z.writestr("06_selected_performance_curve.csv",selected_curve.to_csv(index=False))
        z.writestr("07_independent_benchmark_speed_sweep.csv",benchmark_sweep_df.to_csv(index=False))
        z.writestr("08_volute_stations.csv",volute.to_csv(index=False))
        z.writestr("09_blade_profile_coordinates.csv",blade_profile_table(w).to_csv(index=False))
        z.writestr("10_mechanical_check.csv",pd.DataFrame([mechanical]).to_csv(index=False))
        z.writestr("11_balance_check.csv",pd.DataFrame([balance]).to_csv(index=False))
        z.writestr("12_motor_sizing.csv",pd.DataFrame([motor]).to_csv(index=False))
        z.writestr("13_measurement_checklist.csv",checklist.to_csv(index=False))
        z.writestr("14_reference_envelope.csv",envelope_frame(env).to_csv(index=False))
        z.writestr("impeller_reconstruction.png",fig_png(impeller_figure(w)))
        z.writestr("blade_profile.png",fig_png(blade_figure(w)))
        z.writestr("volute_profile.png",fig_png(volute_figure(volute,w,"Calculated volute / scroll")))
        z.writestr("performance_curves.png",fig_png(curve_figure({"Selected":selected_curve})))
        z.writestr("IMPORTANT_README.txt",(
            "PRELIMINARY REVERSE-ENGINEERING PACKAGE\n\n"
            "This package deliberately separates supplier-drawing dimensions, physical measurements, assumptions and calculated geometry.\n"
            "Do NOT release parts to production while critical aerodynamic geometry is still assumed.\n"
            "The independent Kruger FDA benchmark is used only as a sanity scale and is not a WDL/KQ performance claim.\n"
            "Final production requires actual wheel/scroll measurements, structural review, balance verification, CFD/prototype work as appropriate, and laboratory fan testing.\n"
        ))
        if project_root is not None:
            for asset in ["assets/KQ800_2026_V4_reference.jpeg","assets/WDL800_2020_V1_reference.jpeg"]:
                p=project_root/asset
                if p.exists():
                    z.write(p, f"reference_drawings/{p.name}")
    return bio.getvalue()
