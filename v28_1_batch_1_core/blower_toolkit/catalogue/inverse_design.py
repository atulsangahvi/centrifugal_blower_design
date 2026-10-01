
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from blower_toolkit.engine import (
    FAMILIES,
    MATERIALS,
    Duty,
    PerfPoint,
    acoustics,
    air_density,
    air_viscosity,
    wheel_performance,
)
from blower_toolkit.reverse_engineering import (
    GeometryInput,
    build_wheel,
)


@dataclass
class InverseTarget:
    manufacturer: str
    model: str
    family: str
    arrangement: str
    d2_mm: float
    rpm: float
    basis: str
    airflow_m3h: float = 0.0
    static_pressure_pa: float = 0.0
    input_power_w: float = 0.0
    noise_dba: float = 0.0
    source_file: str = ""
    source_page: int = 0
    temp_c: float = 20.0
    altitude_m: float = 0.0
    rh_pct: float = 50.0
    material: str = "Galvanized Steel"

    @property
    def density(self) -> float:
        return air_density(self.temp_c, self.altitude_m, self.rh_pct)


@dataclass
class InverseLocks:
    d1_mm: float = 0.0
    b2_total_mm: float = 0.0
    b1_total_mm: float = 0.0
    blade_count: int = 0
    beta1_deg: float = 0.0
    beta2_deg: float = 0.0
    cutoff_clearance_mm: float = 0.0
    scroll_internal_width_mm: float = 0.0
    discharge_width_mm: float = 0.0
    discharge_height_mm: float = 0.0
    blade_thickness_mm: float = 0.0
    plate_thickness_mm: float = 0.0
    shaft_diameter_mm: float = 0.0
    hub_diameter_mm: float = 0.0


@dataclass
class InverseConfig:
    mode: str = "Standard"
    pressure_match_weight: float = 1.0
    power_match_weight: float = 0.20
    efficiency_weight: float = 0.12
    noise_weight: float = 0.04
    manufacturability_weight: float = 0.14
    geometry_prior_weight: float = 0.06
    correction_prior_weight: float = 0.10
    use_catalogue_power: bool = False
    use_catalogue_noise: bool = False
    motor_efficiency_pct: float = 90.0
    drive_efficiency_pct: float = 98.0
    allow_empirical_model_correction: bool = True
    endpoint_flow_prior: float = 1.0
    endpoint_pressure_prior: float = 1.0
    seed: int = 27

    @property
    def n_samples(self) -> int:
        return {"Quick": 350, "Standard": 900, "Deep": 2200}.get(self.mode, 900)

    @property
    def n_refine(self) -> int:
        return {"Quick": 2, "Standard": 5, "Deep": 8}.get(self.mode, 5)

    @property
    def refine_iterations(self) -> int:
        return {"Quick": 35, "Standard": 65, "Deep": 100}.get(self.mode, 65)


def supported_catalogue_family(record: dict) -> Optional[str]:
    text = f"{record.get('fan_type','')} {record.get('series','')} {record.get('model','')}".lower()
    if "axial" in text or "inlet ring" in text:
        return None
    if "airfoil" in text:
        return "Airfoil Backward (AF)"
    if "plug" in text:
        return "Plug / Plenum (BC, no volute)"
    if "backward" in text:
        return "Backward Curved (BC)"
    if "forward" in text or "sirocco" in text:
        return "Multi-Blade Sirocco (Cage)"
    if "radial" in text:
        return "Radial Straight (Paddle)"
    return None


def catalogue_arrangement(record: dict) -> str:
    text = str(record.get("arrangement") or "").lower()
    if "dual" in text or "dwdi" in text:
        return "DWDI (double inlet)"
    return "SWSI (single inlet)"


def classify_catalogue_basis(record: dict, stored_curve_points: Optional[pd.DataFrame] = None) -> str:
    """Choose the safest inverse-design interpretation.

    Manufacturer technical tables often show maximum air volume and maximum
    pressure in adjacent columns. Those values should not be assumed to be one
    simultaneous duty point unless a plotted/tested operating point says so.
    """
    if stored_curve_points is not None and not stored_curve_points.empty:
        x = stored_curve_points.dropna(subset=["airflow_m3h", "pressure_pa"]).copy()
        if "curve_name" in x:
            x = x[
                ~x["curve_name"].astype(str).str.contains(
                    "operating row|rated point|illustrative", case=False, regex=True
                )
            ]
        if x["airflow_m3h"].nunique() >= 4:
            return "Digitised Q-P curve"

    status = str(record.get("data_status") or "").lower()
    notes = str(record.get("notes") or "").lower()
    if "simultaneous duty" in status or "simultaneous duty" in notes:
        return "Simultaneous duty point"
    return "Catalogue max-flow / max-pressure endpoints"


def target_from_record(
    record: dict,
    basis: Optional[str] = None,
    temp_c: float = 20.0,
    altitude_m: float = 0.0,
    rh_pct: float = 50.0,
) -> InverseTarget:
    family = supported_catalogue_family(record)
    if family is None:
        raise ValueError("This inverse-design module presently supports centrifugal fans only.")

    d2 = float(record.get("impeller_diameter_mm") or 0.0)
    rpm = float(record.get("rpm") or 0.0)
    if d2 <= 0:
        raise ValueError("Impeller diameter D2 is required. Enter/measure it before optimization.")
    if rpm <= 0:
        raise ValueError("Fan RPM is required. Enter the catalogue or measured speed.")

    material = str(record.get("material") or "Galvanized Steel")
    if material not in MATERIALS:
        material = "Galvanized Steel"

    return InverseTarget(
        manufacturer=str(record.get("manufacturer") or ""),
        model=str(record.get("model") or ""),
        family=family,
        arrangement=catalogue_arrangement(record),
        d2_mm=d2,
        rpm=rpm,
        basis=basis or "Catalogue max-flow / max-pressure endpoints",
        airflow_m3h=float(record.get("airflow_m3h") or 0.0),
        static_pressure_pa=float(record.get("static_pressure_pa") or 0.0),
        input_power_w=float(record.get("input_power_w") or 0.0),
        noise_dba=float(record.get("noise_dba") or 0.0),
        source_file=str(record.get("source_file") or ""),
        source_page=int(record.get("source_page") or 0),
        temp_c=temp_c,
        altitude_m=altitude_m,
        rh_pct=rh_pct,
        material=material,
    )


def _blade_pitch_chord(g: GeometryInput) -> Tuple[float, float, float]:
    r1 = g.d1_mm / 2.0
    r2 = g.d2_mm / 2.0
    beta_m = math.radians(min(90.0, max(10.0, 0.5 * (g.beta1_deg + g.beta2_deg))))
    chord = 1.15 * (r2 - r1) / max(math.sin(beta_m), 0.30)
    pitch = math.pi * (r1 + r2) / max(g.blade_count, 1)
    return pitch, chord, pitch / max(chord, 1e-9)


def _family_pitch_chord_range(family: str) -> Tuple[float, float]:
    code = FAMILIES[family]["code"]
    if code in ("fc", "sirocco"):
        return 0.22, 1.20
    if code in ("af", "bc", "bi", "plug"):
        return 0.45, 1.80
    return 0.45, 2.50


def _default_thickness(d2_mm: float, family: str) -> Tuple[float, float]:
    code = FAMILIES[family]["code"]
    if code in ("fc", "sirocco"):
        blade = min(max(0.6, 0.0025 * d2_mm), 3.0)
        plate = min(max(0.8, 0.0035 * d2_mm), 4.0)
    else:
        blade = min(max(1.0, 0.0035 * d2_mm), 6.0)
        plate = min(max(1.5, 0.0045 * d2_mm), 8.0)
    return blade, plate


def _variable_specs(target: InverseTarget, locks: InverseLocks, cfg: InverseConfig) -> List[dict]:
    fam = FAMILIES[target.family]
    d2 = target.d2_mm
    specs: List[dict] = []

    def add(name: str, lo: float, hi: float, locked: bool = False, value: float = 0.0):
        if locked:
            specs.append({"name": name, "lo": value, "hi": value, "locked": True})
        else:
            specs.append({"name": name, "lo": lo, "hi": hi, "locked": False})

    add("d1d2", *fam["d1d2_rng"], locks.d1_mm > 0, locks.d1_mm / d2 if locks.d1_mm > 0 else 0)
    add("b2d2", *fam["b2d2_rng"], locks.b2_total_mm > 0, locks.b2_total_mm / d2 if locks.b2_total_mm > 0 else 0)
    add("b1_b2", 1.00, 1.35, locks.b1_total_mm > 0 and locks.b2_total_mm > 0,
        locks.b1_total_mm / locks.b2_total_mm if locks.b1_total_mm > 0 and locks.b2_total_mm > 0 else 0)
    add("beta1", *fam["beta1_rng"], locks.beta1_deg > 0, locks.beta1_deg)
    add("beta2", *fam["beta2_rng"], locks.beta2_deg > 0, locks.beta2_deg)
    add("z", float(fam["z_rng"][0]), float(fam["z_rng"][1]), locks.blade_count > 0, float(locks.blade_count))
    add("cutoff_ratio", 0.035, 0.105, locks.cutoff_clearance_mm > 0,
        locks.cutoff_clearance_mm / d2 if locks.cutoff_clearance_mm > 0 else 0)
    add("outlet_velocity", 6.0, 18.0, locks.discharge_width_mm > 0 and locks.discharge_height_mm > 0, 12.0)
    add("scroll_velocity_fraction", 0.50, 0.95, False, 0.0)
    add("scroll_width_ratio", 1.05, 2.25, locks.scroll_internal_width_mm > 0 and locks.b2_total_mm > 0,
        locks.scroll_internal_width_mm / locks.b2_total_mm if locks.scroll_internal_width_mm > 0 and locks.b2_total_mm > 0 else 0)

    if cfg.allow_empirical_model_correction:
        pp = max(cfg.endpoint_pressure_prior, 0.20)
        fp = max(cfg.endpoint_flow_prior, 0.20)
        add("pressure_factor", 0.72 * pp, 1.38 * pp, False, 0.0)
        if target.basis == "Catalogue max-flow / max-pressure endpoints":
            add("flow_factor", 0.78 * fp, 1.22 * fp, False, 0.0)
    else:
        add("pressure_factor", cfg.endpoint_pressure_prior, cfg.endpoint_pressure_prior, True, cfg.endpoint_pressure_prior)
        if target.basis == "Catalogue max-flow / max-pressure endpoints":
            add("flow_factor", cfg.endpoint_flow_prior, cfg.endpoint_flow_prior, True, cfg.endpoint_flow_prior)

    return specs


def _decode(values: np.ndarray, specs: List[dict]) -> Dict[str, float]:
    return {spec["name"]: float(values[i]) for i, spec in enumerate(specs)}


def _bounds(specs: List[dict]) -> List[Tuple[float, float]]:
    out = []
    for s in specs:
        if s["lo"] == s["hi"]:
            eps = max(abs(s["lo"]) * 1e-9, 1e-9)
            out.append((s["lo"] - eps, s["hi"] + eps))
        else:
            out.append((s["lo"], s["hi"]))
    return out


def _candidate_geometry(
    params: Dict[str, float],
    target: InverseTarget,
    locks: InverseLocks,
) -> GeometryInput:
    d2 = target.d2_mm
    d1 = locks.d1_mm if locks.d1_mm > 0 else params["d1d2"] * d2
    b2 = locks.b2_total_mm if locks.b2_total_mm > 0 else params["b2d2"] * d2
    b1 = locks.b1_total_mm if locks.b1_total_mm > 0 else params["b1_b2"] * b2
    z = locks.blade_count if locks.blade_count > 0 else int(round(params["z"]))
    beta1 = locks.beta1_deg if locks.beta1_deg > 0 else params["beta1"]
    beta2 = locks.beta2_deg if locks.beta2_deg > 0 else params["beta2"]
    cutoff = locks.cutoff_clearance_mm if locks.cutoff_clearance_mm > 0 else params["cutoff_ratio"] * d2

    blade_t_default, plate_t_default = _default_thickness(d2, target.family)
    blade_t = locks.blade_thickness_mm if locks.blade_thickness_mm > 0 else blade_t_default
    plate_t = locks.plate_thickness_mm if locks.plate_thickness_mm > 0 else plate_t_default

    q_ref = max(target.airflow_m3h, 100.0) / 3600.0

    if locks.discharge_width_mm > 0 and locks.discharge_height_mm > 0:
        out_w = locks.discharge_width_mm
        out_h = locks.discharge_height_mm
    elif FAMILIES[target.family].get("plenum"):
        out_w = out_h = 0.0
    else:
        area = q_ref / max(params["outlet_velocity"], 4.0)
        # Keep the discharge reasonably compatible with wheel width while
        # preserving the selected total area.
        width_floor_m = max(1.05 * b2 / 1000.0, math.sqrt(area) * 0.75)
        out_w_m = max(width_floor_m, math.sqrt(area))
        out_h_m = area / max(out_w_m, 1e-6)
        out_w, out_h = out_w_m * 1000.0, out_h_m * 1000.0

    if locks.scroll_internal_width_mm > 0:
        scroll_w = locks.scroll_internal_width_mm
    elif FAMILIES[target.family].get("plenum"):
        scroll_w = 0.0
    else:
        scroll_w = max(params["scroll_width_ratio"] * b2, out_w * 0.90)

    shaft = locks.shaft_diameter_mm if locks.shaft_diameter_mm > 0 else max(8.0, 0.045 * d2)
    hub = locks.hub_diameter_mm if locks.hub_diameter_mm > 0 else max(1.8 * shaft, 0.14 * d1)

    return GeometryInput(
        family=target.family,
        arrangement=target.arrangement,
        d2_mm=d2,
        d1_mm=d1,
        b2_total_mm=b2,
        b1_total_mm=b1,
        blade_count=max(3, z),
        beta1_deg=beta1,
        beta2_deg=beta2,
        blade_thickness_mm=blade_t,
        plate_thickness_mm=plate_t,
        hub_diameter_mm=hub,
        shaft_diameter_mm=shaft,
        cutoff_clearance_mm=cutoff,
        scroll_internal_width_mm=scroll_w,
        discharge_width_mm=out_w,
        discharge_height_mm=out_h,
        material=target.material,
    )


def _performance_at(
    wheel,
    target: InverseTarget,
    q_m3h: float,
    rpm: float,
    pressure_factor: float,
) -> PerfPoint:
    p = wheel_performance(
        wheel,
        target.density,
        rpm,
        max(q_m3h, 0.1) / 3600.0,
        visc=air_viscosity(target.temp_c),
    )
    # Empirical pressure factor corrects the mean-line model only. Shaft power
    # is intentionally not multiplied: it remains an independent cross-check.
    return replace(
        p,
        dp_total=p.dp_total * pressure_factor,
        dp_static=p.dp_static * pressure_factor,
        eta_total=min(max(p.eta_total * pressure_factor, 0.0), 0.92),
        eta_static=min(max(p.eta_static * pressure_factor, 0.0), 0.90),
    )


def _endpoint_metrics(
    wheel,
    target: InverseTarget,
    pressure_factor: float,
    flow_factor: float,
) -> dict:
    qmax = max(target.airflow_m3h, 10.0)
    qs = np.linspace(0.025 * qmax, 2.50 * qmax, 34)
    ps = []
    powers = []
    for q in qs:
        p = _performance_at(wheel, target, float(q), target.rpm, pressure_factor)
        ps.append(p.dp_static)
        powers.append(p.p_shaft / 1000.0)

    ps_arr = np.asarray(ps, dtype=float)
    near_shutoff = float(ps_arr[0])

    qfree_raw = float(qs[-1])
    crossed = False
    for i in range(1, len(qs)):
        if ps_arr[i] <= 0.0 < ps_arr[i - 1]:
            qfree_raw = float(
                qs[i - 1]
                + (0.0 - ps_arr[i - 1])
                * (qs[i] - qs[i - 1])
                / (ps_arr[i] - ps_arr[i - 1])
            )
            crossed = True
            break

    return {
        "predicted_qmax_m3h": qfree_raw * flow_factor,
        "predicted_psmax_pa": near_shutoff,
        "zero_crossing_found": crossed,
        "max_sweep_shaft_kw": float(np.nanmax(powers)),
    }


def _target_curve_points(curve_points: Optional[pd.DataFrame]) -> pd.DataFrame:
    if curve_points is None or curve_points.empty:
        return pd.DataFrame()
    x = curve_points.copy()
    required = ["airflow_m3h", "pressure_pa"]
    if not all(c in x.columns for c in required):
        return pd.DataFrame()
    x = x.dropna(subset=required)
    if "curve_name" in x.columns:
        x = x[
            ~x["curve_name"].astype(str).str.contains(
                "operating row|rated point|illustrative", case=False, regex=True
            )
        ]
    return x.sort_values("airflow_m3h")


def _manufacturing_penalties(g: GeometryInput, perf: PerfPoint) -> Tuple[float, dict]:
    fam = FAMILIES[g.family]
    material = MATERIALS.get(g.material, MATERIALS["Galvanized Steel"])
    tip_lim = min(float(fam["tip_max"]), float(material["max_tip"]))
    tip_excess = max(0.0, perf.u2 / max(tip_lim, 1.0) - 1.0)

    pitch, chord, pc = _blade_pitch_chord(g)
    pc_lo, pc_hi = _family_pitch_chord_range(g.family)
    pc_pen = max(0.0, pc_lo - pc) / max(pc_lo, 1e-6) + max(0.0, pc - pc_hi) / max(pc_hi, 1e-6)

    beta2 = math.radians(min(max(g.beta2_deg, 5.0), 175.0))
    blockage = g.blade_count * (g.blade_thickness_mm / 1000.0) / (
        math.pi * (g.d2_mm / 1000.0) * max(math.sin(beta2), 0.15)
    )
    blockage_pen = max(0.0, blockage - 0.16) / 0.16

    v_out = perf.v_out
    if FAMILIES[g.family].get("plenum"):
        # Unhoused plug/plenum wheels intentionally have no defined casing
        # discharge flange velocity; do not penalize v_out=0 as if a volute
        # outlet were missing.
        v_pen = 0.0
    else:
        v_pen = max(0.0, 6.0 - v_out) / 6.0 + max(0.0, v_out - 20.0) / 20.0

    return tip_excess + pc_pen + blockage_pen + v_pen, {
        "tip_speed_ms": perf.u2,
        "tip_limit_ms": tip_lim,
        "pitch_mm": pitch,
        "chord_mm": chord,
        "pitch_chord": pc,
        "blade_blockage_fraction": blockage,
        "outlet_velocity_ms": v_out,
    }


def _geometry_prior_penalty(g: GeometryInput) -> float:
    fam = FAMILIES[g.family]
    vals = [
        (g.d1_mm / g.d2_mm, fam["d1d2"], fam["d1d2_rng"]),
        (g.b2_total_mm / g.d2_mm, fam["b2d2"], fam["b2d2_rng"]),
        (g.beta1_deg, fam["beta1"], fam["beta1_rng"]),
        (g.beta2_deg, fam["beta2"], fam["beta2_rng"]),
        (float(g.blade_count), float(fam["z"]), fam["z_rng"]),
    ]
    p = 0.0
    for value, default, rng in vals:
        span = max(float(rng[1]) - float(rng[0]), 1e-6)
        p += ((float(value) - float(default)) / span) ** 2
    return p / len(vals)


def evaluate_candidate(
    values: np.ndarray,
    specs: List[dict],
    target: InverseTarget,
    locks: InverseLocks,
    cfg: InverseConfig,
    curve_points: Optional[pd.DataFrame] = None,
) -> dict:
    params = _decode(values, specs)
    g = _candidate_geometry(params, target, locks)
    fam = FAMILIES[g.family]

    q_reference = max(target.airflow_m3h, 100.0)
    if target.basis == "Digitised Q-P curve":
        curve_targets = _target_curve_points(curve_points)
        if not curve_targets.empty:
            q_reference = float(curve_targets["airflow_m3h"].median())

    try:
        wheel, _ = build_wheel(
            g,
            q_reference_m3h=q_reference,
            density=target.density,
            rpm=target.rpm,
            casing_method="Area-law rectangular",
            scroll_velocity_fraction_cu2=params.get("scroll_velocity_fraction", 0.75),
        )
    except Exception as exc:
        return {"score": 1e9, "error": str(exc), "geometry": g, **params}

    pressure_factor = params.get("pressure_factor", cfg.endpoint_pressure_prior)
    flow_factor = params.get("flow_factor", cfg.endpoint_flow_prior)

    pressure_errors: List[float] = []
    power_errors: List[float] = []
    endpoint = {}

    if target.basis == "Catalogue max-flow / max-pressure endpoints":
        endpoint = _endpoint_metrics(wheel, target, pressure_factor, flow_factor)
        if target.airflow_m3h <= 0 or target.static_pressure_pa <= 0:
            return {"score": 1e9, "error": "Endpoint mode requires Qmax and Psmax.", "geometry": g, **params}
        pressure_errors.extend([
            (endpoint["predicted_qmax_m3h"] - target.airflow_m3h) / target.airflow_m3h,
            (endpoint["predicted_psmax_pa"] - target.static_pressure_pa) / target.static_pressure_pa,
        ])
        design_q = 0.60 * target.airflow_m3h

    elif target.basis == "Simultaneous duty point":
        if target.airflow_m3h <= 0 or target.static_pressure_pa <= 0:
            return {"score": 1e9, "error": "Duty-point mode requires airflow and pressure.", "geometry": g, **params}
        pp = _performance_at(wheel, target, target.airflow_m3h, target.rpm, pressure_factor)
        pressure_errors.append((pp.dp_static - target.static_pressure_pa) / target.static_pressure_pa)
        design_q = target.airflow_m3h

    else:
        curve_targets = _target_curve_points(curve_points)
        if len(curve_targets) < 4:
            return {"score": 1e9, "error": "Digitised curve mode needs at least 4 valid Q-P points.", "geometry": g, **params}
        for row in curve_targets.itertuples():
            rrpm = float(row.speed_rpm) if hasattr(row, "speed_rpm") and pd.notna(row.speed_rpm) else target.rpm
            pred = _performance_at(wheel, target, float(row.airflow_m3h), rrpm, pressure_factor)
            denom = max(abs(float(row.pressure_pa)), 50.0)
            pressure_errors.append((pred.dp_static - float(row.pressure_pa)) / denom)
            if cfg.use_catalogue_power and hasattr(row, "power_w") and pd.notna(row.power_w) and float(row.power_w) > 0:
                pred_input_w = pred.p_shaft / max(
                    (cfg.motor_efficiency_pct / 100.0) * (cfg.drive_efficiency_pct / 100.0),
                    0.35,
                )
                power_errors.append((pred_input_w - float(row.power_w)) / float(row.power_w))
        design_q = float(curve_targets["airflow_m3h"].median())

    design_perf = _performance_at(wheel, target, max(design_q, 1.0), target.rpm, pressure_factor)

    if cfg.use_catalogue_power and target.input_power_w > 0 and target.basis != "Digitised Q-P curve":
        pred_input_w = design_perf.p_shaft / max(
            (cfg.motor_efficiency_pct / 100.0) * (cfg.drive_efficiency_pct / 100.0),
            0.35,
        )
        power_errors.append((pred_input_w - target.input_power_w) / target.input_power_w)
    else:
        pred_input_w = design_perf.p_shaft / max(
            (cfg.motor_efficiency_pct / 100.0) * (cfg.drive_efficiency_pct / 100.0),
            0.35,
        )

    match_rmse = float(math.sqrt(np.mean(np.square(pressure_errors)))) if pressure_errors else 9.0
    power_rmse = float(math.sqrt(np.mean(np.square(power_errors)))) if power_errors else 0.0

    manu_pen, manu = _manufacturing_penalties(g, design_perf)
    prior_pen = _geometry_prior_penalty(g)

    duty = Duty(
        airflow_m3h=max(design_q, 1.0),
        static_pa=max(design_perf.dp_static, 1.0),
        rpm=target.rpm,
        temp_c=target.temp_c,
        altitude_m=target.altitude_m,
        rh_pct=target.rh_pct,
        density=target.density,
    ).finalize()
    try:
        sound = acoustics(wheel, duty, design_perf, curve=None).lpa_1m
    except Exception:
        sound = 99.0

    if cfg.use_catalogue_noise and target.noise_dba > 0:
        noise_pen = abs(sound - target.noise_dba) / 20.0
    else:
        noise_pen = max(0.0, sound - 78.0) / 20.0

    eta_floor = float(fam["eta_band"][0])
    eta_pen = max(0.0, eta_floor - design_perf.eta_static) / max(eta_floor, 0.2)
    eta_reward = max(0.0, 0.78 - design_perf.eta_static)

    correction_pen = 0.0
    if cfg.allow_empirical_model_correction:
        pp = max(cfg.endpoint_pressure_prior, 1e-6)
        correction_pen += abs(math.log(max(pressure_factor, 1e-6) / pp))
        if target.basis == "Catalogue max-flow / max-pressure endpoints":
            fp = max(cfg.endpoint_flow_prior, 1e-6)
            correction_pen += 0.7 * abs(math.log(max(flow_factor, 1e-6) / fp))

    score = (
        100.0 * cfg.pressure_match_weight * match_rmse
        + 25.0 * cfg.power_match_weight * power_rmse
        + 20.0 * cfg.efficiency_weight * (eta_pen + 0.25 * eta_reward)
        + 12.0 * cfg.noise_weight * noise_pen
        + 35.0 * cfg.manufacturability_weight * manu_pen
        + 18.0 * cfg.geometry_prior_weight * prior_pen
        + 20.0 * cfg.correction_prior_weight * correction_pen
    )

    feasible = (
        match_rmse <= 0.08
        and manu_pen <= 0.15
        and design_perf.eta_static >= 0.25
        and design_perf.dp_static > 0
    )

    row = {
        "score": score,
        "catalogue_match_rmse_pct": 100.0 * match_rmse,
        "power_match_rmse_pct": 100.0 * power_rmse if power_errors else None,
        "feasible": feasible,
        "d1_d2": g.d1_mm / g.d2_mm,
        "d1_mm": g.d1_mm,
        "b2_d2": g.b2_total_mm / g.d2_mm,
        "b2_total_mm": g.b2_total_mm,
        "b1_b2": g.b1_total_mm / max(g.b2_total_mm, 1e-6),
        "b1_total_mm": g.b1_total_mm,
        "beta1_deg": g.beta1_deg,
        "beta2_deg": g.beta2_deg,
        "blade_count": g.blade_count,
        "cutoff_ratio": g.cutoff_clearance_mm / g.d2_mm,
        "cutoff_mm": g.cutoff_clearance_mm,
        "scroll_width_mm": g.scroll_internal_width_mm,
        "discharge_width_mm": g.discharge_width_mm,
        "discharge_height_mm": g.discharge_height_mm,
        "pressure_factor": pressure_factor,
        "flow_factor": flow_factor if target.basis == "Catalogue max-flow / max-pressure endpoints" else None,
        "predicted_static_pa_at_design_q": design_perf.dp_static,
        "design_q_m3h": design_q,
        "shaft_kw_at_design_q": design_perf.p_shaft / 1000.0,
        "estimated_input_w_at_design_q": pred_input_w,
        "eta_static_at_design_q": design_perf.eta_static,
        "eta_total_at_design_q": design_perf.eta_total,
        "predicted_noise_dba": sound,
        "tip_speed_ms": manu["tip_speed_ms"],
        "tip_limit_ms": manu["tip_limit_ms"],
        "pitch_chord": manu["pitch_chord"],
        "blade_blockage_fraction": manu["blade_blockage_fraction"],
        "outlet_velocity_ms": manu["outlet_velocity_ms"],
        "incidence_loss_pa_raw": design_perf.losses.get("incidence", 0.0),
        "volute_loss_pa_raw": design_perf.losses.get("volute", 0.0),
        "geometry": g,
    }
    row.update(endpoint)
    row.update(params)
    return row


def _sample_vectors(specs: List[dict], n: int, seed: int) -> np.ndarray:
    lows = np.array([s["lo"] for s in specs], dtype=float)
    highs = np.array([s["hi"] for s in specs], dtype=float)
    sampler = qmc.LatinHypercube(d=len(specs), seed=seed)
    u = sampler.random(n=n)

    # Manual scaling is deliberate: scipy.stats.qmc.scale rejects equal
    # lower/upper bounds, while equal bounds are exactly how this inverse
    # solver represents physically measured/locked geometry.
    x = lows + u * (highs - lows)

    for j, s in enumerate(specs):
        if s["locked"]:
            x[:, j] = s["lo"]
    return x


def optimize_inverse_design(
    target: InverseTarget,
    locks: Optional[InverseLocks] = None,
    cfg: Optional[InverseConfig] = None,
    curve_points: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, dict]:
    locks = locks or InverseLocks()
    cfg = cfg or InverseConfig()

    if target.d2_mm <= 0 or target.rpm <= 0:
        raise ValueError("D2 and RPM must be positive.")
    if target.basis in ("Catalogue max-flow / max-pressure endpoints", "Simultaneous duty point"):
        if target.airflow_m3h <= 0 or target.static_pressure_pa <= 0:
            raise ValueError("Airflow and pressure are required for this target interpretation.")
    if target.basis == "Digitised Q-P curve" and len(_target_curve_points(curve_points)) < 4:
        raise ValueError("At least 4 digitised Q-P points are required.")

    specs = _variable_specs(target, locks, cfg)
    vectors = _sample_vectors(specs, cfg.n_samples, cfg.seed)

    evaluated = []
    for vec in vectors:
        evaluated.append(evaluate_candidate(vec, specs, target, locks, cfg, curve_points))

    good = [r for r in evaluated if np.isfinite(r.get("score", np.inf)) and r.get("score", 1e9) < 1e8]
    if not good:
        raise ValueError("No candidate geometry could be evaluated. Check target values and locks.")

    good.sort(key=lambda r: r["score"])
    seeds = good[: cfg.n_refine]
    bnds = _bounds(specs)

    refined = []
    for seed_row in seeds:
        x0 = np.array([seed_row[s["name"]] for s in specs], dtype=float)

        def objective(x):
            r = evaluate_candidate(x, specs, target, locks, cfg, curve_points)
            return float(r.get("score", 1e9))

        try:
            opt = minimize(
                objective,
                x0,
                method="Powell",
                bounds=bnds,
                options={"maxiter": cfg.refine_iterations, "xtol": 1e-4, "ftol": 1e-4},
            )
            rr = evaluate_candidate(opt.x, specs, target, locks, cfg, curve_points)
            rr["refined"] = True
            refined.append(rr)
        except Exception:
            pass

    all_rows = good + refined
    # De-duplicate on the key geometry variables.
    dedup = {}
    for r in all_rows:
        key = (
            round(r["d1_d2"], 5),
            round(r["b2_d2"], 5),
            round(r["b1_b2"], 5),
            round(r["beta1_deg"], 3),
            round(r["beta2_deg"], 3),
            int(r["blade_count"]),
            round(r["cutoff_ratio"], 5),
            round(r["pressure_factor"], 4),
            round(float(r.get("flow_factor") or 1.0), 4),
        )
        if key not in dedup or r["score"] < dedup[key]["score"]:
            dedup[key] = r

    rows = sorted(dedup.values(), key=lambda r: r["score"])
    public = []
    for rank, r in enumerate(rows, 1):
        x = {k: v for k, v in r.items() if k != "geometry"}
        x["rank"] = rank
        public.append(x)
    df = pd.DataFrame(public)

    best_row = rows[0]
    best = {
        "target": asdict(target),
        "locks": asdict(locks),
        "config": asdict(cfg),
        "geometry": asdict(best_row["geometry"]),
        "metrics": {k: v for k, v in best_row.items() if k != "geometry"},
        "specs": specs,
    }
    return df, best


def full_curve_for_geometry(
    target: InverseTarget,
    geometry_dict: dict,
    metrics: dict,
    n: int = 61,
    q_max_multiplier: float = 1.50,
) -> pd.DataFrame:
    g = GeometryInput(**geometry_dict)
    q_ref = max(target.airflow_m3h, 100.0)
    wheel, _ = build_wheel(
        g,
        q_reference_m3h=q_ref,
        density=target.density,
        rpm=target.rpm,
        casing_method="Area-law rectangular",
        scroll_velocity_fraction_cu2=float(metrics.get("scroll_velocity_fraction", 0.75)),
    )
    pressure_factor = float(metrics.get("pressure_factor", 1.0))
    flow_factor = float(metrics.get("flow_factor") or 1.0)

    q_end = max(q_ref * q_max_multiplier / max(flow_factor, 0.2), q_ref * 1.1)
    qs_raw = np.linspace(max(0.01 * q_ref, 0.5), q_end, n)
    rows = []
    for qraw in qs_raw:
        p = _performance_at(wheel, target, float(qraw), target.rpm, pressure_factor)
        qplot = qraw * flow_factor if target.basis == "Catalogue max-flow / max-pressure endpoints" else qraw
        rows.append({
            "airflow_m3h": qplot,
            "raw_model_airflow_m3h": qraw,
            "static_pressure_pa": p.dp_static,
            "total_pressure_pa": p.dp_total,
            "shaft_power_kw": p.p_shaft / 1000.0,
            "eta_static": p.eta_static,
            "eta_total": p.eta_total,
            "outlet_velocity_ms": p.v_out,
        })
    return pd.DataFrame(rows)


def identifiability_table(candidates: pd.DataFrame, best_score: Optional[float] = None) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    best_score = float(best_score if best_score is not None else candidates.iloc[0]["score"])
    threshold = best_score + max(2.0, 0.20 * max(best_score, 1.0))
    x = candidates[candidates["score"] <= threshold].copy()
    if len(x) < 5:
        x = candidates.head(min(20, len(candidates))).copy()

    vars_ = [
        ("D1/D2", "d1_d2"),
        ("b2/D2", "b2_d2"),
        ("b1/b2", "b1_b2"),
        ("beta1", "beta1_deg"),
        ("beta2", "beta2_deg"),
        ("Blade count", "blade_count"),
        ("Cutoff/D2", "cutoff_ratio"),
        ("Scroll width", "scroll_width_mm"),
    ]
    rows = []
    for label, col in vars_:
        if col not in x:
            continue
        s = pd.to_numeric(x[col], errors="coerce").dropna()
        if s.empty:
            continue
        lo, hi, med = float(s.min()), float(s.max()), float(s.median())
        spread = hi - lo
        rel = spread / max(abs(med), 1e-6)
        priority = "HIGH" if rel > 0.20 else ("MEDIUM" if rel > 0.08 else "LOW")
        rows.append({
            "parameter": label,
            "near_best_min": lo,
            "near_best_median": med,
            "near_best_max": hi,
            "relative_spread_pct": 100.0 * rel,
            "measurement_priority": priority,
            "interpretation": (
                "Many different values fit the catalogue similarly — measure this physically."
                if priority == "HIGH"
                else "Moderately identifiable; measurement still improves confidence."
                if priority == "MEDIUM"
                else "Near-best solutions agree reasonably well."
            ),
        })
    return pd.DataFrame(rows).sort_values("relative_spread_pct", ascending=False)


def sensitivity_table(
    target: InverseTarget,
    best: dict,
    locks: Optional[InverseLocks] = None,
    cfg: Optional[InverseConfig] = None,
    curve_points: Optional[pd.DataFrame] = None,
    perturb_fraction: float = 0.03,
) -> pd.DataFrame:
    locks = locks or InverseLocks(**best.get("locks", {}))
    cfg = cfg or InverseConfig(**best.get("config", {}))
    specs = best["specs"]
    base = np.array([float(best["metrics"].get(s["name"], s["lo"])) for s in specs])
    base_score = float(best["metrics"]["score"])

    rows = []
    for i, s in enumerate(specs):
        if s["locked"] or s["name"] in ("pressure_factor", "flow_factor"):
            continue
        span = s["hi"] - s["lo"]
        if span <= 0:
            continue
        delta = perturb_fraction * span
        scores = []
        for sign in (-1.0, 1.0):
            xx = base.copy()
            xx[i] = min(max(xx[i] + sign * delta, s["lo"]), s["hi"])
            rr = evaluate_candidate(xx, specs, target, locks, cfg, curve_points)
            scores.append(float(rr["score"]))
        sensitivity = max(scores) - base_score
        rows.append({
            "parameter": s["name"],
            "best_value": base[i],
            "minus_score": scores[0],
            "plus_score": scores[1],
            "score_increase": sensitivity,
            "sensitivity": "HIGH" if sensitivity > 4 else ("MEDIUM" if sensitivity > 1 else "LOW"),
        })
    return pd.DataFrame(rows).sort_values("score_increase", ascending=False)


def manufacturing_confidence(
    target: InverseTarget,
    locks: InverseLocks,
    candidates: pd.DataFrame,
    curve_points: Optional[pd.DataFrame] = None,
) -> dict:
    score = 0
    reasons = []

    if target.basis == "Digitised Q-P curve":
        n = len(_target_curve_points(curve_points))
        score += 45 if n >= 8 else 35
        reasons.append(f"{n} digitised Q-P points constrain the aerodynamic curve.")
    elif target.basis == "Catalogue max-flow / max-pressure endpoints":
        score += 22
        reasons.append("Only Qmax and Psmax endpoints constrain the aerodynamic curve; the inverse problem is non-unique.")
    else:
        score += 18
        reasons.append("One simultaneous duty point is insufficient to uniquely determine hidden geometry.")

    lock_fields = [
        ("D1", locks.d1_mm),
        ("b2", locks.b2_total_mm),
        ("b1", locks.b1_total_mm),
        ("blade count", locks.blade_count),
        ("beta1", locks.beta1_deg),
        ("beta2", locks.beta2_deg),
        ("cutoff", locks.cutoff_clearance_mm),
        ("scroll width", locks.scroll_internal_width_mm),
    ]
    known = [name for name, val in lock_fields if float(val or 0) > 0]
    score += min(32, len(known) * 4)
    if known:
        reasons.append("Measured/locked geometry: " + ", ".join(known) + ".")

    if candidates is not None and not candidates.empty:
        best_err = float(candidates.iloc[0]["catalogue_match_rmse_pct"])
        if best_err <= 3:
            score += 12
        elif best_err <= 8:
            score += 8
        elif best_err <= 15:
            score += 3
        reasons.append(f"Best catalogue match error ≈ {best_err:.1f}%.")

        if bool(candidates.iloc[0].get("feasible", False)):
            score += 5
            reasons.append("Best solution passes current aerodynamic/manufacturing screening.")

    score = min(score, 95)
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 50 else "D"
    message = {
        "A": "Strong reverse-engineering basis. Prototype CFD/test validation is still required.",
        "B": "Good engineering basis; verify remaining high-priority geometry before release.",
        "C": "Useful concept design, but still underdetermined. Measure the high-priority geometry.",
        "D": "Catalogue-constrained concept only. Do not manufacture yet.",
    }[grade]
    return {"score": score, "grade": grade, "message": message, "reasons": reasons}


def peer_endpoint_prior(
    fan_database: pd.DataFrame,
    selected_record: dict,
    max_peers: int = 24,
) -> dict:
    """Estimate empirical correction priors from peer catalogue models.

    The peers are NOT treated as truth. They are used to keep the mean-line
    correction factor in the scale seen across the same manufacturer's family.
    """
    if fan_database is None or fan_database.empty:
        return {"flow_factor": 1.0, "pressure_factor": 1.0, "peer_count": 0}

    family = supported_catalogue_family(selected_record)
    if family is None:
        return {"flow_factor": 1.0, "pressure_factor": 1.0, "peer_count": 0}

    maker = str(selected_record.get("manufacturer") or "")
    ftype = str(selected_record.get("fan_type") or "")
    peers = fan_database.copy()
    peers = peers[
        (peers["manufacturer"].astype(str) == maker)
        & (peers["fan_type"].astype(str) == ftype)
        & (peers["model"].astype(str) != str(selected_record.get("model") or ""))
    ]
    peers = peers.dropna(subset=["impeller_diameter_mm", "airflow_m3h", "static_pressure_pa", "rpm"])
    peers = peers[
        (peers.impeller_diameter_mm > 0)
        & (peers.airflow_m3h > 0)
        & (peers.static_pressure_pa > 0)
        & (peers.rpm > 0)
    ]
    if peers.empty:
        return {"flow_factor": 1.0, "pressure_factor": 1.0, "peer_count": 0}

    if len(peers) > max_peers:
        idx = np.linspace(0, len(peers) - 1, max_peers).astype(int)
        peers = peers.iloc[idx]

    fam = FAMILIES[family]
    flow_factors = []
    pressure_factors = []
    rho = air_density(20.0, 0.0, 50.0)

    for row in peers.to_dict("records"):
        try:
            d2 = float(row["impeller_diameter_mm"])
            qmax = float(row["airflow_m3h"])
            psmax = float(row["static_pressure_pa"])
            rpm = float(row["rpm"])
            arrangement = catalogue_arrangement(row)
            b2 = float(fam["b2d2"]) * d2
            area = (qmax / 3600.0) / 12.0
            out_w_m = max(1.10 * b2 / 1000.0, math.sqrt(area))
            out_h_m = area / out_w_m
            blade_t, plate_t = _default_thickness(d2, family)
            g = GeometryInput(
                family=family,
                arrangement=arrangement,
                d2_mm=d2,
                d1_mm=float(fam["d1d2"]) * d2,
                b2_total_mm=b2,
                b1_total_mm=1.05 * b2,
                blade_count=int(fam["z"]),
                beta1_deg=float(fam["beta1"]),
                beta2_deg=float(fam["beta2"]),
                blade_thickness_mm=blade_t,
                plate_thickness_mm=plate_t,
                hub_diameter_mm=0.15 * float(fam["d1d2"]) * d2,
                shaft_diameter_mm=max(8.0, 0.045 * d2),
                cutoff_clearance_mm=0.05 * d2,
                scroll_internal_width_mm=max(1.25 * b2, 0.9 * out_w_m * 1000),
                discharge_width_mm=out_w_m * 1000,
                discharge_height_mm=out_h_m * 1000,
                material="Galvanized Steel",
            )
            wheel, _ = build_wheel(g, 0.65 * qmax, rho, rpm, "Area-law rectangular", 0.75)

            # Raw endpoint estimate.
            qs = np.linspace(0.025 * qmax, 2.5 * qmax, 28)
            ps = [
                wheel_performance(
                    wheel, rho, rpm, q / 3600.0, visc=air_viscosity(20.0)
                ).dp_static
                for q in qs
            ]
            if ps[0] <= 1:
                continue
            qfree = None
            for i in range(1, len(qs)):
                if ps[i] <= 0 < ps[i - 1]:
                    qfree = qs[i - 1] + (0 - ps[i - 1]) * (qs[i] - qs[i - 1]) / (ps[i] - ps[i - 1])
                    break
            if qfree is None or qfree <= 0:
                continue

            ff = qmax / qfree
            pf = psmax / ps[0]
            if 0.25 <= ff <= 2.0 and 0.45 <= pf <= 12.0:
                flow_factors.append(ff)
                pressure_factors.append(pf)
        except Exception:
            continue

    if len(flow_factors) < 3 or len(pressure_factors) < 3:
        return {"flow_factor": 1.0, "pressure_factor": 1.0, "peer_count": min(len(flow_factors), len(pressure_factors))}

    return {
        "flow_factor": float(np.median(flow_factors)),
        "pressure_factor": float(np.median(pressure_factors)),
        "flow_p10": float(np.quantile(flow_factors, 0.10)),
        "flow_p90": float(np.quantile(flow_factors, 0.90)),
        "pressure_p10": float(np.quantile(pressure_factors, 0.10)),
        "pressure_p90": float(np.quantile(pressure_factors, 0.90)),
        "peer_count": min(len(flow_factors), len(pressure_factors)),
        "note": "Empirical prior from peer catalogue models; not a certified calibration.",
    }
