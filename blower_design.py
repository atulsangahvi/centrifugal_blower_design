"""
Centrifugal Blower Design & Manufacturing Toolkit - SI Units
Forward Curved, Backward Curved / Backward Inclined, and Radial Blade Blowers

This is an engineering preliminary design tool. Final designs must be verified
by prototype testing, AMCA/ISO test procedures, vibration checks, FEA/CFD, and
qualified engineering review before manufacture.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import ezdxf
    HAS_EZDXF = True
except Exception:
    HAS_EZDXF = False

try:
    import cadquery as cq
    HAS_CADQUERY = True
except Exception:
    HAS_CADQUERY = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False


# ----------------------------- Constants -----------------------------
STANDARD_MOTORS_KW = [
    0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
    7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200, 250, 315
]

BLADE_DEFAULTS = {
    "Backward Curved / Backward Inclined": {
        "beta2_deg": 38.0,
        "eta_total": 0.72,
        "phi": 0.18,
        "psi": 0.58,
        "blade_count": 12,
        "power_curve": "non_overloading",
    },
    "Forward Curved": {
        "beta2_deg": 125.0,
        "eta_total": 0.58,
        "phi": 0.16,
        "psi": 0.72,
        "blade_count": 36,
        "power_curve": "overloading",
    },
    "Radial Blade": {
        "beta2_deg": 90.0,
        "eta_total": 0.62,
        "phi": 0.12,
        "psi": 0.62,
        "blade_count": 8,
        "power_curve": "linear",
    },
}

MATERIALS = {
    "Mild Steel IS2062 / S275": {"density": 7850, "allow_stress_mpa": 90, "max_tip_ms": 90},
    "Galvanized Steel": {"density": 7850, "allow_stress_mpa": 80, "max_tip_ms": 75},
    "Stainless Steel 304": {"density": 8000, "allow_stress_mpa": 95, "max_tip_ms": 90},
    "Aluminium 6061-T6": {"density": 2700, "allow_stress_mpa": 70, "max_tip_ms": 70},
}


# ----------------------------- Data Models -----------------------------
@dataclass
class DutyInput:
    airflow_m3h: float
    static_pressure_pa: float
    total_pressure_pa: float
    air_temp_c: float
    altitude_m: float
    density_kgm3: float
    rpm: float
    blade_type: str
    drive_type: str
    motor_eff: float
    drive_eff: float
    design_margin: float
    beta2_deg: float
    beta1_deg: float
    blade_count: int
    outlet_width_ratio: float
    inlet_diameter_ratio: float
    material: str
    blade_thickness_mm: float
    casing_thickness_mm: float
    shaft_allow_shear_mpa: float

@dataclass
class DesignResult:
    q_m3s: float
    density_kgm3: float
    total_pressure_pa: float
    static_pressure_pa: float
    velocity_pressure_pa: float
    rpm: float
    omega: float
    tip_speed_ms: float
    impeller_od_mm: float
    impeller_id_mm: float
    outlet_width_mm: float
    inlet_width_mm: float
    beta1_deg: float
    beta2_deg: float
    blade_count: int
    slip_factor: float
    flow_coeff_phi: float
    pressure_coeff_psi: float
    theoretical_pressure_pa: float
    estimated_total_eff: float
    air_power_kw: float
    shaft_power_kw: float
    motor_input_kw: float
    selected_motor_kw: float
    specific_speed_metric: float
    specific_diameter_metric: float
    outlet_area_m2: float
    outlet_velocity_ms: float
    volute_outlet_width_mm: float
    volute_outlet_height_mm: float
    volute_cutoff_clearance_mm: float
    shaft_diameter_mm: float
    approx_impeller_mass_kg: float
    warnings: List[str]


# ----------------------------- Core Calculations -----------------------------
def standard_air_density(temp_c: float, altitude_m: float) -> float:
    """Approximate dry air density using ISA troposphere."""
    t0 = 288.15
    p0 = 101325.0
    lapse = 0.0065
    r = 287.05
    g = 9.80665
    temp_k_alt = t0 - lapse * altitude_m
    pressure = p0 * (temp_k_alt / t0) ** (g / (r * lapse))
    temp_actual_k = temp_c + 273.15
    return pressure / (r * temp_actual_k)


def selected_motor(power_kw: float) -> float:
    for m in STANDARD_MOTORS_KW:
        if m >= power_kw:
            return m
    return STANDARD_MOTORS_KW[-1]


def stodola_slip_factor(z: int, beta2_deg: float, d1_d2: float) -> float:
    """Practical Stodola-style slip estimate. Conservative bounded value."""
    beta = math.radians(beta2_deg)
    # Works reasonably for preliminary centrifugal impeller sizing.
    sigma = 1.0 - (math.pi * math.sin(beta)) / max(z, 3)
    # Mild correction for high inlet diameter ratio.
    sigma *= max(0.88, min(1.02, 1.0 - 0.10 * (d1_d2 - 0.5)))
    return max(0.55, min(0.95, sigma))


def design_blower(inp: DutyInput) -> DesignResult:
    warnings: List[str] = []
    q = inp.airflow_m3h / 3600.0
    rho = inp.density_kgm3 if inp.density_kgm3 > 0 else standard_air_density(inp.air_temp_c, inp.altitude_m)
    defaults = BLADE_DEFAULTS[inp.blade_type]
    phi = defaults["phi"]
    psi = defaults["psi"]
    eta = defaults["eta_total"]

    # User enters STATIC pressure only. The app estimates outlet velocity pressure
    # from a practical discharge velocity range, then calculates total pressure.
    discharge_velocity = 8.0 if inp.static_pressure_pa < 600 else 10.0 if inp.static_pressure_pa < 1200 else 12.0 if inp.static_pressure_pa < 2200 else 16.0
    velocity_pressure_est = 0.5 * rho * discharge_velocity**2
    total_p = inp.static_pressure_pa + velocity_pressure_est

    omega = 2.0 * math.pi * inp.rpm / 60.0
    # Euler/dimensionless pressure coefficient: DeltaP = psi * rho * U2^2
    u2 = math.sqrt(total_p / max(psi * rho, 1e-9))
    d2 = 2.0 * u2 / omega
    d1 = inp.inlet_diameter_ratio * d2
    b2 = inp.outlet_width_ratio * d2

    # Check continuity at impeller outlet and adjust b2 if very unrealistic.
    outlet_area = math.pi * d2 * b2
    cm2 = q / max(outlet_area, 1e-9)
    cm2_target = phi * u2
    if cm2 > 1.35 * cm2_target:
        b2 = q / (math.pi * d2 * cm2_target)
        warnings.append("Outlet width increased to keep meridional velocity within preliminary design range.")
    elif cm2 < 0.55 * cm2_target:
        warnings.append("Outlet width may be high for the selected duty; check efficiency and casing width.")

    # Recalculate outlet area and meridional/outlet velocity after any automatic width adjustment.
    outlet_area = math.pi * d2 * b2
    cm2 = q / max(outlet_area, 1e-9)

    b1 = 1.15 * b2
    sigma = stodola_slip_factor(inp.blade_count, inp.beta2_deg, d1 / d2)
    theoretical_p = total_p / max(sigma, 1e-6)
    velocity_pressure = max(total_p - inp.static_pressure_pa, 0.0)

    air_power = q * total_p / 1000.0
    shaft_power = air_power / max(eta * inp.drive_eff, 1e-6)
    motor_input = shaft_power / max(inp.motor_eff, 1e-6)
    motor_required = motor_input * (1.0 + inp.design_margin / 100.0)
    motor_kw = selected_motor(motor_required)

    # Fan specific speed / diameter, SI non-dimensional style using pressure head H = dp/rho/g
    h_m = total_p / (rho * 9.80665)
    n_rps = inp.rpm / 60.0
    ns = n_rps * math.sqrt(q) / max(h_m ** 0.75, 1e-9)
    ds = d2 * max(h_m ** 0.25, 1e-9) / max(math.sqrt(q), 1e-9)

    # Volute preliminary sizing
    volute_outlet_area = q / discharge_velocity
    volute_outlet_width = 1.25 * b2
    volute_outlet_height = volute_outlet_area / max(volute_outlet_width, 1e-9)
    cutoff_clearance = 0.06 * d2

    # Shaft torque and diameter by torsion only, preliminary
    torque_nm = (shaft_power * 1000.0) / max(omega, 1e-9)
    tau_allow = inp.shaft_allow_shear_mpa * 1e6
    shaft_d = ((16.0 * torque_nm) / (math.pi * tau_allow)) ** (1.0 / 3.0)
    shaft_d *= 1000.0 * 1.35  # service factor
    shaft_d = max(20.0, math.ceil(shaft_d / 5.0) * 5.0)

    mat = MATERIALS[inp.material]
    max_tip = mat["max_tip_ms"]
    if u2 > max_tip:
        warnings.append(f"Tip speed {u2:.1f} m/s exceeds preliminary limit for {inp.material} ({max_tip} m/s).")
    if inp.blade_type == "Forward Curved" and total_p > 1200:
        warnings.append("Forward-curved fans are generally better for lower pressure AHU duties; check non-overload and stability carefully.")
    if inp.blade_type == "Radial Blade" and eta < 0.65:
        warnings.append("Radial blades are robust for dust/dirty air but usually lower efficiency and noisier than backward-curved blades.")
    if q <= 0 or total_p <= 0 or inp.rpm <= 0:
        warnings.append("Invalid duty inputs. Airflow, pressure and RPM must be positive.")

    # Rough impeller mass estimate: two discs + blades + hub factor
    t = inp.blade_thickness_mm / 1000.0
    disc_t = max(t, 0.003)
    disc_area = math.pi * (d2**2 - (0.35*d1)**2) / 4.0
    blade_area = (d2 - d1) * b2 * 1.25
    mass = mat["density"] * (2 * disc_area * disc_t + inp.blade_count * blade_area * t) * 1.25

    return DesignResult(
        q_m3s=q,
        density_kgm3=rho,
        total_pressure_pa=total_p,
        static_pressure_pa=inp.static_pressure_pa,
        velocity_pressure_pa=velocity_pressure,
        rpm=inp.rpm,
        omega=omega,
        tip_speed_ms=u2,
        impeller_od_mm=d2 * 1000.0,
        impeller_id_mm=d1 * 1000.0,
        outlet_width_mm=b2 * 1000.0,
        inlet_width_mm=b1 * 1000.0,
        beta1_deg=inp.beta1_deg,
        beta2_deg=inp.beta2_deg,
        blade_count=inp.blade_count,
        slip_factor=sigma,
        flow_coeff_phi=phi,
        pressure_coeff_psi=psi,
        theoretical_pressure_pa=theoretical_p,
        estimated_total_eff=eta,
        air_power_kw=air_power,
        shaft_power_kw=shaft_power,
        motor_input_kw=motor_input,
        selected_motor_kw=motor_kw,
        specific_speed_metric=ns,
        specific_diameter_metric=ds,
        outlet_area_m2=outlet_area,
        outlet_velocity_ms=q / max(outlet_area, 1e-9),
        volute_outlet_width_mm=volute_outlet_width * 1000.0,
        volute_outlet_height_mm=volute_outlet_height * 1000.0,
        volute_cutoff_clearance_mm=cutoff_clearance * 1000.0,
        shaft_diameter_mm=shaft_d,
        approx_impeller_mass_kg=mass,
        warnings=warnings,
    )


# ----------------------------- Geometry -----------------------------
def blade_centerline_points(res: DesignResult, n: int = 60) -> List[Tuple[float, float]]:
    r1 = res.impeller_id_mm / 2.0
    r2 = res.impeller_od_mm / 2.0
    beta1 = math.radians(res.beta1_deg)
    beta2 = math.radians(res.beta2_deg)
    rs = np.linspace(r1, r2, n)
    # Log-spiral inspired blade angle interpolation
    theta = np.zeros_like(rs)
    for i in range(1, n):
        r_mid = 0.5 * (rs[i] + rs[i - 1])
        beta = beta1 + (beta2 - beta1) * ((r_mid - r1) / max(r2 - r1, 1e-9))
        dr = rs[i] - rs[i - 1]
        # dtheta/dr = cot(beta)/r ; sign changes for forward/backward visual convention
        theta[i] = theta[i - 1] + (1.0 / math.tan(max(0.05, beta))) * dr / max(r_mid, 1e-9)
    # rotate so blade starts near x-axis
    pts = [(float(r * math.cos(th)), float(r * math.sin(th))) for r, th in zip(rs, theta)]
    return pts


def volute_spiral_points(res: DesignResult, n: int = 120) -> List[Tuple[float, float]]:
    r_imp = res.impeller_od_mm / 2.0
    c = res.volute_cutoff_clearance_mm
    # Area grows with angle; approximate radius growth using casing width.
    width = max(res.volute_outlet_width_mm, 1.0)
    outlet_area_mm2 = res.volute_outlet_width_mm * res.volute_outlet_height_mm
    pts = []
    for th in np.linspace(math.radians(10), math.radians(360), n):
        area = outlet_area_mm2 * (th / (2 * math.pi))
        radial_growth = area / max(width, 1.0)
        r = r_imp + c + radial_growth
        pts.append((float(r * math.cos(th)), float(r * math.sin(th))))
    return pts


def create_dxf(res: DesignResult) -> bytes:
    if not HAS_EZDXF:
        return b"Install ezdxf to generate DXF files."
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    r2 = res.impeller_od_mm / 2.0
    r1 = res.impeller_id_mm / 2.0
    hub_r = max(res.shaft_diameter_mm * 1.6, r1 * 0.25)
    msp.add_circle((0, 0), r2, dxfattribs={"layer": "Impeller_OD"})
    msp.add_circle((0, 0), r1, dxfattribs={"layer": "Impeller_Inlet"})
    msp.add_circle((0, 0), hub_r, dxfattribs={"layer": "Hub"})

    blade = blade_centerline_points(res)
    for k in range(res.blade_count):
        ang = 2 * math.pi * k / res.blade_count
        ca, sa = math.cos(ang), math.sin(ang)
        pts = [(x * ca - y * sa, x * sa + y * ca) for x, y in blade]
        msp.add_lwpolyline(pts, dxfattribs={"layer": "Blade_Centerlines"})

    volute = volute_spiral_points(res)
    msp.add_lwpolyline(volute, dxfattribs={"layer": "Volute_Spiral"})
    # outlet rectangle at end of spiral
    w = res.volute_outlet_width_mm
    h = res.volute_outlet_height_mm
    x0 = r2 + res.volute_cutoff_clearance_mm
    y0 = r2 + h * 0.2
    msp.add_lwpolyline([(x0, y0), (x0 + h, y0), (x0 + h, y0 + w), (x0, y0 + w), (x0, y0)], dxfattribs={"layer": "Volute_Outlet"})

    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def create_step_placeholder(res: DesignResult) -> bytes:
    if not HAS_CADQUERY:
        text = """STEP export needs cadquery installed locally or on the server.\n\nRun:\npip install cadquery\n\nThe app already contains a CAD export hook. For Ansys, export separate bodies:\n- impeller_solid.step\n- volute_solid.step\n- fluid_domain.step\n"""
        return text.encode("utf-8")
    # Basic impeller disk solid placeholder; proper blade solids will be expanded in next iteration.
    d2 = res.impeller_od_mm
    d1 = res.impeller_id_mm
    width = res.outlet_width_mm
    model = cq.Workplane("XY").circle(d2/2).circle(d1/2).extrude(width)
    bio = io.BytesIO()
    cq.exporters.export(model, bio, exportType="STEP")
    return bio.getvalue()


def input_table(inp: DutyInput) -> pd.DataFrame:
    """User-facing inputs. Deliberately excludes total_pressure_pa because user enters static pressure only."""
    rows = [
        ["Airflow", f"{inp.airflow_m3h:,.0f} m³/h"],
        ["Static pressure", f"{inp.static_pressure_pa:,.0f} Pa"],
        ["Fan speed", f"{inp.rpm:,.0f} RPM"],
        ["Blade type", inp.blade_type],
        ["Air temperature", f"{inp.air_temp_c:.1f} °C"],
        ["Altitude", f"{inp.altitude_m:.0f} m"],
        ["Air density used", f"{inp.density_kgm3:.3f} kg/m³"],
        ["Drive type", inp.drive_type],
        ["Drive efficiency", f"{inp.drive_eff:.3f}"],
        ["Motor efficiency", f"{inp.motor_eff:.3f}"],
        ["Motor design margin", f"{inp.design_margin:.1f}%"],
        ["Outlet blade angle β₂", f"{inp.beta2_deg:.1f}°"],
        ["Inlet blade angle β₁", f"{inp.beta1_deg:.1f}°"],
        ["Number of blades", f"{inp.blade_count}"],
        ["Outlet width ratio b₂/D₂", f"{inp.outlet_width_ratio:.3f}"],
        ["Inlet diameter ratio D₁/D₂", f"{inp.inlet_diameter_ratio:.3f}"],
        ["Material", inp.material],
        ["Blade/disc thickness", f"{inp.blade_thickness_mm:.1f} mm"],
        ["Casing thickness", f"{inp.casing_thickness_mm:.1f} mm"],
        ["Allowable shaft shear stress", f"{inp.shaft_allow_shear_mpa:.1f} MPa"],
    ]
    return pd.DataFrame(rows, columns=["Input", "Value"])


def engineering_recommendations(inp: DutyInput, res: DesignResult) -> pd.DataFrame:
    """Actionable guidance so the user can change inputs and improve the design."""
    rows = []
    def add(severity, issue, recommendation, input_to_change):
        rows.append({"Severity": severity, "Issue / Observation": issue, "Recommended corrective action": recommendation, "Input to change": input_to_change})

    d1d2 = res.impeller_id_mm / max(res.impeller_od_mm, 1e-9)
    b2d2 = res.outlet_width_mm / max(res.impeller_od_mm, 1e-9)
    pc = blade_pitch_chord_ratio(res)

    if b2d2 > 0.35:
        add("High", f"b₂/D₂ = {b2d2:.2f}, wheel is very wide for a single inlet/single width blower.", "Increase RPM or use larger wheel arrangement/DIDW/twin fans; reduce b₂/D₂ target only if outlet velocity remains acceptable.", "RPM, b₂/D₂, fan quantity")
    elif b2d2 > 0.25:
        add("Medium", f"b₂/D₂ = {b2d2:.2f}, wider than common preliminary range.", "Check casing width, bearing overhang and manufacturing stiffness; consider a higher RPM or two smaller blowers.", "RPM, b₂/D₂")

    if res.outlet_velocity_ms > 18:
        add("High", f"Outlet velocity = {res.outlet_velocity_ms:.1f} m/s, likely noisy and high duct loss.", "Increase outlet flange area; allow lower discharge velocity; reduce RPM only after checking diameter.", "Outlet velocity target / flange size")
    elif res.outlet_velocity_ms > 16:
        add("Medium", f"Outlet velocity = {res.outlet_velocity_ms:.1f} m/s.", "Try a larger outlet flange or wider volute discharge.", "Flange size")
    elif res.outlet_velocity_ms < 7:
        add("Low", f"Outlet velocity = {res.outlet_velocity_ms:.1f} m/s, low for compact blower.", "This is quieter but physically larger; acceptable if space allows.", "Flange size / RPM")

    if pc < 0.60:
        add("High", f"Blade pitch/chord = {pc:.2f}, blades are crowded.", "Reduce blade count or increase impeller diameter; crowded blades increase blockage, noise and fouling risk.", "Number of blades, RPM")
    elif pc < 0.70:
        add("Medium", f"Blade pitch/chord = {pc:.2f}.", "Slightly reduce blade count or check blade thickness/blockage.", "Number of blades")
    elif pc > 1.50:
        add("Medium", f"Blade pitch/chord = {pc:.2f}, blade guidance may be weak.", "Increase blade count or use a better curved blade profile to reduce slip.", "Number of blades")

    if d1d2 < 0.45:
        add("Medium", f"D₁/D₂ = {d1d2:.2f}, inlet is small.", "Increase inlet ratio to reduce inlet velocity and entry loss, but re-check pressure capability.", "D₁/D₂")
    elif d1d2 > 0.70:
        add("Medium", f"D₁/D₂ = {d1d2:.2f}, inlet is large.", "Reduce inlet ratio if pressure generation or blade length becomes weak.", "D₁/D₂")

    if inp.blade_type == "Forward Curved" and res.static_pressure_pa > 1200:
        add("High", "Forward-curved fan selected for relatively high static pressure.", "Prefer backward-curved/backward-inclined for higher pressure, non-overloading power and better efficiency.", "Blade type")
    if inp.blade_type == "Radial Blade" and inp.airflow_m3h > 25000:
        add("Medium", "Radial blade at large airflow may be noisy and less efficient.", "Use radial only for dust/dirty air; otherwise evaluate backward-curved.", "Blade type")

    snd = estimate_sound_db(inp, res)
    if snd > 95:
        add("High", f"Estimated sound = {snd:.1f} dB(A) at 1 m.", "Reduce tip speed/RPM, use backward-curved blade, lower outlet velocity, add silencers/acoustic lining.", "RPM, blade type, outlet velocity")
    elif snd > 85:
        add("Medium", f"Estimated sound = {snd:.1f} dB(A) at 1 m.", "Review acoustic treatment and duct velocity; avoid operating near unstable region.", "RPM, outlet velocity")

    risk, notes = vibration_risk(inp, res)
    if risk == "High":
        add("High", "Preliminary vibration risk is high.", "Increase cutoff clearance, reduce tip speed/outlet velocity, improve blade pitch, specify dynamic balancing and stiff base frame.", "RPM, blade count, cutoff/design")
    elif risk == "Medium":
        add("Medium", "Preliminary vibration risk is medium.", "Specify dynamic balancing, check bearing span/overhung load, and avoid blade passing frequency resonance.", "Mechanical design")

    mat = MATERIALS[inp.material]
    if res.tip_speed_ms > 0.9 * mat["max_tip_ms"]:
        add("High", f"Tip speed is near material preliminary limit: {res.tip_speed_ms:.1f} m/s.", "Use stronger material, reduce RPM, or increase pressure by improved geometry instead of speed.", "RPM, material")

    if not rows:
        add("OK", "No major preliminary red flags.", "Proceed to detailed CAD, CFD/FEA, prototype testing and AMCA/ISO style validation.", "—")
    return pd.DataFrame(rows)


def plot_impeller_volute_png(res: DesignResult) -> bytes:
    blade = blade_centerline_points(res)
    vol = volute_spiral_points(res)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    r2 = res.impeller_od_mm / 2
    r1 = res.impeller_id_mm / 2
    hub = max(res.shaft_diameter_mm * 1.6, r1 * 0.25)
    ax.add_patch(plt.Circle((0,0), r2, fill=False, linewidth=2))
    ax.add_patch(plt.Circle((0,0), r1, fill=False, linestyle='--'))
    ax.add_patch(plt.Circle((0,0), hub, fill=False, linestyle=':'))
    for k in range(res.blade_count):
        ang = 2 * math.pi * k / res.blade_count
        ca, sa = math.cos(ang), math.sin(ang)
        pts = np.array([(x*ca-y*sa, x*sa+y*ca) for x,y in blade])
        ax.plot(pts[:,0], pts[:,1], linewidth=0.8)
    vol_np = np.array(vol)
    ax.plot(vol_np[:,0], vol_np[:,1], linewidth=2.0)
    # Dimension lines / labels
    ax.annotate(f"D₂ OD = {res.impeller_od_mm:.0f} mm", xy=(-r2, -r2*1.12), xytext=(r2, -r2*1.12), arrowprops=dict(arrowstyle='<->'), ha='center')
    ax.text(0, -r2*1.22, f"D₁ inlet = {res.impeller_id_mm:.0f} mm | b₂ = {res.outlet_width_mm:.0f} mm | blades = {res.blade_count}", ha='center')
    ax.text(0, r2*1.15, f"Outlet flange {res.volute_outlet_width_mm:.0f} × {res.volute_outlet_height_mm:.0f} mm | Vout {res.outlet_velocity_ms:.1f} m/s", ha='center')
    lim = max(np.max(np.abs(vol_np))*1.08, r2*1.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('mm'); ax.set_ylabel('mm'); ax.grid(True)
    ax.set_title('Impeller + volute preliminary layout')
    bio = io.BytesIO(); fig.tight_layout(); fig.savefig(bio, format='png', dpi=180); plt.close(fig); return bio.getvalue()


def plot_blade_profile_png(res: DesignResult) -> bytes:
    blade = np.array(blade_centerline_points(res))
    r_mid = 0.25 * (res.impeller_od_mm + res.impeller_id_mm)
    pitch = math.pi * (2 * r_mid) / max(res.blade_count, 1)
    chord = 0.55 * (res.impeller_od_mm - res.impeller_id_mm)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(blade[:,0], blade[:,1], linewidth=3)
    ax.scatter([blade[0,0], blade[-1,0]], [blade[0,1], blade[-1,1]])
    ax.text(blade[0,0], blade[0,1], f" inlet β₁={res.beta1_deg:.1f}°", va='bottom')
    ax.text(blade[-1,0], blade[-1,1], f" outlet β₂={res.beta2_deg:.1f}°", va='bottom')
    # pitch indicator at mid radius
    ax.annotate(f"Pitch at mean radius ≈ {pitch:.0f} mm", xy=(blade[len(blade)//2,0], blade[len(blade)//2,1]), xytext=(blade[len(blade)//2,0], blade[len(blade)//2,1]+pitch*0.45), arrowprops=dict(arrowstyle='->'))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True); ax.set_xlabel('mm'); ax.set_ylabel('mm')
    ax.set_title(f"Single blade centreline | chord≈{chord:.0f} mm | pitch/chord={pitch/max(chord,1):.2f}")
    bio = io.BytesIO(); fig.tight_layout(); fig.savefig(bio, format='png', dpi=180); plt.close(fig); return bio.getvalue()


def plot_performance_curve_png(res: DesignResult) -> bytes:
    curve = performance_curve(res)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve['Flow_m3h'], curve['TotalPressure_Pa'], label='Total pressure Pa')
    ax2 = ax.twinx()
    ax2.plot(curve['Flow_m3h'], curve['ShaftPower_kW'], linestyle='--', label='Shaft power kW')
    ax.set_xlabel('Flow (m³/h)'); ax.set_ylabel('Pressure (Pa)'); ax2.set_ylabel('Power (kW)')
    ax.grid(True); ax.set_title('Preliminary fan curve')
    bio = io.BytesIO(); fig.tight_layout(); fig.savefig(bio, format='png', dpi=180); plt.close(fig); return bio.getvalue()


def export_manifest_df(inp: DutyInput, res: DesignResult) -> pd.DataFrame:
    return pd.DataFrame([
        ["blower_design_report.pdf", "Complete PDF report including duty, dimensions, sound/vibration, recommendations and plots"],
        ["blower_design_calculations.xlsx", "Inputs, results, practicality checks, recommendations, sound/vibration, fan curve and BOM"],
        ["blower_2d_manufacturing.dxf", "Impeller OD/ID/hub, blade centre lines, volute spiral and outlet rectangle"],
        ["impeller_volute_preview.png", "Same impeller/volute sketch shown on app interface"],
        ["single_blade_profile.png", "Blade angle and pitch/chord visual check"],
        ["fan_curve.png", "Pressure and power curve preview"],
        ["impeller_basic.step_or_instruction.txt", "STEP placeholder or CADQuery instruction depending on server support"],
        ["README_ANSYS_WORKFLOW.txt", "CFD workflow and boundary condition notes"],
    ], columns=["Package file", "Purpose"])


# ----------------------------- Reports -----------------------------
def make_results_df(res: DesignResult) -> pd.DataFrame:
    rows = []
    for k, v in asdict(res).items():
        if k == "warnings":
            continue
        rows.append({"Parameter": k, "Value": v})
    return pd.DataFrame(rows)


def create_excel(inp: DutyInput, res: DesignResult) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        input_table(inp).to_excel(writer, sheet_name="Inputs", index=False)
        make_results_df(res).to_excel(writer, sheet_name="Results", index=False)
        practicality_table(inp, res).to_excel(writer, sheet_name="Practicality", index=False)
        engineering_recommendations(inp, res).to_excel(writer, sheet_name="Recommendations", index=False)
        snd = estimate_sound_db(inp, res)
        risk, notes = vibration_risk(inp, res)
        pd.DataFrame([
            ["Estimated sound at 1 m", f"{snd:.1f} dB(A)", "Preliminary only; not AMCA 300 certified"],
            ["Vibration risk", risk, "Preliminary risk flag"],
            ["Outlet velocity", f"{res.outlet_velocity_ms:.1f} m/s", "Use to correct flange/duct sizing"],
            ["Outlet flange width", f"{res.volute_outlet_width_mm:.0f} mm", "Preliminary volute discharge"],
            ["Outlet flange height", f"{res.volute_outlet_height_mm:.0f} mm", "Preliminary volute discharge"],
        ] + [["Vibration note", n, ""] for n in notes], columns=["Item", "Value", "Note"]).to_excel(writer, sheet_name="Sound_Vibration", index=False)
        performance_curve(res).to_excel(writer, sheet_name="Performance Curve", index=False)
        bom_table(inp, res).to_excel(writer, sheet_name="BOM", index=False)
        export_manifest_df(inp, res).to_excel(writer, sheet_name="Package Manifest", index=False)
    return bio.getvalue()

def create_pdf(inp: DutyInput, res: DesignResult) -> bytes:
    if not HAS_REPORTLAB:
        return b"Install reportlab to generate PDF reports."
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Centrifugal Blower Preliminary Design Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("SI units only. User enters static pressure only; total pressure is calculated as static pressure plus estimated outlet velocity pressure. Preliminary design for engineering review and prototype validation.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    def add_table(df, widths=None):
        data = [list(df.columns)] + df.astype(str).values.tolist()
        tbl = Table(data, colWidths=widths)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTSIZE", (0,0), (-1,-1), 7),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    key_df = pd.DataFrame([
        ["Blade type", inp.blade_type],
        ["Airflow", f"{inp.airflow_m3h:,.0f} m³/h ({res.q_m3s:.3f} m³/s)"],
        ["Static pressure entered", f"{res.static_pressure_pa:.0f} Pa"],
        ["Velocity pressure calculated", f"{res.velocity_pressure_pa:.0f} Pa"],
        ["Total pressure calculated", f"{res.total_pressure_pa:.0f} Pa"],
        ["Outlet velocity", f"{res.outlet_velocity_ms:.1f} m/s"],
        ["Density", f"{res.density_kgm3:.3f} kg/m³"],
        ["RPM", f"{res.rpm:.0f}"],
        ["Impeller OD D₂", f"{res.impeller_od_mm:.1f} mm"],
        ["Inlet diameter D₁", f"{res.impeller_id_mm:.1f} mm"],
        ["Outlet width b₂", f"{res.outlet_width_mm:.1f} mm"],
        ["β₁ / β₂", f"{res.beta1_deg:.1f}° / {res.beta2_deg:.1f}°"],
        ["Blades", str(res.blade_count)],
        ["Outlet flange W × H", f"{res.volute_outlet_width_mm:.0f} × {res.volute_outlet_height_mm:.0f} mm"],
        ["Tip speed", f"{res.tip_speed_ms:.1f} m/s"],
        ["Estimated sound", f"{estimate_sound_db(inp, res):.1f} dB(A) at 1 m"],
        ["Vibration risk", vibration_risk(inp, res)[0]],
        ["Shaft power", f"{res.shaft_power_kw:.2f} kW"],
        ["Selected motor", f"{res.selected_motor_kw:.2f} kW"],
        ["Shaft diameter", f"{res.shaft_diameter_mm:.0f} mm"],
    ], columns=["Parameter", "Value"])
    add_table(key_df, [190, 330])

    story.append(Paragraph("Practicality checks", styles["Heading2"]))
    add_table(practicality_table(inp, res), [100, 70, 95, 45, 210])

    story.append(Paragraph("Engineering recommendations", styles["Heading2"]))
    add_table(engineering_recommendations(inp, res), [45, 160, 220, 90])

    risk, risk_notes = vibration_risk(inp, res)
    if res.warnings or risk_notes:
        story.append(Paragraph("Warnings / Vibration Notes", styles["Heading2"]))
        for w in res.warnings:
            story.append(Paragraph(f"• {w}", styles["BodyText"]))
        for n in risk_notes:
            story.append(Paragraph(f"• {n}", styles["BodyText"]))
        story.append(Spacer(1, 8))

    for title, img_bytes in [
        ("Impeller and volute sketch", plot_impeller_volute_png(res)),
        ("Single blade profile", plot_blade_profile_png(res)),
        ("Preliminary fan curve", plot_performance_curve_png(res)),
    ]:
        story.append(Paragraph(title, styles["Heading2"]))
        img = Image(io.BytesIO(img_bytes), width=500, height=330 if "Impeller" in title else 260)
        story.append(img)
        story.append(Spacer(1, 8))
    doc.build(story)
    return bio.getvalue()

def bom_table(inp: DutyInput, res: DesignResult) -> pd.DataFrame:
    return pd.DataFrame([
        {"Item": "Impeller back plate", "Material": inp.material, "Thickness mm": inp.blade_thickness_mm, "Approx Qty": "1"},
        {"Item": "Impeller shroud/front ring", "Material": inp.material, "Thickness mm": inp.blade_thickness_mm, "Approx Qty": "1"},
        {"Item": "Blades", "Material": inp.material, "Thickness mm": inp.blade_thickness_mm, "Approx Qty": res.blade_count},
        {"Item": "Volute casing", "Material": inp.material, "Thickness mm": inp.casing_thickness_mm, "Approx Qty": "1 set"},
        {"Item": "Shaft", "Material": "EN8/C45 or equivalent", "Thickness mm": "-", "Approx Qty": f"Ø{res.shaft_diameter_mm:.0f} mm preliminary"},
        {"Item": "Motor", "Material": "IE3/IE4 TEFC", "Thickness mm": "-", "Approx Qty": f"{res.selected_motor_kw:.2f} kW"},
    ])


def performance_curve(res: DesignResult) -> pd.DataFrame:
    flows = np.linspace(0.35, 1.25, 25) * res.q_m3s
    rows = []
    for q in flows:
        x = q / res.q_m3s
        # Generic preliminary curves by fan type behavior
        pressure = res.total_pressure_pa * max(0.05, 1.18 - 0.18*x - 0.18*x*x)
        eff = res.estimated_total_eff * max(0.15, 1.0 - 1.8*(x-1.0)**2)
        power = q * pressure / 1000.0 / max(eff, 0.05)
        rows.append({"Flow_m3s": q, "Flow_m3h": q*3600, "TotalPressure_Pa": pressure, "Efficiency": eff, "ShaftPower_kW": power})
    return pd.DataFrame(rows)


def make_zip(inp: DutyInput, res: DesignResult) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("blower_design_report.pdf", create_pdf(inp, res))
        z.writestr("blower_design_calculations.xlsx", create_excel(inp, res))
        z.writestr("blower_2d_manufacturing.dxf", create_dxf(res))
        z.writestr("impeller_volute_preview.png", plot_impeller_volute_png(res))
        z.writestr("single_blade_profile.png", plot_blade_profile_png(res))
        z.writestr("fan_curve.png", plot_performance_curve_png(res))
        z.writestr("impeller_basic.step_or_instruction.txt", create_step_placeholder(res))
        z.writestr("README_ANSYS_WORKFLOW.txt", ansys_workflow_text())
        z.writestr("PACKAGE_MANIFEST.csv", export_manifest_df(inp, res).to_csv(index=False))
    return bio.getvalue()



def estimate_sound_db(inp: DutyInput, res: DesignResult) -> float:
    """Very preliminary A-weighted sound pressure estimate at 1 m.
    This is NOT an AMCA 300 certified sound prediction. It is useful only for
    early comparison of options and for flagging noisy selections.
    """
    base = 62.0
    power_term = 10.0 * math.log10(max(res.shaft_power_kw, 0.1))
    tip_term = 25.0 * math.log10(max(res.tip_speed_ms, 1.0) / 35.0)
    pressure_term = 6.0 * math.log10(max(res.total_pressure_pa, 100.0) / 500.0)
    outlet_term = max(0.0, res.outlet_velocity_ms - 12.0) * 0.9
    blade_corr = {
        "Backward Curved / Backward Inclined": -3.0,
        "Forward Curved": 2.0,
        "Radial Blade": 5.0,
    }.get(inp.blade_type, 0.0)
    pitch_chord = blade_pitch_chord_ratio(res)
    pitch_corr = 4.0 if pitch_chord < 0.55 else 2.0 if pitch_chord < 0.70 else 0.0
    return base + power_term + tip_term + pressure_term + outlet_term + blade_corr + pitch_corr


def blade_pitch_chord_ratio(res: DesignResult) -> float:
    r_mid = 0.25 * (res.impeller_od_mm + res.impeller_id_mm)
    pitch = math.pi * (2 * r_mid) / max(res.blade_count, 1)
    chord = 0.55 * (res.impeller_od_mm - res.impeller_id_mm)
    return pitch / max(chord, 1e-6)


def vibration_risk(inp: DutyInput, res: DesignResult) -> Tuple[str, List[str]]:
    notes = []
    score = 0
    if res.tip_speed_ms > 70:
        score += 2; notes.append("High tip speed increases sensitivity to balance error.")
    elif res.tip_speed_ms > 55:
        score += 1; notes.append("Moderate-high tip speed: dynamic balancing is important.")
    if inp.blade_type == "Forward Curved":
        score += 1; notes.append("Forward-curved wheels have many narrow passages; dust buildup can create unbalance.")
    if inp.blade_type == "Radial Blade":
        score += 1; notes.append("Radial blades create more pulsating discharge and usually higher vibration/noise risk.")
    if blade_pitch_chord_ratio(res) < 0.60:
        score += 1; notes.append("Blade pitch appears tight; watch blockage and blade-passing noise.")
    if res.outlet_velocity_ms > 16:
        score += 1; notes.append("High outlet velocity can cause duct turbulence and vibration.")
    if res.volute_cutoff_clearance_mm < 0.05 * res.impeller_od_mm:
        score += 1; notes.append("Small cutoff clearance can increase blade-passing pulsation.")
    if not notes:
        notes.append("No major vibration risk flags from preliminary geometry. Still balance to ISO/AMCA practice.")
    risk = "Low" if score <= 1 else "Medium" if score <= 3 else "High"
    return risk, notes


def practicality_table(inp: DutyInput, res: DesignResult) -> pd.DataFrame:
    pc = blade_pitch_chord_ratio(res)
    checks = []
    def row(item, value, guide, status, meaning):
        checks.append([item, value, guide, status, meaning])
    d1d2 = res.impeller_id_mm / max(res.impeller_od_mm, 1e-9)
    b2d2 = res.outlet_width_mm / max(res.impeller_od_mm, 1e-9)
    if inp.blade_type == "Forward Curved":
        beta2_guide = "110° to 150°"; beta2_ok = 110 <= res.beta2_deg <= 150
    elif inp.blade_type == "Radial Blade":
        beta2_guide = "85° to 95°"; beta2_ok = 85 <= res.beta2_deg <= 95
    else:
        beta2_guide = "25° to 50°"; beta2_ok = 25 <= res.beta2_deg <= 50
    row("β₂ outlet blade angle", f"{res.beta2_deg:.1f}°", beta2_guide, "OK" if beta2_ok else "Review", "Controls pressure, power curve and non-overloading behaviour.")
    row("β₁ inlet blade angle", f"{res.beta1_deg:.1f}°", "20° to 45° preliminary", "OK" if 20 <= res.beta1_deg <= 45 else "Review", "Should meet incoming air smoothly; wrong β₁ causes shock loss/noise.")
    row("D₁/D₂ inlet ratio", f"{d1d2:.2f}", "0.45 to 0.70 typical", "OK" if 0.45 <= d1d2 <= 0.70 else "Review", "Large inlet improves flow but may reduce pressure capability.")
    row("b₂/D₂ width ratio", f"{b2d2:.2f}", "0.06 to 0.25 common", "OK" if 0.06 <= b2d2 <= 0.25 else "Review", "Too wide means single wheel may be impractical; use DIDW or parallel fans.")
    row("Blade pitch/chord", f"{pc:.2f}", "0.70 to 1.30 preferred", "OK" if 0.70 <= pc <= 1.30 else "Review", "Too low means blades are crowded; too high gives poor guidance/slip.")
    row("Tip speed", f"{res.tip_speed_ms:.1f} m/s", "< material limit", "OK" if not any("Tip speed" in w for w in res.warnings) else "Review", "Higher tip speed increases pressure but raises noise, stress and balance sensitivity.")
    row("Outlet velocity", f"{res.outlet_velocity_ms:.1f} m/s", "8 to 16 m/s typical", "OK" if 8 <= res.outlet_velocity_ms <= 16 else "Review", "High outlet velocity increases velocity pressure, duct loss and noise.")
    return pd.DataFrame(checks, columns=["Input / Check", "Your value", "Practical guide", "Status", "Meaning"])

def ansys_workflow_text() -> str:
    return """ANSYS / CFD workflow guidance

1. Export separate solids:
   - impeller_solid.step
   - stationary_volute.step
   - inlet_duct.step
   - outlet_duct.step
   - fluid_domain.step

2. For Fluent, the most important geometry is the FLUID DOMAIN, not only the metal fan.

3. Use rotating region around the impeller and stationary region in the volute.

4. Suggested boundary conditions:
   - Mass-flow inlet or pressure inlet
   - Pressure outlet
   - Rotating wall / MRF region for preliminary solution
   - Sliding mesh for transient blade-passing study

5. Mesh notes:
   - Inflation layers near blades and volute tongue
   - Fine mesh at blade leading/trailing edges
   - Check y+ according to turbulence model

This toolkit currently creates preliminary geometry. CADQuery/FreeCAD based full blade solids and fluid-domain export are planned inside the same app structure.
"""



# ----------------------------- Automatic Geometry Selection -----------------------------
def _candidate_ranges(blade_type: str):
    """Return practical preliminary search ranges for automatic geometry selection."""
    if blade_type == "Forward Curved":
        return {
            "beta2": [115, 125, 135, 145],
            "beta1": [24, 28, 32, 36],
            "blades": [24, 30, 36, 42, 48],
            "b2d2": [0.08, 0.10, 0.12, 0.15, 0.18, 0.22],
            "d1d2": [0.50, 0.55, 0.60, 0.65],
        }
    if blade_type == "Radial Blade":
        return {
            "beta2": [88, 90, 92],
            "beta1": [22, 28, 34, 40],
            "blades": [6, 8, 10, 12, 14],
            "b2d2": [0.08, 0.10, 0.12, 0.15, 0.18, 0.22],
            "d1d2": [0.45, 0.50, 0.55, 0.60],
        }
    return {
        "beta2": [28, 32, 36, 40, 44, 48],
        "beta1": [22, 26, 30, 34, 38],
        "blades": [8, 10, 12, 14, 16, 18],
        "b2d2": [0.08, 0.10, 0.12, 0.15, 0.18, 0.22],
        "d1d2": [0.45, 0.50, 0.55, 0.60, 0.65],
    }


def optimisation_score(inp: DutyInput, res: DesignResult) -> Tuple[float, List[str]]:
    """Lower score is better. Balances power, sound, vibration and practicality."""
    notes: List[str] = []
    b2d2 = res.outlet_width_mm / max(res.impeller_od_mm, 1e-9)
    d1d2 = res.impeller_id_mm / max(res.impeller_od_mm, 1e-9)
    pc = blade_pitch_chord_ratio(res)
    sound = estimate_sound_db(inp, res)
    vib, _ = vibration_risk(inp, res)
    vib_penalty = {"Low": 0.0, "Medium": 12.0, "High": 35.0}.get(vib, 20.0)

    penalty = 0.0
    if b2d2 < 0.06:
        penalty += 25 * (0.06 - b2d2) / 0.06; notes.append("wheel too narrow")
    if b2d2 > 0.25:
        penalty += 80 * (b2d2 - 0.25) / 0.25; notes.append("wheel too wide")
    if not (0.45 <= d1d2 <= 0.70):
        penalty += 25; notes.append("inlet ratio outside preferred range")
    if pc < 0.70:
        penalty += 35 * (0.70 - pc) / 0.70; notes.append("blade pitch too close")
    if pc > 1.35:
        penalty += 18 * (pc - 1.35) / 1.35; notes.append("blade guidance weak")
    if res.outlet_velocity_ms > 16:
        penalty += 5 * (res.outlet_velocity_ms - 16); notes.append("high outlet velocity")
    if res.outlet_velocity_ms < 7:
        penalty += 4 * (7 - res.outlet_velocity_ms); notes.append("large/low velocity outlet")
    if any("Tip speed" in w for w in res.warnings):
        penalty += 60; notes.append("tip speed above material limit")
    if inp.blade_type == "Forward Curved" and inp.static_pressure_pa > 1200:
        penalty += 40; notes.append("forward curved at high pressure")
    if inp.blade_type == "Radial Blade" and inp.airflow_m3h > 25000:
        penalty += 20; notes.append("radial blade large airflow noise risk")

    # Objective: power is important, then sound, then vibration/practicality penalties.
    score = 1.8 * res.shaft_power_kw + 0.45 * sound + vib_penalty + penalty
    if not notes:
        notes.append("balanced preliminary selection")
    return score, notes


def auto_select_geometry(base_kwargs: Dict, allowed_blade_types: List[str], max_rows: int = 12) -> Tuple[DutyInput, DesignResult, pd.DataFrame]:
    """Search practical discrete geometry options and select the lowest-risk preliminary design."""
    candidates = []
    best = None
    for bt in allowed_blade_types:
        ranges = _candidate_ranges(bt)
        for beta2 in ranges["beta2"]:
            for beta1 in ranges["beta1"]:
                for blades in ranges["blades"]:
                    for b2d2 in ranges["b2d2"]:
                        for d1d2 in ranges["d1d2"]:
                            base_kwargs = dict(
    airflow_m3h=airflow,
    static_pressure_pa=sp,
    total_pressure_pa=0.0,
    air_temp_c=temp_c,
    altitude_m=altitude,
    density_kgm3=density,
    rpm=rpm,
    drive_type=drive_type,
    motor_eff=motor_eff,
    drive_eff=drive_eff,
    design_margin=margin,
    material=material,
    blade_thickness_mm=blade_thk,
    casing_thickness_mm=casing_thk,
    shaft_allow_shear_mpa=shaft_tau,
)

optimisation_df = pd.DataFrame()
if geometry_mode == "Auto optimise geometry":
    allowed_types = list(BLADE_DEFAULTS.keys()) if blade_selection_mode.startswith("Auto") else [manual_blade_type]
    inp, res, optimisation_df = auto_select_geometry(base_kwargs, allowed_types)
    with st.sidebar:
        st.subheader("Auto-selected geometry")
        st.write(f"Blade type: **{inp.blade_type}**")
        st.write(f"β₁ / β₂: **{inp.beta1_deg:.0f}° / {inp.beta2_deg:.0f}°**")
        st.write(f"Blades: **{inp.blade_count}**")
        st.write(f"D₁/D₂: **{inp.inlet_diameter_ratio:.2f}**")
        st.write(f"b₂/D₂ input: **{inp.outlet_width_ratio:.2f}**")
else:
    inp = DutyInput(
        **base_kwargs,
        blade_type=blade_type,
        beta2_deg=beta2,
        beta1_deg=beta1,
        blade_count=int(blades),
        outlet_width_ratio=b2_ratio,
        inlet_diameter_ratio=d1_ratio,
    )
    res = design_blower(inp)

# Output dashboard
c1, c2, c3, c4 = st.columns(4)
c1.metric("Impeller OD", f"{res.impeller_od_mm:.0f} mm")
c2.metric("Outlet width b₂", f"{res.outlet_width_mm:.0f} mm")
c3.metric("Shaft power", f"{res.shaft_power_kw:.2f} kW")
c4.metric("Selected motor", f"{res.selected_motor_kw:.1f} kW")
st.caption(f"Total pressure is calculated from static pressure + outlet velocity pressure: {res.static_pressure_pa:.0f} Pa + {res.velocity_pressure_pa:.0f} Pa = {res.total_pressure_pa:.0f} Pa. Outlet velocity = {res.outlet_velocity_ms:.1f} m/s.")

if res.warnings:
    for w in res.warnings:
        st.warning(w)

tabs = st.tabs(["Design Summary", "Corrective Guidance", "Velocity / Coefficients", "Performance Curves", "2D Geometry", "Mechanical + BOM", "Export Package Review", "Exports", "Ansys Workflow"])

with tabs[0]:
    st.subheader("Main Dimensions")
    summary = pd.DataFrame([
        ["Airflow", f"{res.q_m3s:.3f} m³/s", f"{inp.airflow_m3h:.0f} m³/h"],
        ["Static pressure", f"{res.static_pressure_pa:.0f} Pa", ""],
        ["Calculated velocity pressure", f"{res.velocity_pressure_pa:.0f} Pa", "From outlet velocity"],
        ["Calculated total pressure", f"{res.total_pressure_pa:.0f} Pa", "Static + velocity pressure"],
        ["Blower outlet velocity", f"{res.outlet_velocity_ms:.1f} m/s", "Included in reports"],
        ["Air density", f"{res.density_kgm3:.3f} kg/m³", ""],
        ["Fan speed", f"{res.rpm:.0f} RPM", ""],
        ["Impeller OD D₂", f"{res.impeller_od_mm:.1f} mm", ""],
        ["Inlet diameter D₁", f"{res.impeller_id_mm:.1f} mm", ""],
        ["Outlet width b₂", f"{res.outlet_width_mm:.1f} mm", ""],
        ["Inlet width b₁", f"{res.inlet_width_mm:.1f} mm", ""],
        ["β₁ / β₂", f"{res.beta1_deg:.1f}° / {res.beta2_deg:.1f}°", ""],
        ["Blade count", f"{res.blade_count}", ""],
        ["Volute outlet W × H", f"{res.volute_outlet_width_mm:.0f} × {res.volute_outlet_height_mm:.0f} mm", ""],
        ["Cutoff clearance", f"{res.volute_cutoff_clearance_mm:.1f} mm", ""],
    ], columns=["Parameter", "Value", "Notes"])
    st.dataframe(summary, use_container_width=True)
    if geometry_mode == "Auto optimise geometry":
        st.subheader("Auto optimisation result")
        st.success("The app selected these geometry values by comparing practical candidate designs for lower power, lower sound, lower vibration risk and manufacturable proportions.")
        st.dataframe(optimisation_df, use_container_width=True)
        st.caption("The top row is the selected design. You can switch to Manual geometry override only if you intentionally want to test another option from this table.")

with tabs[1]:
    st.subheader("Corrective action recommendations")
    st.dataframe(engineering_recommendations(inp, res), use_container_width=True)
    st.subheader("Practicality check table")
    st.dataframe(practicality_table(inp, res), use_container_width=True)
    st.info("Change the inputs mentioned in the last column, then re-check the status until critical items become OK or acceptable for your manufacturing method.")

with tabs[2]:
    st.subheader("Aerodynamic Coefficients")
    coeffs = pd.DataFrame([
        ["Flow coefficient φ", res.flow_coeff_phi],
        ["Pressure coefficient ψ", res.pressure_coeff_psi],
        ["Slip factor σ", res.slip_factor],
        ["Estimated total efficiency", res.estimated_total_eff],
        ["Tip speed", res.tip_speed_ms],
        ["Outlet velocity", res.outlet_velocity_ms],
        ["Specific speed metric", res.specific_speed_metric],
        ["Specific diameter metric", res.specific_diameter_metric],
    ], columns=["Parameter", "Value"])
    st.dataframe(coeffs, use_container_width=True)
    st.subheader("Practicality and input meaning")
    st.dataframe(practicality_table(inp, res), use_container_width=True)
    snd = estimate_sound_db(inp, res)
    risk, risk_notes = vibration_risk(inp, res)
    st.metric("Estimated sound at 1 m", f"{snd:.1f} dB(A)")
    st.metric("Preliminary vibration risk", risk)
    for n in risk_notes:
        st.write("- " + n)

with tabs[3]:
    st.subheader("Preliminary Fan Curves")
    curve = performance_curve(res)
    fig, ax = plt.subplots()
    ax.plot(curve["Flow_m3h"], curve["TotalPressure_Pa"], label="Total Pressure Pa")
    ax.set_xlabel("Flow (m³/h)")
    ax.set_ylabel("Pressure (Pa)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(curve["Flow_m3h"], curve["ShaftPower_kW"], label="Shaft Power kW")
    ax2.set_xlabel("Flow (m³/h)")
    ax2.set_ylabel("Power (kW)")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
    st.dataframe(curve, use_container_width=True)

with tabs[4]:
    st.subheader("2D Impeller and Volute Preview")
    blade = blade_centerline_points(res)
    vol = volute_spiral_points(res)
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    r2 = res.impeller_od_mm / 2
    r1 = res.impeller_id_mm / 2
    ax3.add_patch(plt.Circle((0,0), r2, fill=False))
    ax3.add_patch(plt.Circle((0,0), r1, fill=False))
    for k in range(res.blade_count):
        ang = 2 * math.pi * k / res.blade_count
        ca, sa = math.cos(ang), math.sin(ang)
        pts = np.array([(x*ca-y*sa, x*sa+y*ca) for x,y in blade])
        ax3.plot(pts[:,0], pts[:,1])
    vol_np = np.array(vol)
    ax3.plot(vol_np[:,0], vol_np[:,1], linewidth=2)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("mm")
    ax3.set_ylabel("mm")
    ax3.grid(True)
    st.pyplot(fig3)

    st.subheader("Single blade profile and pitch")
    figb, axb = plt.subplots(figsize=(7, 4))
    pts = np.array(blade)
    axb.plot(pts[:,0], pts[:,1], linewidth=2)
    r_mid = 0.25 * (res.impeller_od_mm + res.impeller_id_mm)
    pitch = math.pi * (2 * r_mid) / max(res.blade_count, 1)
    chord = 0.55 * (res.impeller_od_mm - res.impeller_id_mm)
    axb.set_aspect("equal", adjustable="box")
    axb.grid(True)
    axb.set_xlabel("mm")
    axb.set_ylabel("mm")
    axb.set_title(f"Blade centerline: β₁={res.beta1_deg:.1f}°, β₂={res.beta2_deg:.1f}°, pitch/chord={pitch/max(chord,1):.2f}")
    st.pyplot(figb)
    st.info(f"Volute/blower outlet flange: {res.volute_outlet_width_mm:.0f} mm wide × {res.volute_outlet_height_mm:.0f} mm high. Outlet velocity: {res.outlet_velocity_ms:.1f} m/s.")
    st.info("DXF export contains impeller OD, inlet, hub, blade centerlines, volute spiral and outlet rectangle.")

with tabs[5]:
    st.subheader("Mechanical Design + BOM")
    mech = pd.DataFrame([
        ["Tip speed", f"{res.tip_speed_ms:.1f} m/s"],
        ["Preliminary shaft diameter", f"{res.shaft_diameter_mm:.0f} mm"],
        ["Approx impeller mass", f"{res.approx_impeller_mass_kg:.1f} kg"],
        ["Material", inp.material],
        ["Selected motor", f"{res.selected_motor_kw:.2f} kW"],
        ["Estimated sound", f"{estimate_sound_db(inp, res):.1f} dB(A) at 1 m"],
        ["Vibration risk", vibration_risk(inp, res)[0]],
    ], columns=["Parameter", "Value"])
    st.dataframe(mech, use_container_width=True)
    st.subheader("Preliminary BOM")
    st.dataframe(bom_table(inp, res), use_container_width=True)

with tabs[6]:
    st.subheader("Export package review")
    st.write("The same information below is included in the ZIP package so the user can review and correct the design before manufacturing.")
    st.dataframe(export_manifest_df(inp, res), use_container_width=True)
    st.subheader("Inputs included in Excel/PDF")
    st.dataframe(input_table(inp), use_container_width=True)
    st.subheader("Results included in Excel/PDF")
    st.dataframe(make_results_df(res), use_container_width=True)
    st.subheader("Recommendations included in Excel/PDF")
    st.dataframe(engineering_recommendations(inp, res), use_container_width=True)
    st.subheader("Sound and vibration included in Excel/PDF")
    snd = estimate_sound_db(inp, res)
    risk, risk_notes = vibration_risk(inp, res)
    st.metric("Estimated sound at 1 m", f"{snd:.1f} dB(A)")
    st.metric("Vibration risk", risk)
    for n in risk_notes:
        st.write("- " + n)
    st.subheader("Images included in ZIP")
    cimg1, cimg2 = st.columns(2)
    cimg1.image(plot_impeller_volute_png(res), caption="impeller_volute_preview.png")
    cimg2.image(plot_blade_profile_png(res), caption="single_blade_profile.png")

with tabs[7]:
    st.subheader("Download Files")
    st.download_button("Download PDF Report", create_pdf(inp, res), file_name="blower_design_report.pdf")
    st.download_button("Download Excel Calculation", create_excel(inp, res), file_name="blower_design_calculations.xlsx")
    st.download_button("Download DXF Drawing", create_dxf(res), file_name="blower_2d_manufacturing.dxf")
    st.download_button("Download STEP placeholder / CAD instruction", create_step_placeholder(res), file_name="impeller_basic.step_or_instruction.txt")
    st.download_button("Download Complete ZIP Package", make_zip(inp, res), file_name="blower_design_package.zip")

with tabs[8]:
    st.subheader("Ansys / CFD Export Plan")
    st.text(ansys_workflow_text())
    st.info("Next development step: full CadQuery/FreeCAD solid impeller, volute, and fluid-domain export as separate STEP files.")
