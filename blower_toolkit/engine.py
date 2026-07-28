"""
Centrifugal Blower Design & Manufacturing Toolkit v19 — SI units
=================================================================
Physics-first redesign of the v16 toolkit.

Key upgrades over v16
---------------------
1. AERODYNAMICS: impeller sized by iterative velocity-triangle solution with
   Wiesner/Stodola slip and a physical loss model (inlet cone, incidence,
   blade-channel friction, diffusion, volute recovery, disk friction,
   leakage). Pressure matching is DERIVED, not assumed from psi coefficients.
2. FAN FAMILIES: 8 industrial wheel types, each with its own geometric
   proportions (Sirocco cage wheels get their true large-D1/D2, wide-b2
   treatment), SWSI and DWDI arrangements.
3. OFF-DESIGN CURVES: real Euler-line-minus-losses curves. Backward wheels
   come out non-overloading; forward-curved come out overloading — because
   the physics produces it, not because a parabola was fitted.
4. ACOUSTICS: AMCA/ASHRAE (Graham) octave-band method with blade-pass
   frequency increment, off-peak correction and A-weighting — not a
   single-number guess.
5. MECHANICAL: shaft sizing with keyway factor, overhung-impeller first
   critical speed, impeller mass build-up.
6. MANUFACTURING DXF: multi-view dimensioned drawing — impeller plan +
   side section, single-blade press/roll detail with arc data table,
   volute side-plate with spiral station table, inlet cone, title block,
   proper layers. Suitable for shop-floor use after review.
7. CALIBRATION HOOKS: per-family multipliers (pressure / power / sound)
   so the tool can be tuned to YOUR AMCA/ISO test data per product line.

Preliminary engineering tool: validate final designs by AMCA 210 / ISO 5801
testing, balancing per ISO 21940, and qualified review before manufacture.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    HAS_EZDXF = True
except Exception:
    HAS_EZDXF = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage)
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# ----------------------------------------------------------------------------
# Constants and databases
# ----------------------------------------------------------------------------
G = 9.80665
R_AIR = 287.05

STANDARD_MOTORS_KW = [0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0,
                      5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90,
                      110, 132, 160, 200, 250, 315]

MATERIALS = {
    "Mild Steel IS2062 / S275":  {"density": 7850, "max_tip": 105, "E": 205e9},
    "Galvanized Steel":          {"density": 7850, "max_tip": 85,  "E": 205e9},
    "Stainless Steel 304":       {"density": 8000, "max_tip": 100, "E": 193e9},
    "Aluminium 6061-T6":         {"density": 2700, "max_tip": 80,  "E": 69e9},
}

OCTAVES = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
A_WEIGHT = [-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1]

# Graham specific sound power levels Kw per octave band (dB re 1 pW),
# per ASHRAE Applications-style tabulation. Calibrate against your own
# tested product line via the sound calibration hook.
KW_TABLES = {
    "bc_large": [32, 32, 31, 29, 28, 23, 15, 8],    # AF/BC/BI wheel >= 0.9 m
    "bc_small": [36, 38, 36, 34, 33, 28, 20, 15],   # AF/BC/BI wheel  < 0.9 m
    "fc":       [47, 43, 39, 33, 28, 25, 23, 20],   # FC / Sirocco all sizes
    "radial":   [45, 39, 42, 39, 37, 32, 30, 29],   # radial tip / paddle
}

# ----------------------------------------------------------------------------
# Fan family database.
# Geometric proportion defaults + practical ranges are family-specific.
# Loss-model coefficients (k_*) are family-tuned; calibration hooks scale
# the final result to your tested hardware.
# ----------------------------------------------------------------------------
FAMILIES: Dict[str, dict] = {
    "Airfoil Backward (AF)": dict(
        code="af", kw_table="bc", bfi=3,
        beta1=26.0, beta2=38.0, z=10, d1d2=0.66, b2d2=0.20,
        beta1_rng=(18, 35), beta2_rng=(30, 45), z_rng=(9, 12),
        d1d2_rng=(0.55, 0.75), b2d2_rng=(0.12, 0.30), tip_max=110,
        k_in=0.10, k_inc=0.50, k_fr=0.72, k_dif=0.18, k_vol=0.16, k_df=1.0,
        eta_band=(0.76, 0.86),
        guide="Highest efficiency, lowest noise. Best for large AHUs, "
              "package units and plenum retrofits where power matters. "
              "Hollow airfoil blades cost more to fabricate."),
    "Backward Curved (BC)": dict(
        code="bc", kw_table="bc", bfi=3,
        beta1=25.0, beta2=40.0, z=12, d1d2=0.64, b2d2=0.20,
        beta1_rng=(18, 35), beta2_rng=(30, 50), z_rng=(10, 16),
        d1d2_rng=(0.50, 0.72), b2d2_rng=(0.10, 0.28), tip_max=100,
        k_in=0.11, k_inc=0.55, k_fr=1.00, k_dif=0.30, k_vol=0.20, k_df=1.0,
        eta_band=(0.72, 0.82),
        guide="Single-thickness curved plate. Workhorse for AHU / package "
              "unit supply and condenser duties. Non-overloading power."),
    "Backward Inclined (BI flat)": dict(
        code="bi", kw_table="bc", bfi=3,
        beta1=25.0, beta2=42.0, z=11, d1d2=0.62, b2d2=0.19,
        beta1_rng=(18, 35), beta2_rng=(32, 52), z_rng=(9, 14),
        d1d2_rng=(0.50, 0.70), b2d2_rng=(0.10, 0.26), tip_max=95,
        k_in=0.11, k_inc=0.62, k_fr=1.02, k_dif=0.28, k_vol=0.21, k_df=1.0,
        eta_band=(0.68, 0.78),
        guide="Flat inclined blades — cheapest backward wheel to press. "
              "A little below BC efficiency; still non-overloading."),
    "Radial Tip (RT)": dict(
        code="rt", kw_table="radial", bfi=5,
        beta1=28.0, beta2=85.0, z=12, d1d2=0.68, b2d2=0.14,
        beta1_rng=(20, 40), beta2_rng=(75, 90), z_rng=(10, 16),
        d1d2_rng=(0.55, 0.75), b2d2_rng=(0.08, 0.20), tip_max=105,
        k_in=0.12, k_inc=0.55, k_fr=1.02, k_dif=0.30, k_vol=0.23, k_df=1.05,
        eta_band=(0.62, 0.72),
        guide="Backward inlet, radial discharge tip. Higher pressure per "
              "diameter than BC; used for higher-pressure HVAC/process work."),
    "Radial Straight (Paddle)": dict(
        code="paddle", kw_table="radial", bfi=7,
        beta1=90.0, beta2=90.0, z=8, d1d2=0.50, b2d2=0.16,
        beta1_rng=(70, 90), beta2_rng=(85, 90), z_rng=(6, 12),
        d1d2_rng=(0.40, 0.60), b2d2_rng=(0.08, 0.30), tip_max=90,
        k_in=0.15, k_inc=0.50, k_fr=1.30, k_dif=0.45, k_vol=0.30, k_df=1.25,
        eta_band=(0.50, 0.64),
        guide="Flat radial paddles. Dirty/dusty/abrasive air, kitchen "
              "exhaust, material handling. Robust, self-cleaning, noisy."),
    "Forward Curved (FC)": dict(
        code="fc", kw_table="fc", bfi=2,
        beta1=68.0, beta2=145.0, z=32, d1d2=0.80, b2d2=0.35,
        beta1_rng=(50, 85), beta2_rng=(120, 160), z_rng=(24, 44),
        d1d2_rng=(0.72, 0.86), b2d2_rng=(0.25, 0.45), tip_max=55,
        k_in=0.14, k_inc=0.28, k_fr=1.30, k_dif=0.50, k_vol=0.30, k_df=1.0,
        eta_band=(0.52, 0.65),
        guide="Compact wheel, low tip speed, high pressure coefficient. "
              "FCU / small AHU / furnace duty. Power OVERLOADS with flow — "
              "size the motor at max expected flow, not at duty."),
    "Multi-Blade Sirocco (Cage)": dict(
        code="sirocco", kw_table="fc", bfi=2,
        beta1=75.0, beta2=152.0, z=48, d1d2=0.87, b2d2=0.50,
        beta1_rng=(60, 90), beta2_rng=(140, 165), z_rng=(36, 64),
        d1d2_rng=(0.82, 0.90), b2d2_rng=(0.38, 0.62), tip_max=45,
        k_in=0.15, k_inc=0.25, k_fr=1.40, k_dif=0.55, k_vol=0.32, k_df=1.0,
        eta_band=(0.48, 0.62),
        guide="True squirrel-cage: many short shallow blades, D1/D2 ≈ "
              "0.85-0.90, very wide wheel. Fan coil units, compact "
              "double-inlet AHU blowers. Keep out of dusty service."),
    "Plug / Plenum (BC, no volute)": dict(
        code="plug", kw_table="bc", bfi=3,
        beta1=26.0, beta2=40.0, z=11, d1d2=0.64, b2d2=0.20,
        beta1_rng=(18, 35), beta2_rng=(30, 48), z_rng=(9, 14),
        d1d2_rng=(0.50, 0.72), b2d2_rng=(0.10, 0.28), tip_max=95,
        k_in=0.10, k_inc=0.55, k_fr=1.00, k_dif=0.30, k_vol=0.0, k_df=1.0,
        plenum=True, sound_adder=4.0,
        eta_band=(0.58, 0.70),
        guide="Unhoused BC wheel discharging into a plenum (fan-wall / "
              "hygienic AHU). Whole discharge dynamic head is dumped, so "
              "static efficiency is lower, but the package is compact and "
              "easy to clean; N+1 arrays are simple."),
}

ARRANGEMENTS = ["SWSI (single inlet)", "DWDI (double inlet)"]

DEFAULT_CALIBRATION = {name: {"dp": 1.00, "power": 1.00, "sound_db": 0.0}
                       for name in FAMILIES}

# ----------------------------------------------------------------------------
# Air properties
# ----------------------------------------------------------------------------
def air_density(temp_c: float, altitude_m: float, rh_pct: float = 50.0) -> float:
    """ISA pressure at altitude + humid-air correction."""
    t0, p0, lapse = 288.15, 101325.0, 0.0065
    t_alt = max(180.0, t0 - lapse * altitude_m)
    p = p0 * (t_alt / t0) ** (G / (R_AIR * lapse))
    t_k = temp_c + 273.15
    # saturation vapour pressure (Tetens) and humid-air density
    p_ws = 610.78 * math.exp(17.2694 * temp_c / (temp_c + 238.3))
    p_w = min(0.99 * p, max(0.0, rh_pct) / 100.0 * p_ws)
    return (p - p_w) / (R_AIR * t_k) + p_w / (461.5 * t_k)


def air_viscosity(temp_c: float) -> float:
    """Sutherland's law, dynamic viscosity Pa.s."""
    t = temp_c + 273.15
    return 1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4)


def selected_motor(kw: float) -> float:
    for m in STANDARD_MOTORS_KW:
        if m >= kw:
            return m
    return STANDARD_MOTORS_KW[-1]


# ----------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------
@dataclass
class Duty:
    airflow_m3h: float
    static_pa: float
    rpm: float
    temp_c: float = 35.0
    altitude_m: float = 0.0
    rh_pct: float = 50.0
    density: float = 0.0            # 0 -> compute from T/alt/RH

    def finalize(self) -> "Duty":
        if self.density <= 0:
            return replace(self, density=air_density(self.temp_c,
                                                     self.altitude_m,
                                                     self.rh_pct))
        return self

    @property
    def q_m3s(self) -> float:
        return self.airflow_m3h / 3600.0


@dataclass
class Wheel:
    """Complete geometric definition of one blower (per unit, not per side)."""
    family: str
    arrangement: str                # SWSI / DWDI
    d2: float                       # impeller OD, m
    d1: float                       # blade inlet (eye) dia, m
    b2: float                       # TOTAL outlet width, m (both sides if DWDI)
    b1: float                       # TOTAL inlet width, m
    z: int
    beta1: float                    # deg from tangent
    beta2: float                    # deg from tangent
    t_blade: float = 0.003          # m
    t_plate: float = 0.004          # m
    # casing
    r3: float = 0.0                 # volute base circle radius, m
    volute_width: float = 0.0       # casing internal width B, m
    throat_area: float = 0.0        # volute throat area, m^2
    outlet_w: float = 0.0           # discharge flange width, m
    outlet_h: float = 0.0           # discharge flange height, m
    cutoff_clr: float = 0.0         # cutoff clearance, m
    hub_d: float = 0.0
    shaft_d: float = 0.0
    material: str = "Mild Steel IS2062 / S275"

    @property
    def dwdi(self) -> bool:
        return self.arrangement.startswith("DWDI")

    @property
    def outlet_area(self) -> float:
        return max(self.outlet_w * self.outlet_h, 1e-6)


@dataclass
class PerfPoint:
    q: float                        # m3/s at inlet conditions
    dp_total: float                 # Pa (fan total pressure)
    dp_static: float                # Pa (fan static pressure, AMCA convention)
    p_shaft: float                  # W
    eta_total: float
    eta_static: float
    u2: float
    cu2: float
    cm2: float
    cm1: float
    w1: float
    w2: float
    slip: float
    v_out: float
    dp_euler: float
    losses: Dict[str, float] = field(default_factory=dict)


@dataclass
class Mechanical:
    shaft_d_mm: float
    torque_nm: float
    impeller_mass_kg: float
    wr2_kgm2: float                 # rotating inertia (mass moment, kg.m^2)
    tip_speed: float
    critical_rpm: float
    rpm_ratio: float                # operating / first critical
    static_deflection_mm: float


@dataclass
class Acoustics:
    lw_bands: List[float]           # octave-band sound power, dB
    lw_total: float
    lwa_total: float
    lpa_1m: float                   # A-weighted pressure at 1 m (hemispherical)
    bpf_hz: float
    off_peak_corr: float

# ----------------------------------------------------------------------------
# Slip factor
# ----------------------------------------------------------------------------
def slip_factor(z: int, beta2_deg: float, d1d2: float) -> float:
    """Wiesner for backward/radial wheels; Stodola form for forward-curved.

    Wiesner:  sigma = 1 - sqrt(sin(b2)) / z^0.7   (valid to a limit d1/d2)
    Beyond Wiesner's limit ratio the slip degrades (short blades guide less);
    apply his correction. For beta2 > 90 deg (forward) use Stodola
    sigma = 1 - pi*sin(b2)/z which behaves sensibly for many-bladed cages.
    """
    z = max(int(z), 2)
    b = math.radians(min(beta2_deg, 179.0))
    if beta2_deg <= 95.0:
        sig = 1.0 - math.sqrt(max(math.sin(b), 1e-6)) / z ** 0.7
        eps_limit = math.exp(-8.16 * math.sin(b) / z)
        if d1d2 > eps_limit:
            corr = 1.0 - ((d1d2 - eps_limit) / (1.0 - eps_limit)) ** 3
            sig *= max(corr, 0.05)
    else:
        sig = 1.0 - math.pi * abs(math.sin(b)) / z
    return min(max(sig, 0.30), 0.98)


# ----------------------------------------------------------------------------
# Core performance engine — one wheel, one speed, one flow
# ----------------------------------------------------------------------------
def wheel_performance(w: Wheel, duty_density: float, rpm: float, q: float,
                      calibration: Optional[dict] = None,
                      visc: float = 1.85e-5) -> PerfPoint:
    """Velocity-triangle + loss-model performance at flow q (m3/s total).

    DWDI wheels are treated as two mirrored half-wheels: each side takes
    q/2 through width b/2; pressures are identical, power doubles via flow.
    """
    fam = FAMILIES[w.family]
    cal = (calibration or {}).get(w.family, {"dp": 1.0, "power": 1.0})
    rho = duty_density
    sides = 2 if w.dwdi else 1
    qs = max(q, 1e-4) / sides                     # flow per side
    b2s = w.b2 / sides
    b1s = w.b1 / sides

    omega = 2.0 * math.pi * rpm / 60.0
    r1, r2 = w.d1 / 2.0, w.d2 / 2.0
    u1, u2 = omega * r1, omega * r2

    beta1 = math.radians(min(max(w.beta1, 5.0), 175.0))
    beta2 = math.radians(min(max(w.beta2, 5.0), 175.0))

    # blade blockage
    blk1 = max(0.75, 1.0 - w.z * w.t_blade / (math.pi * w.d1 * max(math.sin(beta1), 0.15)))
    blk2 = max(0.80, 1.0 - w.z * w.t_blade / (math.pi * w.d2 * max(math.sin(beta2), 0.15)))

    # leakage (shroud clearance recirculation): volumetric efficiency
    eta_v = 0.985 - 0.020 * (0.3 / max(w.d2, 0.15))     # small wheels leak more
    eta_v = min(max(eta_v, 0.90), 0.99)
    q_imp = qs / eta_v                                   # flow through blading

    # meridional velocities
    cm1 = q_imp / max(math.pi * w.d1 * b1s * blk1, 1e-6)
    cm2 = q_imp / max(math.pi * w.d2 * b2s * blk2, 1e-6)

    # relative velocities
    wu1 = u1                                             # no pre-swirl
    w1 = math.hypot(cm1, wu1)
    sig = slip_factor(w.z, w.beta2, w.d1 / w.d2)
    cu2 = sig * u2 - cm2 / math.tan(beta2)               # cot(b2)=1/tan
    cu2 = max(cu2, 0.05 * u2)
    wu2 = u2 - cu2
    w2 = math.hypot(cm2, wu2)
    c2 = math.hypot(cm2, cu2)

    # Euler pressure rise (per side; identical both sides)
    dp_euler = rho * u2 * cu2

    # ---------------- losses (Pa) ----------------
    c_eye = q_imp / max(math.pi / 4.0 * w.d1 ** 2, 1e-6)
    L_in = fam["k_in"] * 0.5 * rho * c_eye ** 2

    # incidence: tangential mismatch of relative flow vs blade at inlet
    w_shock = abs(wu1 - cm1 / math.tan(beta1))
    L_inc = fam["k_inc"] * 0.5 * rho * w_shock ** 2

    # blade channel friction: f * (L/Dh) * 0.5 rho w_mean^2
    w_mean = 0.5 * (w1 + w2)
    beta_m = 0.5 * (beta1 + beta2)
    path = (r2 - r1) / max(math.sin(min(beta_m, math.radians(90))), 0.25)
    b_m = 0.5 * (b1s + b2s)
    pitch = math.pi * (r1 + r2) / w.z
    dh = 2.0 * b_m * pitch * max(math.sin(beta_m), 0.2) / (b_m + pitch * max(math.sin(beta_m), 0.2))
    re = rho * max(w_mean, 1.0) * max(dh, 1e-3) / visc
    f = 0.30 / max(re, 2e4) ** 0.25                     # Blasius-type
    L_fr = fam["k_fr"] * 4.0 * f * (path / max(dh, 1e-3)) * 0.5 * rho * w_mean ** 2

    # diffusion / separation in the blade channel
    dif = min(max(w1 / max(w2, 0.5) - 1.05, 0.0), 1.2)
    L_dif = fam["k_dif"] * 0.5 * rho * (dif * w_mean) ** 2

    # volute: friction on the swirl + mismatch with throat sizing
    if fam.get("plenum"):
        # unhoused wheel: full discharge kinetic head lost to the plenum
        L_vol = 0.5 * rho * c2 ** 2 * 0.60
        v_out = 0.0
        dyn_out = 0.0
    else:
        c_th = qs / max(w.throat_area / sides if w.dwdi else w.throat_area, 1e-4) \
            if w.throat_area > 0 else 0.7 * cu2
        c_swirl = cu2 * r2 / max(w.r3 if w.r3 > 0 else 1.05 * r2, r2)
        L_vol = fam["k_vol"] * 0.5 * rho * c_swirl ** 2 \
            + 0.30 * 0.5 * rho * (c_swirl - c_th) ** 2
        v_out = q / w.outlet_area if w.outlet_area > 1e-5 else 0.30 * u2
        dyn_out = 0.5 * rho * v_out ** 2

    losses = {"inlet": L_in, "incidence": L_inc, "friction": L_fr,
              "diffusion": L_dif, "volute": L_vol}
    dp_total = (dp_euler - sum(losses.values())) * cal.get("dp", 1.0)
    dp_static = dp_total - dyn_out

    # ---------------- power ----------------
    p_euler = rho * (q / eta_v) * u2 * cu2               # total flow incl. leakage
    # disk friction on backplate+shroud
    re_d = rho * u2 * r2 / visc
    cf = 0.0622 / max(re_d, 1e5) ** 0.2
    p_df = fam["k_df"] * cf * rho * omega ** 3 * r2 ** 5 * sides
    p_shaft = (p_euler + p_df) * cal.get("power", 1.0)
    p_shaft = max(p_shaft, 1.0)

    eta_t = max(q * dp_total, 0.0) / p_shaft
    eta_s = max(q * dp_static, 0.0) / p_shaft

    return PerfPoint(q=q, dp_total=dp_total, dp_static=dp_static,
                     p_shaft=p_shaft, eta_total=min(eta_t, 0.92),
                     eta_static=min(eta_s, 0.90),
                     u2=u2, cu2=cu2, cm2=cm2, cm1=cm1, w1=w1, w2=w2,
                     slip=sig, v_out=v_out, dp_euler=dp_euler, losses=losses)


# ----------------------------------------------------------------------------
# Casing sizing for a given wheel + duty
# ----------------------------------------------------------------------------
def size_casing(w: Wheel, duty: Duty, target_v_out: float) -> Wheel:
    fam = FAMILIES[w.family]
    r2 = w.d2 / 2.0
    if fam.get("plenum"):
        return replace(w, r3=0.0, volute_width=0.0, throat_area=0.0,
                       outlet_w=0.0, outlet_h=0.0, cutoff_clr=0.0)
    # casing width: BC-type volutes are typically ~2x wheel width;
    # wide sirocco/FC wheels use narrower relative casing.
    kb = 2.0 if w.b2 / w.d2 < 0.30 else 1.25
    B = kb * w.b2
    r3 = 1.06 * r2
    # provisional cu2 at duty for spiral sizing
    p = wheel_performance(replace(w, throat_area=0.0, r3=r3,
                                  outlet_w=0.0, outlet_h=0.0),
                          duty.density, duty.rpm, duty.q_m3s)
    cu2 = max(p.cu2, 1.0)
    # log-spiral (constant angular momentum, parallel walls):
    # R(theta) = r3 * exp(q * theta / (2*pi * B * r2 * cu2))
    k_sp = duty.q_m3s / (2.0 * math.pi * B * r2 * cu2)
    r_end = r3 * math.exp(min(k_sp * 2.0 * math.pi, 3.0))
    throat_area = B * (r_end - r3)
    # discharge flange
    area = duty.q_m3s / max(target_v_out, 4.0)
    out_w = max(B, 1.02 * w.b2)
    out_h = area / out_w
    max_h = 1.20 * w.d2
    if out_h > max_h:
        out_h = max_h
        out_w = area / out_h
    cutoff = max(0.05 * w.d2, 0.008)
    return replace(w, r3=r3, volute_width=B, throat_area=throat_area,
                   outlet_w=out_w, outlet_h=out_h, cutoff_clr=cutoff)


# ----------------------------------------------------------------------------
# Design solve: find D2 so delivered STATIC pressure = required, at duty flow
# ----------------------------------------------------------------------------
def size_wheel(duty: Duty, family: str, arrangement: str,
               d1d2: float, b2d2: float, beta1: float, beta2: float, z: int,
               target_v_out: float = 12.0,
               t_blade: float = 0.003, material: str = "Mild Steel IS2062 / S275",
               calibration: Optional[dict] = None,
               auto_beta1: bool = False) -> Tuple[Wheel, PerfPoint, List[str]]:
    """Bisection on D2 until fan static pressure at duty flow matches target."""
    duty = duty.finalize()
    warnings: List[str] = []
    visc = air_viscosity(duty.temp_c)

    def build(d2: float, b1a: float) -> Wheel:
        d1 = d1d2 * d2
        b2 = b2d2 * d2
        # inlet width from eye continuity: keep cm1 ~ 0.9..1.1 x cm2
        b1 = min(max(b2 * 1.05, b2), 1.6 * b2)
        w = Wheel(family=family, arrangement=arrangement, d2=d2, d1=d1,
                  b2=b2, b1=b1, z=int(z), beta1=b1a, beta2=beta2,
                  t_blade=t_blade, material=material)
        return size_casing(w, duty, target_v_out)

    def static_at(d2: float, b1a: float) -> float:
        w = build(d2, b1a)
        return wheel_performance(w, duty.density, duty.rpm, duty.q_m3s,
                                 calibration, visc).dp_static

    def solve_d2(b1a: float) -> Tuple[float, bool]:
        """static(D2) is non-monotonic (losses dominate oversized wheels);
        scan log-spaced diameters for the ASCENDING crossing, then bisect."""
        grid = np.geomspace(0.06, 3.5, 40)
        fs = [static_at(float(d), b1a) - duty.static_pa for d in grid]
        bracket = None
        for i in range(len(grid) - 1):
            if fs[i] < 0.0 <= fs[i + 1]:
                bracket = (float(grid[i]), float(grid[i + 1]))
                break
        if bracket is None:
            for i in range(len(grid) - 1):
                if (fs[i] < 0) != (fs[i + 1] < 0):
                    bracket = (float(grid[i]), float(grid[i + 1]))
                    break
        if bracket is None:
            if all(f < 0 for f in fs):
                return float(grid[int(np.argmax(fs))]), False
            return float(grid[0]), False
        lo, hi = bracket
        f_lo = static_at(lo, b1a) - duty.static_pa
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            fm = static_at(mid, b1a) - duty.static_pa
            if abs(fm) < 0.15:
                return mid, True
            if (fm > 0) == (f_lo > 0):
                lo, f_lo = mid, fm
            else:
                hi = mid
        return 0.5 * (lo + hi), True

    b1a = beta1
    d2, ok = solve_d2(b1a)
    if not ok:
        warnings.append("Duty static pressure could not be matched at this "
                        "speed within D2 = 60 mm .. 3.5 m. Change RPM, split "
                        "flow across fans, or pick another family.")

    # optional: match inlet blade angle to flow angle (+2 deg incidence)
    if auto_beta1:
        w_tmp = build(d2, b1a)
        p_tmp = wheel_performance(w_tmp, duty.density, duty.rpm, duty.q_m3s,
                                  calibration, visc)
        omega = 2 * math.pi * duty.rpm / 60
        u1 = omega * w_tmp.d1 / 2
        b1a = min(max(math.degrees(math.atan2(p_tmp.cm1, u1)) + 2.0,
                      FAMILIES[family]["beta1_rng"][0]),
                  FAMILIES[family]["beta1_rng"][1])
        d2, ok2 = solve_d2(b1a)
        if not ok2:
            warnings.append("Re-solve with matched inlet angle did not "
                            "converge; result uses best available diameter.")

    wheel = build(d2, b1a)
    perf = wheel_performance(wheel, duty.density, duty.rpm, duty.q_m3s,
                             calibration, visc)

    # hub & shaft placeholders (refined in mechanical())
    wheel = replace(wheel, hub_d=max(0.16 * wheel.d1, 0.05))

    mat = MATERIALS[material]
    fam = FAMILIES[family]
    if perf.u2 > min(mat["max_tip"], fam["tip_max"]):
        warnings.append(f"Tip speed {perf.u2:.0f} m/s exceeds the "
                        f"{min(mat['max_tip'], fam['tip_max'])} m/s guide for "
                        f"this family/material — check stress and balancing.")
    if abs(perf.dp_static - duty.static_pa) > max(0.02 * duty.static_pa, 5):
        warnings.append("Sizing did not converge tightly on target static "
                        "pressure; review inputs.")
    lo_e, hi_e = fam["eta_band"]
    if perf.eta_total < lo_e - 0.06:
        warnings.append("Predicted efficiency is well below the normal band "
                        "for this family — geometry ratios are probably "
                        "unfavourable for this duty. Try the optimizer.")
    return wheel, perf, warnings

# ----------------------------------------------------------------------------
# Off-design performance curve
# ----------------------------------------------------------------------------
def performance_curve(w: Wheel, duty: Duty, calibration: Optional[dict] = None,
                      n: int = 33) -> pd.DataFrame:
    duty = duty.finalize()
    visc = air_viscosity(duty.temp_c)
    q_d = duty.q_m3s
    rows = []
    for x in np.linspace(0.20, 1.45, n):
        q = q_d * x
        p = wheel_performance(w, duty.density, duty.rpm, q, calibration, visc)
        rows.append(dict(FlowFrac=x, Flow_m3h=q * 3600, Flow_m3s=q,
                         Total_Pa=max(p.dp_total, 0),
                         Static_Pa=max(p.dp_static, 0),
                         Shaft_kW=p.p_shaft / 1000.0,
                         EtaTotal=max(p.eta_total, 0),
                         EtaStatic=max(p.eta_static, 0)))
    return pd.DataFrame(rows)


def peak_static_efficiency(curve: pd.DataFrame) -> float:
    return float(curve["EtaStatic"].max()) if len(curve) else 0.0


# ----------------------------------------------------------------------------
# Acoustics — AMCA / ASHRAE (Graham) octave-band method
# ----------------------------------------------------------------------------
def _off_peak_correction(eta_ratio: float) -> float:
    r = eta_ratio * 100.0
    if r >= 90: return 0.0
    if r >= 85: return 3.0
    if r >= 75: return 6.0
    if r >= 65: return 9.0
    if r >= 55: return 12.0
    return 15.0


def acoustics(w: Wheel, duty: Duty, perf: PerfPoint,
              curve: Optional[pd.DataFrame] = None,
              calibration: Optional[dict] = None) -> Acoustics:
    duty = duty.finalize()
    fam = FAMILIES[w.family]
    cal_db = (calibration or {}).get(w.family, {}).get("sound_db", 0.0)

    key = fam["kw_table"]
    if key == "bc":
        key = "bc_large" if w.d2 >= 0.90 else "bc_small"
    kw = list(KW_TABLES[key])

    q_cfm = max(duty.q_m3s * 2118.88, 30.0)
    p_inwg = max(perf.dp_static / 249.089, 0.05)
    base = 10.0 * math.log10(q_cfm) + 20.0 * math.log10(p_inwg)

    # off-peak correction from position on the efficiency curve
    eta_pk = peak_static_efficiency(curve) if curve is not None else perf.eta_static
    ratio = perf.eta_static / eta_pk if eta_pk > 1e-3 else 1.0
    c_off = _off_peak_correction(min(ratio, 1.0))

    # blade pass frequency increment in its octave band
    bpf = duty.rpm / 60.0 * w.z
    idx = int(np.argmin([abs(math.log(bpf / f)) for f in OCTAVES])) if bpf > 40 else 2
    bands = [k + base + c_off + cal_db + fam.get("sound_adder", 0.0)
             for k in kw]
    bands[idx] += fam["bfi"]
    if w.dwdi:
        bands = [b + 3.0 for b in bands]

    lw_total = 10.0 * math.log10(sum(10 ** (b / 10.0) for b in bands))
    lwa = 10.0 * math.log10(sum(10 ** ((b + a) / 10.0)
                                for b, a in zip(bands, A_WEIGHT)))
    lpa_1m = lwa - 8.0            # hemispherical radiation, r = 1 m
    return Acoustics(lw_bands=[round(b, 1) for b in bands],
                     lw_total=round(lw_total, 1), lwa_total=round(lwa, 1),
                     lpa_1m=round(lpa_1m, 1), bpf_hz=round(bpf, 1),
                     off_peak_corr=c_off)


# ----------------------------------------------------------------------------
# Mechanical: mass, shaft, first critical speed
# ----------------------------------------------------------------------------
def mechanical(w: Wheel, duty: Duty, perf: PerfPoint,
               tau_allow_mpa: float = 40.0,
               bearing_span_m: float = 0.0,
               overhang_m: float = 0.0) -> Tuple[Wheel, Mechanical]:
    duty = duty.finalize()
    mat = MATERIALS[w.material]
    rho_m, E = mat["density"], mat["E"]
    r1, r2 = w.d1 / 2, w.d2 / 2
    sides = 2 if w.dwdi else 1

    # mass build-up
    a_plate = math.pi * (r2 ** 2 - (0.5 * w.hub_d) ** 2)
    m_back = a_plate * w.t_plate * rho_m
    m_shroud = 0.85 * math.pi * (r2 ** 2 - r1 ** 2) * w.t_plate * rho_m * sides
    chord = 1.15 * (r2 - r1) / max(math.sin(math.radians(
        min(0.5 * (w.beta1 + w.beta2), 90))), 0.35)
    m_blades = w.z * chord * w.b2 * w.t_blade * rho_m
    m_hub = math.pi / 4 * w.hub_d ** 2 * (1.4 * w.hub_d) * rho_m * 0.6
    mass = 1.10 * (m_back + m_shroud + m_blades + m_hub)
    wr2 = 0.6 * mass * r2 ** 2

    # shaft by torsion with keyway + service factor
    omega = 2 * math.pi * duty.rpm / 60
    torque = perf.p_shaft / max(omega, 1e-6)
    d_shaft = (16 * torque * 2.0 / (math.pi * tau_allow_mpa * 1e6)) ** (1 / 3)
    d_shaft = max(0.020, math.ceil(d_shaft * 1000 / 5) * 5 / 1000)

    # first critical speed — overhung impeller on two bearings
    L = bearing_span_m if bearing_span_m > 0 else max(2.2 * w.b2, 0.30)
    a = overhang_m if overhang_m > 0 else max(0.8 * w.b2, 0.12)
    I = math.pi * d_shaft ** 4 / 64
    delta = mass * G * a ** 2 * (L + a) / (3 * 205e9 * I)   # steel shaft
    n_crit = 60 / (2 * math.pi) * math.sqrt(G / max(delta, 1e-9))
    ratio = duty.rpm / max(n_crit, 1.0)

    w2 = replace(w, shaft_d=d_shaft, hub_d=max(w.hub_d, 1.8 * d_shaft))
    return w2, Mechanical(shaft_d_mm=d_shaft * 1000, torque_nm=torque,
                          impeller_mass_kg=mass, wr2_kgm2=wr2,
                          tip_speed=perf.u2, critical_rpm=n_crit,
                          rpm_ratio=ratio,
                          static_deflection_mm=delta * 1000)

# ----------------------------------------------------------------------------
# Manufacturing geometry
# ----------------------------------------------------------------------------
@dataclass
class BladeArc:
    R_mm: float            # blade roll/press radius
    e_mm: float            # arc-centre distance from shaft axis
    center_xy: Tuple[float, float]
    chord_mm: float
    camber_deg: float      # arc included angle
    points: List[Tuple[float, float]]   # centreline, mm, blade #0
    wrap_deg: float        # angular wrap around the axis


def blade_arc_geometry(w: Wheel, n: int = 60) -> BladeArc:
    """Single-arc (Eck) blade between r1 and r2 with end angles beta1/beta2.

    R = (r2^2 - r1^2) / (2 (r2 cos b2 - r1 cos b1)); the arc is constructed
    numerically and the curvature side is chosen so the exit angle matches
    beta2 (this makes the same code valid for backward AND forward blades).
    """
    r1 = w.d1 / 2 * 1000
    r2 = w.d2 / 2 * 1000
    b1 = math.radians(min(max(w.beta1, 5), 175))
    b2 = math.radians(min(max(w.beta2, 5), 175))
    den = 2.0 * (r2 * math.cos(b2) - r1 * math.cos(b1))
    if abs(den) < 1e-6:
        den = 1e-6
    R = abs((r2 ** 2 - r1 ** 2) / den)
    R = min(max(R, 0.15 * (r2 - r1)), 50 * r2)

    p1 = np.array([r1, 0.0])
    d = np.array([math.sin(b1), math.cos(b1)])      # blade tangent at inlet

    best = None
    for sgn in (+1.0, -1.0):
        nvec = sgn * np.array([d[1], -d[0]])
        C = p1 + R * nvec
        phi0 = math.atan2(p1[1] - C[1], p1[0] - C[0])
        for direction in (+1.0, -1.0):
            pts = [tuple(p1)]
            phi = phi0
            ok = False
            for _ in range(4000):
                phi += direction * 0.002
                pt = C + R * np.array([math.cos(phi), math.sin(phi)])
                r = float(np.hypot(*pt))
                pts.append((float(pt[0]), float(pt[1])))
                if r >= r2:
                    ok = True
                    break
                if r < 0.5 * r1:
                    break
            if not ok:
                continue
            # exit blade angle vs local tangent
            v = np.array(pts[-1]) - np.array(pts[-2])
            radial = np.array(pts[-1]) / np.linalg.norm(pts[-1])
            tang = np.array([-radial[1], radial[0]])
            cosang = abs(float(np.dot(v, tang)) / np.linalg.norm(v))
            exit_beta = math.degrees(math.acos(min(max(float(
                np.dot(v, radial)) / np.linalg.norm(v), -1), 1)))
            err = abs(exit_beta - (90 - math.degrees(b2))) \
                if False else abs(math.degrees(
                    math.asin(min(max(abs(float(np.dot(v, radial))
                                          / np.linalg.norm(v)), 0), 1)))
                                  - min(math.degrees(b2), 180 - math.degrees(b2)))
            if best is None or err < best[0]:
                best = (err, pts, C)
    if best is None:      # fall back: integrate dtheta = dr/(r tan(beta))
        pts = [(r1, 0.0)]
        th = 0.0
        rs = np.linspace(r1, r2, n)
        for i in range(1, len(rs)):
            rm = 0.5 * (rs[i] + rs[i - 1])
            frac = (rm - r1) / (r2 - r1)
            beta = b1 + (b2 - b1) * frac
            th += (rs[i] - rs[i - 1]) / (rm * math.tan(beta))
            pts.append((rs[i] * math.cos(th), rs[i] * math.sin(th)))
        C = np.array([0.0, 0.0])
        best = (99, pts, C)
    _, pts, C = best
    # resample to n points
    pts = [pts[int(i)] for i in np.linspace(0, len(pts) - 1, n)]
    chord = math.dist(pts[0], pts[-1])
    camber = 2 * math.degrees(math.asin(min(chord / (2 * R), 1.0)))
    wrap = math.degrees(math.atan2(pts[-1][1], pts[-1][0])
                        - math.atan2(pts[0][1], pts[0][0]))
    return BladeArc(R_mm=R, e_mm=float(np.hypot(*C)),
                    center_xy=(float(C[0]), float(C[1])),
                    chord_mm=chord, camber_deg=camber,
                    points=pts, wrap_deg=wrap)


def volute_stations(w: Wheel, duty: Duty, perf: PerfPoint,
                    step_deg: int = 15) -> pd.DataFrame:
    """Log-spiral R(theta) stations for the volute side plates (mm)."""
    if FAMILIES[w.family].get("plenum") or w.r3 <= 0:
        return pd.DataFrame(columns=["theta_deg", "R_mm"])
    r2 = w.d2 / 2
    cu2 = max(perf.cu2, 1.0)
    k = duty.q_m3s / (2 * math.pi * w.volute_width * r2 * cu2)
    rows = [dict(theta_deg=th,
                 R_mm=round(w.r3 * math.exp(k * math.radians(th)) * 1000, 1))
            for th in range(0, 361, step_deg)]
    return pd.DataFrame(rows)


def inlet_cone_profile(w: Wheel) -> List[Tuple[float, float]]:
    """Simple conical inlet venturi profile (axial z vs radius, mm)."""
    r_th = w.d1 / 2 * 1000 * 0.98          # slight overlap into wheel
    r_in = 1.22 * r_th
    L = 0.55 * r_th
    pts = []
    for t in np.linspace(0, 1, 25):
        # tangent-arc style contraction (quarter-ellipse)
        r = r_th + (r_in - r_th) * (1 - math.sin(t * math.pi / 2))
        pts.append((t * L, r))
    return pts


# ----------------------------------------------------------------------------
# DXF manufacturing drawing (multi-view, dimensioned)
# ----------------------------------------------------------------------------
def _dim(msp, p1, p2, base, text=None):
    d = msp.add_linear_dim(base=base, p1=p1, p2=p2,
                           override={"dimtxt": None} if text is None else None,
                           dimstyle="EZDXF")
    if text:
        d.dimension.dxf.text = text
    d.render()
    return d


def create_dxf(w: Wheel, duty: Duty, perf: PerfPoint, mech: Mechanical,
               project: str = "Blower") -> bytes:
    if not HAS_EZDXF:
        return b"Install ezdxf to generate DXF files."
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    for name, color, lt in [("OUTLINE", 7, "CONTINUOUS"),
                            ("HIDDEN", 8, "DASHED"),
                            ("CENTER", 1, "CENTER"),
                            ("DIM", 1, "CONTINUOUS"),
                            ("TEXT", 3, "CONTINUOUS"),
                            ("TABLE", 4, "CONTINUOUS")]:
        if name not in doc.layers:
            doc.layers.add(name, color=color, linetype=lt)

    D2 = w.d2 * 1000
    D1 = w.d1 * 1000
    r2, r1 = D2 / 2, D1 / 2
    hub_r = w.hub_d / 2 * 1000
    sh_r = w.shaft_d / 2 * 1000
    b2 = w.b2 * 1000
    b1 = w.b1 * 1000
    arc = blade_arc_geometry(w)
    txt_h = max(D2 * 0.018, 5)

    def T(text, pos, h=None, layer="TEXT"):
        msp.add_text(text, height=h or txt_h,
                     dxfattribs={"layer": layer}
                     ).set_placement(pos, align=TextEntityAlignment.LEFT)

    # ---------------- VIEW A: impeller front view ----------------
    for r, layer in [(r2, "OUTLINE"), (r1, "OUTLINE"),
                     (hub_r, "OUTLINE"), (sh_r, "HIDDEN")]:
        msp.add_circle((0, 0), r, dxfattribs={"layer": layer})
    msp.add_line((-1.15 * r2, 0), (1.15 * r2, 0), dxfattribs={"layer": "CENTER"})
    msp.add_line((0, -1.15 * r2), (0, 1.15 * r2), dxfattribs={"layer": "CENTER"})
    for k in range(w.z):
        a = 2 * math.pi * k / w.z
        ca, sa = math.cos(a), math.sin(a)
        pts = [(x * ca - y * sa, x * sa + y * ca) for x, y in arc.points]
        msp.add_lwpolyline(pts, dxfattribs={"layer": "OUTLINE"})
    _dim(msp, (-r2, 0), (r2, 0), (0, -r2 - 3.5 * txt_h),
         text=f"%%c{D2:.0f}")
    _dim(msp, (-r1, 0), (r1, 0), (0, -r2 - 6.5 * txt_h),
         text=f"%%c{D1:.0f}")
    T(f"VIEW A - IMPELLER FRONT  ({w.z} BLADES EQUI-SPACED)",
      (-r2, r2 + 2.5 * txt_h))
    T(f"HUB %%c{2*hub_r:.0f} / SHAFT %%c{2*sh_r:.0f} H7 KEYWAY DIN 6885",
      (-r2, r2 + 1.2 * txt_h))

    # ---------------- VIEW B: side section ----------------
    ox = 2.35 * r2
    sides = 2 if w.dwdi else 1
    msp.add_line((ox, -1.15 * r2), (ox, 1.15 * r2), dxfattribs={"layer": "CENTER"})
    if not w.dwdi:
        # backplate at x = ox, shroud cone from (ox+b2 at r2) to (ox+b1 at r1)
        bp = ox
        msp.add_line((bp, -r2), (bp, r2), dxfattribs={"layer": "OUTLINE"})
        for s in (+1, -1):
            msp.add_line((bp, s * r2), (bp + b2, s * r2),
                         dxfattribs={"layer": "OUTLINE"})              # rim
            msp.add_line((bp + b2, s * r2), (bp + b1, s * r1),
                         dxfattribs={"layer": "OUTLINE"})              # shroud
            msp.add_line((bp + b1, s * r1), (bp + b1 + 0.25 * b1, s * r1),
                         dxfattribs={"layer": "OUTLINE"})              # inlet ring
            msp.add_line((bp, s * hub_r), (bp - 1.2 * 2 * hub_r, s * hub_r),
                         dxfattribs={"layer": "OUTLINE"})              # hub
            msp.add_line((bp - 1.2 * 2 * hub_r, s * sh_r),
                         (bp + 0.4 * b2, s * sh_r), dxfattribs={"layer": "HIDDEN"})
        _dim(msp, (bp, r2), (bp + b2, r2), (bp, r2 + 3.0 * txt_h),
             text=f"b2={b2:.0f}")
        _dim(msp, (bp + b1, r1), (bp, r1), (bp, r1 + 9.0 * txt_h),
             text=f"b1={b1:.0f}")
        T("VIEW B - SWSI SIDE SECTION", (bp - r2 * 0.3, r2 + 2.5 * txt_h + 4 * txt_h))
    else:
        bp = ox                                  # centre plate
        msp.add_line((bp, -r2), (bp, r2), dxfattribs={"layer": "OUTLINE"})
        for s in (+1, -1):
            for side in (+1, -1):
                msp.add_line((bp, s * r2), (bp + side * b2 / 2, s * r2),
                             dxfattribs={"layer": "OUTLINE"})
                msp.add_line((bp + side * b2 / 2, s * r2),
                             (bp + side * b1 / 2 * 1.15, s * r1),
                             dxfattribs={"layer": "OUTLINE"})
        _dim(msp, (bp - b2 / 2, r2), (bp + b2 / 2, r2), (bp, r2 + 3.0 * txt_h),
             text=f"b2 total={b2:.0f}")
        T("VIEW B - DWDI SIDE SECTION (CENTRE PLATE)",
          (bp - r2 * 0.4, r2 + 2.5 * txt_h + 4 * txt_h))

    # ---------------- VIEW C: single blade press detail ----------------
    oy = -(1.9 * r2)
    pts = [(x + 0, y + oy) for x, y in arc.points]
    msp.add_lwpolyline(pts, dxfattribs={"layer": "OUTLINE"})
    msp.add_line(pts[0], pts[-1], dxfattribs={"layer": "CENTER"})
    cx, cy = arc.center_xy
    msp.add_line((cx, cy + oy), pts[len(pts) // 2], dxfattribs={"layer": "DIM"})
    T(f"R{arc.R_mm:.1f}", ((cx + pts[len(pts)//2][0]) / 2,
                           (cy + oy + pts[len(pts)//2][1]) / 2 + txt_h), layer="DIM")
    ylab = oy - 2.0 * txt_h
    for i, line in enumerate([
        "VIEW C - BLADE PRESS/ROLL DETAIL (1 OF %d)" % w.z,
        f"ROLL RADIUS R = {arc.R_mm:.1f} mm   CHORD = {arc.chord_mm:.1f} mm",
        f"CAMBER ANGLE = {arc.camber_deg:.1f} deg   WRAP = {abs(arc.wrap_deg):.1f} deg",
        f"ARC CENTRE AT {arc.e_mm:.1f} mm FROM SHAFT AXIS",
        f"BLADE WIDTH = {b2:.0f} mm (DEVELOPED), THK = {w.t_blade*1000:.1f} mm",
        f"INLET ANGLE B1 = {w.beta1:.1f} deg / OUTLET ANGLE B2 = {w.beta2:.1f} deg (FROM TANGENT)",
    ]):
        T(line, (min(p[0] for p in pts), ylab - i * 1.6 * txt_h), layer="TABLE")

    # ---------------- VIEW D: volute side plate ----------------
    fam = FAMILIES[w.family]
    if not fam.get("plenum") and w.r3 > 0:
        odx = 2.35 * r2 + 3.2 * r2
        st = volute_stations(w, duty, perf)
        spts = [(odx + row.R_mm * math.cos(math.radians(row.theta_deg)),
                 row.R_mm * math.sin(math.radians(row.theta_deg)))
                for row in st.itertuples()]
        msp.add_lwpolyline(spts, dxfattribs={"layer": "OUTLINE"})
        msp.add_circle((odx, 0), r2, dxfattribs={"layer": "HIDDEN"})
        msp.add_circle((odx, 0), w.r3 * 1000, dxfattribs={"layer": "CENTER"})
        # discharge duct: tangential box from theta=0 station upward
        R0 = st.iloc[-1].R_mm
        H = w.outlet_h * 1000
        msp.add_lwpolyline([(odx + R0, 0), (odx + R0, 1.4 * r2),
                            (odx + R0 - H, 1.4 * r2)],
                           dxfattribs={"layer": "OUTLINE"})
        msp.add_line((odx + w.r3 * 1000 + w.cutoff_clr * 1000, 0),
                     (odx + R0 - H, 1.4 * r2), dxfattribs={"layer": "OUTLINE"})
        T("VIEW D - VOLUTE SIDE PLATE (LOG SPIRAL)", (odx - r2, 1.55 * r2 + 2 * txt_h))
        T(f"CASING WIDTH B = {w.volute_width*1000:.0f} mm, "
          f"CUTOFF CLR = {w.cutoff_clr*1000:.0f} mm, "
          f"DISCHARGE {w.outlet_w*1000:.0f} x {w.outlet_h*1000:.0f} mm",
          (odx - r2, 1.55 * r2 + 0.4 * txt_h))
        T("SPIRAL STATIONS (theta deg : R mm)", (odx + 1.35 * R0, 1.3 * r2),
          layer="TABLE")
        for i, row in enumerate(st.itertuples()):
            T(f"{row.theta_deg:>4.0f} : {row.R_mm:8.1f}",
              (odx + 1.35 * R0, 1.3 * r2 - (i + 1) * 1.45 * txt_h), layer="TABLE")

    # ---------------- VIEW E: inlet cone ----------------
    cone = inlet_cone_profile(w)
    oex, oey = 2.35 * r2, -(1.9 * r2)
    for s in (+1, -1):
        msp.add_lwpolyline([(oex + zz, oey + s * rr) for zz, rr in cone],
                           dxfattribs={"layer": "OUTLINE"})
    msp.add_line((oex - 5, oey), (oex + cone[-1][0] + 5, oey),
                 dxfattribs={"layer": "CENTER"})
    T("VIEW E - INLET CONE (SPUN/ROLLED)", (oex, oey + cone[0][1] + 2.5 * txt_h))
    T(f"THROAT %%c{2*cone[0][1]:.0f} / MOUTH %%c{2*cone[-1][1]:.0f} "
      f"/ DEPTH {cone[-1][0]:.0f} mm", (oex, oey + cone[0][1] + 1.0 * txt_h))

    # ---------------- title block ----------------
    tby = -(1.9 * r2) - 14 * txt_h
    tbw = 7.0 * r2
    msp.add_lwpolyline([(-1.2 * r2, tby), (-1.2 * r2 + tbw, tby),
                        (-1.2 * r2 + tbw, tby - 8 * txt_h),
                        (-1.2 * r2, tby - 8 * txt_h), (-1.2 * r2, tby)],
                       dxfattribs={"layer": "OUTLINE"})
    info = [
        f"PROJECT: {project}   FAMILY: {w.family}   ARRANGEMENT: {w.arrangement}",
        f"DUTY: {duty.airflow_m3h:,.0f} m3/h @ {duty.static_pa:.0f} Pa STATIC, "
        f"{duty.rpm:.0f} RPM, rho={duty.finalize().density:.3f} kg/m3",
        f"PREDICTED: Pshaft={perf.p_shaft/1000:.2f} kW  eta_t={perf.eta_total*100:.1f}%  "
        f"TIP={perf.u2:.1f} m/s  MASS~{mech.impeller_mass_kg:.1f} kg",
        f"UNITS: mm  |  UNTOLERANCED DIMS +/-0.5  |  BALANCE ISO 21940 G6.3",
        "PRELIMINARY - VERIFY BY AMCA 210/ISO 5801 TEST BEFORE SERIES PRODUCTION",
    ]
    for i, line in enumerate(info):
        T(line, (-1.15 * r2, tby - (i + 1) * 1.5 * txt_h), layer="TABLE")

    s = io.StringIO()
    doc.write(s)
    return s.getvalue().encode()

# ----------------------------------------------------------------------------
# Preview figures (matplotlib) for the UI / PDF report
# ----------------------------------------------------------------------------
def impeller_figure(w: Wheel, duty: Duty, perf: PerfPoint):
    import matplotlib.pyplot as plt
    arc = blade_arc_geometry(w)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    r2, r1 = w.d2 / 2 * 1000, w.d1 / 2 * 1000
    for r, ls in [(r2, "-"), (r1, "-"), (w.hub_d / 2 * 1000, "--")]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=ls, lw=1.4))
    for k in range(w.z):
        a = 2 * math.pi * k / w.z
        ca, sa = math.cos(a), math.sin(a)
        p = np.array([(x * ca - y * sa, x * sa + y * ca) for x, y in arc.points])
        ax.plot(p[:, 0], p[:, 1], lw=0.9, color="tab:blue")
    st_df = volute_stations(w, duty, perf)
    if len(st_df):
        vx = [row.R_mm * math.cos(math.radians(row.theta_deg)) for row in st_df.itertuples()]
        vy = [row.R_mm * math.sin(math.radians(row.theta_deg)) for row in st_df.itertuples()]
        ax.plot(vx, vy, lw=2, color="tab:red")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"{w.family} — D2 {w.d2*1000:.0f} mm, {w.z} blades, "
                 f"blade R {arc.R_mm:.0f} mm")
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    return fig


def curves_figure(curve: pd.DataFrame, duty: Duty, perf: PerfPoint):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(curve.Flow_m3h, curve.Static_Pa, label="Static")
    axes[0].plot(curve.Flow_m3h, curve.Total_Pa, ls="--", label="Total")
    axes[0].scatter([duty.airflow_m3h], [duty.static_pa], color="red", zorder=5,
                    label="Duty")
    axes[0].set_ylabel("Pressure Pa"); axes[0].legend()
    axes[1].plot(curve.Flow_m3h, curve.Shaft_kW, color="tab:green")
    axes[1].scatter([duty.airflow_m3h], [perf.p_shaft / 1000], color="red", zorder=5)
    axes[1].set_ylabel("Shaft kW")
    axes[2].plot(curve.Flow_m3h, curve.EtaStatic * 100, label="Static")
    axes[2].plot(curve.Flow_m3h, curve.EtaTotal * 100, ls="--", label="Total")
    axes[2].set_ylabel("Efficiency %"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Flow m3/h"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def octave_figure(ac: Acoustics):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([str(f) for f in OCTAVES], ac.lw_bands, color="tab:purple")
    ax.set_xlabel("Octave band Hz"); ax.set_ylabel("Lw dB")
    ax.set_title(f"Sound power spectrum — Lw {ac.lw_total} dB / "
                 f"LwA {ac.lwa_total} dB(A) / ~{ac.lpa_1m} dB(A) at 1 m")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_png(fig) -> bytes:
    import matplotlib.pyplot as plt
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return bio.getvalue()


# ----------------------------------------------------------------------------
# Full design wrapper
# ----------------------------------------------------------------------------
@dataclass
class Design:
    duty: Duty
    wheel: Wheel
    perf: PerfPoint
    mech: Mechanical
    ac: Acoustics
    curve: pd.DataFrame
    motor_kw: float
    warnings: List[str]


def run_design(duty: Duty, family: str, arrangement: str,
               d1d2: float, b2d2: float, beta1: float, beta2: float, z: int,
               target_v_out: float = 12.0, t_blade_mm: float = 3.0,
               material: str = "Mild Steel IS2062 / S275",
               drive_eff: float = 0.97, motor_eff: float = 0.90,
               margin_pct: float = 15.0, tau_mpa: float = 40.0,
               calibration: Optional[dict] = None,
               auto_beta1: bool = False) -> Design:
    duty = duty.finalize()
    w, perf, warns = size_wheel(duty, family, arrangement, d1d2, b2d2,
                                beta1, beta2, z, target_v_out,
                                t_blade_mm / 1000, material, calibration,
                                auto_beta1)
    w, mech = mechanical(w, duty, perf, tau_mpa)
    curve = performance_curve(w, duty, calibration)
    ac = acoustics(w, duty, perf, curve, calibration)
    # motor sizing: forward-curved families overload — size at max curve power
    fam_code = FAMILIES[family]["code"]
    p_size = curve.Shaft_kW.max() if fam_code in ("fc", "sirocco") \
        else perf.p_shaft / 1000
    if fam_code in ("fc", "sirocco") and p_size > 1.1 * perf.p_shaft / 1000:
        warns.append(f"Overloading family: motor sized for {p_size:.2f} kW at "
                     f"max curve flow (duty needs {perf.p_shaft/1000:.2f} kW).")
    motor = selected_motor(p_size / max(drive_eff * motor_eff, 0.5)
                           * (1 + margin_pct / 100))
    if mech.rpm_ratio > 0.75:
        warns.append(f"Operating speed is {mech.rpm_ratio*100:.0f}% of first "
                     "critical — stiffen shaft / shorten overhang.")
    return Design(duty, w, perf, mech, ac, curve, motor, warns)


# ----------------------------------------------------------------------------
# Optimizer — grid over family proportion ranges through the physics engine
# ----------------------------------------------------------------------------
def optimize(duty: Duty, families: List[str], arrangement: str,
             target_v_out: float = 12.0,
             material: str = "Mild Steel IS2062 / S275",
             calibration: Optional[dict] = None,
             w_eff: float = 100.0, w_sound: float = 1.2, w_size: float = 8.0,
             n_keep: int = 40) -> Tuple[Optional[Design], pd.DataFrame]:
    duty = duty.finalize()
    rows, best = [], None
    for fname in families:
        f = FAMILIES[fname]
        d1s = np.linspace(*f["d1d2_rng"], 4)
        b2s = np.linspace(*f["b2d2_rng"], 4)
        be2 = np.linspace(*f["beta2_rng"], 4)
        zs = np.unique(np.linspace(*f["z_rng"], 3).astype(int))
        for d1d2 in d1s:
            for b2d2 in b2s:
                for beta2 in be2:
                    for z in zs:
                        try:
                            w, p, _ = size_wheel(duty, fname, arrangement,
                                                 float(d1d2), float(b2d2),
                                                 f["beta1"], float(beta2),
                                                 int(z), target_v_out,
                                                 material=material,
                                                 calibration=calibration,
                                                 auto_beta1=True)
                        except Exception:
                            continue
                        curve = performance_curve(w, duty, calibration, n=17)
                        ac = acoustics(w, duty, p, curve, calibration)
                        feasible = (p.u2 <= min(MATERIALS[material]["max_tip"],
                                                f["tip_max"]) and
                                    abs(p.dp_static - duty.static_pa)
                                    < max(0.03 * duty.static_pa, 8))
                        score = (w_eff * (1 - p.eta_static)
                                 + w_sound * max(ac.lpa_1m - 60, 0)
                                 + w_size * w.d2
                                 + (0 if feasible else 500))
                        rows.append(dict(Family=fname, D1_D2=round(d1d2, 3),
                                         b2_D2=round(b2d2, 3),
                                         beta2=round(beta2, 1), z=int(z),
                                         D2_mm=round(w.d2 * 1000, 0),
                                         EtaStatic=round(p.eta_static, 3),
                                         Shaft_kW=round(p.p_shaft / 1000, 2),
                                         LpA_1m=ac.lpa_1m,
                                         Tip_ms=round(p.u2, 1),
                                         Feasible=feasible,
                                         Score=round(score, 2)))
                        if feasible and (best is None or score < best[0]):
                            best = (score, fname, float(d1d2), float(b2d2),
                                    float(beta2), int(z))
    df = pd.DataFrame(rows).sort_values(["Feasible", "Score"],
                                        ascending=[False, True]).head(n_keep)
    if best is None:
        return None, df
    _, fname, d1d2, b2d2, beta2, z = best
    des = run_design(duty, fname, arrangement, d1d2, b2d2,
                     FAMILIES[fname]["beta1"], beta2, z, target_v_out,
                     material=material, calibration=calibration,
                     auto_beta1=True)
    return des, df



# ----------------------------------------------------------------------------
# Angle guide and AHRI 431 preliminary rating helper functions
# ----------------------------------------------------------------------------
def check_password() -> bool:
    """Streamlit password gate. Supports either APP_PASSWORD or [auth].password."""
    try:
        import streamlit as st
    except Exception:
        return True
    password = None
    try:
        password = st.secrets.get("APP_PASSWORD", None)
        if password is None and "auth" in st.secrets:
            password = st.secrets["auth"].get("password", None)
    except Exception:
        password = None
    if not password:
        return True
    if st.session_state.get("password_ok", False):
        return True
    st.subheader("Login")
    entered = st.text_input("App password", type="password")
    if entered:
        if entered == password:
            st.session_state["password_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


def angle_guide_figure():
    """Corrected angle convention drawing used by this code.

    Convention: beta2 is the angle between the tangential blade speed U2 and
    the relative discharge velocity W2. This is the same convention used in
    wheel_performance(): cu2 = sigma*u2 - cm2/tan(beta2).
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    cases = [
        ("Backward curved", 38, "beta2 < 90 deg"),
        ("Radial straight", 90, "beta2 = 90 deg"),
        ("Forward / Sirocco", 150, "beta2 > 90 deg"),
    ]
    for ax, (title, beta, note) in zip(axes, cases):
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.35); ax.set_ylim(-0.75, 0.85)
        ax.axhline(0, lw=1, color='0.4')
        ax.axvline(0, lw=1, color='0.85')
        # U2 tangent direction
        ax.arrow(0, 0, 0.9, 0, head_width=0.04, length_includes_head=True, lw=2)
        ax.text(0.92, 0.02, 'U2 tangent / 0 deg', fontsize=8)
        # Vr2 radial outward
        ax.arrow(0, 0, 0, 0.55, head_width=0.04, length_includes_head=True, lw=1.5)
        ax.text(0.03, 0.57, 'Vr2 radial outward', fontsize=8)
        # W2 at beta
        th = math.radians(beta)
        ax.arrow(0, 0, 0.75*math.cos(th), 0.75*math.sin(th),
                 head_width=0.04, length_includes_head=True, lw=2)
        ax.text(0.78*math.cos(th), 0.78*math.sin(th), 'W2', fontsize=9)
        # approximate V2: vector sum W2 + U2 (drawn as absolute leaving direction)
        vx = 0.9 + 0.42*math.cos(th); vy = 0.42*math.sin(th)
        scale = 0.75 / max((vx*vx+vy*vy)**0.5, 1e-6)
        ax.arrow(0, 0, vx*scale, vy*scale, head_width=0.04,
                 length_includes_head=True, lw=1.5, linestyle='--')
        ax.text(vx*scale+0.02, vy*scale, 'V2', fontsize=9)
        # beta arc
        phis = np.linspace(0, th, 60)
        ax.plot(0.23*np.cos(phis), 0.23*np.sin(phis), lw=1.3)
        ax.text(0.27*math.cos(th/2), 0.27*math.sin(th/2), 'beta2', fontsize=9)
        ax.set_title(f"{title}\n{note}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.0, -0.62, "Code formula: cu2 = sigma*U2 - cm2/tan(beta2)", fontsize=8)
    fig.suptitle("Angle definitions used in this toolkit: beta2 measured from tangent U2 to relative velocity W2", fontsize=12)
    fig.tight_layout()
    return fig


def ahri431_rating_frame(des: Design) -> pd.DataFrame:
    """Preliminary AHRI 431 SI rating sheet for central-station AHU supply fans.

    This is not a certification calculation. It organizes the app output in the
    fields normally needed before AMCA 210 / ASHRAE 51 laboratory testing and
    AHRI 430/431 publication.
    """
    w, p, m, ac, duty = des.wheel, des.perf, des.mech, des.ac, des.duty
    rows = [
        ("Standard reference", "AHRI 431 SI / AHRI 430 I-P preliminary format"),
        ("Test method reference", "ANSI/AMCA 210 / ANSI/ASHRAE 51 or ISO 5801 laboratory test required"),
        ("Certification status", "NOT AHRI certified - engineering prediction only"),
        ("Application", "Central-station AHU supply fan / HVAC blower"),
        ("Fan family", w.family),
        ("Arrangement", w.arrangement),
        ("Rated airflow", f"{duty.airflow_m3h:,.0f} m3/h"),
        ("Rated fan static pressure", f"{duty.static_pa:.0f} Pa"),
        ("Predicted fan total pressure", f"{p.dp_total:.0f} Pa"),
        ("Fan speed", f"{duty.rpm:.0f} rpm"),
        ("Air density at rating", f"{duty.density:.3f} kg/m3"),
        ("Shaft / brake power", f"{p.p_shaft/1000:.2f} kW"),
        ("Selected motor", f"{des.motor_kw:.1f} kW"),
        ("Static efficiency", f"{p.eta_static*100:.1f} %"),
        ("Total efficiency", f"{p.eta_total*100:.1f} %"),
        ("Outlet air velocity", f"{p.v_out:.1f} m/s"),
        ("Discharge flange", f"{w.outlet_w*1000:.0f} x {w.outlet_h*1000:.0f} mm" if w.outlet_area > 1e-5 else "Plenum/plug fan - no volute outlet"),
        ("Sound", f"Predicted LwA {ac.lwa_total:.1f} dB(A), LpA@1m {ac.lpa_1m:.1f} dB(A); certified sound test required"),
        ("Mechanical check", f"Tip {p.u2:.1f} m/s; first critical {m.critical_rpm:.0f} rpm; balance ISO 21940"),
        ("Required lab deliverables", "Airflow, fan static pressure, fan total pressure, speed, brake power, efficiency, sound if declared"),
        ("Important warning", "Catalogue/AMCA/AHRI ratings must come from test data; do not publish this as certified performance."),
    ]
    return pd.DataFrame(rows, columns=["AHRI 431 item", "Preliminary value / requirement"])

# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def _results_frame(des: Design) -> pd.DataFrame:
    w, p, m, ac = des.wheel, des.perf, des.mech, des.ac
    duty = des.duty
    arc = blade_arc_geometry(w)
    rows = [
        ("Family / arrangement", f"{w.family} / {w.arrangement}"),
        ("Duty flow", f"{duty.airflow_m3h:,.0f} m3/h"),
        ("Duty static pressure", f"{duty.static_pa:.0f} Pa"),
        ("Speed", f"{duty.rpm:.0f} rpm"),
        ("Air density", f"{duty.density:.3f} kg/m3"),
        ("Impeller OD D2", f"{w.d2*1000:.1f} mm"),
        ("Inlet dia D1", f"{w.d1*1000:.1f} mm  (D1/D2={w.d1/w.d2:.2f})"),
        ("Width b2 (total)", f"{w.b2*1000:.1f} mm  (b2/D2={w.b2/w.d2:.2f})"),
        ("Blades z / beta1 / beta2", f"{w.z} / {w.beta1:.1f} / {w.beta2:.1f} deg"),
        ("Blade roll radius", f"{arc.R_mm:.1f} mm (chord {arc.chord_mm:.0f} mm)"),
        ("Slip factor", f"{p.slip:.3f}"),
        ("Tip speed", f"{p.u2:.1f} m/s"),
        ("Delivered static / total", f"{p.dp_static:.0f} / {p.dp_total:.0f} Pa"),
        ("Euler pressure", f"{p.dp_euler:.0f} Pa"),
        ("Shaft power", f"{p.p_shaft/1000:.2f} kW"),
        ("Total / static efficiency", f"{p.eta_total*100:.1f} / {p.eta_static*100:.1f} %"),
        ("Selected motor", f"{des.motor_kw:.1f} kW"),
        ("Volute width / throat", f"{w.volute_width*1000:.0f} mm / "
                                  f"{w.throat_area*1e6:.0f} mm2"),
        ("Discharge flange", f"{w.outlet_w*1000:.0f} x {w.outlet_h*1000:.0f} mm "
                             f"@ {p.v_out:.1f} m/s"),
        ("Shaft dia / torque", f"{m.shaft_d_mm:.0f} mm / {m.torque_nm:.1f} Nm"),
        ("Impeller mass / WR2", f"{m.impeller_mass_kg:.1f} kg / {m.wr2_kgm2:.2f} kgm2"),
        ("First critical speed", f"{m.critical_rpm:.0f} rpm "
                                 f"(operating at {m.rpm_ratio*100:.0f}%)"),
        ("Sound power Lw / LwA", f"{ac.lw_total} dB / {ac.lwa_total} dB(A)"),
        ("Est. LpA at 1 m", f"{ac.lpa_1m} dB(A)"),
        ("Blade pass frequency", f"{ac.bpf_hz:.0f} Hz"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def losses_frame(perf: PerfPoint) -> pd.DataFrame:
    rows = [(k.title(), round(v, 1),
             f"{v/max(perf.dp_euler,1e-6)*100:.1f}%")
            for k, v in perf.losses.items()]
    rows.append(("Delivered total", round(perf.dp_total, 1),
                 f"{perf.dp_total/max(perf.dp_euler,1e-6)*100:.1f}%"))
    return pd.DataFrame(rows, columns=["Item", "Pa", "% of Euler"])


def create_excel(des: Design, opt_df: Optional[pd.DataFrame] = None) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook()

    def put(ws, df):
        for c, col in enumerate(df.columns, 1):
            cell = ws.cell(1, c, str(col))
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="DDDDDD")
        for ri, row in enumerate(df.itertuples(index=False), 2):
            for ci, v in enumerate(row, 1):
                ws.cell(ri, ci, str(v) if isinstance(v, (list, tuple, dict)) else v)
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(
                12, min(50, max(len(str(c.value or "")) for c in col) + 2))

    ws = wb.active; ws.title = "Design"
    put(ws, _results_frame(des))
    put(wb.create_sheet("Losses"), losses_frame(des.perf))
    put(wb.create_sheet("Curve"), des.curve.round(3))
    oct_df = pd.DataFrame({"Band Hz": OCTAVES, "Lw dB": des.ac.lw_bands,
                           "A-wt": A_WEIGHT})
    put(wb.create_sheet("Octave bands"), oct_df)
    st_df = volute_stations(des.wheel, des.duty, des.perf)
    if len(st_df):
        put(wb.create_sheet("Volute stations"), st_df)
    arc = blade_arc_geometry(des.wheel)
    blade_df = pd.DataFrame(arc.points, columns=["x_mm", "y_mm"]).round(2)
    put(wb.create_sheet("Blade centreline"), blade_df)
    if opt_df is not None and len(opt_df):
        put(wb.create_sheet("Optimizer"), opt_df)
    bio = io.BytesIO(); wb.save(bio)
    return bio.getvalue()


def create_pdf(des: Design) -> bytes:
    if not HAS_REPORTLAB:
        return b"Install reportlab for PDF reports."
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4)
    ss = getSampleStyleSheet()
    story = [Paragraph("Centrifugal Blower Design Report — v18", ss["Title"]),
             Paragraph("Velocity-triangle derived design. Preliminary: "
                       "verify by AMCA 210 / ISO 5801 test.", ss["BodyText"]),
             Spacer(1, 8)]
    data = [["Parameter", "Value"]] + _results_frame(des).values.tolist()
    t = Table(data, colWidths=[190, 300])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story += [t, Spacer(1, 8)]
    ld = [["Loss item", "Pa", "% Euler"]] + losses_frame(des.perf).values.tolist()
    t2 = Table(ld, colWidths=[160, 100, 100])
    t2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                            ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story += [Paragraph("Loss breakdown at duty", ss["Heading2"]), t2, Spacer(1, 8)]
    for w in des.warnings:
        story.append(Paragraph(f"WARNING: {w}", ss["BodyText"]))
    try:
        story += [Spacer(1, 6),
                  RLImage(io.BytesIO(fig_png(impeller_figure(des.wheel, des.duty,
                                                             des.perf))),
                          width=250, height=250),
                  RLImage(io.BytesIO(fig_png(curves_figure(des.curve, des.duty,
                                                           des.perf))),
                          width=470, height=145),
                  RLImage(io.BytesIO(fig_png(octave_figure(des.ac))),
                          width=380, height=195)]
    except Exception:
        pass
    doc.build(story)
    return bio.getvalue()


def make_zip(des: Design, opt_df=None, project="Blower") -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.pdf", create_pdf(des))
        z.writestr("calculations.xlsx", create_excel(des, opt_df))
        z.writestr("manufacturing.dxf",
                   create_dxf(des.wheel, des.duty, des.perf, des.mech, project))
        z.writestr("impeller.png", fig_png(impeller_figure(des.wheel, des.duty,
                                                           des.perf)))
        z.writestr("curves.png", fig_png(curves_figure(des.curve, des.duty,
                                                       des.perf)))
        z.writestr("octaves.png", fig_png(octave_figure(des.ac)))
    return bio.getvalue()


# ----------------------------------------------------------------------------
# Self-test / calibration harness (python blower_design_v17.py)
# ----------------------------------------------------------------------------
def selftest():
    cases = [
        ("Airfoil Backward (AF)", "SWSI (single inlet)", 25000, 900, 1450),
        ("Backward Curved (BC)", "SWSI (single inlet)", 15000, 700, 1450),
        ("Backward Inclined (BI flat)", "SWSI (single inlet)", 12000, 500, 1450),
        ("Radial Tip (RT)", "SWSI (single inlet)", 8000, 1500, 2900),
        ("Radial Straight (Paddle)", "SWSI (single inlet)", 5000, 1200, 2900),
        ("Forward Curved (FC)", "SWSI (single inlet)", 3000, 250, 950),
        ("Multi-Blade Sirocco (Cage)", "DWDI (double inlet)", 2000, 150, 900),
        ("Plug / Plenum (BC, no volute)", "SWSI (single inlet)", 20000, 600, 1450),
    ]
    print(f"{'Family':<32}{'D2 mm':>8}{'kW':>8}{'etaT':>7}{'etaS':>7}"
          f"{'tip':>7}{'LpA':>7}{'dPs err':>9}")
    for fam, arr, q, ps, rpm in cases:
        f = FAMILIES[fam]
        duty = Duty(q, ps, rpm)
        des = run_design(duty, fam, arr, f["d1d2"], f["b2d2"], f["beta1"],
                         f["beta2"], f["z"], auto_beta1=True)
        err = des.perf.dp_static - ps
        print(f"{fam:<32}{des.wheel.d2*1000:>8.0f}{des.perf.p_shaft/1000:>8.2f}"
              f"{des.perf.eta_total:>7.3f}{des.perf.eta_static:>7.3f}"
              f"{des.perf.u2:>7.1f}{des.ac.lpa_1m:>7.1f}{err:>9.1f}")
    # exercise exports on one case
    duty = Duty(15000, 700, 1450)
    f = FAMILIES["Backward Curved (BC)"]
    des = run_design(duty, "Backward Curved (BC)", "SWSI (single inlet)",
                     f["d1d2"], f["b2d2"], f["beta1"], f["beta2"], f["z"],
                     auto_beta1=True)
    dxf = create_dxf(des.wheel, des.duty, des.perf, des.mech, "SELFTEST")
    xls = create_excel(des)
    pdf = create_pdf(des)
    zp = make_zip(des)
    print(f"exports: dxf {len(dxf)//1024} kB, xlsx {len(xls)//1024} kB, "
          f"pdf {len(pdf)//1024} kB, zip {len(zp)//1024} kB")
    print("non-overloading check (BC power at 130% flow / duty power):",
          round(float(des.curve.Shaft_kW.iloc[-3]) / (des.perf.p_shaft/1000), 3))
    fc = FAMILIES["Forward Curved (FC)"]
    dfc = run_design(Duty(3000, 250, 950), "Forward Curved (FC)",
                     "SWSI (single inlet)", fc["d1d2"], fc["b2d2"],
                     fc["beta1"], fc["beta2"], fc["z"], auto_beta1=True)
    print("overloading check (FC power at 130% flow / duty power):",
          round(float(dfc.curve.Shaft_kW.iloc[-3]) / (dfc.perf.p_shaft/1000), 3))


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if __name__ == "__main__" and not _in_streamlit():
    selftest()

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def run_app():
    import streamlit as st
    from blower_toolkit.auth import require_password

    st.set_page_config(page_title="Blower Design Toolkit v24", layout="wide")
    require_password()
    st.title("Centrifugal Blower Design & Manufacturing Toolkit v24")
    st.caption("Velocity-triangle physics engine · 8 wheel families · AHRI 431 preliminary rating · "
               "SWSI/DWDI · octave-band acoustics · dimensioned DXF output")

    if "calibration" not in st.session_state:
        st.session_state.calibration = json.loads(json.dumps(DEFAULT_CALIBRATION))

    with st.sidebar:
        st.header("Duty")
        airflow = st.number_input("Airflow (m³/h)", 50.0, 500000.0, 15000.0, 250.0)
        sp = st.number_input("Static pressure (Pa)", 20.0, 6000.0, 700.0, 25.0)
        rpm = st.number_input("Speed (RPM)", 100.0, 6000.0, 1450.0, 10.0)
        st.header("Air")
        temp = st.number_input("Temperature (°C)", -20.0, 90.0, 35.0, 1.0)
        alt = st.number_input("Altitude (m)", 0.0, 4000.0, 0.0, 50.0)
        rh = st.number_input("Relative humidity (%)", 0.0, 100.0, 50.0, 5.0)
        rho_auto = air_density(temp, alt, rh)
        use_auto = st.checkbox(f"Use computed density ({rho_auto:.3f} kg/m³)", True)
        rho = rho_auto if use_auto else st.number_input("Density (kg/m³)",
                                                        0.4, 1.5, 1.2, 0.01)
        st.header("Selection")
        mode = st.radio("Mode", ["Manual geometry", "Optimize geometry"])
        fam_pick = st.selectbox("Fan family",
                                (["All families"] if mode == "Optimize geometry"
                                 else []) + list(FAMILIES.keys()))
        arrangement = st.selectbox("Arrangement", ARRANGEMENTS)
        st.header("Outlet & build")
        v_out = st.number_input("Target outlet velocity (m/s)", 5.0, 20.0, 12.0, 0.5)
        material = st.selectbox("Material", list(MATERIALS.keys()))
        t_blade = st.number_input("Blade thickness (mm)", 0.8, 8.0, 3.0, 0.5)
        st.header("Drive & motor")
        drive_eff = st.number_input("Drive efficiency", 0.80, 1.00, 0.97, 0.01)
        motor_eff = st.number_input("Motor efficiency", 0.75, 0.98, 0.91, 0.01)
        margin = st.number_input("Motor margin (%)", 0.0, 40.0, 15.0, 1.0)
        project = st.text_input("Project / drawing title", "HVAC Blower")

    duty = Duty(airflow, sp, rpm, temp, alt, rh,
                rho if not use_auto else 0.0).finalize()
    cal = st.session_state.calibration

    opt_df = None
    if mode == "Optimize geometry":
        fams = list(FAMILIES.keys()) if fam_pick == "All families" else [fam_pick]
        with st.spinner("Running physics optimizer over family proportion "
                        "ranges…"):
            des, opt_df = optimize(duty, fams, arrangement, v_out, material, cal)
        if des is None:
            st.error("No feasible geometry found — change speed, arrangement "
                     "or family.")
            st.dataframe(opt_df, use_container_width=True)
            st.stop()
    else:
        f = FAMILIES[fam_pick]
        with st.expander("Geometry (family defaults pre-filled)", expanded=True):
            c = st.columns(5)
            d1d2 = c[0].number_input("D1/D2", 0.30, 0.92, float(f["d1d2"]), 0.01)
            b2d2 = c[1].number_input("b2/D2", 0.04, 0.65, float(f["b2d2"]), 0.01)
            beta2 = c[2].number_input("β2 (° from tangent)", 20.0, 170.0,
                                      float(f["beta2"]), 1.0)
            z = int(c[3].number_input("Blades z", 4, 72, int(f["z"]), 1))
            ab1 = c[4].checkbox("Match β1 to flow", True)
            beta1 = f["beta1"] if ab1 else c[4].number_input(
                "β1 (°)", 10.0, 95.0, float(f["beta1"]), 1.0)
        des = run_design(duty, fam_pick, arrangement, d1d2, b2d2, beta1,
                         beta2, z, v_out, t_blade, material, drive_eff,
                         motor_eff, margin, calibration=cal, auto_beta1=ab1)

    w, p, m, ac = des.wheel, des.perf, des.mech, des.ac
    st.info(FAMILIES[w.family]["guide"])
    cc = st.columns(6)
    cc[0].metric("Impeller D2", f"{w.d2*1000:.0f} mm")
    cc[1].metric("Shaft power", f"{p.p_shaft/1000:.2f} kW")
    cc[2].metric("Static efficiency", f"{p.eta_static*100:.1f} %")
    cc[3].metric("Motor", f"{des.motor_kw:.1f} kW")
    cc[4].metric("LpA @ 1 m", f"{ac.lpa_1m:.0f} dB(A)")
    cc[5].metric("Tip speed", f"{p.u2:.0f} m/s")
    for wa in des.warnings:
        st.warning(wa)

    tabs = st.tabs(["Design", "Angle Guide", "AHRI 431", "Losses", "Curves", "Acoustics",
                    "Geometry & Drawing", "Mechanical",
                    "Optimizer", "Calibration", "Exports"])
    with tabs[0]:
        st.dataframe(_results_frame(des), use_container_width=True,
                     hide_index=True)
    with tabs[1]:
        st.pyplot(angle_guide_figure())
        st.markdown("""
**Angle convention used by the calculations**

- `beta2` is measured from the **tangent direction U2** to the **relative discharge velocity W2**.
- Backward curved: `beta2 < 90 deg`.
- Radial straight: `beta2 = 90 deg`.
- Forward curved / Sirocco: `beta2 > 90 deg`.

This is why the code uses: `cu2 = slip*U2 - cm2/tan(beta2)`.
""")
    with tabs[2]:
        st.subheader("AHRI 431 / AHRI 430 preliminary AHU rating sheet")
        st.warning("This is not AHRI certification. Use this tab to prepare rating data for laboratory testing. Final AHU fan ratings must be validated by ANSI/AMCA 210 / ANSI/ASHRAE 51 or ISO 5801 test data and published under the applicable AHRI program only after qualification.")
        st.dataframe(ahri431_rating_frame(des), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(losses_frame(p), use_container_width=True, hide_index=True)
        st.caption("Where the Euler pressure goes at the duty point. Large "
                   "incidence → wrong β1 or D1; large diffusion → channel "
                   "loaded too hard (raise z or b2); large volute → casing "
                   "too tight.")
    with tabs[4]:
        st.pyplot(curves_figure(des.curve, duty, p))
        st.dataframe(des.curve.round(2), use_container_width=True,
                     hide_index=True)
    with tabs[5]:
        st.pyplot(octave_figure(ac))
        st.dataframe(pd.DataFrame({"Band Hz": OCTAVES,
                                   "Lw dB": ac.lw_bands}),
                     use_container_width=True, hide_index=True)
        st.caption(f"Graham method · BPF {ac.bpf_hz:.0f} Hz gets +BFI · "
                   f"off-peak correction {ac.off_peak_corr:.0f} dB. Duct "
                   "attenuation and room effect are NOT included.")
    with tabs[6]:
        st.pyplot(impeller_figure(w, duty, p))
        arc = blade_arc_geometry(w)
        g1, g2 = st.columns(2)
        g1.markdown(f"**Blade press data** — roll radius R = "
                    f"{arc.R_mm:.1f} mm, chord {arc.chord_mm:.0f} mm, camber "
                    f"{arc.camber_deg:.1f}°, arc centre {arc.e_mm:.1f} mm "
                    f"from axis.")
        st_df = volute_stations(w, duty, p)
        if len(st_df):
            g2.markdown("**Volute spiral stations**")
            g2.dataframe(st_df, hide_index=True, height=240)
    with tabs[7]:
        md = pd.DataFrame([
            ("Shaft diameter", f"{m.shaft_d_mm:.0f} mm"),
            ("Torque", f"{m.torque_nm:.1f} Nm"),
            ("Impeller mass", f"{m.impeller_mass_kg:.1f} kg"),
            ("WR² (inertia)", f"{m.wr2_kgm2:.3f} kg·m²"),
            ("First critical speed", f"{m.critical_rpm:.0f} rpm"),
            ("Operating / critical", f"{m.rpm_ratio*100:.0f} %"),
            ("Static deflection", f"{m.static_deflection_mm:.3f} mm"),
        ], columns=["Item", "Value"])
        st.dataframe(md, use_container_width=True, hide_index=True)
        st.caption("Overhung-impeller model. Keep operating speed below "
                   "~75% of first critical; balance to ISO 21940 G6.3 "
                   "(G2.5 for high speed).")
    with tabs[8]:
        if opt_df is not None:
            st.dataframe(opt_df, use_container_width=True, hide_index=True)
        else:
            st.info("Switch mode to 'Optimize geometry' to compare "
                    "alternatives through the physics engine.")
    with tabs[9]:
        st.markdown("Tune predictions to **your** AMCA/ISO test data per "
                    "family: `dp` scales delivered pressure, `power` scales "
                    "shaft power, `sound_db` shifts all octave bands.")
        fam_c = st.selectbox("Family to calibrate", list(FAMILIES.keys()))
        c = st.columns(3)
        cal[fam_c]["dp"] = c[0].number_input("dp ×", 0.70, 1.30,
                                             float(cal[fam_c]["dp"]), 0.01)
        cal[fam_c]["power"] = c[1].number_input("power ×", 0.70, 1.30,
                                                float(cal[fam_c]["power"]), 0.01)
        cal[fam_c]["sound_db"] = c[2].number_input("sound +dB", -10.0, 10.0,
                                                   float(cal[fam_c]["sound_db"]),
                                                   0.5)
        st.download_button("Download calibration JSON",
                           json.dumps(cal, indent=2), "calibration.json")
        up = st.file_uploader("Load calibration JSON", type="json")
        if up:
            try:
                loaded = json.load(up)
                for k in cal:
                    if k in loaded:
                        cal[k].update(loaded[k])
                st.success("Calibration loaded — rerun the design.")
            except Exception as e:
                st.error(f"Could not read file: {e}")
    with tabs[10]:
        st.download_button("DXF manufacturing drawing",
                           create_dxf(w, duty, p, m, project),
                           "blower_manufacturing.dxf")
        st.download_button("Excel calculation book", create_excel(des, opt_df),
                           "blower_calculations.xlsx")
        st.download_button("PDF report", create_pdf(des), "blower_report.pdf")
        st.download_button("Complete ZIP package",
                           make_zip(des, opt_df, project), "blower_package.zip")


