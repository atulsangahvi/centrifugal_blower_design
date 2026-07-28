
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
import pandas as pd


@dataclass
class FanWallInputs:
    airflow_m3h: float
    static_pressure_pa: float
    operating_hours_per_year: float
    electricity_cost_per_kwh: float
    required_redundancy: int = 1
    diversity_factor: float = 1.0
    design_margin_pct: float = 10.0
    installation_loss_pct: float = 5.0
    old_system_efficiency_pct: float = 45.0
    old_motor_efficiency_pct: float = 88.0
    old_drive_efficiency_pct: float = 90.0
    old_control_efficiency_pct: float = 95.0
    new_system_misc_efficiency_pct: float = 98.0
    maintenance_saving_per_year: float = 0.0
    retrofit_cost: float = 0.0


def _safe_eff(value_pct: float, floor: float = 1e-3) -> float:
    return max(floor, min(1.0, value_pct / 100.0))


def system_effective_duty(inp: FanWallInputs) -> tuple[float, float]:
    q = inp.airflow_m3h * max(inp.diversity_factor, 0.0)
    q *= 1.0 + inp.design_margin_pct / 100.0
    p = inp.static_pressure_pa * (1.0 + inp.installation_loss_pct / 100.0)
    return q, p


def fan_curve_power_kw(
    airflow_m3h: float,
    pressure_pa: float,
    fan_efficiency_pct: float,
    motor_efficiency_pct: float,
    misc_efficiency_pct: float = 100.0,
) -> float:
    q_m3s = airflow_m3h / 3600.0
    air_kw = q_m3s * pressure_pa / 1000.0
    total_eff = (
        _safe_eff(fan_efficiency_pct)
        * _safe_eff(motor_efficiency_pct)
        * _safe_eff(misc_efficiency_pct)
    )
    return air_kw / total_eff


def affinity_speed_fraction(
    required_airflow_m3h: float,
    fan_airflow_m3h_at_full_speed: float,
    active_fans: int,
) -> float:
    if fan_airflow_m3h_at_full_speed <= 0 or active_fans <= 0:
        return float("inf")
    return required_airflow_m3h / (fan_airflow_m3h_at_full_speed * active_fans)


def evaluate_fan_wall(
    fans: pd.DataFrame,
    inp: FanWallInputs,
    fan_type: str = "All",
    verified_only: bool = False,
    min_speed_fraction: float = 0.35,
    max_speed_fraction: float = 1.00,
    max_fans: int = 30,
) -> pd.DataFrame:
    """Evaluate commercial fans as parallel fan-wall candidates.

    Uses catalogue airflow and pressure summary data for preliminary screening.
    Where complete curves are available, the calling UI should prefer curve-based
    selection. Results remain preliminary until verified against a published curve.
    """
    if fans is None or fans.empty:
        return pd.DataFrame()

    q_req, p_req = system_effective_duty(inp)
    df = fans.copy()
    df = df.dropna(subset=["airflow_m3h", "static_pressure_pa"])
    if fan_type != "All":
        df = df[df["fan_type"] == fan_type]
    if verified_only:
        df = df[df["data_status"] == "Verified from catalogue"]

    rows = []
    for _, fan in df.iterrows():
        q_fan = float(fan.airflow_m3h)
        p_fan = float(fan.static_pressure_pa)
        if q_fan <= 0 or p_fan <= 0:
            continue

        # Pressure scales approximately with speed squared.
        min_speed_for_pressure = math.sqrt(max(p_req / p_fan, 0.0))
        for total_fans in range(1 + max(inp.required_redundancy, 0), max_fans + 1):
            active_fans = total_fans - max(inp.required_redundancy, 0)
            if active_fans <= 0:
                continue

            speed_for_flow = affinity_speed_fraction(q_req, q_fan, active_fans)
            speed_fraction = max(speed_for_flow, min_speed_for_pressure)
            if not (min_speed_fraction <= speed_fraction <= max_speed_fraction):
                continue

            delivered_q = q_fan * active_fans * speed_fraction
            delivered_p = p_fan * speed_fraction ** 2

            fan_eff = float(fan.efficiency_pct) if pd.notna(fan.efficiency_pct) else 65.0
            motor_eff = 92.0 if str(fan.motor_type).upper() == "EC" else 90.0

            full_speed_input_kw = (
                float(fan.input_power_w) / 1000.0
                if pd.notna(fan.input_power_w) and float(fan.input_power_w) > 0
                else fan_curve_power_kw(
                    q_fan,
                    p_fan,
                    fan_eff,
                    motor_eff,
                    inp.new_system_misc_efficiency_pct,
                )
            )
            # Fan-law approximation for variable speed.
            active_power_kw = active_fans * full_speed_input_kw * speed_fraction ** 3

            old_input_kw = fan_curve_power_kw(
                q_req,
                p_req,
                inp.old_system_efficiency_pct,
                inp.old_motor_efficiency_pct,
                inp.old_drive_efficiency_pct * inp.old_control_efficiency_pct / 100.0,
            )

            annual_new_kwh = active_power_kw * inp.operating_hours_per_year
            annual_old_kwh = old_input_kw * inp.operating_hours_per_year
            energy_saved_kwh = max(0.0, annual_old_kwh - annual_new_kwh)
            annual_energy_saving = energy_saved_kwh * inp.electricity_cost_per_kwh
            total_annual_saving = annual_energy_saving + inp.maintenance_saving_per_year
            payback_years = (
                inp.retrofit_cost / total_annual_saving
                if total_annual_saving > 0 and inp.retrofit_cost > 0
                else None
            )

            face_area_each_m2 = None
            if pd.notna(fan.width_mm) and pd.notna(fan.height_mm):
                face_area_each_m2 = float(fan.width_mm) * float(fan.height_mm) / 1e6
            elif pd.notna(fan.impeller_diameter_mm):
                d = float(fan.impeller_diameter_mm) / 1000.0
                face_area_each_m2 = d * d

            total_face_area_m2 = (
                face_area_each_m2 * total_fans if face_area_each_m2 else None
            )
            wall_velocity_m_s = (
                (q_req / 3600.0) / total_face_area_m2
                if total_face_area_m2 and total_face_area_m2 > 0
                else None
            )

            score = 100.0
            score -= abs(delivered_q - q_req) / max(q_req, 1e-9) * 30.0
            score -= abs(delivered_p - p_req) / max(p_req, 1e-9) * 25.0
            score -= max(0.0, total_fans - 8) * 1.5
            score -= max(0.0, speed_fraction - 0.9) * 15.0
            if fan.data_status != "Verified from catalogue":
                score -= 10.0
            if wall_velocity_m_s is not None and wall_velocity_m_s > 3.5:
                score -= (wall_velocity_m_s - 3.5) * 8.0
            if total_annual_saving > 0:
                score += min(10.0, total_annual_saving / 10000.0)

            rows.append({
                "manufacturer": fan.manufacturer,
                "model": fan.model,
                "fan_type": fan.fan_type,
                "motor_type": fan.motor_type,
                "total_fans": total_fans,
                "active_fans_at_design": active_fans,
                "standby_fans": inp.required_redundancy,
                "speed_fraction": speed_fraction,
                "speed_pct": speed_fraction * 100.0,
                "required_airflow_m3h": q_req,
                "delivered_airflow_m3h": delivered_q,
                "required_pressure_pa": p_req,
                "delivered_pressure_pa": delivered_p,
                "new_input_power_kw": active_power_kw,
                "old_input_power_kw": old_input_kw,
                "annual_new_energy_kwh": annual_new_kwh,
                "annual_old_energy_kwh": annual_old_kwh,
                "annual_energy_saved_kwh": energy_saved_kwh,
                "annual_energy_saving": annual_energy_saving,
                "maintenance_saving_per_year": inp.maintenance_saving_per_year,
                "total_annual_saving": total_annual_saving,
                "retrofit_cost": inp.retrofit_cost,
                "simple_payback_years": payback_years,
                "wall_velocity_m_s": wall_velocity_m_s,
                "selection_score": score,
                "source_file": fan.source_file,
                "source_page": fan.source_page,
                "data_status": fan.data_status,
                "notes": (
                    "Preliminary parallel fan-wall screening using published summary "
                    "ratings and fan-law scaling. Verify on full manufacturer curve."
                ),
            })

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["selection_score", "total_annual_saving", "total_fans"],
            ascending=[False, False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def fan_wall_control_table(
    selected: pd.Series,
    airflow_steps_pct: list[int] | None = None,
) -> pd.DataFrame:
    if airflow_steps_pct is None:
        airflow_steps_pct = [25, 40, 50, 60, 75, 85, 100]

    active = int(selected["active_fans_at_design"])
    q_design = float(selected["required_airflow_m3h"])
    p_design = float(selected["required_pressure_pa"])
    design_speed = float(selected["speed_fraction"])
    design_power = float(selected["new_input_power_kw"])

    rows = []
    for load_pct in airflow_steps_pct:
        load_frac = load_pct / 100.0
        q = q_design * load_frac
        # Keep all active fans online until speed becomes very low.
        active_now = active
        speed = design_speed * load_frac
        while active_now > 1 and speed < 0.35:
            active_now -= 1
            speed = design_speed * load_frac * active / active_now
        speed = min(speed, 1.0)
        pressure = p_design * speed ** 2 / max(design_speed ** 2, 1e-9)
        power = design_power * (active_now / active) * (speed / max(design_speed, 1e-9)) ** 3
        rows.append({
            "load_pct": load_pct,
            "target_airflow_m3h": q,
            "active_fans": active_now,
            "standby_fans": int(selected["standby_fans"]),
            "speed_pct": speed * 100.0,
            "estimated_pressure_pa": pressure,
            "estimated_power_kw": power,
            "control_note": (
                "Constant airflow / pressure sequence using EC speed control. "
                "Stage fans to avoid sustained very-low-speed operation."
            ),
        })
    return pd.DataFrame(rows)


def n_plus_one_check(selected: pd.Series) -> pd.DataFrame:
    total = int(selected["total_fans"])
    standby = int(selected["standby_fans"])
    active = total - standby
    q_req = float(selected["required_airflow_m3h"])
    q_del = float(selected["delivered_airflow_m3h"])
    p_req = float(selected["required_pressure_pa"])
    p_del = float(selected["delivered_pressure_pa"])

    flow_margin = 100.0 * (q_del - q_req) / max(q_req, 1e-9)
    pressure_margin = 100.0 * (p_del - p_req) / max(p_req, 1e-9)
    return pd.DataFrame([
        {
            "check": "Redundancy count",
            "status": "PASS" if standby >= 1 else "WARN",
            "message": f"{standby} standby fan(s) provided; {active} active at design duty.",
        },
        {
            "check": "Airflow duty",
            "status": "PASS" if flow_margin >= -1.0 else "FAIL",
            "message": f"Airflow margin {flow_margin:.1f}%.",
        },
        {
            "check": "Pressure duty",
            "status": "PASS" if pressure_margin >= -1.0 else "FAIL",
            "message": f"Pressure margin {pressure_margin:.1f}%.",
        },
        {
            "check": "Operating speed",
            "status": "PASS" if 35.0 <= float(selected["speed_pct"]) <= 95.0 else "WARN",
            "message": f"Design speed {float(selected['speed_pct']):.1f}% of catalogue speed.",
        },
        {
            "check": "Wall velocity",
            "status": (
                "PASS"
                if pd.notna(selected["wall_velocity_m_s"]) and float(selected["wall_velocity_m_s"]) <= 3.5
                else "WARN"
            ),
            "message": (
                f"Estimated fan-wall face velocity {float(selected['wall_velocity_m_s']):.2f} m/s."
                if pd.notna(selected["wall_velocity_m_s"])
                else "Insufficient dimensional data to estimate wall velocity."
            ),
        },
    ])
