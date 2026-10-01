
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AxisCalibration:
    """Pixel-to-engineering coordinate calibration for one plot."""

    x_left_px: float
    x_right_px: float
    y_top_px: float
    y_bottom_px: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_log: bool = False
    y_log: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.x_right_px <= self.x_left_px:
            errors.append("Right plot boundary must be greater than the left boundary.")
        if self.y_bottom_px <= self.y_top_px:
            errors.append("Bottom plot boundary must be below the top boundary.")
        if self.x_max <= self.x_min:
            errors.append("X-axis maximum must be greater than minimum.")
        if self.y_max <= self.y_min:
            errors.append("Y-axis maximum must be greater than minimum.")
        if self.x_log and self.x_min <= 0:
            errors.append("Logarithmic X-axis requires a positive minimum.")
        if self.y_log and self.y_min <= 0:
            errors.append("Logarithmic Y-axis requires a positive minimum.")
        return errors


def _interpolate_axis(frac: float, minimum: float, maximum: float, logarithmic: bool) -> float:
    frac = float(np.clip(frac, 0.0, 1.0))
    if logarithmic:
        return float(10 ** (np.log10(minimum) + frac * (np.log10(maximum) - np.log10(minimum))))
    return float(minimum + frac * (maximum - minimum))


def pixel_to_data(x_px: float, y_px: float, calibration: AxisCalibration) -> tuple[float, float]:
    """Convert a pixel click to graph coordinates.

    Image pixel Y increases downward, whereas graph Y increases upward.
    """
    errors = calibration.validate()
    if errors:
        raise ValueError("; ".join(errors))

    x_fraction = (x_px - calibration.x_left_px) / (
        calibration.x_right_px - calibration.x_left_px
    )
    y_fraction = (calibration.y_bottom_px - y_px) / (
        calibration.y_bottom_px - calibration.y_top_px
    )
    x_value = _interpolate_axis(
        x_fraction, calibration.x_min, calibration.x_max, calibration.x_log
    )
    y_value = _interpolate_axis(
        y_fraction, calibration.y_min, calibration.y_max, calibration.y_log
    )
    return x_value, y_value


def convert_clicks(
    clicks: Iterable[dict],
    calibration: AxisCalibration,
    model: str,
    curve_name: str,
    speed_rpm: float | None,
    source_file: str,
    source_page: int | None,
    curve_kind: str = "static_pressure",
    control_setting: str = "",
    point_status: str = "Needs review",
) -> pd.DataFrame:
    rows = []
    for index, click in enumerate(clicks, 1):
        x_value, y_value = pixel_to_data(click["x"], click["y"], calibration)
        rows.append({
            "point_no": index,
            "pixel_x": float(click["x"]),
            "pixel_y": float(click["y"]),
            "model": model,
            "curve_name": curve_name,
            "curve_kind": curve_kind,
            "speed_rpm": speed_rpm,
            "control_setting": control_setting,
            "airflow_m3h": x_value,
            "pressure_pa": y_value,
            "power_w": None,
            "efficiency_pct": None,
            "noise_dba": None,
            "source_file": source_file,
            "source_page": source_page,
            "point_status": point_status,
            "notes": "Digitised from graph image; verify calibration and curve identity.",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("airflow_m3h").reset_index(drop=True)


def curve_quality_checks(points: pd.DataFrame) -> pd.DataFrame:
    """Return engineering and data-quality warnings for a digitised fan curve."""
    checks: list[dict] = []
    if points is None or points.empty:
        return pd.DataFrame([{
            "check": "Curve points",
            "status": "FAIL",
            "message": "No points have been captured.",
        }])

    q = points.dropna(subset=["airflow_m3h", "pressure_pa"]).sort_values("airflow_m3h")
    checks.append({
        "check": "Point count",
        "status": "PASS" if len(q) >= 5 else "WARN",
        "message": f"{len(q)} points captured; 8–20 points normally gives a useful curve.",
    })

    duplicate_q = q["airflow_m3h"].duplicated().any()
    checks.append({
        "check": "Duplicate airflow",
        "status": "WARN" if duplicate_q else "PASS",
        "message": "Duplicate airflow values detected." if duplicate_q else "Airflow values are unique.",
    })

    monotonic_q = bool(q["airflow_m3h"].is_monotonic_increasing)
    checks.append({
        "check": "Airflow order",
        "status": "PASS" if monotonic_q else "FAIL",
        "message": "Airflow increases from left to right." if monotonic_q else "Airflow ordering is inconsistent.",
    })

    dp = np.diff(q["pressure_pa"].to_numpy(dtype=float))
    rising_fraction = float(np.mean(dp > 0)) if len(dp) else 0.0
    checks.append({
        "check": "Pressure trend",
        "status": "PASS" if rising_fraction <= 0.20 else "WARN",
        "message": (
            f"{rising_fraction * 100:.0f}% of adjacent segments rise with airflow. "
            "A stable centrifugal-fan pressure curve normally falls overall, though local humps may occur."
        ),
    })

    nonnegative = bool((q[["airflow_m3h", "pressure_pa"]] >= 0).all().all())
    checks.append({
        "check": "Non-negative coordinates",
        "status": "PASS" if nonnegative else "FAIL",
        "message": "All airflow and pressure values are non-negative." if nonnegative else "Negative values found.",
    })

    q_span = float(q.airflow_m3h.max() - q.airflow_m3h.min())
    p_span = float(q.pressure_pa.max() - q.pressure_pa.min())
    checks.append({
        "check": "Curve span",
        "status": "PASS" if q_span > 0 and p_span > 0 else "FAIL",
        "message": f"Airflow span {q_span:.1f} m³/h; pressure span {p_span:.1f} Pa.",
    })

    return pd.DataFrame(checks)


def fan_law_scale_curve(
    points: pd.DataFrame,
    old_speed_rpm: float,
    new_speed_rpm: float,
    old_density_kg_m3: float = 1.20,
    new_density_kg_m3: float = 1.20,
) -> pd.DataFrame:
    """Scale a curve using classical fan laws for the same fan geometry.

    Q2/Q1 = N2/N1
    P2/P1 = (rho2/rho1)*(N2/N1)^2
    Power2/Power1 = (rho2/rho1)*(N2/N1)^3
    """
    if old_speed_rpm <= 0 or new_speed_rpm <= 0:
        raise ValueError("Fan speeds must be positive.")
    if old_density_kg_m3 <= 0 or new_density_kg_m3 <= 0:
        raise ValueError("Air densities must be positive.")

    ratio_n = new_speed_rpm / old_speed_rpm
    ratio_rho = new_density_kg_m3 / old_density_kg_m3

    out = points.copy()
    out["airflow_m3h"] = out["airflow_m3h"] * ratio_n
    out["pressure_pa"] = out["pressure_pa"] * ratio_rho * ratio_n ** 2
    if "power_w" in out:
        out["power_w"] = out["power_w"] * ratio_rho * ratio_n ** 3
    out["speed_rpm"] = new_speed_rpm
    out["curve_name"] = out["curve_name"].astype(str) + f" scaled to {new_speed_rpm:g} rpm"
    out["point_status"] = "Fan-law scaled"
    out["notes"] = (
        "Scaled from the same fan geometry using classical fan laws. "
        "Validate against manufacturer limits, Reynolds effects, motor limits and test data."
    )
    return out
