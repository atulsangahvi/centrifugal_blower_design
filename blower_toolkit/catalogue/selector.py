
from __future__ import annotations

import numpy as np
import pandas as pd


def interpolate_curve(curve: pd.DataFrame, airflow_m3h: float) -> dict | None:
    if curve is None or curve.empty:
        return None
    x = curve.dropna(subset=["airflow_m3h", "pressure_pa"]).copy()
    x = x.sort_values("airflow_m3h").drop_duplicates("airflow_m3h")
    if len(x) < 2:
        return None
    q = float(airflow_m3h)
    qmin, qmax = float(x.airflow_m3h.min()), float(x.airflow_m3h.max())
    if q < qmin or q > qmax:
        return None

    result = {
        "airflow_m3h": q,
        "pressure_pa": float(np.interp(q, x.airflow_m3h, x.pressure_pa)),
        "within_curve": True,
    }
    for field in ["power_w", "efficiency_pct", "noise_dba"]:
        y = x.dropna(subset=[field])
        if len(y) >= 2 and y.airflow_m3h.min() <= q <= y.airflow_m3h.max():
            result[field] = float(np.interp(q, y.airflow_m3h, y[field]))
        else:
            result[field] = None
    return result


def select_by_curves(
    fans: pd.DataFrame,
    curves: pd.DataFrame,
    airflow_m3h: float,
    static_pressure_pa: float,
    fan_type: str = "All",
    verified_only: bool = False,
    tolerance_pct: float = 25.0,
) -> pd.DataFrame:
    if fans.empty or curves.empty:
        return pd.DataFrame()

    candidates = fans.copy()
    if fan_type != "All":
        candidates = candidates[candidates.fan_type == fan_type]
    if verified_only:
        candidates = candidates[candidates.data_status == "Verified from catalogue"]

    rows = []
    for _, fan in candidates.iterrows():
        model_curves = curves[curves.model == fan.model]
        if model_curves.empty:
            continue
        group_cols = ["curve_name", "speed_rpm", "control_setting"]
        for keys, curve in model_curves.groupby(group_cols, dropna=False):
            point = interpolate_curve(curve, airflow_m3h)
            if not point:
                continue
            pressure_error = 100.0 * (point["pressure_pa"] - static_pressure_pa) / max(static_pressure_pa, 1e-9)
            if abs(pressure_error) > tolerance_pct:
                continue
            score = max(0.0, 100.0 - abs(pressure_error) * 2.5)
            efficiency = point.get("efficiency_pct")
            if efficiency is not None:
                score += min(10.0, efficiency / 10.0)
            rows.append({
                "manufacturer": fan.manufacturer,
                "model": fan.model,
                "fan_type": fan.fan_type,
                "curve_name": keys[0],
                "speed_rpm": keys[1],
                "control_setting": keys[2],
                "required_airflow_m3h": airflow_m3h,
                "predicted_pressure_pa": point["pressure_pa"],
                "required_pressure_pa": static_pressure_pa,
                "pressure_margin_pa": point["pressure_pa"] - static_pressure_pa,
                "pressure_error_pct": pressure_error,
                "predicted_power_w": point.get("power_w"),
                "predicted_efficiency_pct": point.get("efficiency_pct"),
                "predicted_noise_dba": point.get("noise_dba"),
                "selection_score": score,
                "source_file": fan.source_file,
                "source_page": fan.source_page,
                "data_status": fan.data_status,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "predicted_efficiency_pct"],
        ascending=[False, False],
        na_position="last",
    )
