from __future__ import annotations
import numpy as np
import pandas as pd

def rank_matches(df: pd.DataFrame, airflow_m3h: float, static_pressure_pa: float,
                 rpm: float|None=None, fan_type: str|None=None, n=20) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d=df.copy()
    for c in ['airflow_m3h','static_pressure_pa','rpm','input_power_w','noise_dba','impeller_diameter_mm']:
        d[c]=pd.to_numeric(d[c], errors='coerce')
    d=d.dropna(subset=['airflow_m3h','static_pressure_pa'])
    if fan_type and fan_type != 'All':
        d=d[d.fan_type.fillna('').str.contains(fan_type, case=False, regex=False)]
    if d.empty: return d
    qerr=(d.airflow_m3h-airflow_m3h).abs()/max(airflow_m3h,1)
    perr=(d.static_pressure_pa-static_pressure_pa).abs()/max(static_pressure_pa,1)
    rerr=0 if not rpm else (d.rpm-rpm).abs()/max(rpm,1)
    d['match_score']=100*np.exp(-(1.4*qerr+1.8*perr+0.5*rerr))
    d['flow_error_pct']=100*(d.airflow_m3h-airflow_m3h)/max(airflow_m3h,1)
    d['pressure_error_pct']=100*(d.static_pressure_pa-static_pressure_pa)/max(static_pressure_pa,1)
    return d.sort_values('match_score',ascending=False).head(n)
