
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from blower_toolkit.auth import require_password

from blower_toolkit.catalogue import CatalogueDB
from blower_toolkit.catalogue.fanwall import (
    FanWallInputs,
    evaluate_fan_wall,
    fan_wall_control_table,
    n_plus_one_check,
)

st.set_page_config(page_title="AHU & Cooling Tower Fan Wall", layout="wide")
require_password()
st.title("AHU & Cooling Tower Fan-Wall Selector v23")
st.caption(
    "Parallel EC fan selection, N+1 redundancy, preliminary control staging, "
    "annual energy comparison, retrofit savings and payback."
)

ROOT = Path(__file__).resolve().parents[1]
db = CatalogueDB(ROOT / "data" / "catalogue.db")
fans = db.all()

if fans.empty:
    st.error("The manufacturer knowledge base is empty.")
    st.stop()

with st.sidebar:
    st.header("Application")
    application = st.selectbox(
        "System type",
        ["AHU supply / return fan wall", "Cooling tower axial fan wall"],
    )
    fan_type_options = ["All"] + sorted(fans.fan_type.dropna().unique().tolist())
    fan_type = st.selectbox("Commercial fan type", fan_type_options)
    verified_only = st.checkbox("Use verified models only", value=False)

st.subheader("1. Required duty")
a, b, c, d = st.columns(4)
airflow = a.number_input("Required airflow (m³/h)", 100.0, 2_000_000.0, 100_000.0, 1000.0)
pressure = b.number_input("Required static pressure (Pa)", 10.0, 10_000.0, 600.0, 10.0)
diversity = c.number_input("Diversity factor", 0.1, 1.5, 1.0, 0.05)
margin = d.number_input("Design airflow margin (%)", 0.0, 50.0, 10.0, 1.0)

e, f, g = st.columns(3)
installation_loss = e.number_input("Installation / system-effect allowance (%)", 0.0, 50.0, 5.0, 1.0)
redundancy = f.number_input("Standby fans required", 0, 5, 1, 1)
max_fans = g.number_input("Maximum total fans to consider", 2, 60, 24, 1)

st.subheader("2. Operating and economic data")
h1, h2, h3, h4 = st.columns(4)
hours = h1.number_input("Operating hours per year", 0.0, 8760.0, 6000.0, 100.0)
electricity_cost = h2.number_input("Electricity cost per kWh", 0.0, 10.0, 0.12, 0.01)
retrofit_cost = h3.number_input("Installed retrofit cost", 0.0, 100_000_000.0, 0.0, 1000.0)
maintenance_saving = h4.number_input("Annual maintenance saving", 0.0, 10_000_000.0, 0.0, 500.0)

st.subheader("3. Existing fan-system baseline")
o1, o2, o3, o4 = st.columns(4)
old_fan_eff = o1.number_input("Existing fan efficiency (%)", 10.0, 90.0, 45.0, 1.0)
old_motor_eff = o2.number_input("Existing motor efficiency (%)", 40.0, 99.0, 88.0, 1.0)
old_drive_eff = o3.number_input("Existing belt / drive efficiency (%)", 40.0, 100.0, 90.0, 1.0)
old_control_eff = o4.number_input("Existing control efficiency (%)", 40.0, 100.0, 95.0, 1.0)

inp = FanWallInputs(
    airflow_m3h=airflow,
    static_pressure_pa=pressure,
    operating_hours_per_year=hours,
    electricity_cost_per_kwh=electricity_cost,
    required_redundancy=int(redundancy),
    diversity_factor=diversity,
    design_margin_pct=margin,
    installation_loss_pct=installation_loss,
    old_system_efficiency_pct=old_fan_eff,
    old_motor_efficiency_pct=old_motor_eff,
    old_drive_efficiency_pct=old_drive_eff,
    old_control_efficiency_pct=old_control_eff,
    maintenance_saving_per_year=maintenance_saving,
    retrofit_cost=retrofit_cost,
)

if st.button("Evaluate fan-wall options", type="primary"):
    st.session_state["fanwall_results"] = evaluate_fan_wall(
        fans,
        inp,
        fan_type=fan_type,
        verified_only=verified_only,
        max_fans=int(max_fans),
    )

results = st.session_state.get("fanwall_results", pd.DataFrame())

if results.empty:
    st.info(
        "No evaluated result is stored yet, or no catalogue summary rating could meet "
        "the duty within the configured fan count. Add more verified models and complete curves."
    )
else:
    st.subheader("4. Ranked fan-wall options")
    show_cols = [
        "manufacturer", "model", "fan_type", "motor_type",
        "total_fans", "active_fans_at_design", "standby_fans",
        "speed_pct", "required_airflow_m3h", "delivered_airflow_m3h",
        "required_pressure_pa", "delivered_pressure_pa",
        "new_input_power_kw", "old_input_power_kw",
        "annual_energy_saved_kwh", "total_annual_saving",
        "simple_payback_years", "wall_velocity_m_s",
        "selection_score", "data_status",
    ]
    st.dataframe(
        results[[c for c in show_cols if c in results]],
        use_container_width=True,
        hide_index=True,
    )

    models = [
        f"{r.model} — {int(r.total_fans)} total / {int(r.active_fans_at_design)} active"
        for _, r in results.head(25).iterrows()
    ]
    selected_label = st.selectbox("Inspect one option", models)
    idx = models.index(selected_label)
    selected = results.head(25).iloc[idx]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total fans", int(selected.total_fans))
    m2.metric("Design speed", f"{selected.speed_pct:.1f}%")
    m3.metric("New input power", f"{selected.new_input_power_kw:.1f} kW")
    m4.metric("Annual saving", f"{selected.total_annual_saving:,.0f}")
    m5.metric(
        "Simple payback",
        "N/A" if pd.isna(selected.simple_payback_years) else f"{selected.simple_payback_years:.2f} yr",
    )

    tabs = st.tabs([
        "Engineering Checks",
        "Control Sequence",
        "Energy & ROI",
        "Arrangement Guidance",
        "Export",
    ])

    with tabs[0]:
        st.dataframe(
            n_plus_one_check(selected),
            use_container_width=True,
            hide_index=True,
        )
        st.warning(
            "This module presently screens catalogue summary ratings. Before purchase "
            "or manufacturing release, verify the duty on a published or tested curve, "
            "including air density, system effect, acoustic interaction and motor limits."
        )

    with tabs[1]:
        control = fan_wall_control_table(selected)
        st.dataframe(control, use_container_width=True, hide_index=True)
        st.line_chart(control.set_index("load_pct")[["estimated_power_kw"]])
        st.caption(
            "Recommended sequence: maintain the required static pressure or airflow, "
            "stage fans to avoid very low speed, rotate lead duty, and alarm on loss of "
            "feedback, motor fault or excessive power."
        )

    with tabs[2]:
        energy = pd.DataFrame([
            {
                "Scenario": "Existing system",
                "Input power kW": selected.old_input_power_kw,
                "Annual energy kWh": selected.annual_old_energy_kwh,
                "Annual energy cost": selected.annual_old_energy_kwh * electricity_cost,
            },
            {
                "Scenario": "Proposed fan wall",
                "Input power kW": selected.new_input_power_kw,
                "Annual energy kWh": selected.annual_new_energy_kwh,
                "Annual energy cost": selected.annual_new_energy_kwh * electricity_cost,
            },
        ])
        st.dataframe(energy, use_container_width=True, hide_index=True)
        st.bar_chart(energy.set_index("Scenario")[["Annual energy kWh"]])
        st.write(
            {
                "Annual energy saving": float(selected.annual_energy_saving),
                "Annual maintenance saving": float(selected.maintenance_saving_per_year),
                "Total annual saving": float(selected.total_annual_saving),
                "Retrofit cost": float(selected.retrofit_cost),
                "Simple payback years": (
                    None if pd.isna(selected.simple_payback_years)
                    else float(selected.simple_payback_years)
                ),
            }
        )

    with tabs[3]:
        if application.startswith("AHU"):
            st.markdown(
                """
### AHU fan-wall arrangement checks

- Provide uniform upstream flow and adequate distance from filters, coils and dampers.
- Avoid placing a partition edge directly in front of a fan inlet.
- Use blanking plates around the fan modules to prevent bypass.
- Include removable modules, isolation, non-return provisions where needed and safe access.
- Rotate lead fans to equalise running hours.
- Check acoustic interaction and blade-passing-frequency coincidence between adjacent fans.
- For hygienic AHUs, eliminate dust traps and provide cleanable surfaces.
"""
            )
        else:
            st.markdown(
                """
### Cooling-tower axial fan-wall checks

- Confirm the published fan is suitable for wet, corrosive cooling-tower duty.
- Verify IP rating, coating system, salt-mist resistance and bearing sealing.
- Check discharge recirculation, wind effects and airflow distribution through the heat-transfer surface.
- Ensure the fan frame does not create excessive blockage or bypass.
- Confirm drainage, electrical isolation, safe access and module replacement clearances.
- Validate fan-wall interaction experimentally because nearby fans may change the free-air curve.
"""
            )

    with tabs[4]:
        detail = selected.to_frame(name="value").reset_index().rename(columns={"index": "parameter"})
        st.download_button(
            "Download selected option CSV",
            detail.to_csv(index=False).encode("utf-8"),
            f"{selected.model}_fanwall_selection.csv",
            "text/csv",
        )
        st.download_button(
            "Download control sequence CSV",
            fan_wall_control_table(selected).to_csv(index=False).encode("utf-8"),
            f"{selected.model}_fanwall_control.csv",
            "text/csv",
        )
        st.download_button(
            "Download all ranked options CSV",
            results.to_csv(index=False).encode("utf-8"),
            "fanwall_ranked_options.csv",
            "text/csv",
        )
