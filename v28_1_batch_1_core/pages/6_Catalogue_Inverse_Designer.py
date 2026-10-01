
from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from blower_toolkit.auth import require_password
from blower_toolkit.catalogue import CatalogueDB
from blower_toolkit.catalogue.inverse_design import (
    InverseConfig,
    InverseLocks,
    catalogue_arrangement,
    classify_catalogue_basis,
    full_curve_for_geometry,
    identifiability_table,
    manufacturing_confidence,
    optimize_inverse_design,
    peer_endpoint_prior,
    sensitivity_table,
    supported_catalogue_family,
    target_from_record,
)
from blower_toolkit.catalogue.manufacturing_bridge import embedded_pdf_path, render_pdf_page_png

st.set_page_config(page_title="Catalogue Inverse Designer", layout="wide")
require_password()

st.title("Catalogue Blower → Inverse Design & Manufacturing Geometry v27")
st.caption(
    "Keep the manufacturer information that is actually known, solve for hidden impeller/scroll geometry, "
    "show non-uniqueness, and transfer the best candidate into the detailed manufacturing engine."
)

ROOT = Path(__file__).resolve().parents[1]
db = CatalogueDB(ROOT / "data" / "catalogue.db")
fans = db.all()
curves = db.curves()

centrifugal = fans[
    fans.apply(lambda r: supported_catalogue_family(r.to_dict()) is not None, axis=1)
].copy()

if centrifugal.empty:
    st.error("No supported centrifugal catalogue models are stored.")
    st.stop()

# Prefer the model selected in Catalogue Knowledge Base.
initial = st.session_state.get("catalogue_selected_model")
model_list = centrifugal.model.tolist()
initial_index = model_list.index(initial) if initial in model_list else 0

st.subheader("1. Select the catalogue blower")
c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
maker_list = ["All"] + sorted(centrifugal.manufacturer.dropna().unique().tolist())
maker = c1.selectbox("Manufacturer", maker_list)

view = centrifugal.copy()
if maker != "All":
    view = view[view.manufacturer == maker]

type_list = ["All"] + sorted(view.fan_type.dropna().unique().tolist())
fan_type = c2.selectbox("Fan type", type_list)
if fan_type != "All":
    view = view[view.fan_type == fan_type]

models = view.model.tolist()
if not models:
    st.warning("No model matches these filters.")
    st.stop()

if initial in models:
    selected_index = models.index(initial)
else:
    selected_index = 0
model = c3.selectbox("Catalogue model", models, index=selected_index)
st.session_state["catalogue_selected_model"] = model

r = view[view.model == model].iloc[0]
record = r.to_dict()
model_curve_points = curves[curves.model == model].copy() if not curves.empty else pd.DataFrame()

left, right = st.columns([0.46, 0.54])
with left:
    st.markdown("#### Manufacturer information currently stored")
    summary = pd.DataFrame([
        ("Manufacturer", r.manufacturer),
        ("Model", r.model),
        ("Fan type", r.fan_type),
        ("Arrangement", r.arrangement),
        ("D2 / nominal impeller diameter", r.impeller_diameter_mm),
        ("Catalogue air-flow summary", r.airflow_m3h),
        ("Catalogue pressure summary", r.static_pressure_pa),
        ("RPM", r.rpm),
        ("Input power", r.input_power_w),
        ("Noise", r.noise_dba),
        ("Material", r.material),
        ("Source PDF", r.source_file),
        ("PDF page", r.source_page),
        ("Extraction status", r.data_status),
    ], columns=["Parameter", "Value"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

with right:
    pdf_path = embedded_pdf_path(ROOT, r.source_file)
    if pdf_path and pd.notna(r.source_page):
        try:
            st.image(
                render_pdf_page_png(pdf_path, int(r.source_page), zoom=1.05),
                caption=f"{r.source_file} — source page {int(r.source_page)}",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Could not render source page: {exc}")
    else:
        st.info("The source PDF/page is not available in this build.")

st.warning(
    "Important: a catalogue table that lists 'Air Flow' and 'Air Pressure' does not prove that those two "
    "maximum values occur at the same operating point. For Seemtek/Longwell summary tables the safer default "
    "is to treat them as Qmax and Psmax endpoints unless you digitise the plotted Q-P curve."
)

# ---------------------------------------------------------------------------
# 2. Define the target interpretation.
# ---------------------------------------------------------------------------
st.subheader("2. Tell the optimizer what the catalogue data actually means")

auto_basis = classify_catalogue_basis(record, model_curve_points)
basis_options = [
    "Catalogue max-flow / max-pressure endpoints",
    "Simultaneous duty point",
    "Digitised Q-P curve",
]
basis = st.radio(
    "Target basis",
    basis_options,
    index=basis_options.index(auto_basis) if auto_basis in basis_options else 0,
    help=(
        "Endpoint mode fits the predicted free-delivery flow and near-shutoff static pressure separately. "
        "Simultaneous mode uses Q and P at one operating point. Curve mode is best when 4+ actual plotted points are stored."
    ),
)

t1, t2, t3, t4 = st.columns(4)
target_q = t1.number_input(
    "Catalogue Q value (m³/h)",
    min_value=0.0,
    value=float(r.airflow_m3h or 0.0),
    step=max(1.0, float(r.airflow_m3h or 100.0) * 0.02),
)
target_p = t2.number_input(
    "Catalogue static-pressure value (Pa)",
    min_value=0.0,
    value=float(r.static_pressure_pa or 0.0),
    step=max(1.0, float(r.static_pressure_pa or 100.0) * 0.02),
)
target_rpm = t3.number_input(
    "Fan speed (rpm)",
    min_value=0.0,
    value=float(r.rpm or 0.0),
    step=10.0,
)
target_d2 = t4.number_input(
    "D2 / nominal wheel diameter (mm)",
    min_value=0.0,
    value=float(r.impeller_diameter_mm or 0.0),
    step=1.0,
)

a1, a2, a3 = st.columns(3)
temp = a1.number_input("Air temperature (°C)", -40.0, 150.0, 20.0, 1.0)
altitude = a2.number_input("Altitude (m)", -500.0, 5000.0, 0.0, 50.0)
rh = a3.number_input("RH (%)", 0.0, 100.0, 50.0, 5.0)

if basis == "Digitised Q-P curve":
    valid_curve = model_curve_points.dropna(subset=["airflow_m3h", "pressure_pa"]).copy()
    valid_curve = valid_curve[
        ~valid_curve.curve_name.astype(str).str.contains(
            "operating row|rated point|illustrative", case=False, regex=True
        )
    ] if "curve_name" in valid_curve else valid_curve
    if len(valid_curve) < 4:
        st.error(
            "This model does not yet have at least 4 genuine digitised Q-P points. "
            "Use Performance Curve Digitiser first, or use endpoint mode."
        )
    else:
        st.success(f"{len(valid_curve)} digitised Q-P points will constrain the inverse design.")
        st.dataframe(
            valid_curve[["airflow_m3h", "pressure_pa", "speed_rpm", "power_w", "point_status"]],
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# 3. Measured/known internal geometry locks.
# ---------------------------------------------------------------------------
st.subheader("3. Lock any dimensions you actually know")
st.caption(
    "Leave a value at zero if it is unknown. Every real measurement collapses the inverse-design search space "
    "and makes the manufacturing result much more credible."
)

with st.expander("Measured / known internal geometry", expanded=False):
    l1, l2, l3, l4 = st.columns(4)
    lock_d1 = l1.number_input("Known D1 (mm; 0=unknown)", 0.0, 5000.0, 0.0, 1.0)
    lock_b2 = l2.number_input("Known b2 total (mm; 0=unknown)", 0.0, 5000.0, 0.0, 1.0)
    lock_b1 = l3.number_input("Known b1 total (mm; 0=unknown)", 0.0, 5000.0, 0.0, 1.0)
    lock_z = l4.number_input("Known blade count (0=unknown)", 0, 200, 0, 1)

    l5, l6, l7, l8 = st.columns(4)
    lock_b1a = l5.number_input("Known beta1 (deg; 0=unknown)", 0.0, 175.0, 0.0, 1.0)
    lock_b2a = l6.number_input("Known beta2 (deg; 0=unknown)", 0.0, 175.0, 0.0, 1.0)
    lock_cut = l7.number_input("Known cutoff clearance (mm; 0=unknown)", 0.0, 1000.0, 0.0, 1.0)
    lock_scroll = l8.number_input("Known scroll internal width (mm; 0=unknown)", 0.0, 5000.0, 0.0, 1.0)

    l9, l10, l11, l12 = st.columns(4)
    lock_outw = l9.number_input("Known discharge clear width (mm; 0=unknown)", 0.0, 5000.0, 0.0, 1.0)
    lock_outh = l10.number_input("Known discharge clear height (mm; 0=unknown)", 0.0, 5000.0, 0.0, 1.0)
    lock_blade_t = l11.number_input("Known blade thickness (mm; 0=unknown)", 0.0, 30.0, 0.0, 0.1)
    lock_plate_t = l12.number_input("Known plate thickness (mm; 0=unknown)", 0.0, 30.0, 0.0, 0.1)

locks = InverseLocks(
    d1_mm=lock_d1,
    b2_total_mm=lock_b2,
    b1_total_mm=lock_b1,
    blade_count=int(lock_z),
    beta1_deg=lock_b1a,
    beta2_deg=lock_b2a,
    cutoff_clearance_mm=lock_cut,
    scroll_internal_width_mm=lock_scroll,
    discharge_width_mm=lock_outw,
    discharge_height_mm=lock_outh,
    blade_thickness_mm=lock_blade_t,
    plate_thickness_mm=lock_plate_t,
)

# ---------------------------------------------------------------------------
# 4. Empirical prior and optimization settings.
# ---------------------------------------------------------------------------
st.subheader("4. Optimization strategy")

record_for_prior = dict(record)
record_for_prior.update(
    airflow_m3h=target_q,
    static_pressure_pa=target_p,
    rpm=target_rpm,
    impeller_diameter_mm=target_d2,
)
peer_prior = peer_endpoint_prior(fans, record_for_prior)

p1, p2, p3, p4 = st.columns(4)
p1.metric("Peer models used", peer_prior.get("peer_count", 0))
p2.metric("Empirical flow correction prior", f"{peer_prior.get('flow_factor',1.0):.3f}")
p3.metric("Empirical pressure correction prior", f"{peer_prior.get('pressure_factor',1.0):.3f}")
p4.metric("Family", supported_catalogue_family(record_for_prior) or "Unsupported")

st.caption(
    "The empirical correction prior is calculated from other models in the same manufacturer's fan family. "
    "It corrects limitations of the preliminary mean-line loss model; it is not a substitute for prototype testing."
)

s1, s2, s3, s4 = st.columns(4)
effort = s1.selectbox("Optimization effort", ["Quick", "Standard", "Deep"], index=1)
use_power = s2.checkbox(
    "Use catalogue input power as a secondary constraint",
    value=False,
    help="Only enable this if the power value corresponds to the target condition; endpoint tables may not.",
)
use_noise = s3.checkbox("Use catalogue noise as a secondary constraint", value=False)
allow_corr = s4.checkbox("Allow empirical model correction", value=True)

s5, s6 = st.columns(2)
motor_eff = s5.number_input("Assumed motor efficiency (%)", 50.0, 99.5, 90.0, 1.0)
drive_eff = s6.number_input("Assumed drive efficiency (%)", 50.0, 100.0, 98.0, 1.0)

family = supported_catalogue_family(record_for_prior)
if family is None:
    st.error("This model is not supported by the centrifugal inverse-design engine.")
    st.stop()

target_record = dict(record_for_prior)
target_record["fan_type"] = r.fan_type
target_record["arrangement"] = r.arrangement
target_record["input_power_w"] = float(r.input_power_w or 0.0)
target_record["noise_dba"] = float(r.noise_dba or 0.0)
target_record["material"] = r.material
target_record["source_file"] = r.source_file
target_record["source_page"] = r.source_page

target = target_from_record(
    target_record,
    basis=basis,
    temp_c=temp,
    altitude_m=altitude,
    rh_pct=rh,
)
target.d2_mm = target_d2
target.rpm = target_rpm
target.airflow_m3h = target_q
target.static_pressure_pa = target_p

cfg = InverseConfig(
    mode=effort,
    use_catalogue_power=use_power,
    use_catalogue_noise=use_noise,
    motor_efficiency_pct=motor_eff,
    drive_efficiency_pct=drive_eff,
    allow_empirical_model_correction=allow_corr,
    endpoint_flow_prior=float(peer_prior.get("flow_factor", 1.0)),
    endpoint_pressure_prior=float(peer_prior.get("pressure_factor", 1.0)),
)

if st.button("Solve hidden geometry from catalogue constraints", type="primary"):
    with st.spinner(
        f"Running {effort.lower()} inverse optimization: fixed D2/RPM, varying hidden blade and scroll geometry..."
    ):
        try:
            candidates, best = optimize_inverse_design(
                target,
                locks=locks,
                cfg=cfg,
                curve_points=model_curve_points,
            )
            st.session_state["inverse_design_candidates"] = candidates
            st.session_state["inverse_design_best"] = best
            st.session_state["inverse_design_model"] = model
            st.session_state["inverse_design_target"] = target
            st.session_state["inverse_design_locks"] = locks
            st.session_state["inverse_design_cfg"] = cfg
        except Exception as exc:
            st.exception(exc)

candidates = st.session_state.get("inverse_design_candidates")
best = st.session_state.get("inverse_design_best")
stored_model = st.session_state.get("inverse_design_model")

if best and candidates is not None and not candidates.empty and stored_model == model:
    st.divider()
    st.subheader("5. Best inverse-designed geometry")

    bm = best["metrics"]
    bg = best["geometry"]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Catalogue match error", f"{bm['catalogue_match_rmse_pct']:.1f}%")
    m2.metric("D1 / D2", f"{bm['d1_d2']:.3f}")
    m3.metric("b2 / D2", f"{bm['b2_d2']:.3f}")
    m4.metric("beta1", f"{bm['beta1_deg']:.1f}°")
    m5.metric("beta2", f"{bm['beta2_deg']:.1f}°")
    m6.metric("Blades", f"{int(bm['blade_count'])}")

    n1, n2, n3, n4, n5, n6 = st.columns(6)
    n1.metric("Static efficiency", f"{100*bm['eta_static_at_design_q']:.1f}%")
    n2.metric("Shaft power", f"{bm['shaft_kw_at_design_q']:.3f} kW")
    n3.metric("Est. input", f"{bm['estimated_input_w_at_design_q']:.0f} W")
    n4.metric("Noise estimate", f"{bm['predicted_noise_dba']:.1f} dB(A)")
    n5.metric("Tip speed", f"{bm['tip_speed_ms']:.1f} m/s")
    n6.metric("Pitch/chord", f"{bm['pitch_chord']:.3f}")

    if basis == "Catalogue max-flow / max-pressure endpoints":
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Catalogue Qmax", f"{target_q:,.0f} m³/h")
        e2.metric("Predicted Qmax", f"{bm.get('predicted_qmax_m3h',0):,.0f} m³/h")
        e3.metric("Catalogue Psmax", f"{target_p:,.0f} Pa")
        e4.metric("Predicted Psmax", f"{bm.get('predicted_psmax_pa',0):,.0f} Pa")

    result_tabs = st.tabs([
        "Geometry",
        "Target vs Predicted",
        "Top Candidates",
        "What Must Be Measured",
        "Sensitivity",
        "Manufacturing Confidence",
        "Export / Continue",
    ])

    with result_tabs[0]:
        geometry_table = pd.DataFrame([
            ("D2", bg["d2_mm"], "mm", "LOCKED from catalogue/user"),
            ("D1", bg["d1_mm"], "mm", "Measured lock" if locks.d1_mm > 0 else "Inverse optimized"),
            ("D1/D2", bg["d1_mm"]/bg["d2_mm"], "", "Inverse constraint"),
            ("b2 total", bg["b2_total_mm"], "mm", "Measured lock" if locks.b2_total_mm > 0 else "Inverse optimized"),
            ("b2/D2", bg["b2_total_mm"]/bg["d2_mm"], "", "Inverse constraint"),
            ("b1 total", bg["b1_total_mm"], "mm", "Measured lock" if locks.b1_total_mm > 0 else "Inverse optimized"),
            ("Blade count", bg["blade_count"], "", "Measured lock" if locks.blade_count > 0 else "Inverse optimized"),
            ("beta1", bg["beta1_deg"], "deg from tangent", "Measured lock" if locks.beta1_deg > 0 else "Inverse optimized"),
            ("beta2", bg["beta2_deg"], "deg from tangent", "Measured lock" if locks.beta2_deg > 0 else "Inverse optimized"),
            ("Cutoff clearance", bg["cutoff_clearance_mm"], "mm", "Measured lock" if locks.cutoff_clearance_mm > 0 else "Inverse optimized"),
            ("Scroll internal width", bg["scroll_internal_width_mm"], "mm", "Measured lock" if locks.scroll_internal_width_mm > 0 else "Inverse optimized"),
            ("Discharge clear width", bg["discharge_width_mm"], "mm", "Measured lock" if locks.discharge_width_mm > 0 else "Calculated from outlet-velocity target"),
            ("Discharge clear height", bg["discharge_height_mm"], "mm", "Measured lock" if locks.discharge_height_mm > 0 else "Calculated from outlet-velocity target"),
            ("Blade thickness", bg["blade_thickness_mm"], "mm", "Measured lock" if locks.blade_thickness_mm > 0 else "Manufacturing starting value"),
            ("Plate thickness", bg["plate_thickness_mm"], "mm", "Measured lock" if locks.plate_thickness_mm > 0 else "Manufacturing starting value"),
            ("Model pressure correction", bm["pressure_factor"], "×", "Empirical model coefficient — not a physical dimension"),
            ("Model flow correction", bm.get("flow_factor"), "×", "Endpoint-only empirical coefficient"),
        ], columns=["Parameter", "Value", "Unit", "Provenance"])
        st.dataframe(geometry_table, use_container_width=True, hide_index=True)

    with result_tabs[1]:
        predicted_curve = full_curve_for_geometry(target, bg, bm)
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.plot(
            predicted_curve["airflow_m3h"],
            predicted_curve["static_pressure_pa"],
            label="Inverse-designed predicted curve",
        )

        if basis == "Catalogue max-flow / max-pressure endpoints":
            ax.scatter([0, target_q], [target_p, 0], s=70, label="Catalogue endpoint constraints")
        elif basis == "Simultaneous duty point":
            ax.scatter([target_q], [target_p], s=80, label="Catalogue/user duty point")
        else:
            valid_curve = model_curve_points.dropna(subset=["airflow_m3h", "pressure_pa"]).copy()
            valid_curve = valid_curve[
                ~valid_curve.curve_name.astype(str).str.contains(
                    "operating row|rated point|illustrative", case=False, regex=True
                )
            ] if "curve_name" in valid_curve else valid_curve
            ax.scatter(valid_curve.airflow_m3h, valid_curve.pressure_pa, s=50, label="Digitised catalogue curve")

        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Airflow (m³/h)")
        ax.set_ylabel("Static pressure (Pa)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        st.dataframe(predicted_curve.round(4), use_container_width=True, hide_index=True)

    with result_tabs[2]:
        display_cols = [
            "rank","score","catalogue_match_rmse_pct","feasible",
            "d1_d2","b2_d2","b1_b2","beta1_deg","beta2_deg","blade_count",
            "cutoff_ratio","scroll_width_mm","discharge_width_mm","discharge_height_mm",
            "eta_static_at_design_q","shaft_kw_at_design_q","predicted_noise_dba",
            "tip_speed_ms","pitch_chord","pressure_factor","flow_factor"
        ]
        st.dataframe(
            candidates[[c for c in display_cols if c in candidates.columns]].head(50),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Do not automatically assume rank 1 is the original manufacturer's hidden geometry. "
            "Several different geometries can reproduce the same sparse catalogue information."
        )

    with result_tabs[3]:
        ident = identifiability_table(candidates)
        st.dataframe(ident, use_container_width=True, hide_index=True)
        high = ident[ident.measurement_priority == "HIGH"] if not ident.empty else pd.DataFrame()
        if not high.empty:
            st.error(
                "Highest-value physical measurements before manufacturing: "
                + ", ".join(high.parameter.head(5).tolist())
            )
        else:
            st.success("Near-best solutions are relatively well clustered for the current constraints.")

    with result_tabs[4]:
        sens = sensitivity_table(
            target,
            best,
            locks=locks,
            cfg=cfg,
            curve_points=model_curve_points,
        )
        st.dataframe(sens, use_container_width=True, hide_index=True)

    with result_tabs[5]:
        confidence = manufacturing_confidence(target, locks, candidates, model_curve_points)
        c1, c2 = st.columns([0.2, 0.8])
        c1.metric("Confidence", f"{confidence['score']}/100")
        c1.metric("Grade", confidence["grade"])
        c2.info(confidence["message"])
        for reason in confidence["reasons"]:
            st.write("• " + reason)
        st.warning(
            "Even a high catalogue fit is not a production release. Final geometry still requires shaft/stress checks, "
            "balance limits, controlled drawings, CFD/prototype review and AMCA 210 / ISO 5801-type testing."
        )

    with result_tabs[6]:
        predicted_curve = full_curve_for_geometry(target, bg, bm)
        ident = identifiability_table(candidates)
        sens = sensitivity_table(target, best, locks=locks, cfg=cfg, curve_points=model_curve_points)

        export_bio = io.BytesIO()
        with zipfile.ZipFile(export_bio, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("inverse_design_target.json", json.dumps(best["target"], indent=2))
            z.writestr("best_hidden_geometry.json", json.dumps(bg, indent=2))
            z.writestr("best_metrics.json", json.dumps({k:v for k,v in bm.items() if not isinstance(v, (dict,list))}, indent=2, default=str))
            z.writestr("top_candidates.csv", candidates.head(100).to_csv(index=False))
            z.writestr("predicted_curve.csv", predicted_curve.to_csv(index=False))
            z.writestr("identifiability_measurement_priority.csv", ident.to_csv(index=False))
            z.writestr("sensitivity.csv", sens.to_csv(index=False))
            z.writestr(
                "IMPORTANT.txt",
                "Inverse-designed geometry is one engineering solution consistent with the selected catalogue constraints. "
                "It is not claimed to be the manufacturer's proprietary internal geometry. Measure critical dimensions and validate by CFD/prototype testing before manufacture."
            )

        st.download_button(
            "Download inverse-design result package",
            export_bio.getvalue(),
            f"{model}_inverse_design_v27.zip",
            "application/zip",
        )

        if st.button("Send best geometry to Reverse Engineer / Manufacturing page", type="primary"):
            handoff = {
                "supported": True,
                "source": "Catalogue inverse optimizer v27",
                "manufacturer": target.manufacturer,
                "model": target.model,
                "source_file": target.source_file,
                "source_page": target.source_page,
                "family": target.family,
                "arrangement": target.arrangement,
                "known_d2_mm": target.d2_mm,
                "known_airflow_m3h": target.airflow_m3h,
                "known_static_pressure_pa": target.static_pressure_pa,
                "known_rpm": target.rpm,
                "known_power_w": target.input_power_w,
                "known_noise_dba": target.noise_dba,
                "known_material": target.material,
                "calculated_d1_mm": bg["d1_mm"],
                "calculated_b2_mm": bg["b2_total_mm"],
                "calculated_b1_mm": bg["b1_total_mm"],
                "calculated_blade_count": int(bg["blade_count"]),
                "calculated_beta1_deg": bg["beta1_deg"],
                "calculated_beta2_deg": bg["beta2_deg"],
                "calculated_cutoff_mm": bg["cutoff_clearance_mm"],
                "notes": (
                    f"v27 inverse-optimized hidden geometry. Catalogue match RMSE "
                    f"{bm['catalogue_match_rmse_pct']:.1f}%. Treat as calculated, not manufacturer-measured."
                ),
            }
            st.session_state["catalogue_manufacture_handoff"] = handoff
            st.session_state["inverse_optimizer_geometry_handoff"] = {
                "geometry": bg,
                "metrics": bm,
                "model": model,
            }
            try:
                st.switch_page("pages/5_Reverse_Engineer_Blower.py")
            except Exception:
                st.success("Transferred. Select Reverse Engineer Blower from the left navigation.")
