from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from blower_toolkit.auth import require_password
from blower_toolkit.engine import FAMILIES, MATERIALS, wheel_performance, air_density
from blower_toolkit.reverse_engineering import (
    REFERENCE_PRESETS,
    ReferenceEnvelope,
    GeometryInput,
    OperatingInput,
    TestCalibration,
    DriveMechanicalInput,
    preset_to_envelope,
    provisional_geometry,
    envelope_frame,
    geometry_provenance_table,
    readiness_score,
    mm_per_pixel,
    measure_line_mm,
    fit_circle_three_points,
    benchmark_speed_sweep,
    raw_meanline_curve,
    normalize_curve_to_benchmark,
    calibrate_curve_to_test,
    comparison_check,
    operating_point,
    solve_speed_for_duty,
    scale_curve_speed,
    area_law_volute_stations,
    log_spiral_volute_stations,
    reference_trace_to_stations,
    center_hung_mechanical_check,
    balance_residual_unbalance,
    motor_selection_for_operating_range,
    measurement_checklist,
    envelope_fit_checks,
    blade_profile_table,
    impeller_figure,
    blade_figure,
    volute_figure,
    curve_figure,
    make_manufacturing_package,
)

st.set_page_config(page_title="Reverse Engineer Existing Blower", layout="wide")
require_password()

ROOT = Path(__file__).resolve().parents[1]

st.title("Reverse Engineer Existing Blower & Manufacturing Geometry — v25 High-Effort Rebuild")
optimizer_notice = st.session_state.get("inverse_optimizer_geometry_handoff")
if optimizer_notice:
    st.success(
        f"Catalogue inverse optimizer v27 geometry loaded for {optimizer_notice.get('model','')}. "
        "These dimensions are calculated inverse-design values, not manufacturer-measured geometry."
    )
catalogue_handoff = st.session_state.get("catalogue_manufacture_handoff")
if catalogue_handoff:
    st.success(f"Catalogue reference loaded: {catalogue_handoff.get('manufacturer','')} {catalogue_handoff.get('model','')}. Known catalogue values are preserved; missing geometry starts as calculated/low-confidence.")
st.caption(
    "Drawing measurement → geometry provenance → independent benchmark sanity check → "
    "mean-line performance → test calibration → centre-hung shaft/drive check → blade/scroll manufacturing data."
)

st.warning(
    "The two uploaded WDL/KQ800 sheets are outline drawings, not impeller detail drawings. "
    "This page therefore never treats D2, D1, wheel width, blade count or blade angles as verified unless you classify them as measured/verified."
)

# -----------------------------------------------------------------------------
# Shared session state
# -----------------------------------------------------------------------------
for key, default in {
    "re_measure_clicks": [],
    "re_measurements": [],
    "re_scale_mm_per_px": None,
    "re_results": None,
    "re_uploaded_drawing_bytes": None,
    "re_uploaded_drawing_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

workflow = st.tabs([
    "1 Reference Drawing",
    "2 Measure Drawing",
    "3 Impeller / Scroll Geometry",
    "4 Performance & RPM",
    "5 Shaft / Drive / Balance",
    "6 Manufacturing Output",
])

# -----------------------------------------------------------------------------
# 1. Reference drawing
# -----------------------------------------------------------------------------
with workflow[0]:
    st.subheader("Supplier outline drawing and verified external envelope")

    st.markdown("#### Drawing source")
    st.caption(
        "You can use one of the built-in WDL/KQ800 reference drawings or upload any new "
        "blower drawing. An uploaded drawing is carried automatically into the measurement tab."
    )

    source_mode = st.radio(
        "Choose drawing source",
        ["Built-in reference drawing", "Upload new drawing"],
        horizontal=True,
        key="re_drawing_source_mode",
    )

    preset_name = "Custom reference"

    if source_mode == "Built-in reference drawing":
        preset_name = st.selectbox(
            "Reference drawing preset",
            list(REFERENCE_PRESETS.keys()),
            key="re_reference_preset",
        )
        p = REFERENCE_PRESETS[preset_name]
        env = preset_to_envelope(preset_name)
        asset_path = ROOT / p["asset"]
        if asset_path.exists():
            st.image(str(asset_path), caption=preset_name, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Drawing", env.drawing_code or "—")
        c2.metric("Declared fan weight", f"{env.declared_fan_weight_kg:.0f} kg")
        c3.metric("Shaft end", f"Ø{env.shaft_diameter_mm:.0f} mm")
        c4.metric("Balance", env.balance_grade)
        st.info(
            f"Drawing-stated fan standard: {env.fan_standard}. {p.get('fan_standard_note','')} "
            "The KQ800 2026 sheet visibly specifies impeller dynamic balance G4.0."
        )

    else:
        uploaded_reference = st.file_uploader(
            "Upload a new blower drawing",
            type=["png", "jpg", "jpeg", "webp"],
            key="reverse_reference_upload",
            help="Upload an outline drawing, impeller drawing, casing drawing, scan, or clear photograph.",
        )

        if uploaded_reference is not None:
            uploaded_bytes = uploaded_reference.getvalue()
            st.session_state["re_uploaded_drawing_bytes"] = uploaded_bytes
            st.session_state["re_uploaded_drawing_name"] = uploaded_reference.name

        if st.session_state.get("re_uploaded_drawing_bytes"):
            try:
                uploaded_image = Image.open(
                    io.BytesIO(st.session_state["re_uploaded_drawing_bytes"])
                ).convert("RGB")
                st.image(
                    uploaded_image,
                    caption=st.session_state.get("re_uploaded_drawing_name") or "Uploaded drawing",
                    use_container_width=True,
                )
                st.success(
                    "New drawing loaded. Open the '2 Measure Drawing' tab to calibrate and measure it."
                )
                if st.button("Clear uploaded drawing", key="re_clear_uploaded_drawing"):
                    st.session_state["re_uploaded_drawing_bytes"] = None
                    st.session_state["re_uploaded_drawing_name"] = None
                    st.session_state["re_measure_clicks"] = []
                    st.session_state["re_measurements"] = []
                    st.session_state["re_scale_mm_per_px"] = None
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not open uploaded drawing: {exc}")

        else:
            st.info("Upload a drawing above to begin a new reverse-engineering job.")

        env = ReferenceEnvelope(
            reference_name=st.session_state.get("re_uploaded_drawing_name") or "Custom reference"
        )

    st.markdown("#### Editable drawing dimensions")
    e1, e2, e3, e4 = st.columns(4)
    env.base_side_length_mm = e1.number_input("Base side length (mm)", 0.0, 10000.0, float(env.base_side_length_mm))
    env.frame_inner_width_mm = e2.number_input("Frame inner/base width (mm)", 0.0, 10000.0, float(env.frame_inner_width_mm))
    env.overall_height_mm = e3.number_input("Overall height (mm)", 0.0, 10000.0, float(env.overall_height_mm))
    env.shaft_center_height_mm = e4.number_input("Shaft centre height (mm)", 0.0, 10000.0, float(env.shaft_center_height_mm))

    e5, e6, e7, e8 = st.columns(4)
    env.overall_axial_width_mm = e5.number_input("Overall axial width (mm)", 0.0, 10000.0, float(env.overall_axial_width_mm))
    env.body_axial_width_mm = e6.number_input("Body axial width (mm)", 0.0, 10000.0, float(env.body_axial_width_mm))
    env.inner_body_axial_width_mm = e7.number_input("Inner/body clear axial width (mm)", 0.0, 10000.0, float(env.inner_body_axial_width_mm))
    env.shaft_diameter_mm = e8.number_input("Drawing shaft diameter (mm)", 0.0, 500.0, float(env.shaft_diameter_mm))

    e9, e10, e11, e12 = st.columns(4)
    env.discharge_clear_width_mm = e9.number_input("Discharge clear width (mm)", 0.0, 10000.0, float(env.discharge_clear_width_mm))
    env.discharge_clear_height_mm = e10.number_input("Discharge clear height (mm)", 0.0, 10000.0, float(env.discharge_clear_height_mm))
    env.discharge_flange_outer_width_mm = e11.number_input("Flange outside width (mm)", 0.0, 10000.0, float(env.discharge_flange_outer_width_mm))
    env.discharge_flange_outer_height_mm = e12.number_input("Flange outside height (mm)", 0.0, 10000.0, float(env.discharge_flange_outer_height_mm))

    st.dataframe(envelope_frame(env), use_container_width=True, hide_index=True)
    st.session_state["re_env"] = env
    st.session_state["re_preset_name"] = preset_name

# -----------------------------------------------------------------------------
# 2. Drawing measurement tool
# -----------------------------------------------------------------------------
with workflow[1]:
    env = st.session_state.get("re_env", ReferenceEnvelope("Custom reference"))
    st.subheader("Scale and measure features directly from the outline drawing")
    st.caption(
        "Use a known dimension such as 1330 mm to calibrate the displayed drawing. "
        "Then measure any straight feature with 2 clicks or a circular feature with 3 clicks. "
        "Perspective/skew in a raster drawing can introduce error, so measured values remain medium-confidence until checked physically."
    )

    preset_name = st.session_state.get("re_preset_name", "Custom reference")
    image = None
    image_caption = None

    # Prefer a drawing uploaded in Tab 1.
    if st.session_state.get("re_uploaded_drawing_bytes"):
        try:
            image = Image.open(
                io.BytesIO(st.session_state["re_uploaded_drawing_bytes"])
            ).convert("RGB")
            image_caption = st.session_state.get("re_uploaded_drawing_name") or "Uploaded drawing"
        except Exception as exc:
            st.error(f"Could not reopen uploaded drawing: {exc}")

    # Otherwise use the selected built-in reference.
    if image is None and preset_name in REFERENCE_PRESETS:
        p = ROOT / REFERENCE_PRESETS[preset_name]["asset"]
        if p.exists():
            image = Image.open(p).convert("RGB")
            image_caption = preset_name

    st.caption(
        "Drawing currently loaded: "
        + (image_caption if image_caption else "none")
        + ". You can also replace it here."
    )

    up = st.file_uploader(
        "Replace / upload drawing for measurement",
        type=["png", "jpg", "jpeg", "webp"],
        key="reverse_measure_upload",
    )
    if up:
        new_bytes = up.getvalue()
        st.session_state["re_uploaded_drawing_bytes"] = new_bytes
        st.session_state["re_uploaded_drawing_name"] = up.name
        image = Image.open(io.BytesIO(new_bytes)).convert("RGB")
        image_caption = up.name

    if image is None:
        st.info("Upload a new drawing in Tab 1 or choose a built-in reference drawing.")
    else:
        mode = st.radio("Measurement mode", ["Calibrate scale", "Line measurement", "Circle diameter (3 points)"], horizontal=True)
        expected_points = 2 if mode != "Circle diameter (3 points)" else 3

        if mode == "Calibrate scale":
            known_mm = st.number_input("Known dimension represented by your two clicks (mm)", 1.0, 10000.0, 1330.0)
        else:
            known_mm = 0.0

        # Resize to one fixed pixel width before annotation/click capture so the
        # calibration and the drawn cursor markers use the same coordinate system.
        display_width = 1050
        if image.width != display_width:
            display_height = int(round(image.height * display_width / image.width))
            annotated = image.resize((display_width, display_height))
        else:
            annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for p in st.session_state.re_measure_clicks:
            x, y = p
            r = 6
            draw.ellipse((x-r, y-r, x+r, y+r), fill="red", outline="white")
        if len(st.session_state.re_measure_clicks) >= 2:
            draw.line(st.session_state.re_measure_clicks[:2], fill="red", width=3)

        value = streamlit_image_coordinates(annotated, key="reverse_measure_image", width=display_width)
        if value:
            pt = (float(value["x"]), float(value["y"]))
            last = st.session_state.re_measure_clicks[-1] if st.session_state.re_measure_clicks else None
            if last != pt and len(st.session_state.re_measure_clicks) < expected_points:
                st.session_state.re_measure_clicks.append(pt)
                st.rerun()

        b1, b2, b3 = st.columns(3)
        if b1.button("Undo click", disabled=not st.session_state.re_measure_clicks):
            st.session_state.re_measure_clicks.pop(); st.rerun()
        if b2.button("Clear clicks", disabled=not st.session_state.re_measure_clicks):
            st.session_state.re_measure_clicks = []; st.rerun()
        b3.metric("Clicks", f"{len(st.session_state.re_measure_clicks)}/{expected_points}")

        clicks = st.session_state.re_measure_clicks
        if len(clicks) == expected_points:
            if mode == "Calibrate scale":
                scale = mm_per_pixel(clicks[0], clicks[1], known_mm)
                st.success(f"Scale = {scale:.4f} mm per displayed pixel")
                if st.button("Accept this calibration", type="primary"):
                    st.session_state.re_scale_mm_per_px = scale
                    st.session_state.re_measure_clicks = []
                    st.rerun()
            elif st.session_state.re_scale_mm_per_px is None:
                st.error("Calibrate the scale first.")
            else:
                scale = st.session_state.re_scale_mm_per_px
                if mode == "Line measurement":
                    result_mm = measure_line_mm(clicks[0], clicks[1], scale)
                    result_kind = "Line"
                else:
                    _, radius_px = fit_circle_three_points(clicks)
                    result_mm = 2.0 * radius_px * scale
                    result_kind = "Circle diameter"
                st.success(f"Measured {result_kind}: {result_mm:.1f} mm")
                label = st.text_input("Measurement label", "D2" if result_kind.startswith("Circle") else "Feature")
                if st.button("Save measurement", type="primary"):
                    st.session_state.re_measurements.append({
                        "Label": label,
                        "Value_mm": result_mm,
                        "Method": result_kind,
                        "Source": "Measured from scaled drawing image",
                        "Confidence": "MEDIUM",
                    })
                    st.session_state.re_measure_clicks = []
                    st.rerun()

        st.metric("Current drawing scale", "Not calibrated" if st.session_state.re_scale_mm_per_px is None else f"{st.session_state.re_scale_mm_per_px:.4f} mm/pixel")
        if st.session_state.re_measurements:
            st.dataframe(pd.DataFrame(st.session_state.re_measurements), use_container_width=True, hide_index=True)
            if st.button("Clear saved drawing measurements"):
                st.session_state.re_measurements = []; st.rerun()

# -----------------------------------------------------------------------------
# 3. Geometry & provenance
# -----------------------------------------------------------------------------
with workflow[2]:
    env = st.session_state.get("re_env", ReferenceEnvelope("Custom reference"))
    st.subheader("Impeller, blade and scroll geometry — every value gets a provenance tag")

    handoff = st.session_state.get("catalogue_manufacture_handoff")
    handoff_d2 = float(handoff.get("known_d2_mm",0.0)) if handoff else 0.0
    default_g, default_prov = provisional_geometry(handoff_d2 or 800.0, env.shaft_diameter_mm or 60.0)
    if handoff:
        default_g.family = handoff.get("family", default_g.family)
        default_g.arrangement = handoff.get("arrangement", default_g.arrangement)
        default_g.d1_mm = float(handoff.get("calculated_d1_mm",default_g.d1_mm))
        default_g.b2_total_mm = float(handoff.get("calculated_b2_mm",default_g.b2_total_mm))
        default_g.b1_total_mm = float(handoff.get("calculated_b1_mm",default_g.b1_total_mm))
        default_g.blade_count = int(handoff.get("calculated_blade_count",default_g.blade_count))
        default_g.beta1_deg = float(handoff.get("calculated_beta1_deg",default_g.beta1_deg))
        default_g.beta2_deg = float(handoff.get("calculated_beta2_deg",default_g.beta2_deg))
        default_g.cutoff_clearance_mm = float(handoff.get("calculated_cutoff_mm",default_g.cutoff_clearance_mm))
    default_g.scroll_internal_width_mm = env.discharge_clear_width_mm or default_g.scroll_internal_width_mm
    default_g.discharge_width_mm = env.discharge_clear_width_mm or default_g.discharge_width_mm
    default_g.discharge_height_mm = env.discharge_clear_height_mm or default_g.discharge_height_mm
    default_g.shaft_diameter_mm = env.shaft_diameter_mm or default_g.shaft_diameter_mm

    optimizer_handoff = st.session_state.get("inverse_optimizer_geometry_handoff")
    optimized_loaded = bool(
        optimizer_handoff
        and optimizer_handoff.get("model") == (handoff or {}).get("model")
    )
    if optimized_loaded:
        og = optimizer_handoff.get("geometry", {})
        for attr in [
            "d2_mm","d1_mm","b2_total_mm","b1_total_mm","blade_count",
            "beta1_deg","beta2_deg","blade_thickness_mm","plate_thickness_mm",
            "hub_diameter_mm","shaft_diameter_mm","cutoff_clearance_mm",
            "scroll_internal_width_mm","discharge_width_mm","discharge_height_mm"
        ]:
            if attr in og and og[attr] is not None:
                setattr(default_g, attr, og[attr])

    # Apply any drawing measurement with an exact recognized label.
    measurements = {str(x["Label"]).strip().lower(): float(x["Value_mm"]) for x in st.session_state.get("re_measurements", [])}
    for key, attr in {"d2":"d2_mm", "d1":"d1_mm", "b2":"b2_total_mm", "b1":"b1_total_mm", "shaft":"shaft_diameter_mm", "shaft diameter":"shaft_diameter_mm"}.items():
        if key in measurements:
            setattr(default_g, attr, measurements[key])

    g1, g2, g3 = st.columns(3)
    fam_options=list(FAMILIES.keys()); default_family=default_g.family if default_g.family in fam_options else "Multi-Blade Sirocco (Cage)"
    family = g1.selectbox("Fan family", fam_options, index=fam_options.index(default_family))
    arr_options=["DWDI (double inlet)","SWSI (single inlet)"]; default_arr=default_g.arrangement if default_g.arrangement in arr_options else arr_options[0]
    arrangement = g2.selectbox("Arrangement", arr_options, index=arr_options.index(default_arr))
    material = g3.selectbox("Impeller material", list(MATERIALS.keys()), index=list(MATERIALS.keys()).index("Galvanized Steel"))

    fam = FAMILIES[family]
    d2_source = f"Manufacturer catalogue page {handoff.get('source_page','')}" if handoff else "Nominal model designation — VERIFY physically"
    calc_source = (
        "Inverse-optimized to catalogue constraints in v27 — CALCULATED, not manufacturer-measured"
        if optimized_loaded
        else "Calculated starting geometry from app fan-family design ranges"
        if handoff
        else "Assumed family default"
    )
    rows = [
        {"Parameter":"D2", "Value":default_g.d2_mm, "Unit":"mm", "Source":d2_source, "Confidence":"HIGH" if handoff else "LOW"},
        {"Parameter":"D1", "Value":default_g.d1_mm if handoff else fam["d1d2"]*default_g.d2_mm, "Unit":"mm", "Source":calc_source, "Confidence":"MEDIUM" if optimized_loaded else "LOW"},
        {"Parameter":"b2 total", "Value":default_g.b2_total_mm if handoff else fam["b2d2"]*default_g.d2_mm, "Unit":"mm", "Source":calc_source, "Confidence":"MEDIUM" if optimized_loaded else "LOW"},
        {"Parameter":"b1 total", "Value":default_g.b1_total_mm if handoff else 1.05*fam["b2d2"]*default_g.d2_mm, "Unit":"mm", "Source":calc_source, "Confidence":"MEDIUM" if optimized_loaded else "LOW"},
        {"Parameter":"Blade count", "Value":float(default_g.blade_count if handoff else fam["z"]), "Unit":"count", "Source":calc_source, "Confidence":"MEDIUM" if optimized_loaded else "LOW"},
        {"Parameter":"beta1", "Value":float(default_g.beta1_deg if handoff else fam["beta1"]), "Unit":"deg from tangent", "Source":calc_source, "Confidence":"MEDIUM" if optimized_loaded else "LOW"},
        {"Parameter":"beta2", "Value":float(default_g.beta2_deg if handoff else fam["beta2"]), "Unit":"deg from tangent", "Source":calc_source, "Confidence":"MEDIUM" if optimized_loaded else "LOW"},
        {"Parameter":"Blade thickness", "Value":2.0, "Unit":"mm", "Source":"Assumed fabrication value", "Confidence":"LOW"},
        {"Parameter":"Plate thickness", "Value":3.0, "Unit":"mm", "Source":"Assumed fabrication value", "Confidence":"LOW"},
        {"Parameter":"Hub diameter", "Value":max(120.0, 2.0*(env.shaft_diameter_mm or 60.0)), "Unit":"mm", "Source":"Assumed provisional value", "Confidence":"LOW"},
        {"Parameter":"Shaft diameter", "Value":env.shaft_diameter_mm or 60.0, "Unit":"mm", "Source":"Supplier outline drawing" if env.shaft_diameter_mm else "Assumed", "Confidence":"HIGH" if env.shaft_diameter_mm else "LOW"},
        {"Parameter":"Cutoff clearance", "Value":0.05*default_g.d2_mm, "Unit":"mm", "Source":"Assumed 5% D2 — measure actual tongue", "Confidence":"LOW"},
        {"Parameter":"Scroll internal width", "Value":env.discharge_clear_width_mm or 1007.0, "Unit":"mm", "Source":"Inferred from drawing clear discharge width — VERIFY", "Confidence":"MEDIUM"},
        {"Parameter":"Discharge width", "Value":env.discharge_clear_width_mm or 1007.0, "Unit":"mm", "Source":"Supplier outline drawing", "Confidence":"HIGH" if env.discharge_clear_width_mm else "LOW"},
        {"Parameter":"Discharge height", "Value":env.discharge_clear_height_mm or 1007.0, "Unit":"mm", "Source":"Supplier outline drawing", "Confidence":"HIGH" if env.discharge_clear_height_mm else "LOW"},
    ]

    # Override recognized measured drawing values.
    for row in rows:
        k = row["Parameter"].lower()
        alias = {"d2":"d2", "d1":"d1", "b2 total":"b2", "b1 total":"b1", "shaft diameter":"shaft"}.get(k)
        if alias and alias in measurements:
            row["Value"] = measurements[alias]
            row["Source"] = "Measured from scaled drawing image"
            row["Confidence"] = "MEDIUM"

    edited = st.data_editor(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Source": st.column_config.TextColumn(help="State exactly where this value came from."),
            "Confidence": st.column_config.SelectboxColumn(options=["HIGH", "MEDIUM", "LOW"]),
        },
        key="reverse_geometry_editor",
    )

    vals = {r.Parameter: float(r.Value) for r in edited.itertuples()}
    source_map = {r.Parameter: {"source": str(r.Source), "confidence": str(r.Confidence)} for r in edited.itertuples()}
    g = GeometryInput(
        family=family,
        arrangement=arrangement,
        d2_mm=vals["D2"], d1_mm=vals["D1"], b2_total_mm=vals["b2 total"], b1_total_mm=vals["b1 total"],
        blade_count=int(round(vals["Blade count"])), beta1_deg=vals["beta1"], beta2_deg=vals["beta2"],
        blade_thickness_mm=vals["Blade thickness"], plate_thickness_mm=vals["Plate thickness"],
        hub_diameter_mm=vals["Hub diameter"], shaft_diameter_mm=vals["Shaft diameter"],
        cutoff_clearance_mm=vals["Cutoff clearance"], scroll_internal_width_mm=vals["Scroll internal width"],
        discharge_width_mm=vals["Discharge width"], discharge_height_mm=vals["Discharge height"], material=material,
    )
    prov = geometry_provenance_table(g, source_map)
    st.session_state["re_geometry"] = g
    st.session_state["re_provenance"] = prov

    # Physical meaning / practical ranges.
    st.markdown("#### Geometry practicality")
    practical = pd.DataFrame([
        ["D1/D2", g.d1_mm/g.d2_mm, f"Family guide {fam['d1d2_rng'][0]:.2f}–{fam['d1d2_rng'][1]:.2f}", "PASS" if fam['d1d2_rng'][0] <= g.d1_mm/g.d2_mm <= fam['d1d2_rng'][1] else "WARN"],
        ["b2/D2", g.b2_total_mm/g.d2_mm, f"Family guide {fam['b2d2_rng'][0]:.2f}–{fam['b2d2_rng'][1]:.2f}", "PASS" if fam['b2d2_rng'][0] <= g.b2_total_mm/g.d2_mm <= fam['b2d2_rng'][1] else "WARN"],
        ["beta1", g.beta1_deg, f"{fam['beta1_rng'][0]}–{fam['beta1_rng'][1]}° from tangent", "PASS" if fam['beta1_rng'][0] <= g.beta1_deg <= fam['beta1_rng'][1] else "WARN"],
        ["beta2", g.beta2_deg, f"{fam['beta2_rng'][0]}–{fam['beta2_rng'][1]}° from tangent", "PASS" if fam['beta2_rng'][0] <= g.beta2_deg <= fam['beta2_rng'][1] else "WARN"],
        ["Blade count", g.blade_count, f"{fam['z_rng'][0]}–{fam['z_rng'][1]}", "PASS" if fam['z_rng'][0] <= g.blade_count <= fam['z_rng'][1] else "WARN"],
    ], columns=["Parameter","Current","Guide","Status"])
    st.dataframe(practical, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# 4. Performance and RPM
# -----------------------------------------------------------------------------
with workflow[3]:
    env = st.session_state.get("re_env", ReferenceEnvelope("Custom reference"))
    g = st.session_state.get("re_geometry")
    prov = st.session_state.get("re_provenance")
    if g is None:
        st.info("Complete the geometry tab first.")
    else:
        st.subheader("Performance prediction with an independent commercial sanity benchmark")
        st.info(
            "RPM is not shown on the WDL/KQ800 outline drawings. The default 600 rpm is therefore only a starting point. "
            "The table below immediately shows how strongly flow, pressure and power change with RPM."
        )
        handoff = st.session_state.get("catalogue_manufacture_handoff")
        catalogue_rpm=float(handoff.get("known_rpm",0.0)) if handoff else 0.0
        catalogue_q=float(handoff.get("known_airflow_m3h",0.0)) if handoff else 0.0
        catalogue_sp=float(handoff.get("known_static_pressure_pa",0.0)) if handoff else 0.0
        if handoff and (catalogue_q>0 or catalogue_sp>0):
            st.success(f"Catalogue target: Q={catalogue_q:,.0f} m³/h, SP={catalogue_sp:,.0f} Pa, RPM={catalogue_rpm:,.0f}. Refine calculated geometry until the model matches this known duty.")
        p1,p2,p3,p4 = st.columns(4)
        rpm = p1.number_input("Fan shaft speed (rpm)", 100.0, 3000.0, catalogue_rpm if catalogue_rpm>=100 else 600.0, 10.0)
        temp = p2.number_input("Air temperature (°C)", -40.0, 150.0, 35.0, 1.0)
        altitude = p3.number_input("Altitude (m)", -500.0, 5000.0, 0.0, 50.0)
        rh = p4.number_input("RH (%)", 0.0, 100.0, 50.0, 5.0)
        rho = air_density(temp, altitude, rh)
        st.metric("Calculated air density", f"{rho:.3f} kg/m³")

        # Add RPM provenance so readiness is honest.
        prov2 = pd.concat([prov, pd.DataFrame([{"Parameter":"RPM","Value":rpm,"Unit":"rpm","Source":"User entered — verify pulley ratio / tachometer","Confidence":"LOW"}])], ignore_index=True)

        area = (g.discharge_width_mm/1000.0)*(g.discharge_height_mm/1000.0)
        sweep = benchmark_speed_sweep(g.d2_mm, rho, area, 300, 900, 50)
        st.markdown("#### Independent 800-class forward-curved similarity sanity scale")
        st.caption(
            "This is not WDL/KQ performance. It scales one published, tested Kruger FDA forward-curved DIDW selection point by fan laws to your D2, RPM and density. "
            "Its job is to catch impossible reverse-engineering results."
        )
        st.dataframe(sweep.round(2), use_container_width=True, hide_index=True)

        casing_method = st.selectbox("Calculated casing model", ["Area-law rectangular", "Log spiral / free-vortex", "Reference discharge only"])
        op = OperatingInput(rpm=rpm, temp_c=temp, altitude_m=altitude, rh_pct=rh, density_kg_m3=rho, min_flow_fraction=0.25, max_flow_fraction=1.45, curve_points=49)

        # Optional test calibration.
        st.markdown("#### Optional calibration to your own measured/test point")
        cal_on = st.checkbox("I have a test point for this blower / prototype")
        test = TestCalibration(enabled=cal_on)
        if cal_on:
            t1,t2,t3,t4 = st.columns(4)
            test.test_rpm = t1.number_input("Test rpm", 100.0, 3000.0, rpm)
            test.test_airflow_m3h = t2.number_input("Measured airflow (m³/h)", 100.0, 500000.0, 50000.0, 500.0)
            test.test_pressure_pa = t3.number_input("Measured pressure (Pa)", 1.0, 10000.0, 800.0, 10.0)
            test.pressure_kind = t4.selectbox("Pressure value is", ["Static", "Total"])
            t5,t6,t7 = st.columns(3)
            test.test_shaft_power_kw = t5.number_input("Measured shaft power (kW; 0 if unknown)", 0.0, 500.0, 0.0, 0.5)
            test.test_motor_input_kw = t6.number_input("Measured motor input kW (0 if shaft power known)", 0.0, 500.0, 0.0, 0.5)
            test.motor_efficiency_pct = t7.number_input("Motor efficiency for input→shaft (%)", 50.0, 99.0, 90.0, 1.0)

        if st.button("Run high-effort reverse-engineering performance model", type="primary"):
            try:
                wheel, raw, meta = raw_meanline_curve(g, op, casing_method)
                normalized, norm_meta = normalize_curve_to_benchmark(raw, g, op)
                if test.enabled:
                    calibrated, cal_meta = calibrate_curve_to_test(raw, g, op, test)
                    selected = calibrated
                    selected_label = "Test-calibrated mean-line"
                    selected_meta = cal_meta
                else:
                    selected = normalized
                    selected_label = "Benchmark-normalized mean-line"
                    selected_meta = norm_meta

                comparison = comparison_check(raw, g, op)
                st.session_state.re_results = {
                    "wheel": wheel,
                    "raw": raw,
                    "normalized": normalized,
                    "selected": selected,
                    "selected_label": selected_label,
                    "meta": meta,
                    "selected_meta": selected_meta,
                    "comparison": comparison,
                    "sweep": sweep,
                    "op": op,
                    "prov": prov2,
                    "casing_method": casing_method,
                    "test": test,
                }
                st.rerun()
            except Exception as exc:
                st.exception(exc)

        res = st.session_state.get("re_results")
        if res:
            raw = res["raw"]; norm = res["normalized"]; selected = res["selected"]
            st.markdown("#### Raw physics vs sanity-normalized result")
            st.pyplot(curve_figure({"Raw mean-line":raw, "Benchmark-normalized":norm, res["selected_label"]:selected}), clear_figure=True)
            st.dataframe(res["comparison"].round(3), use_container_width=True, hide_index=True)
            st.caption(
                "If the raw model differs from the independent benchmark by more than roughly 40%, the app flags it for review rather than quietly trusting the result. "
                "A real test point supersedes the commercial sanity normalization."
            )

            # Report operating point.
            q_default = float(catalogue_q) if handoff and catalogue_q>0 else float(res["meta"]["q_reference_m3h"])
            qmin=float(selected.Flow_m3h.min()); qmax=float(selected.Flow_m3h.max())
            report_q = st.number_input("Airflow at which to report performance (m³/h)", qmin, qmax, min(max(q_default,qmin),qmax), 500.0)
            point = operating_point(selected, report_q)
            st.session_state["re_selected_q"] = report_q
            st.session_state["re_selected_point"] = point

            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Airflow", f"{report_q:,.0f} m³/h")
            m2.metric("Static pressure", f"{point['Static_Pa']:,.0f} Pa")
            m3.metric("Total pressure", f"{point['Total_Pa']:,.0f} Pa")
            m4.metric("Shaft power", f"{point['Shaft_kW']:.1f} kW")
            m5.metric("Total efficiency", f"{100*point['EtaTotal']:.1f}%")
            m6.metric("Outlet velocity", f"{point['OutletVelocity_ms']:.1f} m/s")

            # Inverse duty-to-RPM solver.
            st.markdown("#### Required duty → estimated RPM")
            s1,s2,s3,s4 = st.columns(4)
            target_q=s1.number_input("Target airflow (m³/h)", 100.0, 500000.0, report_q, 500.0)
            target_sp=s2.number_input("Target static pressure (Pa)", 1.0, 10000.0, max(1.0,float(point['Static_Pa'])), 10.0)
            nmin=s3.number_input("Search minimum rpm",100.0,3000.0,300.0,10.0)
            nmax=s4.number_input("Search maximum rpm",100.0,3000.0,1000.0,10.0)
            if st.button("Find closest fan speed"):
                try:
                    sol=solve_speed_for_duty(selected, rpm, target_q, target_sp, nmin, nmax, 2.0)
                    st.session_state["re_speed_solution"]=sol
                    st.success(f"Closest speed ≈ {sol['rpm']:.0f} rpm → {sol['static_pressure_pa']:.0f} Pa at {target_q:,.0f} m³/h; shaft ≈ {sol['shaft_power_kw']:.1f} kW")
                except Exception as exc:
                    st.error(str(exc))

# -----------------------------------------------------------------------------
# 5. Mechanical / drive / balance
# -----------------------------------------------------------------------------
with workflow[4]:
    env = st.session_state.get("re_env", ReferenceEnvelope("Custom reference"))
    g = st.session_state.get("re_geometry")
    res = st.session_state.get("re_results")
    point = st.session_state.get("re_selected_point")
    if g is None or res is None or point is None:
        st.info("Run the performance model first.")
    else:
        st.subheader("Centre-hung DIDW shaft, belt drive, motor and balance check")
        st.info(
            "The original v24 shaft routine was an overhung-wheel approximation. This reverse-engineering page instead uses a centre-hung DIDW model between two bearings, with optional pulley overhang/belt load."
        )
        d1,d2,d3,d4 = st.columns(4)
        drive_type=d1.selectbox("Drive", ["Belt drive","Direct / coupling"])
        bearing_span=d2.number_input("Bearing centre span (mm; MEASURE)",0.0,5000.0,0.0,10.0)
        pulley_d=d3.number_input("Pulley pitch diameter (mm; belt drive)",0.0,2000.0,0.0,10.0,disabled=drive_type!="Belt drive")
        pulley_over=d4.number_input("Pulley overhang from drive bearing (mm)",0.0,1000.0,0.0,10.0,disabled=drive_type!="Belt drive")

        d5,d6,d7,d8 = st.columns(4)
        imp_mass=d5.number_input("Actual impeller mass (kg; 0=estimate)",0.0,2000.0,0.0,5.0)
        shaft_d=d6.number_input("Shaft diameter for check (mm)",20.0,300.0,float(env.shaft_diameter_mm or g.shaft_diameter_mm),5.0)
        belt_factor=d7.number_input("Total belt radial-load factor × tangential",1.0,6.0,2.5,0.1,disabled=drive_type!="Belt drive")
        balance_grade=d8.text_input("Balance grade",env.balance_grade or "G4.0")

        d9,d10,d11 = st.columns(3)
        drive_eff=d9.number_input("Drive efficiency (%)",50.0,100.0,95.0,1.0)
        motor_eff=d10.number_input("Motor efficiency (%)",50.0,99.0,92.0,1.0)
        margin=d11.number_input("Motor margin (%)",0.0,50.0,15.0,1.0)

        max_q=st.number_input("Maximum airflow the system can reach (for FC motor-overload sizing) m³/h",float(res["selected"].Flow_m3h.min()),float(res["selected"].Flow_m3h.max()),min(float(res["selected"].Flow_m3h.max()),1.20*float(st.session_state.re_selected_q)),500.0)

        mech_in=DriveMechanicalInput(
            drive_type=drive_type,bearing_span_mm=bearing_span,pulley_pitch_diameter_mm=pulley_d,pulley_overhang_mm=pulley_over,
            belt_pull_factor=belt_factor,impeller_mass_kg=imp_mass,shaft_diameter_mm=shaft_d,
            drive_efficiency_pct=drive_eff,motor_efficiency_pct=motor_eff,motor_margin_pct=margin,balance_grade=balance_grade,
        )
        rpm_for_mech = st.session_state.get("re_speed_solution",{}).get("rpm",res["op"].rpm)
        # If a solved RPM exists, scale the selected curve for motor sizing; otherwise use model speed.
        mech_curve=scale_curve_speed(res["selected"],res["op"].rpm,rpm_for_mech) if abs(rpm_for_mech-res["op"].rpm)>1e-6 else res["selected"]
        q_for_mech=st.session_state.get("re_speed_solution",{}).get("airflow_m3h",st.session_state.re_selected_q)
        shaft_kw=operating_point(mech_curve,q_for_mech)["Shaft_kW"]
        mechanical=center_hung_mechanical_check(g,rpm_for_mech,shaft_kw,mech_in)
        motor=motor_selection_for_operating_range(mech_curve,q_for_mech,max_q,mech_in)
        balance=balance_residual_unbalance(balance_grade,rpm_for_mech,mechanical["impeller_mass_kg"],2)

        st.session_state["re_mech_in"]=mech_in
        st.session_state["re_mechanical"]=mechanical
        st.session_state["re_motor"]=motor
        st.session_state["re_balance"]=balance

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Torque",f"{mechanical['torque_nm']:.0f} N·m")
        m2.metric("Required shaft",f"{mechanical['shaft_diameter_required_mm']:.0f} mm")
        m3.metric("Checked shaft",f"{mechanical['shaft_diameter_used_mm']:.0f} mm")
        m4.metric("Selected motor",f"{motor['selected_standard_motor_kw']:.1f} kW")
        m5.metric("Residual unbalance total",f"{balance['U_total_gmm']:.0f} g·mm" if balance['U_total_gmm'] else "N/A")

        c1,c2 = st.columns(2)
        with c1:
            st.dataframe(pd.DataFrame([mechanical]).T.rename(columns={0:"Value"}),use_container_width=True)
        with c2:
            st.dataframe(pd.DataFrame([motor]).T.rename(columns={0:"Value"}),use_container_width=True)
            st.dataframe(pd.DataFrame([balance]).T.rename(columns={0:"Value"}),use_container_width=True)

        if bearing_span <= 0:
            st.error("Bearing span is still unknown. Shaft critical speed and reactions are not release-ready.")
        if drive_type=="Belt drive" and (pulley_d<=0 or pulley_over<=0):
            st.error("Pulley size/overhang are still unknown. Belt load and shaft bending are not release-ready.")

# -----------------------------------------------------------------------------
# 6. Manufacturing output
# -----------------------------------------------------------------------------
with workflow[5]:
    env=st.session_state.get("re_env",ReferenceEnvelope("Custom reference"))
    g=st.session_state.get("re_geometry")
    prov=st.session_state.get("re_provenance")
    res=st.session_state.get("re_results")
    mech=st.session_state.get("re_mechanical")
    motor=st.session_state.get("re_motor")
    balance=st.session_state.get("re_balance")
    mech_in=st.session_state.get("re_mech_in")
    selected_point=st.session_state.get("re_selected_point")
    if g is None or res is None:
        st.info("Complete the geometry and performance tabs first.")
    else:
        st.subheader("Blade, volute, envelope fit and manufacturing-reference package")
        wheel=res["wheel"]
        qsel=st.session_state.get("re_selected_q",res["meta"]["q_reference_m3h"])
        psel=wheel_performance(wheel,res["meta"]["density_kg_m3"],res["op"].rpm,qsel/3600.0)

        method=st.selectbox("Volute geometry to export",["Area-law rectangular","Log spiral / free-vortex","Reference casing trace CSV"])
        if method=="Area-law rectangular":
            volute=area_law_volute_stations(wheel,qsel/3600.0,psel,10,0.75)
            title="Area-law rectangular volute (design alternative)"
        elif method=="Log spiral / free-vortex":
            volute=log_spiral_volute_stations(wheel,qsel/3600.0,psel,10)
            title="Log-spiral/free-vortex volute (design alternative)"
        else:
            trace_up=st.file_uploader("Upload traced casing stations CSV with columns theta_deg,R_mm",type=["csv"],key="trace_csv")
            if trace_up:
                volute=reference_trace_to_stations(pd.read_csv(trace_up)); title="Reference casing trace"
            else:
                volute=pd.DataFrame(); title="Reference casing trace"
                st.info("Upload traced stations to use this method.")

        st.caption(
            "A calculated scroll is a design alternative, not an exact copy of the WDL/KQ casing. "
            "For an exact replacement, trace or physically measure the actual casing station radii and tongue geometry."
        )

        r1,r2,r3=st.tabs(["Impeller","Single blade","Volute / scroll"])
        with r1:
            st.pyplot(impeller_figure(wheel),clear_figure=True)
        with r2:
            st.pyplot(blade_figure(wheel),clear_figure=True)
            st.dataframe(blade_profile_table(wheel).round(3),use_container_width=True,hide_index=True)
        with r3:
            if not volute.empty:
                st.pyplot(volute_figure(volute,wheel,title),clear_figure=True)
                st.dataframe(volute.round(3),use_container_width=True,hide_index=True)

        prov_for_ready=res.get("prov",prov)
        ready=readiness_score(prov_for_ready,calibrated=bool(res.get("test") and res["test"].enabled))
        st.markdown(f"### Manufacturing readiness: Grade {ready['grade']} — {ready['score']:.0f}%")
        st.write(ready["message"])
        checklist=measurement_checklist(prov_for_ready,mech_in)
        st.dataframe(checklist,use_container_width=True,hide_index=True)
        if not volute.empty:
            st.markdown("#### External envelope fit")
            st.dataframe(envelope_fit_checks(env,g,volute).round(2),use_container_width=True,hide_index=True)

        if mech is None or motor is None or balance is None:
            st.warning("Complete the Shaft / Drive / Balance tab before creating the full manufacturing package.")
        elif volute.empty:
            st.warning("Choose or upload a volute definition before creating the package.")
        else:
            package=make_manufacturing_package(
                env,g,prov_for_ready,wheel,res["raw"],res["selected"],res["sweep"],selected_point or {},volute,
                mech,balance,motor,checklist,ready,res["selected_meta"],ROOT,
            )
            st.download_button(
                "Download v25 high-effort reverse-engineering package",
                package,
                "KQ800_reverse_engineering_manufacturing_package.zip",
                "application/zip",
                type="primary",
            )
            st.markdown(
                "The ZIP includes PDF + Excel engineering reports, DXF reference drawing, raw/selected performance curves, independent benchmark speed sweep, blade coordinates, volute stations, mechanical/balance/motor checks, provenance, measurement checklist, plots, and copies of the two uploaded reference drawings."
            )
