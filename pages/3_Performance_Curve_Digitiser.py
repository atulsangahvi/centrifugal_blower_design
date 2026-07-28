
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from blower_toolkit.auth import require_password
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from blower_toolkit.catalogue import (
    AxisCalibration,
    CatalogueDB,
    convert_clicks,
    curve_quality_checks,
    fan_law_scale_curve,
)

st.set_page_config(page_title="Performance Curve Digitiser", layout="wide")
require_password()
st.title("Performance Curve Digitiser v22")
st.caption(
    "Calibrate a published graph, capture curve points, validate them, apply fan-law "
    "scaling, and save the verified points to the manufacturer knowledge base."
)

ROOT = Path(__file__).resolve().parents[1]
db = CatalogueDB(ROOT / "data" / "catalogue.db")
fans = db.all()

if fans.empty:
    st.error("Add fan models to the catalogue knowledge base before digitising curves.")
    st.stop()

with st.sidebar:
    st.header("Curve identity")
    model = st.selectbox("Model", fans.model.tolist())
    fan = fans[fans.model == model].iloc[0]
    curve_name = st.text_input("Curve name", "Published P-Q curve")
    curve_kind = st.selectbox(
        "Y-axis quantity",
        ["static_pressure", "total_pressure", "power", "efficiency", "noise"],
    )
    default_rpm = float(fan.rpm) if pd.notna(fan.rpm) else 0.0
    speed_rpm = st.number_input("Curve speed (rpm)", min_value=0.0, value=default_rpm)
    control_setting = st.text_input("Control setting", "")
    source_file = st.text_input("Source file", str(fan.source_file or ""))
    source_page = st.number_input(
        "Source PDF page", min_value=0, value=int(fan.source_page or 0), step=1
    )
    point_status = st.selectbox(
        "Point status",
        ["Needs review", "Verified from catalogue"],
    )

uploaded = st.file_uploader(
    "Upload a cropped or full catalogue curve image",
    type=["png", "jpg", "jpeg", "webp"],
)

if "digitizer_clicks" not in st.session_state:
    st.session_state.digitizer_clicks = []

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    width, height = image.size

    st.subheader("1. Calibrate the graph axes")
    st.info(
        "Enter the pixel coordinates of the plot rectangle. Pixel (0,0) is the "
        "top-left corner of the uploaded image. Use the live cursor readout below "
        "to locate the left, right, top and bottom plot boundaries."
    )

    c1, c2, c3, c4 = st.columns(4)
    x_left = c1.number_input("Plot left X pixel", 0.0, float(width), 0.0)
    x_right = c2.number_input("Plot right X pixel", 0.0, float(width), float(width))
    y_top = c3.number_input("Plot top Y pixel", 0.0, float(height), 0.0)
    y_bottom = c4.number_input("Plot bottom Y pixel", 0.0, float(height), float(height))

    a1, a2, a3, a4 = st.columns(4)
    x_min = a1.number_input("Airflow axis minimum (m³/h)", value=0.0)
    x_max = a2.number_input("Airflow axis maximum (m³/h)", value=10000.0)
    y_min = a3.number_input("Y-axis minimum", value=0.0)
    y_max = a4.number_input("Y-axis maximum", value=1000.0)
    l1, l2 = st.columns(2)
    x_log = l1.checkbox("Logarithmic airflow axis")
    y_log = l2.checkbox("Logarithmic Y-axis")

    calibration = AxisCalibration(
        x_left_px=x_left,
        x_right_px=x_right,
        y_top_px=y_top,
        y_bottom_px=y_bottom,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_log=x_log,
        y_log=y_log,
    )
    errors = calibration.validate()
    for error in errors:
        st.error(error)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    if not errors:
        draw.rectangle(
            [x_left, y_top, x_right, y_bottom],
            outline="red",
            width=max(2, int(min(width, height) / 400)),
        )
    for click in st.session_state.digitizer_clicks:
        r = max(3, int(min(width, height) / 250))
        draw.ellipse(
            [click["x"] - r, click["y"] - r, click["x"] + r, click["y"] + r],
            fill="red",
            outline="white",
        )

    st.subheader("2. Capture the curve")
    st.caption(
        "Click sequentially along one published curve. The app stores the original "
        "pixel point and its calibrated engineering coordinate."
    )
    value = streamlit_image_coordinates(
        annotated,
        key="curve_digitizer_image",
        width=min(width, 1200),
    )
    if value and not errors:
        click = {"x": float(value["x"]), "y": float(value["y"])}
        last = st.session_state.digitizer_clicks[-1] if st.session_state.digitizer_clicks else None
        if last != click:
            st.session_state.digitizer_clicks.append(click)
            st.rerun()

    b1, b2, b3 = st.columns(3)
    if b1.button("Undo last point", disabled=not st.session_state.digitizer_clicks):
        st.session_state.digitizer_clicks.pop()
        st.rerun()
    if b2.button("Clear all points", disabled=not st.session_state.digitizer_clicks):
        st.session_state.digitizer_clicks = []
        st.rerun()
    b3.metric("Captured points", len(st.session_state.digitizer_clicks))

    if st.session_state.digitizer_clicks and not errors:
        points = convert_clicks(
            st.session_state.digitizer_clicks,
            calibration,
            model=model,
            curve_name=curve_name,
            speed_rpm=speed_rpm or None,
            source_file=source_file,
            source_page=source_page or None,
            curve_kind=curve_kind,
            control_setting=control_setting,
            point_status=point_status,
        )

        st.subheader("3. Review digitised points")
        edited = st.data_editor(
            points,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
        )

        qcol, pcol = st.columns(2)
        with qcol:
            st.markdown("#### Reconstructed curve")
            if curve_kind in ("static_pressure", "total_pressure"):
                st.line_chart(
                    edited.set_index("airflow_m3h")["pressure_pa"]
                )
            else:
                st.info(
                    "The database currently stores pressure in the main Y field. "
                    "Edit the appropriate power, efficiency or noise column before saving."
                )
        with pcol:
            st.markdown("#### Quality checks")
            st.dataframe(
                curve_quality_checks(edited),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("4. Save or scale")
        s1, s2 = st.columns(2)
        replace_curve = s1.checkbox(
            "Replace an existing curve with the same model, curve name and RPM"
        )
        if s1.button("Save curve to knowledge base", type="primary"):
            save_df = edited.drop(
                columns=["point_no", "pixel_x", "pixel_y"], errors="ignore"
            )
            n = db.add_curve_points(save_df, replace_curve=replace_curve)
            st.success(f"Saved {n} curve points.")
            st.session_state.digitizer_clicks = []

        with s2:
            st.markdown("#### Fan-law preview")
            new_speed = st.number_input(
                "New speed (rpm)",
                min_value=1.0,
                value=float(speed_rpm or 1.0),
                key="new_speed",
            )
            old_density = st.number_input(
                "Original air density (kg/m³)", min_value=0.1, value=1.20
            )
            new_density = st.number_input(
                "New air density (kg/m³)", min_value=0.1, value=1.20
            )
            if st.button("Generate scaled curve"):
                scaled = fan_law_scale_curve(
                    edited.drop(
                        columns=["point_no", "pixel_x", "pixel_y"], errors="ignore"
                    ),
                    old_speed_rpm=float(speed_rpm),
                    new_speed_rpm=float(new_speed),
                    old_density_kg_m3=float(old_density),
                    new_density_kg_m3=float(new_density),
                )
                st.dataframe(scaled, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download scaled curve CSV",
                    scaled.to_csv(index=False).encode("utf-8"),
                    f"{model}_scaled_curve.csv",
                    "text/csv",
                )
else:
    st.markdown(
        """
### Recommended workflow

1. Open the manufacturer PDF and export or screenshot only the curve graph.
2. Upload the graph image here.
3. Use the cursor coordinates to identify the plot rectangle.
4. Enter the published axis minimum and maximum values.
5. Click 8–20 points along one curve.
6. Review the reconstructed curve and quality warnings.
7. Save it as **Needs review**.
8. Compare several digitised points manually against the source graph.
9. Change the status to **Verified from catalogue** only after checking.
"""
    )
