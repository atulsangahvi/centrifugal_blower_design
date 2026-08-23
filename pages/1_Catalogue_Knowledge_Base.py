
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from blower_toolkit.auth import require_password

from blower_toolkit.catalogue import (
    CatalogueDB,
    MODEL_COLUMNS,
    CURVE_COLUMNS,
    DOCUMENT_COLUMNS,
    extract_candidates_from_pdf,
    rank_matches,
    select_by_curves,
)

st.set_page_config(page_title="Fan Manufacturer Knowledge Base", layout="wide")
require_password()
st.title("Fan Manufacturer Knowledge Base v25")
st.caption(
    "Catalogue registry, model database, curve storage, real-duty interpolation, "
    "traceability, and data-quality control."
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "catalogue.db"
SEED_MODELS = ROOT / "data" / "seed_catalogue_v21.csv"
SEED_DOCS = ROOT / "data" / "catalogue_manifest.csv"
SEED_CURVES = ROOT / "data" / "seed_curves_v21.csv"

db = CatalogueDB(DB_PATH)
if db.documents().empty and SEED_DOCS.exists():
    db.register_documents(pd.read_csv(SEED_DOCS))
if db.all().empty and SEED_MODELS.exists():
    db.upsert_dataframe(pd.read_csv(SEED_MODELS))
if db.curves().empty and SEED_CURVES.exists():
    db.add_curve_points(pd.read_csv(SEED_CURVES))

with st.sidebar:
    st.header("Database status")
    st.metric("Registered documents", len(db.documents()))
    st.metric("Stored models", len(db.all()))
    st.metric("Curve points", len(db.curves()))
    st.warning(
        "Machine-extracted values remain 'Needs review' until checked against "
        "the original catalogue page."
    )
    st.download_button(
        "Download model CSV",
        db.export_csv(),
        "fan_models.csv",
        "text/csv",
    )
    st.download_button(
        "Download curve CSV",
        db.export_curve_csv(),
        "fan_curve_points.csv",
        "text/csv",
    )

tabs = st.tabs([
    "Dashboard",
    "Catalogue Registry",
    "Model Library",
    "Import PDF / CSV",
    "Curve Data",
    "Duty Selection",
    "Review & Edit",
    "Data Dictionary",
])

with tabs[0]:
    q = db.quality_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manufacturers", int(db.all().manufacturer.nunique()) if len(db.all()) else 0)
    c2.metric("Verified models", int((db.all().data_status == "Verified from catalogue").sum()) if len(db.all()) else 0)
    c3.metric("Models with duty ratings", int((db.all().airflow_m3h.notna() & db.all().static_pressure_pa.notna()).sum()) if len(db.all()) else 0)
    c4.metric("Models with full curves", int(db.curves().model.nunique()) if len(db.curves()) else 0)
    st.subheader("Manufacturer data quality")
    st.dataframe(q, use_container_width=True, hide_index=True)
    st.info(
        "v21 separates three different data layers: catalogue endpoint ratings, "
        "digitised performance-curve points, and drawing dimensions. This prevents "
        "maximum airflow and maximum pressure from being mistaken for one simultaneous duty point."
    )

with tabs[1]:
    st.subheader("Registered source catalogues")
    docs = db.documents()
    st.dataframe(
        docs[[
            "id", "manufacturer", "title", "filename", "document_type",
            "product_scope", "publication_code", "page_count", "source_status"
        ]] if not docs.empty else docs,
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("### Register another catalogue")
    upload = st.file_uploader("Choose PDF for registry", type=["pdf"], key="registry_pdf")
    if upload:
        manufacturer = st.text_input("Manufacturer", "Seemtek", key="registry_maker")
        title = st.text_input("Catalogue title", Path(upload.name).stem)
        document_type = st.selectbox(
            "Document type",
            ["Comprehensive catalogue", "Product brochure", "Application brochure",
             "Impeller brochure", "Retrofit brochure", "Testing / standards", "Other"],
        )
        scope = st.text_input("Product scope")
        pub_code = st.text_input("Publication code")
        if st.button("Register source document"):
            from pypdf import PdfReader
            import io
            pages = len(PdfReader(io.BytesIO(upload.getvalue())).pages)
            row = pd.DataFrame([{
                "manufacturer": manufacturer,
                "title": title,
                "filename": upload.name,
                "document_type": document_type,
                "product_scope": scope,
                "publication_code": pub_code,
                "revision": "",
                "page_count": pages,
                "source_status": "Uploaded / awaiting extraction",
                "notes": "",
            }])
            db.register_documents(row)
            st.success("Catalogue registered.")
            st.rerun()

with tabs[2]:
    df = db.all()
    c1, c2, c3, c4 = st.columns(4)
    makers = ["All"] + sorted(df.manufacturer.dropna().unique().tolist()) if len(df) else ["All"]
    types = ["All"] + sorted(df.fan_type.dropna().unique().tolist()) if len(df) else ["All"]
    maker = c1.selectbox("Manufacturer", makers)
    ftype = c2.selectbox("Fan type", types)
    text = c3.text_input("Model / series contains")
    status_options = ["All"] + sorted(df.data_status.dropna().unique().tolist()) if len(df) else ["All"]
    status = c4.selectbox("Data status", status_options)

    view = df.copy()
    if maker != "All":
        view = view[view.manufacturer == maker]
    if ftype != "All":
        view = view[view.fan_type == ftype]
    if status != "All":
        view = view[view.data_status == status]
    if text:
        mask = (
            view.model.fillna("").str.contains(text, case=False)
            | view.series.fillna("").str.contains(text, case=False)
        )
        view = view[mask]
    columns = [
        "manufacturer", "series", "model", "fan_type", "arrangement", "motor_type",
        "impeller_diameter_mm", "wheel_width_mm", "blade_count",
        "airflow_m3h", "static_pressure_pa", "rpm", "input_power_w",
        "efficiency_pct", "noise_dba", "material", "ip_rating",
        "source_file", "source_page", "data_status",
    ]
    st.dataframe(view[[c for c in columns if c in view]], use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Import manufacturer data")
    manufacturer = st.text_input("Manufacturer name", "Seemtek", key="import_maker")
    uploads = st.file_uploader(
        "Upload PDF or structured CSV files",
        type=["pdf", "csv"],
        accept_multiple_files=True,
        key="model_import",
    )
    if uploads:
        frames = []
        all_warnings = []
        for up in uploads:
            if up.name.lower().endswith(".csv"):
                x = pd.read_csv(up)
                for c in MODEL_COLUMNS:
                    if c not in x:
                        x[c] = None
                frames.append(x[MODEL_COLUMNS])
            else:
                with st.spinner(f"Extracting model candidates from {up.name}..."):
                    result = extract_candidates_from_pdf(
                        up.getvalue(), up.name, manufacturer
                    )
                frames.append(result.candidates)
                all_warnings.extend([f"{up.name}: {w}" for w in result.warnings])
                st.caption(
                    f"{up.name}: {result.page_count} pages; "
                    f"text extracted from {result.extracted_pages} pages."
                )
        for warning in all_warnings:
            st.warning(warning)
        staged = (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame(columns=MODEL_COLUMNS)
        )
        st.warning(
            "Review all fields against the catalogue. Image-heavy tables and drawings "
            "cannot be trusted from text extraction alone."
        )
        edited = st.data_editor(
            staged,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "data_status": st.column_config.SelectboxColumn(
                    options=["Needs review", "Verified from catalogue", "Rejected"]
                )
            },
        )
        if st.button("Save reviewed model rows", type="primary"):
            keep = edited[edited.data_status != "Rejected"]
            n = db.upsert_dataframe(keep)
            st.success(f"Saved or updated {n} model records.")

with tabs[4]:
    st.subheader("Digitised performance curves")
    fans = db.all()
    if fans.empty:
        st.info("Add model records first.")
    else:
        model = st.selectbox("Model", fans.model.tolist(), key="curve_model")
        existing = db.curves(model)
        if not existing.empty:
            # Streamlit's built-in line_chart can fail when Pandas returns
            # MultiIndex columns from pivot_table (notably with newer
            # Pandas/Streamlit combinations on Python 3.14).  Flatten the
            # curve-name/RPM column index into ordinary unique strings first.
            chart_source = existing.dropna(
                subset=["airflow_m3h", "pressure_pa"]
            ).copy()

            if not chart_source.empty:
                chart_df = chart_source.pivot_table(
                    index="airflow_m3h",
                    columns=["curve_name", "speed_rpm"],
                    values="pressure_pa",
                    aggfunc="mean",
                ).sort_index()

                if isinstance(chart_df.columns, pd.MultiIndex):
                    flat_names = []
                    used_names = {}
                    for curve_name, speed_rpm in chart_df.columns.to_list():
                        curve_label = str(curve_name) if pd.notna(curve_name) else "Curve"
                        if pd.notna(speed_rpm):
                            try:
                                speed_label = f"{float(speed_rpm):g} rpm"
                            except (TypeError, ValueError):
                                speed_label = f"{speed_rpm} rpm"
                            base_name = f"{curve_label} | {speed_label}"
                        else:
                            base_name = curve_label

                        # Guarantee unique plain-string column names.
                        count = used_names.get(base_name, 0)
                        used_names[base_name] = count + 1
                        flat_names.append(
                            base_name if count == 0 else f"{base_name} ({count + 1})"
                        )

                    chart_df.columns = flat_names
                else:
                    chart_df.columns = [str(c) for c in chart_df.columns]

                if not chart_df.empty and len(chart_df.columns) > 0:
                    st.line_chart(
                        chart_df,
                        x_label="Airflow (m³/h)",
                        y_label="Pressure (Pa)",
                    )
                else:
                    st.info("No plottable pressure-curve points are available for this model.")
            else:
                st.info("Stored curve records do not yet contain valid airflow/pressure points.")

            st.dataframe(existing, use_container_width=True, hide_index=True)
        else:
            st.info("No curve points stored for this model.")

        st.markdown("### Add or replace curve points")
        st.caption(
            "Upload CSV columns: model, curve_name, curve_kind, speed_rpm, "
            "control_setting, airflow_m3h, pressure_pa, power_w, efficiency_pct, "
            "noise_dba, source_file, source_page, point_status, notes."
        )
        curve_upload = st.file_uploader("Curve-point CSV", type=["csv"], key="curve_csv")
        if curve_upload:
            curve_df = pd.read_csv(curve_upload)
        else:
            curve_df = pd.DataFrame([{
                "model": model,
                "curve_name": "Published P-Q curve",
                "curve_kind": "static_pressure",
                "speed_rpm": fans.loc[fans.model == model, "rpm"].iloc[0],
                "control_setting": "",
                "airflow_m3h": None,
                "pressure_pa": None,
                "power_w": None,
                "efficiency_pct": None,
                "noise_dba": None,
                "source_file": fans.loc[fans.model == model, "source_file"].iloc[0],
                "source_page": fans.loc[fans.model == model, "source_page"].iloc[0],
                "point_status": "Needs review",
                "notes": "",
            }])
        for c in CURVE_COLUMNS:
            if c not in curve_df:
                curve_df[c] = None
        edited_curve = st.data_editor(
            curve_df[CURVE_COLUMNS],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )
        replace = st.checkbox("Replace matching model / curve / speed before saving")
        if st.button("Save curve points", type="primary"):
            n = db.add_curve_points(edited_curve, replace_curve=replace)
            st.success(f"Saved {n} curve points.")
            st.rerun()

with tabs[5]:
    st.subheader("Commercial fan duty selection")
    a, b, c, d = st.columns(4)
    airflow = a.number_input("Required airflow (m³/h)", 1.0, 1_000_000.0, 10_000.0, 100.0)
    pressure = b.number_input("Required static pressure (Pa)", 1.0, 20_000.0, 500.0, 10.0)
    types = ["All"] + sorted(db.all().fan_type.dropna().unique().tolist())
    fan_type = c.selectbox("Fan type", types, key="select_type")
    tolerance = d.slider("Pressure tolerance (%)", 5, 50, 25)
    verified_only = st.checkbox("Use verified models only")

    curve_matches = select_by_curves(
        db.all(), db.curves(), airflow, pressure,
        fan_type=fan_type,
        verified_only=verified_only,
        tolerance_pct=float(tolerance),
    )
    if not curve_matches.empty:
        st.success("Curve-based matches found. These interpolate actual stored curve points.")
        st.dataframe(curve_matches, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "No digitised curve contains this airflow within the selected tolerance. "
            "Showing endpoint-rating benchmarks instead."
        )
        endpoint = rank_matches(
            db.all(), airflow, pressure, None, fan_type
        )
        if endpoint.empty:
            st.info("No comparable endpoint ratings are stored.")
        else:
            st.dataframe(endpoint, use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Review and maintain model records")
    all_df = db.all()
    if all_df.empty:
        st.info("Database is empty.")
    else:
        model = st.selectbox("Select model", all_df.model.tolist(), key="edit_model")
        row = all_df[all_df.model == model].iloc[0]
        edit_df = pd.DataFrame([{c: row.get(c) for c in MODEL_COLUMNS}])
        updated = st.data_editor(
            edit_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
        )
        c1, c2 = st.columns(2)
        if c1.button("Save changes", type="primary"):
            db.upsert_dataframe(updated)
            st.success("Record updated.")
            st.rerun()
        if c2.button("Delete model"):
            db.delete_models([row.id])
            st.success("Record deleted.")
            st.rerun()

with tabs[7]:
    st.markdown(
        """
### Data layers

**Model record**  
One row per commercial fan or impeller. Endpoint values may be catalogue maxima and are not automatically a valid simultaneous duty point.

**Curve point**  
Digitised Q-P, power, efficiency, or sound data at a stated RPM or control setting. These points support interpolation and real duty selection.

**Dimension record**  
Named drawing dimensions with page traceability and optional tolerances.

### Data status

- **Verified from catalogue** — manually checked against the source page.
- **Needs review** — machine-extracted, incomplete, or awaiting visual confirmation.
- **Rejected** — extraction candidate intentionally excluded.

### Engineering rule

A commercial match is not a certified selection unless the entered duty lies on a verified published curve, at the correct air density, speed, installation category, and test arrangement.
"""
    )
    dictionary = pd.DataFrame({
        "Table": ["documents", "fans", "curve_points", "dimensions"],
        "Purpose": [
            "Source catalogue registry and provenance",
            "Fan identity, endpoint ratings, materials, electrical and dimensional summary",
            "Digitised performance maps for interpolation and selection",
            "Drawing dimensions and tolerances",
        ],
    })
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
