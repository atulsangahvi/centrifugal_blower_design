
# Centrifugal Blower Design & Manufacturer Intelligence Toolkit v23

This release combines the existing centrifugal blower design engine with the first serious version of a manufacturer catalogue intelligence platform.

## Run on Streamlit Cloud

Use:

```text
app.py
```

The multipage navigation includes:

- Blower Design
- Fan Manufacturer Knowledge Base
- Development Roadmap



## v23 — AHU and cooling-tower fan-wall selector

v23 adds a working parallel-fan screening and retrofit module:

- AHU supply / return fan walls
- Cooling-tower axial fan walls
- Multiple fans operating in parallel
- Configurable N+1 or greater redundancy
- Flow and installation margins
- Variable-speed fan-law screening
- Preliminary fan staging sequence
- Existing fan, motor, belt and control efficiency baseline
- Annual energy and maintenance savings
- Simple payback calculation
- Fan-wall face-velocity estimate where dimensions exist
- Engineering checks and arrangement guidance
- CSV export for selected option, control sequence and all ranked candidates

Important: v23 screens published catalogue summary ratings. Final selections must use
verified full curves and account for system effect, density, fan interaction, noise,
motor limits and recognised test data.

## v22 — Visual performance-curve digitiser

v22 adds a working graph digitisation page:

- Upload PNG/JPG/WebP curve images.
- Calibrate graph pixel boundaries and engineering axes.
- Support linear or logarithmic axes.
- Click sequential points on a published curve.
- Reconstruct and edit the Q-P dataset.
- Run data-quality and engineering trend checks.
- Save digitised points with model, RPM, source file and source page.
- Scale a verified curve to another speed and air density using classical fan laws.
- Export scaled curves as CSV.

Install the two additional dependencies in `requirements.txt`:

- Pillow
- streamlit-image-coordinates


## What v21 adds

### 1. Source catalogue registry
Every source PDF is registered by manufacturer, title, filename, product scope, page count, publication code and review status.

The starter manifest includes the uploaded Seemtek catalogues:

- Product Comprehensive Catalog
- Romulus backward-curved fan brochure
- Achelous backward-curved fan brochure
- Axial Fan Product Catalog
- Forward Curved Centrifugal Blower Catalog
- AHU Fan Retrofit
- Cooling Tower EC Fan Retrofit
- Impeller Brochure: Vulturnus, AeroPlus and Aeolus

The source PDFs themselves are not redistributed in this ZIP. Upload them through the app when performing extraction or keep them in your controlled engineering document library.

### 2. Normalised database schema

The SQLite database now separates:

- `documents`: source catalogue provenance
- `fans`: model identity and summary data
- `curve_points`: digitised Q-P, power, efficiency and sound points
- `dimensions`: drawing dimensions and tolerances
- `metadata`: database schema version

This separation is essential because a catalogue's maximum airflow and maximum static pressure are normally opposite endpoints, not one simultaneous duty point.

### 3. Curve-based duty selection

When curve points exist, the selector interpolates pressure at the requested airflow and ranks only curves that contain that airflow. Endpoint benchmarking is used only as a fallback.

### 4. Data-quality controls

Statuses include:

- Verified from catalogue
- Needs review
- Rejected
- Illustrative only

Every model and curve point retains source filename and page.

### 5. PDF / CSV importer

The importer detects several Seemtek, Longwell and generic fan model-number patterns. It extracts only clearly labelled values and sends all results to a review table.

Image-heavy PDF pages still require visual checking. Automatic extraction is not treated as verified engineering data.

## Starter data

A small number of Seemtek records are included as examples from clearly readable catalogue tables and drawings. A demonstration curve is marked `Illustrative only`; replace it with digitised source data before engineering use.

## Recommended GitHub structure

```text
app.py
blower_design.py
requirements.txt
README.md
.streamlit/
blower_toolkit/
pages/
data/
```

Do not upload your real `.streamlit/secrets.toml`. Use Streamlit Cloud Secrets.

## Next development work

1. Vision-assisted graph digitiser with axis calibration.
2. Drawing-dimension capture and tolerance review.
3. Density, speed and diameter correction using fan laws.
4. AHU fan-wall selection and N+1 redundancy.
5. AHU retrofit annual-energy and ROI calculator.
6. Cooling tower EC axial fan retrofit selection.
7. Commercial impeller geometry benchmarking.
8. Manufacturing CAD, DXF, STEP, BOM and costing.
9. Calibration with AMCA 210 / ISO 5801 test results.
10. Proprietary product-series generator for your manufactured fans.

## Engineering limitation

This toolkit is for preliminary design, catalogue intelligence and product development. Final fan ratings, sound values, structural suitability and manufacturing release require controlled drawings, material verification, balancing, prototype testing and recognised laboratory methods.
