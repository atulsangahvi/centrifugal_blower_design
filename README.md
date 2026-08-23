## v25.2 reverse-engineering drawing upload improvement

The Reverse Engineer Existing Blower page now has a prominent **Upload new drawing**
option in the first **Reference Drawing** tab.

The uploaded image is stored in Streamlit session state and is automatically reused
in the **Measure Drawing** tab. The measurement tab also retains its own
replace/upload control.

This fixes the earlier UX problem where the upload control existed only inside the
measurement tab and was easy to miss.

## v25.1 hotfix

Fixed the Reverse Engineer Existing Blower page runtime error:

`NameError: name 'air_density' is not defined`

The page now imports `air_density` from `blower_toolkit.engine` before calculating
air density from temperature, altitude and relative humidity.


# Centrifugal Blower Design & Manufacturer Intelligence Toolkit v25 — High-Effort Reverse-Engineering Rebuild

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
- Performance Curve Digitiser
- AHU Cooling Tower Fan Wall
- Reverse Engineer Existing Blower




## v25 — High-Effort Reverse Engineering & Manufacturing Mode

This v25 rebuild replaces the earlier quick reverse-engineering attempt. The main correction is methodological: an outline drawing is no longer treated as if it contains hidden impeller geometry or a complete fan rating.

The new `Reverse Engineer Existing Blower` page includes:

- two exact uploaded WDL/KQ800 outline-drawing presets, including the 2026 KQ800 V4.0 and 2020 WDL-800 V1.0 revisions
- drawing scale calibration plus 2-point linear and 3-point circular measurement tools
- a provenance/confidence tag for every aerodynamic dimension
- explicit `D2`, `D1`, total wheel width, blade count, beta1, beta2, blade/plate thickness, tongue clearance and scroll width inputs
- an independent commercial forward-curved DIDW similarity benchmark used only as a sanity scale
- the existing velocity-triangle/loss-model mean-line calculation as a separate raw physics result
- benchmark-normalized mean-line mode and a higher-confidence user-test calibration mode
- an inverse duty-to-RPM search
- forward-curved motor-overload sizing over the declared operating-flow range, not only one point
- centre-hung DIDW shaft mechanics with bearing span, pulley diameter, pulley overhang and belt load
- G-grade residual-unbalance calculation; the KQ800 drawing preset uses its stated G4.0 requirement
- circular-arc blade manufacturing coordinates
- three scroll choices: area-law rectangular, free-vortex/log spiral, or user-traced reference casing stations
- an external-envelope fit check
- a manufacturing-readiness grade plus a list of measurements still required before release
- PDF, Excel, DXF, CSV and PNG manufacturing-reference exports

### Why an independent benchmark is included

A reverse-engineering model can produce a mathematically converged but physically implausible result if hidden dimensions are guessed. v25 therefore compares the raw model against a published forward-curved DIDW reference scale. This does **not** claim that WDL/KQ has the same performance; it simply prevents an 800-class fan from being accepted at an obviously wrong order of magnitude without a warning.

### Release rule

Any dimension tagged `LOW` confidence or `Assumed` must be physically measured or replaced by a controlled manufacturer/detail drawing before production release. A test-calibrated curve substantially improves performance confidence, but it does not replace mechanical drawing verification.


## v24 — Shared application authentication

The password stored in Streamlit Secrets now protects every page, not only the
main blower-design page.

Supported Secrets formats:

```toml
APP_PASSWORD = "your_password"
```

or:

```toml
[auth]
password = "your_password"
```

The user logs in once per Streamlit session. The same authenticated session then
covers Blower Design, Catalogue Knowledge Base, Development Roadmap, Performance
Curve Digitiser, and AHU Cooling Tower Fan Wall. A Log out button appears in the
sidebar after authentication.

If no password is configured, the app stays open for local development.

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
