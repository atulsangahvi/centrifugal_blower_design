
## v28.1 - GitHub browser commit-safe release

The previous v28 package contained 101 project files. GitHub's browser uploader
allows a maximum of 100 files per upload/commit operation.

v28.1 reduces the project to fewer than 100 files by replacing the 13 Airtek
split PDFs with the original 350-page Airtek catalogue, which is only
17.88 MiB and therefore remains below
GitHub's 25 MiB browser file-size limit.

All Airtek database, curve, dimension, and model source references now point to
the original Airtek PDF and original page number.

For maximum reliability in the browser, four optional upload batches are also
provided separately so you can commit the project in several smaller commits.

## v28 - Airtek Power V.2024 catalogue integration

The 350-page **Airtek Power Product Manual V.2024** is now embedded as 13 logical,
GitHub-safe PDF parts instead of one large catalogue file.

Structured Airtek data added in this release:

- **364 product models** across EC/AC backward-curved centrifugal, forward-curved centrifugal, single-inlet, double-inlet, axial and duct fan families
- **963 published numeric Q-P/rpm/power points** extracted from the catalogue's printed performance tables
- **212 Airtek models with structured published performance points**
- **619 nomenclature-derived D2/H dimension records** with source-page traceability
- AC 50 Hz ratings are preferred as the summary row when the catalogue provides both 50 and 60 Hz; the alternate ratings are retained in `data/airtek_alternate_ratings_v28.csv`
- every model, curve point and dimension maps to the exact embedded split PDF and local page

Airtek centrifugal models with four or more published Q-P points can be sent directly
to **Catalogue Inverse Designer** using the published curve as the aerodynamic constraint.
This is preferable to treating catalogue maximum airflow and pressure values as one duty point.

The Catalogue Duty Search is also upgraded: when stored Q-P points exist it interpolates
manufacturer pressure at the requested airflow inside the published curve range.

Audit files:

- `data/airtek_models_v28.csv`
- `data/airtek_curve_points_v28.csv`
- `data/airtek_dimensions_v28.csv`
- `data/airtek_alternate_ratings_v28.csv`
- `data/airtek_pdf_split_mapping_v28.csv`


### Small plug/plenum compatibility fix

Airtek's backward-curved EC/AC ranges include unhoused/plug-style wheels. The geometry validator now allows plenum families to operate without a volute discharge flange or scroll width, because the aerodynamic engine already treats their discharge kinetic energy as a plenum loss. This prevents the catalogue inverse-design workflow from crashing simply because a plug fan has no volute outlet.

## v27.2 SAFE - GitHub browser upload

All individual files are kept below **18 MB**, giving comfortable margin below
GitHub's browser-upload size limit. Large catalogue PDFs are split by page range;
no intended catalogue page content is removed.


## v27.2 - GitHub browser-upload compatible catalogue assets

GitHub's browser uploader rejects large individual files. The previously embedded
Seemtek Forward Curved, Longwell EC Backward Curved, and Seemtek Axial catalogues
were above 25 MB.

v27.2 splits only those large PDFs into page-range parts while preserving the PDF
page content. The SQLite model database, dimensions, curve records, and CSV sources
are automatically remapped to the correct PDF part and local page number.

See `data/catalogue_pdf_split_mapping_v27_2.csv` and
`GITHUB_UPLOAD_INSTRUCTIONS.txt`.


## v27.1 — Longwell technical-table parsing correction

During v27 validation, the earlier generic numeric parser was found to shift Longwell
columns whenever a table cell contained `/` for unavailable current/noise values.
That could turn a speed such as 2550 rpm into the current column and corrupt the
following fields.

v27.1 replaces those Longwell rows with named-column parsing directly from the
technical tables:

- AC Forward Curved
- EC Dual Inlet Forward Curved
- EC Backward Curved

The corrected audit dataset is stored at:

`data/longwell_models_corrected_v27.csv`

The inverse designer still treats adjacent Air Flow / Air Pressure summary values as
endpoint constraints by default unless an actual Q-P curve or confirmed simultaneous
operating point is available.


## v27 — Catalogue-constrained inverse design

This release turns a catalogue blower into a genuine inverse engineering problem.

The new **Catalogue Inverse Designer** does not automatically assume that adjacent
catalogue "Air Flow" and "Air Pressure" numbers occur at the same point. It supports:

1. **Catalogue max-flow / max-pressure endpoints** — the safe default for many
   manufacturer technical tables. The solver fits free-delivery airflow and
   near-shutoff static pressure separately.
2. **Simultaneous duty point** — use when Q and P are known to belong to the same
   operating point.
3. **Digitised Q-P curve** — preferred when four or more actual plotted curve points
   are stored.

At fixed D2 and RPM the inverse solver varies hidden geometry:

- D1/D2
- b2/D2
- b1/b2
- beta1
- beta2
- blade count
- cutoff/tongue clearance
- discharge velocity/area
- scroll internal width
- scroll throat-velocity relationship

The objective combines catalogue matching, efficiency, optional power/noise
constraints, tip-speed limits, blade blockage, pitch/chord manufacturability and
geometry priors.

Measured dimensions can be locked so optimization cannot change them.

The app also exposes a central truth of reverse engineering: **the inverse problem can
be non-unique**. Several hidden geometries may reproduce the same sparse catalogue
data. v27 therefore adds:

- top-candidate comparison
- identifiability ranges
- a ranked "what should I measure next?" table
- parameter sensitivity analysis
- manufacturing-confidence grade
- explicit empirical model-correction coefficients
- direct transfer of the best candidate into Reverse Engineer Blower for blade,
  scroll, shaft, balancing, CAD/DXF and manufacturing development

A good catalogue fit is one engineering solution consistent with the available
constraints; it is not claimed to reproduce the manufacturer's proprietary internal
geometry. Prototype CFD and fan testing remain required before production release.

## v26 — Embedded manufacturer catalogue browser + manufacturing bridge

This build physically embeds 10 Seemtek/Longwell PDF catalogues, extracts 231 model/impeller records, shows the exact source page inside the app, supports catalogue duty searching, and can send any selected catalogue model into Reverse Engineer Blower. Known catalogue values are retained; missing geometry is calculated from the appropriate family ranges and explicitly marked low-confidence until optimized/measured.

## v25.3 catalogue chart hotfix

Fixed a `KeyError` in **Catalogue Knowledge Base → Digitised performance curves**.

The failure was caused by passing a Pandas `MultiIndex` column structure directly
from `pivot_table()` into Streamlit's `st.line_chart()`.  The chart data is now
cleaned, sorted, and flattened to ordinary unique string column names before
rendering.  Invalid/empty airflow-pressure records are handled without crashing.

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
