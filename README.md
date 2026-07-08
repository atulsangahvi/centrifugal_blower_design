# Centrifugal Blower Design & Manufacturing Toolkit v18

SI-unit Streamlit app for preliminary centrifugal blower design for AHU, HVAC package units, cooling towers, exhaust and process-air applications.

## What is new in v18

- v17 physics-based engine retained:
  - velocity-triangle impeller sizing
  - Wiesner / Stodola slip factor
  - inlet, incidence, friction, diffusion and volute loss breakdown
  - off-design performance curves
  - octave-band acoustic estimate
  - shaft, mass and first critical speed checks
  - SWSI and DWDI arrangements
  - AF, BC, BI, radial tip, radial straight, forward curved, Sirocco/cage and plug/plenum families
- Streamlit password gate using `APP_PASSWORD` or `[auth].password` in secrets.
- Corrected angle guide tab showing the exact `beta2` convention used by the code.
- AHRI 431 / AHRI 430 preliminary AHU rating tab.
- Strong warnings that final performance must be validated by AMCA 210 / ANSI/ASHRAE 51 or ISO 5801 testing before catalogue or AHRI publication.
- `ezdxf` added for manufacturing DXF export.

## Main file path

For Streamlit Cloud, set the main file path to either:

```text
app.py
```

or directly:

```text
blower_design.py
```

Using `app.py` is recommended.

## Streamlit secrets

In Streamlit Cloud, add one of these:

```toml
APP_PASSWORD = "your_password"
```

or:

```toml
[auth]
password = "your_password"
```

If no password is configured, the app runs without login.

## Important engineering warning

This app is a preliminary engineering and manufacturing aid. It is not a certified selection program.

Final fan performance, AHU fan rating, sound data and catalogue publication should be based on qualified laboratory testing, typically:

- ANSI/AMCA 210 / ANSI/ASHRAE 51 for airflow performance testing
- ISO 5801 where applicable
- AHRI 430 / AHRI 431 for central-station AHU supply fan rating presentation
- ISO 21940 for balancing

Do not publish the calculated values as certified AMCA/AHRI data until the product is tested and qualified.

## Files

- `app.py` - Streamlit entrypoint wrapper
- `blower_design.py` - complete app and calculation engine
- `requirements.txt` - Python package dependencies
- `.streamlit/secrets.example.toml` - example secrets format only

## Notes for DXF export

DXF export requires `ezdxf`. If the output says to install `ezdxf`, confirm that `requirements.txt` was deployed and reboot Streamlit.

## Suggested validation workflow

1. Use optimiser to generate preliminary geometry.
2. Review angle guide, loss breakdown, curves, acoustic estimate and mechanical checks.
3. Export DXF and review manufacturing details.
4. Build prototype.
5. Test airflow, pressure, power and sound.
6. Use the Calibration tab to tune pressure, power and sound multipliers by fan family.
7. Freeze product line curves only after repeated test confirmation.
