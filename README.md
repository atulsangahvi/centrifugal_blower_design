# Centrifugal Blower Design & Manufacturing Toolkit

SI-unit Streamlit app for preliminary design of:
- Forward-curved centrifugal blowers
- Backward-curved / backward-inclined centrifugal blowers
- Radial-blade centrifugal blowers

## Features
- Duty inputs: airflow, static pressure, total pressure, temperature, density, RPM
- Blade type selection
- Impeller preliminary sizing: D1, D2, b1, b2, beta angles, blade count
- Slip factor estimate
- Motor power and standard motor selection
- Volute preliminary sizing
- Shaft preliminary sizing
- Fan performance curves
- BOM
- PDF report
- Excel report
- DXF export
- CAD/Ansys workflow notes

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important Engineering Note
This is a preliminary design and manufacturing toolkit skeleton. Final fan designs must be validated using AMCA/ISO test methods, vibration testing, stress checks, CFD/FEA, and prototype testing.

## Next Engineering Upgrades
1. Full CadQuery / FreeCAD STEP solid model generation
2. Separate fluid-domain STEP export for Ansys Fluent
3. Better blade surface generation for forward/backward/radial blades
4. Bellmouth and inlet-box loss models
5. AMCA test-data correction module
6. Noise spectrum prediction
7. Bearing catalogue selection
8. Full fabrication drawings and flat patterns
