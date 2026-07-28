
import streamlit as st

st.set_page_config(page_title="Development Roadmap", layout="wide")
st.title("Blower Design & Manufacturer Intelligence Roadmap")

st.markdown("""
### Working modules in v21
- Centrifugal blower preliminary design engine
- Modular Streamlit architecture
- SQLite manufacturer knowledge base
- Source-catalogue registry
- PDF/CSV model candidate importer
- Separate performance-curve storage
- Curve interpolation for real duty selection
- Interactive graph calibration and curve digitisation
- Curve quality checks and fan-law scaling
- AHU and cooling-tower fan-wall preliminary selector
- N+1 redundancy, control staging, energy and ROI analysis
- Endpoint benchmarking fallback
- Data-quality and provenance controls
- Seemtek and Longwell starter records

### Next engineering releases
1. **Vision-assisted automatic curve tracing** building on the manual calibrated digitiser.
2. **Drawing dimension capture** into structured dimension records.
3. **Fan-law scaling** with density, speed and diameter corrections.
4. **Fan array / fan wall selection**, redundancy, diversity and control.
5. **AHU retrofit calculator** including old fan, motor, belt and annual energy baseline.
6. **Cooling tower axial fan selection** including airflow distribution and corrosion duty.
7. **Impeller geometry benchmarking** against commercial wheel families.
8. **Manufacturing outputs**: DXF, STEP, BOM, shaft, hub, inlet cone and housing.
9. **Test-data calibration** using AMCA 210 / ISO 5801 laboratory results.
10. **Costing and product-series generator** for your own manufactured range.
""")

st.warning(
    "The application remains a design-development tool. Final commercial ratings "
    "must be validated by laboratory testing and controlled manufacturing drawings."
)
