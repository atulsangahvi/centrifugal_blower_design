# Centrifugal Blower Design & Manufacturing Toolkit

Run:
```bash
pip install streamlit numpy pandas matplotlib openpyxl reportlab ezdxf
streamlit run app.py
```

Optional STEP support:
```bash
pip install cadquery
```

## Password on Streamlit Cloud
Add in **App settings → Secrets**:
```toml
APP_PASSWORD = "your_password_here"
```
If `APP_PASSWORD` is not defined, the app remains open for local testing.

## Notes
The app uses SI units only. User enters **static pressure only**; the app calculates discharge velocity pressure and total pressure.
