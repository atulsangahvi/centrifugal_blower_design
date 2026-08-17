# Reverse-engineering reference notes

## Uploaded KQ800 / WDL-800 drawings

The v25 preset values are transcribed from the two drawings supplied by the user. The 2026 KQ800 V4.0 drawing states JB/T 9068-2017 and dynamic balance grade G4.0. It shows a 60 mm shaft end, while the 2020 WDL-800 V1.0 drawing shows approximately 55 mm.

The model designation `800` is **not** treated as a verified impeller outside diameter. D2 remains a hypothesis until physically measured or confirmed by a controlled detail drawing.

## Independent fan-performance sanity scale

The optional benchmark normalization uses one published Kruger FDA500 selection example solely to check order of magnitude. The official catalogue identifies the FDA series as DIDW forward-curved fans and states that fan size is impeller diameter in mm. The example lists 20,000 m3/h, 737 Pa total pressure, 828 rpm, 6.5 kW absorbed shaft power, and 62% total efficiency at standard density.

This benchmark is not used as a claim about WDL/KQ performance. Test-calibration of the actual fan supersedes benchmark normalization.

## Release rule

No value tagged `LOW` confidence should be used for production release. Critical aerodynamic geometry (D2, D1, wheel width, blade count, beta1, beta2, blade thickness, tongue clearance and scroll width) must be measured or verified. Final performance requires a controlled test method such as AMCA 210 / ISO 5801.
