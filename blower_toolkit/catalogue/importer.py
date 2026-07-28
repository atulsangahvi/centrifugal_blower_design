
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd
from pypdf import PdfReader

from .database import MODEL_COLUMNS


MODEL_PATTERNS = [
    r"\b[A-Z]\dE-\dA\d{3}-[A-Z0-9-]{2,}\b",
    r"\b(?:BAF|IN)-[A-Z]?\d{3}-\d{2,3}[A-Z0-9-]*\b",
    r"\bSMT\d{2}C\d{4}E-\d+[A-Z]?\b",
    r"\b(?:LW|RH|DK|GDF|DD|DDF|KDD|KDE)[A-Z0-9-]{5,}\b",
]
MODEL_RE = re.compile("|".join(f"(?:{p})" for p in MODEL_PATTERNS))


@dataclass
class PDFImportResult:
    candidates: pd.DataFrame
    page_count: int
    extracted_pages: int
    warnings: list[str]


def _empty_row():
    return {c: None for c in MODEL_COLUMNS}


def _find_value(text: str, patterns: list[str]):
    for pattern in patterns:
        m = re.search(pattern, text, re.I | re.S)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, TypeError):
                return m.group(1).strip()
    return None


def infer_family(model: str, page_text: str = ""):
    u = f"{model} {page_text[:700]}".upper()
    if "DUAL INLET" in u:
        arrangement = "DWDI / dual inlet"
    elif "SINGLE INLET" in u:
        arrangement = "SWSI / single inlet"
    elif "PLUG" in u or "BACKWARD CURVED" in u:
        arrangement = "Plenum / unhoused"
    else:
        arrangement = None

    if "AXIAL" in u or model.startswith(("A3E", "G3E", "W3E")):
        fan_type = "Axial"
    elif "FORWARD" in u or model.startswith("S3E"):
        fan_type = "Forward curved / Sirocco"
    elif "AIRFOIL" in u or model.startswith("BAF"):
        fan_type = "Backward curved airfoil"
    elif "BACKWARD" in u or model.startswith(("IN-", "SMT")):
        fan_type = "Backward curved / plug fan"
    else:
        fan_type = None

    if re.search(r"\bEC\b", u) or model.startswith(("S3E", "A3E", "G3E", "W3E")):
        motor_type = "EC"
    elif re.search(r"\bDC\b", u):
        motor_type = "DC"
    elif re.search(r"\bAC\b", u):
        motor_type = "AC"
    else:
        motor_type = None

    return fan_type, arrangement, motor_type


def extract_candidates_from_pdf(
    pdf_bytes: bytes,
    filename: str,
    manufacturer: str = "Unknown",
) -> PDFImportResult:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    rows: list[dict] = []
    warnings: list[str] = []
    extracted_pages = 0

    for page_no, page in enumerate(reader.pages, 1):
        raw = page.extract_text() or ""
        text = raw.replace("\x0e", " ").replace("\u00a0", " ")
        if text.strip():
            extracted_pages += 1
        models = list(dict.fromkeys(m.group(0) for m in MODEL_RE.finditer(text)))

        for model in models:
            pos = text.find(model)
            block = text[max(0, pos - 500): pos + 1800]
            fan_type, arrangement, motor_type = infer_family(model, text)
            row = _empty_row()
            row.update(
                manufacturer=manufacturer,
                series=re.sub(r"[-_][^-_]*$", "", model),
                model=model,
                fan_type=fan_type,
                arrangement=arrangement,
                motor_type=motor_type,
                source_file=filename,
                source_page=page_no,
                data_status="Needs review",
            )

            row["voltage_v"] = _find_value(block, [
                r"(?:VAC|Voltage)\s*[:\s]\s*(\d+(?:\.\d+)?)",
                r"\b(115|220|230|380|400|460)\s*V(?:AC)?\b",
            ])
            row["current_a"] = _find_value(block, [
                r"(?:Current)\s*(?:\(A\))?\s*[:\s]\s*(\d+(?:\.\d+)?)",
            ])
            row["input_power_w"] = _find_value(block, [
                r"(?:Input\s*Power|Power)\s*(?:\(W\))?\s*[:\s]\s*(\d+(?:\.\d+)?)",
            ])
            row["rpm"] = _find_value(block, [
                r"(?:Rated\s*Speed|Speed)\s*(?:RPM)?\s*[:\s]\s*(\d+(?:\.\d+)?)",
                r"\b(\d{3,5})\s*RPM\b",
            ])
            row["airflow_m3h"] = _find_value(block, [
                r"(?:Air\s*(?:Volume|Flow)|Airflow)\s*(?:\(m[³3]/h\))?\s*[:\s]\s*(\d+(?:\.\d+)?)",
            ])
            row["static_pressure_pa"] = _find_value(block, [
                r"(?:Static\s*Pressure|Pressure)\s*(?:\(Pa\))?\s*[:\s]\s*(\d+(?:\.\d+)?)",
            ])
            row["noise_dba"] = _find_value(block, [
                r"(?:Noise)\s*(?:\[?dB\(A\)\]?)?\s*[:\s]\s*(\d+(?:\.\d+)?)",
            ])
            row["efficiency_pct"] = _find_value(block, [
                r"(?:Efficiency|η)\s*[:\s]\s*(\d+(?:\.\d+)?)\s*%",
            ])

            material = re.search(
                r"(?:Impeller\s*)?Material\s*[:：]\s*([A-Za-z0-9+ /_-]+)",
                block,
                re.I,
            )
            row["material"] = material.group(1).strip() if material else None
            ip = re.search(r"\bIP\d{2}\b", block, re.I)
            row["ip_rating"] = ip.group(0).upper() if ip else None
            ins = re.search(r"(?:Insulation\s*class)\s*[:：]\s*([A-Z])", block, re.I)
            row["insulation_class"] = ins.group(1).upper() if ins else None
            rows.append(row)

    if extracted_pages < max(1, len(reader.pages) // 3):
        warnings.append(
            "Much of this PDF appears image-based. Model and table extraction will require manual review or later OCR/vision-assisted processing."
        )
    if not rows:
        warnings.append(
            "No supported model-number patterns were found in extracted PDF text."
        )

    return PDFImportResult(
        candidates=pd.DataFrame(rows, columns=MODEL_COLUMNS),
        page_count=len(reader.pages),
        extracted_pages=extracted_pages,
        warnings=warnings,
    )


def curve_template(model: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "model": model,
            "curve_name": "Published P-Q curve",
            "curve_kind": "static_pressure",
            "speed_rpm": None,
            "control_setting": "",
            "airflow_m3h": None,
            "pressure_pa": None,
            "power_w": None,
            "efficiency_pct": None,
            "noise_dba": None,
            "source_file": "",
            "source_page": None,
            "point_status": "Needs review",
            "notes": "",
        }]
    )
