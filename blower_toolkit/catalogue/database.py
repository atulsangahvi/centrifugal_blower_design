
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

MODEL_COLUMNS = [
    "manufacturer", "series", "model", "fan_type", "arrangement", "motor_type",
    "impeller_diameter_mm", "wheel_width_mm", "blade_count", "blade_family",
    "airflow_m3h", "static_pressure_pa", "total_pressure_pa", "rpm",
    "input_power_w", "shaft_power_w", "efficiency_pct", "noise_dba",
    "voltage_v", "frequency_hz", "current_a", "material", "ip_rating",
    "insulation_class", "control", "operating_temp", "width_mm", "height_mm",
    "depth_mm", "weight_kg", "source_file", "source_page", "data_status", "notes"
]

DOCUMENT_COLUMNS = [
    "manufacturer", "title", "filename", "document_type", "product_scope",
    "publication_code", "revision", "page_count", "source_status", "notes"
]

CURVE_COLUMNS = [
    "model", "curve_name", "curve_kind", "speed_rpm", "control_setting",
    "airflow_m3h", "pressure_pa", "power_w", "efficiency_pct", "noise_dba",
    "source_file", "source_page", "point_status", "notes"
]


class CatalogueDB:
    """SQLite-backed manufacturer knowledge base with traceability."""

    SCHEMA_VERSION = 2

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self):
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _init(self):
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    manufacturer TEXT NOT NULL,
                    title TEXT NOT NULL,
                    filename TEXT UNIQUE NOT NULL,
                    document_type TEXT,
                    product_scope TEXT,
                    publication_code TEXT,
                    revision TEXT,
                    page_count INTEGER,
                    source_status TEXT DEFAULT 'Registered',
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS fans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    manufacturer TEXT,
                    series TEXT,
                    model TEXT UNIQUE NOT NULL,
                    fan_type TEXT,
                    arrangement TEXT,
                    motor_type TEXT,
                    impeller_diameter_mm REAL,
                    wheel_width_mm REAL,
                    blade_count INTEGER,
                    blade_family TEXT,
                    airflow_m3h REAL,
                    static_pressure_pa REAL,
                    total_pressure_pa REAL,
                    rpm REAL,
                    input_power_w REAL,
                    shaft_power_w REAL,
                    efficiency_pct REAL,
                    noise_dba REAL,
                    voltage_v REAL,
                    frequency_hz TEXT,
                    current_a REAL,
                    material TEXT,
                    ip_rating TEXT,
                    insulation_class TEXT,
                    control TEXT,
                    operating_temp TEXT,
                    width_mm REAL,
                    height_mm REAL,
                    depth_mm REAL,
                    weight_kg REAL,
                    source_file TEXT,
                    source_page INTEGER,
                    data_status TEXT DEFAULT 'Needs review',
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS curve_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    curve_name TEXT DEFAULT 'Published P-Q curve',
                    curve_kind TEXT DEFAULT 'static_pressure',
                    speed_rpm REAL,
                    control_setting TEXT,
                    airflow_m3h REAL NOT NULL,
                    pressure_pa REAL NOT NULL,
                    power_w REAL,
                    efficiency_pct REAL,
                    noise_dba REAL,
                    source_file TEXT,
                    source_page INTEGER,
                    point_status TEXT DEFAULT 'Needs review',
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(model) REFERENCES fans(model) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS dimensions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    dimension_code TEXT NOT NULL,
                    value_mm REAL,
                    tolerance_plus_mm REAL,
                    tolerance_minus_mm REAL,
                    description TEXT,
                    source_file TEXT,
                    source_page INTEGER,
                    data_status TEXT DEFAULT 'Needs review',
                    UNIQUE(model, dimension_code),
                    FOREIGN KEY(model) REFERENCES fans(model) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_fans_maker_type
                    ON fans(manufacturer, fan_type);
                CREATE INDEX IF NOT EXISTS idx_curve_model
                    ON curve_points(model, speed_rpm, airflow_m3h);
                """
            )
            con.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

    @staticmethod
    def _clean(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        return value

    def upsert_dataframe(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        count = 0
        with self.connect() as con:
            for _, row in df.iterrows():
                rec = {c: self._clean(row.get(c)) for c in MODEL_COLUMNS}
                if not rec["model"]:
                    continue
                cols = list(rec)
                vals = [rec[c] for c in cols]
                q = ",".join("?" for _ in cols)
                updates = ",".join(
                    f"{c}=excluded.{c}" for c in cols if c != "model"
                )
                con.execute(
                    f"""
                    INSERT INTO fans ({','.join(cols)}) VALUES ({q})
                    ON CONFLICT(model) DO UPDATE SET
                    {updates}, updated_at=CURRENT_TIMESTAMP
                    """,
                    vals,
                )
                count += 1
        return count

    def register_documents(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        count = 0
        with self.connect() as con:
            for _, row in df.iterrows():
                rec = {c: self._clean(row.get(c)) for c in DOCUMENT_COLUMNS}
                if not rec["filename"] or not rec["title"] or not rec["manufacturer"]:
                    continue
                cols = list(rec)
                vals = [rec[c] for c in cols]
                q = ",".join("?" for _ in cols)
                updates = ",".join(
                    f"{c}=excluded.{c}" for c in cols if c != "filename"
                )
                con.execute(
                    f"""
                    INSERT INTO documents ({','.join(cols)}) VALUES ({q})
                    ON CONFLICT(filename) DO UPDATE SET
                    {updates}, updated_at=CURRENT_TIMESTAMP
                    """,
                    vals,
                )
                count += 1
        return count

    def add_curve_points(self, df: pd.DataFrame, replace_curve: bool = False) -> int:
        if df is None or df.empty:
            return 0
        clean = df.copy()
        for c in CURVE_COLUMNS:
            if c not in clean:
                clean[c] = None
        clean = clean[CURVE_COLUMNS]
        clean = clean.dropna(subset=["model", "airflow_m3h", "pressure_pa"])
        if clean.empty:
            return 0

        with self.connect() as con:
            if replace_curve:
                keys = clean[["model", "curve_name", "speed_rpm"]].drop_duplicates()
                for _, k in keys.iterrows():
                    if pd.isna(k["speed_rpm"]):
                        con.execute(
                            "DELETE FROM curve_points WHERE model=? AND curve_name=? AND speed_rpm IS NULL",
                            (k["model"], k["curve_name"]),
                        )
                    else:
                        con.execute(
                            "DELETE FROM curve_points WHERE model=? AND curve_name=? AND speed_rpm=?",
                            (k["model"], k["curve_name"], float(k["speed_rpm"])),
                        )
            rows = []
            for _, row in clean.iterrows():
                rows.append(tuple(self._clean(row[c]) for c in CURVE_COLUMNS))
            con.executemany(
                f"INSERT INTO curve_points ({','.join(CURVE_COLUMNS)}) VALUES ({','.join('?' for _ in CURVE_COLUMNS)})",
                rows,
            )
        return len(clean)

    def add_dimensions(self, df: pd.DataFrame) -> int:
        required = [
            "model", "dimension_code", "value_mm", "tolerance_plus_mm",
            "tolerance_minus_mm", "description", "source_file", "source_page",
            "data_status"
        ]
        if df is None or df.empty:
            return 0
        x = df.copy()
        for c in required:
            if c not in x:
                x[c] = None
        x = x.dropna(subset=["model", "dimension_code"])
        with self.connect() as con:
            for _, r in x.iterrows():
                vals = [self._clean(r[c]) for c in required]
                con.execute(
                    f"""
                    INSERT INTO dimensions ({','.join(required)})
                    VALUES ({','.join('?' for _ in required)})
                    ON CONFLICT(model, dimension_code) DO UPDATE SET
                    value_mm=excluded.value_mm,
                    tolerance_plus_mm=excluded.tolerance_plus_mm,
                    tolerance_minus_mm=excluded.tolerance_minus_mm,
                    description=excluded.description,
                    source_file=excluded.source_file,
                    source_page=excluded.source_page,
                    data_status=excluded.data_status
                    """,
                    vals,
                )
        return len(x)

    def all(self) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(
                "SELECT * FROM fans ORDER BY manufacturer, series, model", con
            )

    def documents(self) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(
                "SELECT * FROM documents ORDER BY manufacturer, title", con
            )

    def curves(self, model: str | None = None) -> pd.DataFrame:
        with self.connect() as con:
            if model:
                return pd.read_sql_query(
                    """
                    SELECT * FROM curve_points
                    WHERE model=?
                    ORDER BY curve_name, speed_rpm, airflow_m3h
                    """,
                    con,
                    params=(model,),
                )
            return pd.read_sql_query(
                "SELECT * FROM curve_points ORDER BY model, curve_name, speed_rpm, airflow_m3h",
                con,
            )

    def dimensions(self, model: str | None = None) -> pd.DataFrame:
        with self.connect() as con:
            if model:
                return pd.read_sql_query(
                    "SELECT * FROM dimensions WHERE model=? ORDER BY dimension_code",
                    con,
                    params=(model,),
                )
            return pd.read_sql_query(
                "SELECT * FROM dimensions ORDER BY model, dimension_code", con
            )

    def delete_models(self, ids: Iterable[int]):
        ids = [int(x) for x in ids]
        if not ids:
            return
        with self.connect() as con:
            con.executemany("DELETE FROM fans WHERE id=?", [(x,) for x in ids])

    def quality_summary(self) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(
                """
                SELECT manufacturer,
                       COUNT(*) AS models,
                       SUM(CASE WHEN data_status='Verified from catalogue' THEN 1 ELSE 0 END) AS verified_models,
                       SUM(CASE WHEN airflow_m3h IS NOT NULL AND static_pressure_pa IS NOT NULL THEN 1 ELSE 0 END) AS duty_ratings,
                       SUM(CASE WHEN efficiency_pct IS NOT NULL THEN 1 ELSE 0 END) AS efficiency_records,
                       SUM(CASE WHEN noise_dba IS NOT NULL THEN 1 ELSE 0 END) AS noise_records
                FROM fans
                GROUP BY manufacturer
                ORDER BY manufacturer
                """,
                con,
            )

    def export_csv(self) -> bytes:
        return self.all().to_csv(index=False).encode("utf-8")

    def export_curve_csv(self) -> bytes:
        return self.curves().to_csv(index=False).encode("utf-8")
