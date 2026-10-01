from __future__ import annotations
from pathlib import Path
from blower_toolkit.engine import FAMILIES

def family_for_record(record):
    t=f"{record.get('fan_type','')} {record.get('series','')} {record.get('model','')}".lower()
    if 'axial' in t or 'inlet ring' in t:
        return None
    if 'airfoil' in t:
        return 'Airfoil Backward (AF)'
    if 'plug' in t:
        return 'Plug / Plenum (BC, no volute)'
    if 'backward' in t:
        return 'Backward Curved (BC)'
    if 'forward' in t or 'sirocco' in t:
        return 'Multi-Blade Sirocco (Cage)'
    if 'radial' in t:
        return 'Radial Straight (Paddle)'
    return None

def arrangement_for_record(record):
    a=str(record.get('arrangement') or '').lower()
    return 'DWDI (double inlet)' if ('dual' in a or 'dwdi' in a) else 'SWSI (single inlet)'

def manufacturing_handoff(record):
    fam_name=family_for_record(record)
    if fam_name is None:
        return dict(
            supported=False,
            source='Manufacturer catalogue',
            manufacturer=record.get('manufacturer'),
            model=record.get('model'),
            source_file=record.get('source_file'),
            source_page=record.get('source_page'),
            notes='Current centrifugal inverse-design engine does not support this product type.'
        )
    fam=FAMILIES[fam_name]
    d2=float(record.get('impeller_diameter_mm') or 0) or 400.0
    return dict(
        supported=True,
        source='Manufacturer catalogue',
        manufacturer=record.get('manufacturer'),
        model=record.get('model'),
        source_file=record.get('source_file'),
        source_page=record.get('source_page'),
        family=fam_name,
        arrangement=arrangement_for_record(record),
        known_d2_mm=d2,
        known_airflow_m3h=float(record.get('airflow_m3h') or 0),
        known_static_pressure_pa=float(record.get('static_pressure_pa') or 0),
        known_rpm=float(record.get('rpm') or 0),
        known_power_w=float(record.get('input_power_w') or 0),
        known_noise_dba=float(record.get('noise_dba') or 0),
        known_material=record.get('material'),
        calculated_d1_mm=float(fam['d1d2'])*d2,
        calculated_b2_mm=float(fam['b2d2'])*d2,
        calculated_b1_mm=1.05*float(fam['b2d2'])*d2,
        calculated_blade_count=int(fam['z']),
        calculated_beta1_deg=float(fam['beta1']),
        calculated_beta2_deg=float(fam['beta2']),
        calculated_cutoff_mm=0.05*d2,
        notes='Catalogue values preserved; missing internal geometry seeded from app family ranges and should be inverse-optimized/verified.'
    )

def embedded_pdf_path(root,source_file):
    p=Path(root)/'assets'/'catalogues'/str(source_file); return p if p.exists() else None

def render_pdf_page_png(pdf_path,page_number,zoom=1.2):
    import fitz
    doc=fitz.open(pdf_path)
    try:
        i=max(0,min(int(page_number)-1,len(doc)-1)); pix=doc.load_page(i).get_pixmap(matrix=fitz.Matrix(zoom,zoom),alpha=False); return pix.tobytes('png')
    finally: doc.close()

def pdf_page_text(pdf_path,page_number):
    import fitz
    doc=fitz.open(pdf_path)
    try:
        i=max(0,min(int(page_number)-1,len(doc)-1)); return doc.load_page(i).get_text('text')
    finally: doc.close()
