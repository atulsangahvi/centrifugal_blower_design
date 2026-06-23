"""
Centrifugal Blower Design & Manufacturing Toolkit - SI Units
Forward Curved, Backward Curved / Backward Inclined, and Radial Blade Blowers

Preliminary engineering tool for sizing, learning, manufacturing sketches, and prototype planning.
Final product designs must be validated by AMCA/ISO fan testing, vibration balancing, CFD/FEA,
and qualified engineering review before manufacture.
"""
from __future__ import annotations

import io, math, zipfile
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import ezdxf
    HAS_EZDXF = True
except Exception:
    HAS_EZDXF = False

try:
    import cadquery as cq
    HAS_CADQUERY = True
except Exception:
    HAS_CADQUERY = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# ----------------------------- Password -----------------------------
def check_password() -> bool:
    """Password gate. Add APP_PASSWORD = "your_password" in Streamlit secrets.
    For local testing, app remains open if no APP_PASSWORD is defined.
    """
    try:
        required = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        required = ""
    if not required:
        return True
    if st.session_state.get("password_ok"):
        return True
    st.title("Centrifugal Blower Design Toolkit")
    pw = st.text_input("Enter app password", type="password")
    if st.button("Login"):
        if pw == required:
            st.session_state["password_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

# ----------------------------- Constants -----------------------------
STANDARD_MOTORS_KW = [0.18,0.25,0.37,0.55,0.75,1.1,1.5,2.2,3.0,4.0,5.5,7.5,11,15,18.5,22,30,37,45,55,75,90,110,132,160,200,250,315]

BLADE_DEFAULTS = {
    "Backward Curved / Backward Inclined": {"beta1": 28.0, "beta2": 38.0, "eta_total": 0.72, "phi": 0.18, "psi": 0.58, "blade_count": 12, "b2_d2": 0.14, "d1_d2": 0.55, "power_curve":"non_overloading"},
    "Forward Curved": {"beta1": 38.0, "beta2": 125.0, "eta_total": 0.58, "phi": 0.16, "psi": 0.72, "blade_count": 36, "b2_d2": 0.18, "d1_d2": 0.58, "power_curve":"overloading"},
    "Radial Blade": {"beta1": 30.0, "beta2": 90.0, "eta_total": 0.62, "phi": 0.12, "psi": 0.62, "blade_count": 8, "b2_d2": 0.12, "d1_d2": 0.50, "power_curve":"linear"},
}

MATERIALS = {
    "Mild Steel IS2062 / S275": {"density":7850, "allow_stress_mpa":90, "max_tip_ms":90},
    "Galvanized Steel": {"density":7850, "allow_stress_mpa":80, "max_tip_ms":75},
    "Stainless Steel 304": {"density":8000, "allow_stress_mpa":95, "max_tip_ms":90},
    "Aluminium 6061-T6": {"density":2700, "allow_stress_mpa":70, "max_tip_ms":70},
}

PRACTICAL_RANGES = {
    "Backward Curved / Backward Inclined": {"beta1":(20,45), "beta2":(25,55), "z":(8,18), "b2d2":(0.06,0.25), "d1d2":(0.45,0.70), "pitch_chord":(1.0,2.2), "tip":(30,85), "outlet_v":(8,20)},
    "Forward Curved": {"beta1":(25,55), "beta2":(100,150), "z":(24,64), "b2d2":(0.10,0.35), "d1d2":(0.45,0.75), "pitch_chord":(0.45,1.2), "tip":(18,60), "outlet_v":(7,16)},
    "Radial Blade": {"beta1":(20,55), "beta2":(80,100), "z":(6,12), "b2d2":(0.06,0.22), "d1d2":(0.40,0.65), "pitch_chord":(1.2,3.0), "tip":(30,85), "outlet_v":(10,24)},
}

# ----------------------------- Dataclasses -----------------------------
@dataclass
class DutyInput:
    airflow_m3h: float
    static_pressure_pa: float
    air_temp_c: float
    altitude_m: float
    density_kgm3: float
    rpm: float
    blade_type: str
    drive_type: str
    motor_eff: float
    drive_eff: float
    design_margin: float
    beta2_deg: float
    beta1_deg: float
    blade_count: int
    outlet_width_ratio: float
    inlet_diameter_ratio: float
    material: str
    blade_thickness_mm: float
    casing_thickness_mm: float
    shaft_allow_shear_mpa: float
    outlet_aspect_ratio: float
    unbalance_grade_mm_s: float
    isolation_quality: str

@dataclass
class DesignResult:
    q_m3s: float; density_kgm3: float; total_pressure_pa: float; static_pressure_pa: float; velocity_pressure_pa: float
    rpm: float; omega: float; tip_speed_ms: float
    impeller_od_mm: float; impeller_id_mm: float; outlet_width_mm: float; inlet_width_mm: float
    beta1_deg: float; beta2_deg: float; blade_count: int; blade_pitch_mm: float; blade_chord_mm: float; pitch_chord_ratio: float
    slip_factor: float; flow_coeff_phi: float; pressure_coeff_psi: float; theoretical_pressure_pa: float; estimated_total_eff: float
    air_power_kw: float; shaft_power_kw: float; motor_input_kw: float; selected_motor_kw: float
    specific_speed_metric: float; specific_diameter_metric: float
    impeller_outlet_area_m2: float; impeller_outlet_velocity_ms: float
    flange_area_m2: float; flange_width_mm: float; flange_height_mm: float; flange_velocity_ms: float
    velocity_pressure_at_flange_pa: float; volute_cutoff_clearance_mm: float; shaft_diameter_mm: float; approx_impeller_mass_kg: float
    estimated_sound_power_dba: float; estimated_sound_pressure_1m_dba: float; vibration_risk_score: float; vibration_risk_label: str
    warnings: List[str]; recommendations: List[str]

# ----------------------------- Calculations -----------------------------
def standard_air_density(temp_c: float, altitude_m: float) -> float:
    t0=288.15; p0=101325.0; lapse=0.0065; r=287.05; g=9.80665
    temp_k_alt = t0 - lapse * altitude_m
    pressure = p0 * (temp_k_alt/t0) ** (g/(r*lapse))
    return pressure / (r*(temp_c+273.15))

def selected_motor(power_kw: float) -> float:
    return next((m for m in STANDARD_MOTORS_KW if m >= power_kw), STANDARD_MOTORS_KW[-1])

def stodola_slip_factor(z:int, beta2_deg:float, d1_d2:float)->float:
    beta=math.radians(beta2_deg)
    sigma = 1.0 - (math.pi*math.sin(beta))/max(z,3)
    sigma *= max(0.88, min(1.02, 1.0 - 0.10*(d1_d2-0.5)))
    return max(0.55, min(0.95, sigma))

def estimate_total_pressure_from_static(q:float, sp:float, rho:float, flange_area_guess:float|None=None)->Tuple[float,float,float]:
    # First guess discharge velocity from static pressure class; then iterate once after geometry.
    v = 8.0 if sp < 700 else 11.0 if sp < 1400 else 14.0 if sp < 2500 else 18.0
    if flange_area_guess and flange_area_guess > 0:
        v = q / flange_area_guess
    vp = 0.5 * rho * v*v
    return sp + vp, vp, v

def noise_estimate(res_base, inp: DutyInput) -> Tuple[float,float]:
    # Preliminary empirical screening only, not AMCA sound rating.
    q = max(res_base.q_m3s, 0.01); sp=max(res_base.static_pressure_pa,1)
    swl = 45 + 10*math.log10(q) + 20*math.log10(sp/100.0)
    if inp.blade_type == "Forward Curved": swl += 3
    if inp.blade_type == "Radial Blade": swl += 6
    if res_base.tip_speed_ms > 60: swl += 3
    if res_base.flange_velocity_ms > 18: swl += 4
    if res_base.pitch_chord_ratio < PRACTICAL_RANGES[inp.blade_type]["pitch_chord"][0]: swl += 2
    spl_1m = swl - 8  # very rough free-field + casing effect assumption
    return swl, spl_1m

def vibration_risk(res_base, inp:DutyInput)->Tuple[float,str,List[str]]:
    score=0.0; notes=[]; rng=PRACTICAL_RANGES[inp.blade_type]
    if res_base.tip_speed_ms > rng["tip"][1]: score+=25; notes.append("High tip speed raises balance and stress sensitivity.")
    if res_base.flange_velocity_ms > rng["outlet_v"][1]: score+=15; notes.append("High outlet velocity can create turbulence and duct vibration.")
    if res_base.pitch_chord_ratio < rng["pitch_chord"][0]: score+=12; notes.append("Blade pitch is tight; blade-passing tone and blockage risk increase.")
    if res_base.pitch_chord_ratio > rng["pitch_chord"][1]: score+=8; notes.append("Blade pitch is wide; pulsation and non-uniform flow risk increase.")
    if inp.blade_type == "Forward Curved": score+=8; notes.append("Forward-curved wheels need careful operation away from unstable/overload region.")
    if inp.blade_type == "Radial Blade": score+=10; notes.append("Radial blades are robust but usually noisier and more pulsating.")
    score += min(25, inp.unbalance_grade_mm_s*2.0)
    if inp.isolation_quality == "Poor / rigid mounting": score += 15
    elif inp.isolation_quality == "Average": score += 7
    label = "Low" if score < 25 else "Medium" if score < 55 else "High"
    return score, label, notes

def design_blower(inp:DutyInput)->DesignResult:
    warnings=[]; rec=[]
    q=inp.airflow_m3h/3600.0
    rho=inp.density_kgm3 if inp.density_kgm3>0 else standard_air_density(inp.air_temp_c, inp.altitude_m)
    defaults=BLADE_DEFAULTS[inp.blade_type]
    phi=defaults["phi"]; psi=defaults["psi"]; eta=defaults["eta_total"]
    # static pressure only. Initial total pressure estimate from assumed outlet velocity.
    total_p, vp_initial, _ = estimate_total_pressure_from_static(q, inp.static_pressure_pa, rho)
    omega=2*math.pi*inp.rpm/60.0
    u2=math.sqrt(total_p/max(psi*rho,1e-9))
    d2=2*u2/max(omega,1e-9)
    d1=inp.inlet_diameter_ratio*d2
    b2=inp.outlet_width_ratio*d2
    outlet_area=math.pi*d2*b2
    cm2=q/max(outlet_area,1e-9)
    cm2_target=phi*u2
    if cm2 > 1.35*cm2_target:
        b2=q/(math.pi*d2*cm2_target)
        warnings.append("Outlet width increased automatically because selected b₂/D₂ gave excessive impeller meridional velocity.")
    elif cm2 < 0.55*cm2_target:
        rec.append("Selected b₂/D₂ is wide for this duty; consider smaller b₂/D₂, lower RPM, or a different fan size after prototype curve check.")
    outlet_area=math.pi*d2*b2
    impeller_outlet_v=q/max(outlet_area,1e-9)
    # outlet flange dimensions and real flange velocity pressure.
    target_flange_v = 8.0 if inp.static_pressure_pa < 700 else 11.0 if inp.static_pressure_pa < 1400 else 14.0 if inp.static_pressure_pa < 2500 else 18.0
    flange_area=max(q/target_flange_v, 1e-6)
    flange_h=math.sqrt(flange_area/max(inp.outlet_aspect_ratio,0.1))
    flange_w=inp.outlet_aspect_ratio*flange_h
    flange_v=q/flange_area
    vp_flange=0.5*rho*flange_v*flange_v
    total_p=inp.static_pressure_pa+vp_flange
    # recalc once with better total pressure
    u2=math.sqrt(total_p/max(psi*rho,1e-9)); d2=2*u2/max(omega,1e-9); d1=inp.inlet_diameter_ratio*d2; b2=max(inp.outlet_width_ratio*d2, b2)
    outlet_area=math.pi*d2*b2; impeller_outlet_v=q/max(outlet_area,1e-9)
    b1=1.15*b2
    sigma=stodola_slip_factor(inp.blade_count, inp.beta2_deg, d1/d2)
    theoretical_p=total_p/max(sigma,1e-6)
    air_power=q*total_p/1000.0
    shaft_power=air_power/max(eta*inp.drive_eff,1e-6)
    motor_input=shaft_power/max(inp.motor_eff,1e-6)
    motor_kw=selected_motor(motor_input*(1+inp.design_margin/100))
    h_m=total_p/(rho*9.80665); n_rps=inp.rpm/60.0
    ns=n_rps*math.sqrt(q)/max(h_m**0.75,1e-9); ds=d2*max(h_m**0.25,1e-9)/max(math.sqrt(q),1e-9)
    cutoff=0.06*d2
    torque=(shaft_power*1000)/max(omega,1e-9); tau=inp.shaft_allow_shear_mpa*1e6
    shaft_d=((16*torque)/(math.pi*tau))**(1/3)*1000*1.35
    shaft_d=max(20, math.ceil(shaft_d/5)*5)
    mat=MATERIALS[inp.material]
    t=inp.blade_thickness_mm/1000.0; disc_t=max(t,0.003)
    disc_area=math.pi*(d2*d2-(0.35*d1)**2)/4
    blade_chord=max((d2-d1)*0.62, 0.001)
    blade_area=blade_chord*b2*1.25
    mass=mat["density"]*(2*disc_area*disc_t+inp.blade_count*blade_area*t)*1.25
    pitch=math.pi*d2/max(inp.blade_count,1)
    pitch_chord=pitch/max(blade_chord,1e-9)

    # Practical checks
    rng=PRACTICAL_RANGES[inp.blade_type]
    if u2 > mat["max_tip_ms"]: warnings.append(f"Tip speed {u2:.1f} m/s exceeds preliminary material limit {mat['max_tip_ms']} m/s.")
    if not (rng["beta1"][0] <= inp.beta1_deg <= rng["beta1"][1]): warnings.append("β₁ is outside the usual practical range for this blade family.")
    if not (rng["beta2"][0] <= inp.beta2_deg <= rng["beta2"][1]): warnings.append("β₂ is outside the usual practical range for this blade family.")
    if not (rng["z"][0] <= inp.blade_count <= rng["z"][1]): warnings.append("Blade count is outside the usual practical range for this blade family.")
    if not (rng["b2d2"][0] <= b2/d2 <= rng["b2d2"][1]): warnings.append("Calculated b₂/D₂ is outside practical preliminary range; consider different RPM, fan size, or multiple fans.")
    if not (rng["d1d2"][0] <= d1/d2 <= rng["d1d2"][1]): warnings.append("D₁/D₂ is outside practical preliminary range; inlet eye may be too small or too large.")
    if not (rng["pitch_chord"][0] <= pitch_chord <= rng["pitch_chord"][1]): warnings.append("Blade pitch/chord ratio is outside practical range; check blockage, tone, and efficiency.")
    if flange_v > rng["outlet_v"][1]: rec.append("Discharge flange velocity is high; increase outlet flange area or add diffuser to reduce noise and pressure loss.")
    if inp.blade_type == "Forward Curved" and inp.static_pressure_pa > 1200: rec.append("For this pressure, compare with backward-curved fan; forward-curved may become wide and overload-sensitive.")
    if b2/d2 > 0.45: rec.append("Wheel is very wide. Consider DIDW construction, two fans in parallel, or lower RPM/larger diameter.")

    dummy = type('X', (), {})()
    for k,v in {"q_m3s":q,"static_pressure_pa":inp.static_pressure_pa,"tip_speed_ms":u2,"flange_velocity_ms":flange_v,"pitch_chord_ratio":pitch_chord}.items(): setattr(dummy,k,v)
    swl,spl=noise_estimate(dummy, inp)
    vib_score,vib_label,vib_notes=vibration_risk(dummy, inp); rec += vib_notes

    return DesignResult(q,rho,total_p,inp.static_pressure_pa,vp_flange,inp.rpm,omega,u2,d2*1000,d1*1000,b2*1000,b1*1000,inp.beta1_deg,inp.beta2_deg,inp.blade_count,pitch*1000,blade_chord*1000,pitch_chord,sigma,phi,psi,theoretical_p,eta,air_power,shaft_power,motor_input,motor_kw,ns,ds,outlet_area,impeller_outlet_v,flange_area,flange_w*1000,flange_h*1000,flange_v,vp_flange,cutoff*1000,shaft_d,mass,swl,spl,vib_score,vib_label,warnings,rec)

# ----------------------------- Geometry -----------------------------
def blade_centerline_points(res:DesignResult,n:int=80)->List[Tuple[float,float]]:
    r1=res.impeller_id_mm/2; r2=res.impeller_od_mm/2
    beta1=math.radians(res.beta1_deg); beta2=math.radians(res.beta2_deg)
    rs=np.linspace(r1,r2,n); theta=np.zeros_like(rs)
    sign=-1 if res.beta2_deg < 90 else 1
    for i in range(1,n):
        rmid=0.5*(rs[i]+rs[i-1]); frac=(rmid-r1)/max(r2-r1,1e-9); beta=beta1+(beta2-beta1)*frac
        dr=rs[i]-rs[i-1]; theta[i]=theta[i-1]+sign*(1/math.tan(max(0.08,min(math.pi-0.08,beta))))*dr/max(rmid,1e-9)
    return [(float(r*math.cos(th)),float(r*math.sin(th))) for r,th in zip(rs,theta)]

def blade_outline_points(res:DesignResult)->np.ndarray:
    c=np.array(blade_centerline_points(res,80))
    # thickness shown exaggerated enough to see: actual blade thickness is not encoded in res, use chord scaling for visual
    th=max(3.0, res.blade_chord_mm*0.035)
    tang=np.gradient(c,axis=0); norm=np.column_stack([-tang[:,1],tang[:,0]])
    nrm=np.linalg.norm(norm,axis=1); norm=norm/np.maximum(nrm[:,None],1e-9)
    return np.vstack([c+0.5*th*norm, (c-0.5*th*norm)[::-1], c[:1]+0.5*th*norm[:1]])

def volute_spiral_points(res:DesignResult,n:int=140)->List[Tuple[float,float]]:
    r_imp=res.impeller_od_mm/2; c=res.volute_cutoff_clearance_mm; width=max(res.flange_width_mm,1)
    outlet_area_mm2=res.flange_width_mm*res.flange_height_mm
    pts=[]
    for th in np.linspace(math.radians(10),math.radians(360),n):
        area=outlet_area_mm2*(th/(2*math.pi)); radial_growth=area/max(width,1)
        r=r_imp+c+radial_growth
        pts.append((float(r*math.cos(th)),float(r*math.sin(th))))
    return pts

def practicality_table(inp:DutyInput,res:DesignResult)->pd.DataFrame:
    rng=PRACTICAL_RANGES[inp.blade_type]
    rows=[
        ("β₁ inlet blade angle",res.beta1_deg,rng["beta1"],"Controls entry shock. Too low/high causes inlet separation and noise."),
        ("β₂ outlet blade angle",res.beta2_deg,rng["beta2"],"Controls pressure curve and power behavior. <90° backward; 90° radial; >90° forward."),
        ("Blade count z",res.blade_count,rng["z"],"Too few gives pulsation; too many gives blockage and close pitch."),
        ("D₁/D₂ inlet diameter ratio",res.impeller_id_mm/res.impeller_od_mm,rng["d1d2"],"Eye size compared with wheel OD. Too small chokes inlet; too large weakens pressure rise."),
        ("b₂/D₂ outlet width ratio",res.outlet_width_mm/res.impeller_od_mm,rng["b2d2"],"Wheel width. Very high ratio means physically wide fan or need DIDW/parallel fans."),
        ("Pitch/chord ratio",res.pitch_chord_ratio,rng["pitch_chord"],"Spacing between blades relative to blade length. Too close = blockage/tone; too open = pulsation."),
        ("Tip speed",res.tip_speed_ms,rng["tip"],"Higher tip speed gives pressure but increases stress, noise and balancing demand."),
        ("Flange outlet velocity",res.flange_velocity_ms,rng["outlet_v"],"High discharge velocity increases velocity pressure, duct loss and noise."),
    ]
    out=[]
    for name,val,rr,meaning in rows:
        lo,hi=rr; ok=lo<=val<=hi
        out.append({"Input / Check":name,"Your value":round(val,3),"Practical guide":f"{lo} to {hi}","Status":"OK" if ok else "Review", "Physical meaning / recommendation":meaning})
    return pd.DataFrame(out)

def create_dxf(res:DesignResult)->bytes:
    if not HAS_EZDXF: return b"Install ezdxf to generate DXF files."
    doc=ezdxf.new("R2010"); msp=doc.modelspace(); r2=res.impeller_od_mm/2; r1=res.impeller_id_mm/2; hub=max(res.shaft_diameter_mm*1.6,r1*0.25)
    for r,layer in [(r2,"Impeller_OD_D2"),(r1,"Inlet_Eye_D1"),(hub,"Hub")]: msp.add_circle((0,0),r,dxfattribs={"layer":layer})
    blade=blade_centerline_points(res)
    for k in range(res.blade_count):
        a=2*math.pi*k/res.blade_count; ca,sa=math.cos(a),math.sin(a); pts=[(x*ca-y*sa,x*sa+y*ca) for x,y in blade]
        msp.add_lwpolyline(pts,dxfattribs={"layer":"Blade_Centerlines"})
    vol=volute_spiral_points(res); msp.add_lwpolyline(vol,dxfattribs={"layer":"Volute_Spiral"})
    # outlet flange rectangle
    x0=r2+res.volute_cutoff_clearance_mm; y0=r2*0.55; W=res.flange_width_mm; H=res.flange_height_mm
    msp.add_lwpolyline([(x0,y0),(x0+H,y0),(x0+H,y0+W),(x0,y0+W),(x0,y0)],dxfattribs={"layer":"Outlet_Flange_WxH"})
    stream=io.StringIO(); doc.write(stream); return stream.getvalue().encode()

def create_step_placeholder(res:DesignResult)->bytes:
    if not HAS_CADQUERY:
        return b"STEP export needs cadquery installed locally/server. Run: pip install cadquery. Export separate impeller_solid.step, volute_solid.step, fluid_domain.step for Ansys."
    model=cq.Workplane("XY").circle(res.impeller_od_mm/2).circle(res.impeller_id_mm/2).extrude(res.outlet_width_mm)
    bio=io.BytesIO(); cq.exporters.export(model,bio,exportType="STEP"); return bio.getvalue()

# ----------------------------- Reports -----------------------------
def performance_curve(res:DesignResult)->pd.DataFrame:
    flows=np.linspace(0.35,1.25,25)*res.q_m3s; rows=[]
    for q in flows:
        x=q/res.q_m3s
        pressure=res.total_pressure_pa*max(0.05,1.18-0.18*x-0.18*x*x)
        eff=res.estimated_total_eff*max(0.15,1.0-1.8*(x-1.0)**2)
        power=q*pressure/1000/max(eff,0.05)
        rows.append({"Flow_m3s":q,"Flow_m3h":q*3600,"TotalPressure_Pa":pressure,"StaticPressure_approx_Pa":max(0,pressure-res.velocity_pressure_at_flange_pa),"Efficiency":eff,"ShaftPower_kW":power})
    return pd.DataFrame(rows)

def bom_table(inp:DutyInput,res:DesignResult)->pd.DataFrame:
    return pd.DataFrame([
        {"Item":"Impeller back plate","Material":inp.material,"Thickness mm":inp.blade_thickness_mm,"Approx Qty":"1"},
        {"Item":"Impeller shroud/front ring","Material":inp.material,"Thickness mm":inp.blade_thickness_mm,"Approx Qty":"1"},
        {"Item":"Blades","Material":inp.material,"Thickness mm":inp.blade_thickness_mm,"Approx Qty":res.blade_count},
        {"Item":"Volute casing","Material":inp.material,"Thickness mm":inp.casing_thickness_mm,"Approx Qty":"1 set"},
        {"Item":"Outlet flange","Material":inp.material,"Thickness mm":inp.casing_thickness_mm,"Approx Qty":f"{res.flange_width_mm:.0f} x {res.flange_height_mm:.0f} mm"},
        {"Item":"Shaft","Material":"EN8/C45 or equivalent","Thickness mm":"-","Approx Qty":f"Ø{res.shaft_diameter_mm:.0f} mm preliminary"},
        {"Item":"Motor","Material":"IE3/IE4 TEFC","Thickness mm":"-","Approx Qty":f"{res.selected_motor_kw:.2f} kW"},
    ])

def results_df(res:DesignResult)->pd.DataFrame:
    return pd.DataFrame([{"Parameter":k,"Value":v} for k,v in asdict(res).items() if k not in ["warnings","recommendations"]])

def create_excel(inp:DutyInput,res:DesignResult)->bytes:
    bio=io.BytesIO()
    with pd.ExcelWriter(bio,engine="openpyxl") as writer:
        pd.DataFrame([asdict(inp)]).T.reset_index().rename(columns={"index":"Input",0:"Value"}).to_excel(writer,"Inputs",index=False)
        results_df(res).to_excel(writer,"Results",index=False)
        practicality_table(inp,res).to_excel(writer,"Practicality",index=False)
        performance_curve(res).to_excel(writer,"Performance Curve",index=False)
        bom_table(inp,res).to_excel(writer,"BOM",index=False)
    return bio.getvalue()

def create_pdf(inp:DutyInput,res:DesignResult)->bytes:
    if not HAS_REPORTLAB: return b"Install reportlab to generate PDF reports."
    bio=io.BytesIO(); doc=SimpleDocTemplate(bio,pagesize=A4); styles=getSampleStyleSheet(); story=[]
    story += [Paragraph("Centrifugal Blower Preliminary Design Report",styles["Title"]),Spacer(1,8),Paragraph("SI units only. Static pressure is user input; total pressure is calculated as static pressure plus estimated discharge velocity pressure.",styles["BodyText"]),Spacer(1,10)]
    kv={"Blade type":inp.blade_type,"Airflow":f"{inp.airflow_m3h:,.0f} m³/h ({res.q_m3s:.3f} m³/s)","Static pressure input":f"{res.static_pressure_pa:.0f} Pa","Calculated velocity pressure":f"{res.velocity_pressure_pa:.0f} Pa","Calculated total pressure":f"{res.total_pressure_pa:.0f} Pa","Density":f"{res.density_kgm3:.3f} kg/m³","RPM":f"{res.rpm:.0f}","Impeller OD D₂":f"{res.impeller_od_mm:.1f} mm","Inlet diameter D₁":f"{res.impeller_id_mm:.1f} mm","Outlet width b₂":f"{res.outlet_width_mm:.1f} mm","β₁ / β₂":f"{res.beta1_deg:.1f}° / {res.beta2_deg:.1f}°","Blades":str(res.blade_count),"Blade pitch":f"{res.blade_pitch_mm:.1f} mm","Pitch/chord":f"{res.pitch_chord_ratio:.2f}","Tip speed":f"{res.tip_speed_ms:.1f} m/s","Outlet flange W x H":f"{res.flange_width_mm:.0f} x {res.flange_height_mm:.0f} mm","Air velocity at flange":f"{res.flange_velocity_ms:.1f} m/s","Estimated sound pressure @1 m":f"{res.estimated_sound_pressure_1m_dba:.1f} dBA","Vibration risk":f"{res.vibration_risk_label} ({res.vibration_risk_score:.0f}/100)","Shaft power":f"{res.shaft_power_kw:.2f} kW","Selected motor":f"{res.selected_motor_kw:.2f} kW","Shaft diameter":f"{res.shaft_diameter_mm:.0f} mm"}
    data=[["Parameter","Value"]]+[[k,v] for k,v in kv.items()]
    tbl=Table(data,colWidths=[190,290]); tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")]))
    story.append(tbl); story.append(Spacer(1,10))
    story.append(Paragraph("Practicality / Recommendations",styles["Heading2"]))
    for w in res.warnings: story.append(Paragraph(f"• WARNING: {w}",styles["BodyText"]))
    for r in res.recommendations: story.append(Paragraph(f"• {r}",styles["BodyText"]))
    doc.build(story); return bio.getvalue()

def ansys_workflow_text()->str:
    return """ANSYS / CFD workflow guidance

1. Export separate solids: impeller_solid.step, stationary_volute.step, inlet_duct.step, outlet_duct.step, fluid_domain.step.
2. For Fluent, the most important geometry is the FLUID DOMAIN, not only the metal fan.
3. Use rotating region around the impeller and stationary region in the volute.
4. Suggested boundary conditions: pressure inlet or mass-flow inlet, pressure outlet, MRF for preliminary, sliding mesh for transient blade-passing.
5. Mesh: inflation layers near blades/tongue, fine leading/trailing edges, y+ according to turbulence model.
"""

def make_zip(inp:DutyInput,res:DesignResult)->bytes:
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("blower_design_report.pdf",create_pdf(inp,res)); z.writestr("blower_design_calculations.xlsx",create_excel(inp,res)); z.writestr("blower_2d_manufacturing.dxf",create_dxf(res)); z.writestr("impeller_basic.step_or_instruction.txt",create_step_placeholder(res)); z.writestr("README_ANSYS_WORKFLOW.txt",ansys_workflow_text())
    return bio.getvalue()

# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Centrifugal Blower Design Toolkit", layout="wide")
if not check_password(): st.stop()
st.title("Centrifugal Blower Design & Manufacturing Toolkit")
st.caption("Forward curved, backward curved / backward inclined, and radial blade blowers — SI units only")

with st.sidebar:
    st.header("Duty Inputs - Static Pressure Only")
    airflow=st.number_input("Airflow (m³/h)",min_value=100.0,value=10000.0,step=500.0)
    sp=st.number_input("Static pressure required (Pa)",min_value=10.0,value=900.0,step=50.0,help="Enter only static pressure. The app calculates velocity pressure and total pressure from outlet air velocity.")
    rpm=st.number_input("Fan speed (RPM)",min_value=100.0,value=1450.0,step=50.0)
    blade_type=st.selectbox("Blade type",list(BLADE_DEFAULTS.keys()))
    st.header("Air Properties")
    temp_c=st.number_input("Air temperature (°C)",value=35.0,step=1.0); altitude=st.number_input("Altitude (m)",value=0.0,step=100.0)
    auto_density=standard_air_density(temp_c,altitude); use_auto=st.checkbox(f"Use calculated density ({auto_density:.3f} kg/m³)",value=True)
    density=auto_density if use_auto else st.number_input("Air density (kg/m³)",value=1.20,step=0.01)
    st.header("Impeller Geometry Inputs")
    defaults=BLADE_DEFAULTS[blade_type]
    beta1=st.number_input("β₁ inlet blade angle (deg)",value=float(defaults["beta1"]),step=1.0,help="Angle of blade at inlet eye. It should match incoming relative air direction to avoid entry shock.")
    beta2=st.number_input("β₂ outlet blade angle (deg)",value=float(defaults["beta2"]),step=1.0,help="Angle at impeller outlet. <90 backward, 90 radial, >90 forward curved.")
    blades=st.number_input("Number of blades z",min_value=3,value=int(defaults["blade_count"]),step=1)
    b2_ratio=st.number_input("Initial outlet width ratio b₂/D₂",min_value=0.03,max_value=0.80,value=float(defaults["b2_d2"]),step=0.01)
    d1_ratio=st.number_input("Inlet diameter ratio D₁/D₂",min_value=0.25,max_value=0.85,value=float(defaults["d1_d2"]),step=0.01)
    outlet_ar=st.number_input("Outlet flange aspect ratio W/H",min_value=0.5,max_value=4.0,value=1.5,step=0.1)
    st.header("Mechanical / Drive / Vibration")
    material=st.selectbox("Impeller/casing material",list(MATERIALS.keys()))
    blade_thk=st.number_input("Blade / disc thickness (mm)",min_value=1.0,value=3.0,step=0.5); casing_thk=st.number_input("Casing thickness (mm)",min_value=1.0,value=3.0,step=0.5)
    drive_type=st.selectbox("Drive type",["Direct drive","Belt drive","Coupling drive"]); drive_eff=st.number_input("Drive efficiency",min_value=0.70,max_value=1.00,value=0.95 if drive_type=="Belt drive" else 0.98,step=0.01)
    motor_eff=st.number_input("Motor efficiency",min_value=0.70,max_value=0.99,value=0.90,step=0.01); margin=st.number_input("Motor design margin (%)",min_value=0.0,value=15.0,step=1.0)
    shaft_tau=st.number_input("Allowable shaft shear stress (MPa)",min_value=20.0,value=40.0,step=5.0)
    unbalance=st.number_input("Assumed residual unbalance severity (mm/s)",min_value=0.0,value=2.5,step=0.5,help="Screening value only. Final impeller must be dynamically balanced.")
    isolation=st.selectbox("Mounting / isolation quality",["Good isolation and aligned drive","Average","Poor / rigid mounting"])

inp=DutyInput(airflow,sp,temp_c,altitude,density,rpm,blade_type,drive_type,motor_eff,drive_eff,margin,beta2,beta1,int(blades),b2_ratio,d1_ratio,material,blade_thk,casing_thk,shaft_tau,outlet_ar,unbalance,isolation)
res=design_blower(inp)

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Calculated total pressure",f"{res.total_pressure_pa:.0f} Pa",f"VP {res.velocity_pressure_pa:.0f} Pa")
c2.metric("Impeller OD D₂",f"{res.impeller_od_mm:.0f} mm")
c3.metric("Outlet flange",f"{res.flange_width_mm:.0f}×{res.flange_height_mm:.0f} mm")
c4.metric("Flange air velocity",f"{res.flange_velocity_ms:.1f} m/s")
c5.metric("Sound @1 m",f"{res.estimated_sound_pressure_1m_dba:.0f} dBA")
for w in res.warnings: st.warning(w)

with st.expander("What do β₁, β₂, D₁/D₂ and b₂/D₂ physically mean?", expanded=True):
    st.markdown("""
- **β₁ inlet blade angle**: blade angle where air enters the impeller eye. If wrong, air hits the blade with shock, increasing noise, loss and vibration.
- **β₂ outlet blade angle**: blade angle where air leaves the wheel. **Backward < 90°** gives stable/non-overloading power, **radial ≈ 90°** is robust for dirty air, **forward > 90°** gives compact high flow but can overload and be less stable.
- **D₁/D₂**: inlet eye diameter divided by impeller OD. Too small causes inlet choking; too large reduces pressure-producing annulus.
- **b₂/D₂**: outlet width divided by impeller OD. Too low causes high velocity/choking; too high makes a very wide wheel and may require DIDW or multiple fans.
- **Pitch/chord ratio**: circumferential spacing between blades divided by blade length. Too close means blockage and blade-passing tone; too wide means pulsation and non-uniform discharge.
""")

tabs=st.tabs(["Design Summary","Practicality & Guidance","Impeller Sketch","Blade Profile Sketch","Performance Curves","Noise + Vibration","Mechanical + BOM","Exports","Ansys Workflow"])
with tabs[0]:
    summary=pd.DataFrame([
        ["Airflow",f"{res.q_m3s:.3f} m³/s",f"{inp.airflow_m3h:.0f} m³/h"],["Static pressure input",f"{res.static_pressure_pa:.0f} Pa","User input"],["Velocity pressure calculated",f"{res.velocity_pressure_pa:.0f} Pa",f"0.5ρV² at flange"],["Total pressure calculated",f"{res.total_pressure_pa:.0f} Pa","SP + VP"],["Impeller OD D₂",f"{res.impeller_od_mm:.1f} mm", ""],["Inlet diameter D₁",f"{res.impeller_id_mm:.1f} mm",f"D₁/D₂={res.impeller_id_mm/res.impeller_od_mm:.2f}"],["Outlet width b₂",f"{res.outlet_width_mm:.1f} mm",f"b₂/D₂={res.outlet_width_mm/res.impeller_od_mm:.2f}"],["β₁ / β₂",f"{res.beta1_deg:.1f}° / {res.beta2_deg:.1f}°", ""],["Blade count",f"{res.blade_count}",f"Pitch={res.blade_pitch_mm:.1f} mm"],["Outlet flange W×H",f"{res.flange_width_mm:.0f}×{res.flange_height_mm:.0f} mm",f"Velocity={res.flange_velocity_ms:.1f} m/s"],["Shaft power / motor",f"{res.shaft_power_kw:.2f} / {res.selected_motor_kw:.2f} kW", ""]],columns=["Parameter","Value","Notes"])
    st.dataframe(summary,use_container_width=True)
with tabs[1]:
    st.subheader("Practicality Check Table")
    st.dataframe(practicality_table(inp,res),use_container_width=True)
    st.subheader("Recommendations")
    if not res.recommendations and not res.warnings: st.success("No major preliminary warnings. Still validate by prototype fan test and balancing.")
    for r in res.recommendations: st.info(r)
with tabs[2]:
    st.subheader("Full Impeller Sketch with Diameters, Blades and Outlet Flange")
    blade=blade_centerline_points(res); vol=np.array(volute_spiral_points(res)); fig,ax=plt.subplots(figsize=(8,8)); r2=res.impeller_od_mm/2; r1=res.impeller_id_mm/2; hub=max(res.shaft_diameter_mm*1.6,r1*0.25)
    ax.add_patch(plt.Circle((0,0),r2,fill=False,linewidth=2)); ax.add_patch(plt.Circle((0,0),r1,fill=False,linestyle="--")); ax.add_patch(plt.Circle((0,0),hub,fill=False,linestyle=":"))
    for k in range(res.blade_count):
        a=2*math.pi*k/res.blade_count; ca,sa=math.cos(a),math.sin(a); pts=np.array([(x*ca-y*sa,x*sa+y*ca) for x,y in blade]); ax.plot(pts[:,0],pts[:,1],linewidth=0.8)
    ax.plot(vol[:,0],vol[:,1],linewidth=2); x0=r2+res.volute_cutoff_clearance_mm; y0=r2*0.55; W=res.flange_width_mm; H=res.flange_height_mm
    ax.plot([x0,x0+H,x0+H,x0,x0],[y0,y0,y0+W,y0+W,y0],linewidth=2)
    ax.annotate(f"D₂={res.impeller_od_mm:.0f} mm",xy=(r2,0),xytext=(0,-r2*1.15),arrowprops=dict(arrowstyle="->")); ax.annotate(f"D₁={res.impeller_id_mm:.0f} mm",xy=(r1,0),xytext=(-r2*0.9,-r2*0.8),arrowprops=dict(arrowstyle="->")); ax.text(x0,y0+W+20,f"Outlet {W:.0f}×{H:.0f} mm\nV={res.flange_velocity_ms:.1f} m/s")
    ax.set_aspect("equal"); ax.grid(True); ax.set_xlabel("mm"); ax.set_ylabel("mm"); st.pyplot(fig)
with tabs[3]:
    st.subheader("Single Blade Profile: Angles, Pitch and Chord")
    outline=blade_outline_points(res); center=np.array(blade_centerline_points(res)); figb,axb=plt.subplots(figsize=(8,5)); axb.plot(outline[:,0],outline[:,1]); axb.plot(center[:,0],center[:,1],linestyle="--"); axb.scatter([center[0,0],center[-1,0]],[center[0,1],center[-1,1]])
    axb.annotate(f"β₁={res.beta1_deg:.0f}°",xy=center[0],xytext=center[0]+np.array([30,30]),arrowprops=dict(arrowstyle="->")); axb.annotate(f"β₂={res.beta2_deg:.0f}°",xy=center[-1],xytext=center[-1]+np.array([-120,40]),arrowprops=dict(arrowstyle="->")); axb.text(center[:,0].mean(),center[:,1].mean(),f"Chord≈{res.blade_chord_mm:.0f} mm\nPitch at OD≈{res.blade_pitch_mm:.0f} mm\nPitch/chord={res.pitch_chord_ratio:.2f}")
    axb.set_aspect("equal"); axb.grid(True); axb.set_xlabel("mm"); axb.set_ylabel("mm"); st.pyplot(figb)
with tabs[4]:
    curve=performance_curve(res); figp,axp=plt.subplots(); axp.plot(curve.Flow_m3h,curve.TotalPressure_Pa,label="Total Pressure"); axp.plot(curve.Flow_m3h,curve.StaticPressure_approx_Pa,label="Static Pressure approx"); axp.set_xlabel("Flow (m³/h)"); axp.set_ylabel("Pressure (Pa)"); axp.grid(True); axp.legend(); st.pyplot(figp)
    figpow,axpow=plt.subplots(); axpow.plot(curve.Flow_m3h,curve.ShaftPower_kW,label="Shaft Power kW"); axpow.set_xlabel("Flow (m³/h)"); axpow.set_ylabel("kW"); axpow.grid(True); axpow.legend(); st.pyplot(figpow); st.dataframe(curve,use_container_width=True)
with tabs[5]:
    st.subheader("Preliminary Noise and Vibration Screening")
    nv=pd.DataFrame([["Estimated sound power",f"{res.estimated_sound_power_dba:.1f} dBA","Screening only, not AMCA 300"],["Estimated sound pressure at 1 m",f"{res.estimated_sound_pressure_1m_dba:.1f} dBA","Depends strongly on casing, duct, room and silencer"],["Vibration risk",f"{res.vibration_risk_label} ({res.vibration_risk_score:.0f}/100)","Based on tip speed, outlet velocity, blade pitch, fan type, unbalance and mounting"],["Blade passing frequency",f"{res.rpm/60*res.blade_count:.1f} Hz","Can create tonal noise if casing/duct resonates"]],columns=["Check","Value","Note"])
    st.dataframe(nv,use_container_width=True)
with tabs[6]:
    st.dataframe(bom_table(inp,res),use_container_width=True)
with tabs[7]:
    st.download_button("Download PDF Report",create_pdf(inp,res),"blower_design_report.pdf")
    st.download_button("Download Excel Calculation",create_excel(inp,res),"blower_design_calculations.xlsx")
    st.download_button("Download DXF Drawing",create_dxf(res),"blower_2d_manufacturing.dxf")
    st.download_button("Download STEP placeholder / CAD instruction",create_step_placeholder(res),"impeller_basic.step_or_instruction.txt")
    st.download_button("Download Complete ZIP Package",make_zip(inp,res),"blower_design_package.zip")
with tabs[8]: st.text(ansys_workflow_text())
