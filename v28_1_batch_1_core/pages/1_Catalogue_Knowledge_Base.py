from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from blower_toolkit.auth import require_password
from blower_toolkit.catalogue import CatalogueDB, MODEL_COLUMNS, extract_candidates_from_pdf
from blower_toolkit.catalogue.manufacturing_bridge import embedded_pdf_path, manufacturing_handoff, pdf_page_text, render_pdf_page_png
st.set_page_config(page_title='Fan Manufacturer Knowledge Base',layout='wide'); require_password()
st.title('Fan Manufacturer Knowledge Base v28')
st.caption('Embedded manufacturer catalogues (including Airtek Power V.2024), extracted model library, published Q-P point storage, exact source-page viewer, duty search, and manufacturing-equivalent bridge.')
ROOT=Path(__file__).resolve().parents[1]; db=CatalogueDB(ROOT/'data'/'catalogue.db'); fans=db.all(); docs=db.documents(); curves=db.curves(); dims=db.dimensions(); emb=len(list((ROOT/'assets'/'catalogues').glob('*.pdf')))
with st.sidebar:
 st.header('Catalogue database'); st.metric('Manufacturers',int(fans.manufacturer.nunique()) if len(fans) else 0); st.metric('Embedded PDF files / parts',emb); st.metric('Extracted models',len(fans)); st.metric('Models with published Q-P points',int(curves.model.nunique()) if len(curves) else 0); st.metric('Stored Q-P / operating points',len(curves)); st.caption('Large catalogues are split into GitHub-safe PDF parts. Exact source pages remain viewable. Verify critical dimensions visually before production release.')
tabs=st.tabs(['Catalogue Browser','Model Detail','Duty Search','Catalogue Sources','Curve Data','Import More Catalogues','Database Review'])
with tabs[0]:
 st.subheader('Browse actual catalogue models'); view=fans.copy(); c1,c2,c3,c4=st.columns(4); maker=c1.selectbox('Manufacturer',['All']+sorted(view.manufacturer.dropna().unique().tolist()));
 if maker!='All': view=view[view.manufacturer==maker]
 typ=c2.selectbox('Fan type',['All']+sorted(view.fan_type.dropna().unique().tolist()));
 if typ!='All': view=view[view.fan_type==typ]
 arr=c3.selectbox('Arrangement',['All']+sorted(view.arrangement.dropna().unique().tolist()));
 if arr!='All': view=view[view.arrangement==arr]
 q=c4.text_input('Model / series contains');
 if q: view=view[view.model.fillna('').str.contains(q,case=False,regex=False)|view.series.fillna('').str.contains(q,case=False,regex=False)]
 st.write(f'**{len(view)} model(s)**'); cols=['manufacturer','series','model','fan_type','arrangement','motor_type','impeller_diameter_mm','airflow_m3h','static_pressure_pa','rpm','input_power_w','noise_dba','voltage_v','material','ip_rating','source_file','source_page','data_status']; st.dataframe(view[[c for c in cols if c in view]],use_container_width=True,hide_index=True)
 if len(view):
  m=st.selectbox('Open model',view.model.tolist());
  if st.button('Open model detail',type='primary'): st.session_state['catalogue_selected_model']=m; st.info('Open the Model Detail tab above.')
with tabs[1]:
 st.subheader('Manufacturer model detail')
 if len(fans):
  ml=fans.model.tolist(); initial=st.session_state.get('catalogue_selected_model'); idx=ml.index(initial) if initial in ml else 0; m=st.selectbox('Model',ml,index=idx,key='detail_model'); st.session_state['catalogue_selected_model']=m; r=fans[fans.model==m].iloc[0]
  mc=curves[curves.model==m].copy() if not curves.empty else pd.DataFrame()
  a,b,c,d,e=st.columns(5); a.metric('Manufacturer',str(r.manufacturer)); b.metric('Impeller','—' if pd.isna(r.impeller_diameter_mm) else f'{r.impeller_diameter_mm:g} mm'); c.metric('Airflow','—' if pd.isna(r.airflow_m3h) else f'{r.airflow_m3h:,.0f} m³/h');
  if not pd.isna(r.static_pressure_pa): d.metric('Static pressure',f'{r.static_pressure_pa:,.0f} Pa')
  elif len(mc): d.metric('Published pressure points',f'{mc.pressure_pa.min():,.0f}–{mc.pressure_pa.max():,.0f} Pa')
  else: d.metric('Static pressure','—')
  e.metric('RPM','—' if pd.isna(r.rpm) else f'{r.rpm:,.0f}')
  L,R=st.columns([.42,.58])
  with L:
   info=pd.DataFrame([(k,getattr(r,k,None)) for k in ['model','series','fan_type','arrangement','motor_type','impeller_diameter_mm','wheel_width_mm','airflow_m3h','static_pressure_pa','rpm','input_power_w','efficiency_pct','noise_dba','voltage_v','frequency_hz','current_a','material','ip_rating','insulation_class','control','operating_temp','source_file','source_page','data_status','notes']],columns=['Parameter','Value']); st.dataframe(info,use_container_width=True,hide_index=True)
   md=dims[dims.model==m] if not dims.empty else pd.DataFrame();
   if len(md): st.markdown('#### Catalogue dimensions'); st.dataframe(md[['dimension_code','value_mm','description','source_page','data_status']],use_container_width=True,hide_index=True)
   if len(mc):
    st.markdown('#### Stored performance / operating points')
    st.dataframe(mc[['curve_name','speed_rpm','airflow_m3h','pressure_pa','power_w','noise_dba','point_status']],use_container_width=True,hide_index=True)
    chart_src=mc.dropna(subset=['airflow_m3h','pressure_pa']).copy()
    if len(chart_src)>=2:
     chart_src['curve_label']=chart_src['curve_name'].fillna('Published curve').astype(str)
     chart=chart_src.pivot_table(index='airflow_m3h',columns='curve_label',values='pressure_pa',aggfunc='mean').sort_index()
     chart.columns=[str(x) for x in chart.columns]
     if len(chart.columns): st.line_chart(chart,x_label='Airflow (m³/h)',y_label='Static pressure (Pa)')
  with R:
   st.markdown('#### Original catalogue page'); pp=embedded_pdf_path(ROOT,r.source_file)
   if pp and pd.notna(r.source_page):
    try: st.image(render_pdf_page_png(pp,int(r.source_page)),caption=f'{r.source_file} — PDF page {int(r.source_page)}',use_container_width=True)
    except Exception as ex: st.error(str(ex))
  st.markdown('### Manufacture / design an equivalent')
  h=manufacturing_handoff(r.to_dict())
  if not h.get('supported',True):
   st.warning('This product type is not yet supported by the centrifugal inverse-design/manufacturing engine.')
  else:
   st.warning(
    "For many manufacturer technical tables, the published Air Flow and Air Pressure values are maximum endpoints, "
    "not necessarily one simultaneous duty point. The inverse designer handles this explicitly."
   )
   handoff_rows=[
    ('Known catalogue','D2 / nominal diameter',h['known_d2_mm'],'mm'),
    ('Catalogue summary','Air flow',h['known_airflow_m3h'],'m³/h'),
    ('Catalogue summary','Static pressure',h['known_static_pressure_pa'] if h['known_static_pressure_pa'] else ('Use published Q-P points' if len(mc) else '—'),'Pa'),
    ('Known catalogue','RPM',h['known_rpm'],'rpm'),
    ('Calculated start','D1',h['calculated_d1_mm'],'mm'),
    ('Calculated start','b2',h['calculated_b2_mm'],'mm'),
    ('Calculated start','Blade count',h['calculated_blade_count'],'count'),
    ('Calculated start','beta1',h['calculated_beta1_deg'],'deg'),
    ('Calculated start','beta2',h['calculated_beta2_deg'],'deg')
   ]
   st.dataframe(pd.DataFrame(handoff_rows,columns=['Source','Parameter','Value','Unit']),use_container_width=True,hide_index=True)
   ib1,ib2=st.columns(2)
   if ib1.button('Inverse-optimize hidden geometry from catalogue data',type='primary'):
    st.session_state['catalogue_selected_model']=m
    st.session_state['catalogue_manufacture_handoff']=h
    try: st.switch_page('pages/6_Catalogue_Inverse_Designer.py')
    except Exception: st.success('Model selected. Open Catalogue Inverse Designer from the left navigation.')
   if ib2.button('Open directly in Reverse Engineer'):
    st.session_state['catalogue_manufacture_handoff']=h
    try: st.switch_page('pages/5_Reverse_Engineer_Blower.py')
    except Exception: st.info('Select Reverse Engineer Blower from the left navigation.')
with tabs[2]:
 st.subheader('Search all catalogue models by required duty')
 c1,c2,c3,c4=st.columns(4); rq=c1.number_input('Required airflow (m³/h)',1.,2e6,5000.,100.); rp=c2.number_input('Required static pressure (Pa)',1.,30000.,500.,10.); rr=c3.number_input('Preferred RPM (0=ignore)',0.,20000.,0.,50.); typ=c4.selectbox('Fan type',['All']+sorted(fans.fan_type.dropna().unique().tolist()),key='duty_type')
 results=[]
 # First use genuine stored Q-P points: interpolate pressure at the requested airflow only inside the published Q range.
 if len(curves):
  valid_curves=curves.dropna(subset=['model','airflow_m3h','pressure_pa']).copy()
  for (model,curve_name),grp in valid_curves.groupby(['model','curve_name'],dropna=False):
   grp=grp.sort_values('airflow_m3h').drop_duplicates('airflow_m3h')
   if len(grp)<2 or rq<float(grp.airflow_m3h.min()) or rq>float(grp.airflow_m3h.max()): continue
   fr=fans[fans.model==model]
   if fr.empty: continue
   fr=fr.iloc[0]
   if typ!='All' and fr.fan_type!=typ: continue
   pred_p=float(np.interp(rq,grp.airflow_m3h.astype(float),grp.pressure_pa.astype(float)))
   med_rpm=float(pd.to_numeric(grp.speed_rpm,errors='coerce').median()) if grp.speed_rpm.notna().any() else (float(fr.rpm) if not pd.isna(fr.rpm) else np.nan)
   rpm_err=0.0 if rr<=0 or pd.isna(med_rpm) else 100*abs(med_rpm-rr)/rr
   p_err=100*abs(pred_p-rp)/rp
   results.append(dict(manufacturer=fr.manufacturer,model=model,fan_type=fr.fan_type,arrangement=fr.arrangement,impeller_diameter_mm=fr.impeller_diameter_mm,required_airflow_m3h=rq,predicted_static_pressure_pa=pred_p,curve_name=curve_name,curve_rpm=med_rpm,input_power_w=fr.input_power_w,noise_dba=fr.noise_dba,pressure_error_pct=p_err,rpm_error_pct=rpm_err,duty_score=.90*p_err+.10*rpm_err,selection_basis='Published Q-P points',source_file=grp.iloc[0].source_file,source_page=grp.iloc[0].source_page))
 # Add summary-rated models that do not have a usable curve at this Q.
 u=fans.dropna(subset=['airflow_m3h','static_pressure_pa']).copy()
 if typ!='All': u=u[u.fan_type==typ]
 for _,fr in u.iterrows():
  if any(x['model']==fr.model for x in results): continue
  flow_err=100*abs(float(fr.airflow_m3h)-rq)/rq; p_err=100*abs(float(fr.static_pressure_pa)-rp)/rp; rpm_err=0.0 if rr<=0 or pd.isna(fr.rpm) else 100*abs(float(fr.rpm)-rr)/rr
  results.append(dict(manufacturer=fr.manufacturer,model=fr.model,fan_type=fr.fan_type,arrangement=fr.arrangement,impeller_diameter_mm=fr.impeller_diameter_mm,required_airflow_m3h=rq,predicted_static_pressure_pa=fr.static_pressure_pa,curve_name='—',curve_rpm=fr.rpm,input_power_w=fr.input_power_w,noise_dba=fr.noise_dba,pressure_error_pct=p_err,rpm_error_pct=rpm_err,duty_score=.45*flow_err+.45*p_err+.10*rpm_err,selection_basis='Catalogue summary rating',source_file=fr.source_file,source_page=fr.source_page))
 if results:
  ranked=pd.DataFrame(results).sort_values('duty_score').head(40)
  st.dataframe(ranked,use_container_width=True,hide_index=True)
  st.success("Where published Q-P points exist, pressure is interpolated at your requested airflow inside the manufacturer's stored curve range. Summary-only models are clearly labelled separately.")
 else: st.info('No stored catalogue curve/rating covers this duty. Try a different airflow/pressure or digitise another manufacturer curve.')
with tabs[3]: st.subheader('Embedded catalogue sources'); st.dataframe(docs,use_container_width=True,hide_index=True); st.success(f'{emb} PDF catalogues are physically included under assets/catalogues.')
with tabs[4]:
 st.subheader('Stored performance / operating points'); st.dataframe(curves,use_container_width=True,hide_index=True)
with tabs[5]:
 st.subheader('Import additional catalogues'); maker=st.text_input('Manufacturer'); ups=st.file_uploader('Upload PDF / CSV',type=['pdf','csv'],accept_multiple_files=True)
 if ups:
  frames=[]
  for up in ups:
   if up.name.lower().endswith('.csv'):
    x=pd.read_csv(up); [x.__setitem__(c,None) for c in MODEL_COLUMNS if c not in x]; frames.append(x[MODEL_COLUMNS])
   else: frames.append(extract_candidates_from_pdf(up.getvalue(),up.name,maker or 'Unknown').candidates)
  if frames:
   ed=st.data_editor(pd.concat(frames,ignore_index=True),use_container_width=True,hide_index=True,num_rows='dynamic');
   if st.button('Save reviewed imported rows'): st.success(f'Saved/updated {db.upsert_dataframe(ed)} records.')
with tabs[6]: st.subheader('Database quality'); st.dataframe(db.quality_summary(),use_container_width=True,hide_index=True); st.markdown('Catalogue values define known duty/envelope where published. Hidden blade, scroll, shaft and tolerance details remain design/measurement work and are explicitly tagged as calculated or assumed.')
