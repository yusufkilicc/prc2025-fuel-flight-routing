"""
Faz 5 (v2) — Polished route decision web app.
FastAPI backend (OpenAP scoring) + single-page Tailwind/MapLibre frontend.
Run:  python web_app.py     (then open http://localhost:8600)
"""
import os, sys, math, json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")
sys.path.insert(0, os.path.join(_PKG, "src"))
import route_decision as rd
try:
    import airportsdata
    _IATA = {k: v.get("iata") or "" for k, v in airportsdata.load("ICAO").items()}
except Exception:
    _IATA = {}

app = FastAPI(title="Fuel & Emissions-Aware Route Comparison")
SHAP = json.load(open(os.path.join(_PKG, "models", "shap_importance.json"), encoding="utf-8"))

AC_NAMES = {
    "A20N": "A320neo", "A21N": "A321neo", "A319": "A319", "A320": "A320", "A321": "A321",
    "A318": "A318", "A332": "A330-200", "A333": "A330-300", "A359": "A350-900", "A388": "A380",
    "B737": "737-700", "B738": "737-800", "B739": "737-900", "B38M": "737 MAX 8", "B39M": "737 MAX 9",
    "B744": "747-400", "B748": "747-8", "B752": "757-200", "B763": "767-300", "B772": "777-200",
    "B77L": "777F", "B77W": "777-300ER", "B788": "787-8", "B789": "787-9", "A306": "A300-600", "MD11": "MD-11",
}

def _load():
    apt = pd.read_parquet(os.path.join(ROOT, "apt.parquet"))
    fl = pd.read_parquet(os.path.join(ROOT, "flightlist_train.parquet"))
    used = pd.unique(pd.concat([fl["origin_icao"], fl["destination_icao"]]))
    names = (pd.concat([
        fl[["origin_icao", "origin_name"]].rename(columns={"origin_icao": "icao", "origin_name": "name"}),
        fl[["destination_icao", "destination_name"]].rename(columns={"destination_icao": "icao", "destination_name": "name"})])
        .drop_duplicates("icao").set_index("icao")["name"])
    apt = apt[apt["icao"].isin(used)].dropna(subset=["latitude", "longitude"]).copy()
    apt["name"] = apt["icao"].map(names).fillna(apt["icao"])
    apt = apt.sort_values("icao").reset_index(drop=True)
    ac = sorted(fl["aircraft_type"].dropna().unique())
    return apt, ac

APT, AC = _load()
APT_IDX = APT.set_index("icao")

@app.get("/api/airports")
def airports():
    rows = [{"icao": r.icao, "iata": _IATA.get(r.icao, ""), "name": str(r.name_),
             "lat": float(r.latitude), "lon": float(r.longitude)}
            for r in APT.rename(columns={"name": "name_"}).itertuples()]
    acs = [{"code": c, "name": AC_NAMES.get(c, c)} for c in AC]
    return {"airports": rows, "aircraft": acs, "shap": SHAP}

@app.get("/api/score")
def score(o: str = Query(...), d: str = Query(...), ac: str = "A359", headwind: float = 0.0):
    if o not in APT_IDX.index or d not in APT_IDX.index or o == d:
        return JSONResponse({"error": "invalid airports"}, status_code=400)
    O, D = APT_IDX.loc[o], APT_IDX.loc[d]
    rows = rd.scenario_routes(O.latitude, O.longitude, D.latitude, D.longitude, ac, base_headwind_kt=headwind)
    out = []
    for r in rows:
        out.append({
            "label": r["label"], "subtitle": r["subtitle"], "recommended": r["recommended"],
            "dist_km": r["dist_km"], "fuel_kg": r["fuel_kg"], "co2_kg": r["co2_kg"],
            "fuel_lo": r.get("fuel_lo"), "fuel_hi": r.get("fuel_hi"),
            "time_min": r["time_min"], "cost_usd": r["cost_usd"], "cruise_ft": r["cruise_ft"],
            "path": [[float(p[1]), float(p[0])] for p in r["path"]],  # [lon,lat] for maplibre
        })
    return {
        "origin": {"icao": o, "iata": _IATA.get(o, ""), "name": str(O["name"]), "lat": float(O.latitude), "lon": float(O.longitude)},
        "dest": {"icao": d, "iata": _IATA.get(d, ""), "name": str(D["name"]), "lat": float(D.latitude), "lon": float(D.longitude)},
        "aircraft": AC_NAMES.get(ac, ac), "scenarios": out,
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Fuel & Emissions-Aware Route Comparison</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<script src="https://unpkg.com/maplibre-gl@5.6.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@5.6.1/dist/maplibre-gl.css" rel="stylesheet"/>
<style>
  :root{ --ink:#0f172a; --sub:#64748b; --line:#e2e8f0; --brand:#2563eb; --good:#0ea5e9; }
  *{ font-family:'Inter',system-ui,sans-serif; }
  body{ background:#f6f8fb; color:var(--ink); }
  .card{ background:#fff; border:1px solid var(--line); border-radius:16px; box-shadow:0 1px 2px rgba(15,23,42,.04),0 8px 24px rgba(15,23,42,.04); }
  select{ appearance:none; background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E"); background-repeat:no-repeat; background-position:right .7rem center; }
  .kpi-val{ font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  #map{ border-radius:16px; }
  .maplibregl-ctrl-attrib{ font-size:9px; opacity:.5; }
  .seg{ transition:all .18s ease; }
  .seg:hover{ transform:translateY(-2px); box-shadow:0 10px 30px rgba(15,23,42,.10); }
  .ring-rec{ border-color:#bfdbfe; box-shadow:0 0 0 3px rgba(37,99,235,.12); }
  .pulse{ animation:p 1.2s ease-in-out infinite; } @keyframes p{0%,100%{opacity:.45}50%{opacity:1}}
  select:focus-visible, button:focus-visible, input:focus-visible{ outline:2px solid #2563eb; outline-offset:2px; }
  @media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
</style>
</head>
<body class="min-h-screen">
<div class="max-w-7xl mx-auto px-6 py-7">

  <!-- header -->
  <div class="flex items-start justify-between flex-wrap gap-4 mb-6">
    <div>
      <h1 class="text-[26px] font-extrabold tracking-tight flex items-center gap-2">
        <span class="text-2xl">✈️</span> Fuel &amp; Emissions-Aware Route Comparison</h1>
      <p class="text-sm text-slate-500 mt-1 max-w-2xl">Decision layer over an OpenAP-physics + ML hybrid trained on the PRC&nbsp;2025 dataset. Compare candidate routes between two airports by fuel, CO₂, time and cost.</p>
    </div>
    <div class="text-right text-xs text-slate-400 leading-5">
      <div class="font-semibold text-slate-500">PRC Data Challenge 2025</div>
      <div>OpenAP · LightGBM · MapLibre</div>
    </div>
  </div>

  <!-- controls -->
  <div class="card p-4 mb-5">
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div><label for="origin" class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Origin</label>
        <select id="origin" aria-label="Origin airport" class="mt-1 w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm font-medium bg-white"></select></div>
      <div><label for="dest" class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Destination</label>
        <select id="dest" aria-label="Destination airport" class="mt-1 w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm font-medium bg-white"></select></div>
      <div><label for="ac" class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Aircraft</label>
        <select id="ac" aria-label="Aircraft type" class="mt-1 w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm font-medium bg-white"></select></div>
      <div><label for="wind" class="text-xs font-semibold text-slate-500 uppercase tracking-wide" title="Extra headwind applied uniformly to all scenarios. Negative = tailwind.">Extra headwind (kt) ⓘ</label>
        <div class="flex items-center gap-3 mt-1">
          <input id="wind" type="range" min="-80" max="80" step="10" value="0" aria-label="Extra headwind in knots" class="w-full accent-blue-600"/>
          <span id="windv" class="text-sm font-semibold w-10 text-right tabular-nums">0</span></div></div>
    </div>
  </div>

  <!-- KPI row -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
    <div class="card p-4"><div class="text-xs font-semibold text-slate-400 uppercase tracking-wide">Distance (great-circle)</div>
      <div id="k_dist" class="kpi-val text-2xl font-extrabold mt-1">—</div></div>
    <div class="card p-4"><div class="text-xs font-semibold text-slate-400 uppercase tracking-wide">Recommended fuel</div>
      <div id="k_fuel" class="kpi-val text-2xl font-extrabold mt-1 text-blue-600">—</div></div>
    <div class="card p-4"><div class="text-xs font-semibold text-slate-400 uppercase tracking-wide">CO₂</div>
      <div id="k_co2" class="kpi-val text-2xl font-extrabold mt-1">—</div></div>
    <div class="card p-4"><div class="text-xs font-semibold text-slate-400 uppercase tracking-wide">Flight time</div>
      <div id="k_time" class="kpi-val text-2xl font-extrabold mt-1">—</div></div>
  </div>

  <!-- map -->
  <div class="card p-2 mb-5 relative">
    <div id="map" style="height:460px;width:100%"></div>
    <div id="route-pill" class="absolute top-4 left-4 bg-white/90 backdrop-blur px-3 py-1.5 rounded-full text-xs font-semibold border border-slate-200 shadow-sm"></div>
    <button id="globe-toggle" aria-label="Toggle globe/flat projection" class="absolute top-4 right-14 bg-white/90 backdrop-blur px-3 py-1.5 rounded-full text-xs font-semibold border border-slate-200 shadow-sm hover:bg-white">🌐 Globe</button>
    <div class="absolute bottom-4 left-4 bg-white/90 backdrop-blur px-3 py-2 rounded-lg text-[11px] border border-slate-200 shadow-sm space-y-1">
      <div class="flex items-center gap-2"><span style="display:inline-block;width:18px;height:0;border-top:3px dashed #2563eb"></span> Recommended route</div>
      <div class="flex items-center gap-2"><span style="display:inline-block;width:18px;height:0;border-top:2px dashed #94a3b8"></span> Alternatives</div>
      <div class="flex items-center gap-2"><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#10b981"></span> Origin <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#ef4444;margin-left:6px"></span> Destination</div>
    </div>
  </div>

  <!-- scenario cards + SHAP -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-6">
    <div class="lg:col-span-2">
      <h2 class="text-sm font-bold text-slate-500 uppercase tracking-wide mb-3">Candidate routes</h2>
      <div id="cards" class="grid grid-cols-1 md:grid-cols-3 gap-4"></div>
    </div>
    <div>
      <h2 class="text-sm font-bold text-slate-500 uppercase tracking-wide mb-3">Why this estimate</h2>
      <div id="shap" class="card p-5"></div>
    </div>
  </div>

  <!-- honesty -->
  <div class="card p-4 text-xs text-slate-500 leading-6">
    <span class="font-semibold text-slate-600">Model limits.</span>
    Scores come from the <b>OpenAP</b> physics model — a hypothetical (not-yet-flown) route has no real telemetry.
    Mass is <b>estimated</b> (load-factor assumption), the largest fuel driver. The <i>wind/fuel-optimal</i> and
    <i>published-ATC</i> routes are <b>illustrative</b> (no live wind grid or ATC route data). The ML model validated on the
    PRC dataset predicts fuel for <i>flown</i> trajectories; this tool is its <b>extrapolation</b>. Confidence drops far from
    the training distribution (medium/long-haul commercial jets).
  </div>
</div>

<script>
const fmt = (n)=> n.toLocaleString('en-US');
const tMin = (m)=>{const h=Math.floor(m/60), x=Math.round(m%60); return h>0?`${h}h ${x}m`:`${x}m`;};
const code = (a)=> a.iata || a.icao;
let AIR=[], SHAPDATA=null, map, isGlobe=true;

async function init(){
  const r = await (await fetch('/api/airports')).json();
  AIR = r.airports; SHAPDATA = r.shap;
  const byIcao = Object.fromEntries(AIR.map(a=>[a.icao,a]));
  const opt = (a)=>`<option value="${a.icao}">${code(a)} — ${a.name}</option>`;
  document.getElementById('origin').innerHTML = AIR.map(opt).join('');
  document.getElementById('dest').innerHTML   = AIR.map(opt).join('');
  document.getElementById('ac').innerHTML = r.aircraft.map(a=>`<option value="${a.code}">${a.name}</option>`).join('');
  if(byIcao['LTFM']) document.getElementById('origin').value='LTFM';
  if(byIcao['KJFK']) document.getElementById('dest').value='KJFK';
  document.getElementById('ac').value = r.aircraft.find(a=>a.code==='A359')?'A359':r.aircraft[0].code;
  // shareable URL state (D1.7): ?o=&d=&ac=&w=
  const q = new URLSearchParams(location.search);
  if(q.get('o') && byIcao[q.get('o')]) document.getElementById('origin').value=q.get('o');
  if(q.get('d') && byIcao[q.get('d')]) document.getElementById('dest').value=q.get('d');
  const acSel=document.getElementById('ac');
  if(q.get('ac') && [...acSel.options].some(o=>o.value===q.get('ac'))) acSel.value=q.get('ac');
  if(q.get('w')!==null && q.get('w')!==''){ document.getElementById('wind').value=q.get('w'); document.getElementById('windv').textContent=q.get('w'); }
  renderShap();

  map = new maplibregl.Map({
    container:'map',
    style:'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    center:[0,30], zoom:1.4, attributionControl:true
  });
  map.addControl(new maplibregl.NavigationControl({showCompass:false}),'top-right');
  map.on('style.load', ()=>{ map.setProjection({type:'globe'}); });
  map.on('load', run);

  document.getElementById('globe-toggle').addEventListener('click',()=>{
    isGlobe=!isGlobe;
    map.setProjection({type: isGlobe?'globe':'mercator'});
    document.getElementById('globe-toggle').textContent = isGlobe?'🌐 Globe':'🗺️ Flat';
  });
  ['origin','dest','ac'].forEach(id=>document.getElementById(id).addEventListener('change',run));
  document.getElementById('wind').addEventListener('input',e=>{document.getElementById('windv').textContent=e.target.value;});
  document.getElementById('wind').addEventListener('change',run);
}

function renderShap(){
  if(!SHAPDATA) return;
  const max = Math.max(...SHAPDATA.drivers.map(d=>d.value));
  const bars = SHAPDATA.drivers.map((d,i)=>{
    const w = Math.round(100*d.value/max);
    const c = i===0?'#2563eb':'#93c5fd';
    return `<div class="mb-2.5">
      <div class="flex justify-between text-xs mb-1"><span class="font-medium text-slate-600">${d.feature}</span>
        <span class="tabular-nums text-slate-400">${d.value.toFixed(2)}</span></div>
      <div class="h-2 rounded-full bg-slate-100 overflow-hidden"><div style="width:${w}%;background:${c}" class="h-full rounded-full"></div></div>
    </div>`;}).join('');
  document.getElementById('shap').innerHTML =
    `<div class="text-[15px] font-bold mb-0.5">${SHAPDATA.title}</div>
     <div class="text-xs text-slate-400 mb-4">${SHAPDATA.subtitle}</div>${bars}
     <div class="text-[11px] text-slate-400 mt-3 leading-5">Mean |SHAP| on log-fuel. Physics (OpenAP) dominates, then interval duration, aircraft and mass.</div>`;
}

function setBusy(b){ ['k_dist','k_fuel','k_co2','k_time'].forEach(id=>document.getElementById(id).classList.toggle('pulse',b)); }
function showError(msg){
  document.getElementById('route-pill').innerHTML = `<span class="text-red-600">⚠ ${msg}</span>`;
  ['k_dist','k_fuel','k_co2','k_time'].forEach(id=>document.getElementById(id).textContent='—');
}

let _ctrl = null;
async function run(){
  const o=document.getElementById('origin').value, d=document.getElementById('dest').value;
  const ac=document.getElementById('ac').value, w=document.getElementById('wind').value;
  history.replaceState(null, '', `?o=${o}&d=${d}&ac=${ac}&w=${w}`);  // shareable URL (D1.7)
  if(o===d){ showError('Pick two different airports'); return; }
  if(_ctrl) _ctrl.abort();              // onceki istegi iptal et (yaris onleme)
  _ctrl = new AbortController();
  setBusy(true);
  let data;
  try{
    const resp = await fetch(`/api/score?o=${o}&d=${d}&ac=${ac}&headwind=${w}`, {signal:_ctrl.signal});
    data = await resp.json();
  }catch(e){
    if(e.name === 'AbortError') return;  // yenisi geldi, sessizce cik
    setBusy(false); showError('Request failed — is the server running?'); return;
  }
  setBusy(false);
  if(data.error){ showError('Invalid selection'); return; }
  render(data);
}

function render(data){
  const recs = data.scenarios;
  const rec = recs.find(s=>s.recommended) || recs[0];
  document.getElementById('k_dist').textContent = fmt(rec.dist_km)+' km';
  document.getElementById('k_fuel').innerHTML = fmt(rec.fuel_kg)+' kg' + (rec.fuel_lo
     ? `<div class="text-[11px] font-medium text-slate-400 mt-0.5">range ${fmt(rec.fuel_lo)}–${fmt(rec.fuel_hi)} kg <span class="text-slate-300">· mass band</span></div>` : '');
  document.getElementById('k_co2').textContent  = (rec.co2_kg/1000).toFixed(1)+' t';
  document.getElementById('k_time').textContent = tMin(rec.time_min);
  document.getElementById('route-pill').innerHTML =
    `<span class="text-slate-500">${code(data.origin)}</span> → <span class="text-slate-500">${code(data.dest)}</span> · <span class="text-blue-600">${data.aircraft}</span>`;

  // cards
  const order = ['Great-circle','Wind / fuel-optimal','Published ATC route'];
  recs.sort((a,b)=>order.indexOf(a.label)-order.indexOf(b.label));
  document.getElementById('cards').innerHTML = recs.map(s=>{
    const rr = s.recommended;
    return `<div class="card seg p-5 ${rr?'ring-rec':''}">
      <div class="flex items-center justify-between">
        <div class="font-bold text-[15px]">${s.label}</div>
        ${rr?'<span class="text-[11px] font-bold text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded-full">Recommended</span>':''}
      </div>
      <div class="text-xs text-slate-400 mt-0.5 mb-4">${s.subtitle}</div>
      ${rowHtml('Distance', fmt(s.dist_km)+' km')}
      ${rowHtml('Fuel', fmt(s.fuel_kg)+' kg', rr)}
      ${s.fuel_lo ? `<div class="text-[11px] text-slate-400 -mt-1 mb-1 text-right">range ${fmt(s.fuel_lo)}–${fmt(s.fuel_hi)} kg</div>` : ''}
      ${rowHtml('CO₂', (s.co2_kg/1000).toFixed(1)+' t')}
      ${rowHtml('Time', tMin(s.time_min))}
      ${rowHtml('Cruise', 'FL'+Math.round(s.cruise_ft/100))}
    </div>`;
  }).join('');

  drawMap(data, rec);
}
function rowHtml(k,v,strong){ return `<div class="flex items-center justify-between py-1.5 border-t border-slate-100 first:border-t-0">
   <span class="text-sm text-slate-500">${k}</span>
   <span class="text-sm ${strong?'font-extrabold text-blue-600':'font-semibold'} tabular-nums">${v}</span></div>`; }

function drawMap(data, rec){
  // remove old layers/sources
  ['rec-line','alt-line','pts','labels'].forEach(id=>{ if(map.getLayer(id)) map.removeLayer(id); });
  ['rec','alt','pts'].forEach(id=>{ if(map.getSource(id)) map.removeSource(id); });

  const others = data.scenarios.filter(s=>!s.recommended);
  map.addSource('alt',{type:'geojson',data:{type:'FeatureCollection',
     features: others.map(s=>({type:'Feature',geometry:{type:'LineString',coordinates:s.path}}))}});
  map.addLayer({id:'alt-line',type:'line',source:'alt',
     paint:{'line-color':'#94a3b8','line-width':1.5,'line-dasharray':[2,2],'line-opacity':.55}});

  map.addSource('rec',{type:'geojson',data:{type:'Feature',geometry:{type:'LineString',coordinates:rec.path}}});
  map.addLayer({id:'rec-line',type:'line',source:'rec',
     paint:{'line-color':'#2563eb','line-width':3,'line-dasharray':[1.6,1.4]}});

  const pts={type:'FeatureCollection',features:[
     {type:'Feature',properties:{icao:code(data.origin),kind:'o'},geometry:{type:'Point',coordinates:[data.origin.lon,data.origin.lat]}},
     {type:'Feature',properties:{icao:code(data.dest),kind:'d'},geometry:{type:'Point',coordinates:[data.dest.lon,data.dest.lat]}}]};
  map.addSource('pts',{type:'geojson',data:pts});
  map.addLayer({id:'pts',type:'circle',source:'pts',
     paint:{'circle-radius':6,'circle-color':['match',['get','kind'],'o','#10b981','#ef4444'],
            'circle-stroke-width':2,'circle-stroke-color':'#fff'}});
  map.addLayer({id:'labels',type:'symbol',source:'pts',
     layout:{'text-field':['get','icao'],'text-font':['Open Sans Bold'],'text-size':12,
             'text-offset':[0,-1.4],'text-anchor':'bottom'},
     paint:{'text-color':'#0f172a','text-halo-color':'#fff','text-halo-width':1.5}});

  // fit bounds — antimeridian-safe: boylamlari surekli hale getir (Pasifik rotalari icin)
  const p = rec.path;
  const lons=[p[0][0]], lats=[p[0][1]];
  for(let i=1;i<p.length;i++){
    let x=p[i][0]; const prev=lons[i-1];
    while(x-prev>180) x-=360; while(x-prev<-180) x+=360;
    lons.push(x); lats.push(p[i][1]);
  }
  map.fitBounds([[Math.min(...lons),Math.min(...lats)],[Math.max(...lons),Math.max(...lats)]],
                {padding:70,duration:600});
}
init();
</script>
</body></html>"""

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8600"))
    print(f"Route decision web app -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
