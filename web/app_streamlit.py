"""
Faz 5 — Rota karar-destek arayuzu (Streamlit + pydeck).
Calistir:  streamlit run app.py
A->B + ucak tipi sec -> dunya haritasinda aday rotalar + yakit/CO2/sure/maliyet kiyasi.
Skorlama motoru: OpenAP fizigi (hipotetik rotada telemetri yok).
"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")
sys.path.insert(0, os.path.join(_PKG, "src"))
import route_decision as rd
st.set_page_config(page_title="Yakit & Emisyon Farkindali Rota Karsilastirma", layout="wide")

@st.cache_data
def load_airports():
    apt = pd.read_parquet(os.path.join(ROOT, "apt.parquet"))
    fl = pd.read_parquet(os.path.join(ROOT, "flightlist_train.parquet"))
    used = pd.unique(pd.concat([fl["origin_icao"], fl["destination_icao"]]))
    names = (pd.concat([fl[["origin_icao", "origin_name"]].rename(columns={"origin_icao": "icao", "origin_name": "name"}),
                        fl[["destination_icao", "destination_name"]].rename(columns={"destination_icao": "icao", "destination_name": "name"})])
             .drop_duplicates("icao").set_index("icao")["name"])
    apt = apt[apt["icao"].isin(used)].copy()
    apt["name"] = apt["icao"].map(names).fillna(apt["icao"])
    apt["label"] = apt["icao"] + " — " + apt["name"].astype(str)
    return apt.sort_values("label").reset_index(drop=True), sorted(fl["aircraft_type"].dropna().unique())

apt, ac_types = load_airports()
icao_label = dict(zip(apt["icao"], apt["label"]))
labels = apt["label"].tolist()

st.title("✈️ Yakit & Emisyon Farkindali Rota Karsilastirma")
st.caption("PRC 2025 verisiyle egitilen fizik (OpenAP) + ML hibrit yaklasiminin karar katmani — "
           "iki nokta arasi aday rotalari yakit / CO₂ / sure / maliyet acisindan kiyaslar.")

with st.sidebar:
    st.header("Ucus secimi")
    def_o = labels.index(icao_label.get("EHAM", labels[0])) if "EHAM" in icao_label else 0
    def_d = labels.index(icao_label.get("KJFK", labels[1])) if "KJFK" in icao_label else 1
    o_lab = st.selectbox("Kalkis", labels, index=def_o)
    d_lab = st.selectbox("Varis", labels, index=def_d)
    ac = st.selectbox("Ucak tipi", ac_types, index=(ac_types.index("A359") if "A359" in ac_types else 0))
    mode = st.radio("Kiyas turu", ["Cruise irtifasi", "Lateral rota"], index=0)
    headwind = st.slider("Karsi ruzgar (kt)", -80, 80, 0, 10)
    st.markdown("**Kriter agirliklari**")
    w_fuel = st.slider("Yakit", 0.0, 1.0, 0.5, 0.1)
    w_time = st.slider("Sure", 0.0, 1.0, 0.3, 0.1)
    w_cost = st.slider("Maliyet", 0.0, 1.0, 0.2, 0.1)

o_icao = o_lab.split(" — ")[0]; d_icao = d_lab.split(" — ")[0]
o = apt[apt["icao"] == o_icao].iloc[0]; d = apt[apt["icao"] == d_icao].iloc[0]

if o_icao == d_icao:
    st.warning("Kalkis ve varis ayni. Farkli havalimanlari secin.")
    st.stop()

if mode == "Cruise irtifasi":
    rows = rd.compare_altitudes(o.latitude, o.longitude, d.latitude, d.longitude, ac, headwind_kt=headwind)
    key = "cruise_ft"; key_label = lambda r: f"FL{int(r['cruise_ft']/100)}"
else:
    rows = rd.compare_routes(o.latitude, o.longitude, d.latitude, d.longitude, ac, headwind_kt=headwind)
    key = "label"; key_label = lambda r: r["label"]

# cok-kriter skor (normalize + agirlikli; dusuk = iyi)
def mcda(rows):
    arr = {k: np.array([r[k] for r in rows], float) for k in ["fuel_kg", "time_min", "cost_usd"]}
    norm = {k: (v - v.min()) / (v.max() - v.min() + 1e-9) for k, v in arr.items()}
    wsum = w_fuel + w_time + w_cost + 1e-9
    score = (w_fuel*norm["fuel_kg"] + w_time*norm["time_min"] + w_cost*norm["cost_usd"]) / wsum
    return score
scores = mcda(rows)
best_i = int(np.argmin(scores))
for i, r in enumerate(rows):
    r["mcda"] = round(float(scores[i]), 3); r["secim"] = "✅ ONERILEN" if i == best_i else ""

# --- harita ---
COLORS = [[0, 170, 90], [30, 110, 220], [220, 90, 60], [150, 80, 200]]
arc_df = []
for i, r in enumerate(rows):
    p = r["path"]
    col = [0, 200, 100] if i == best_i else COLORS[(i + 1) % len(COLORS)]
    for j in range(len(p) - 1):
        arc_df.append({"s_lat": p[j][0], "s_lon": p[j][1], "t_lat": p[j+1][0], "t_lon": p[j+1][1],
                       "r": col[0], "g": col[1], "b": col[2], "w": 5 if i == best_i else 2})
arc_df = pd.DataFrame(arc_df)
pts_df = pd.DataFrame([{"lat": o.latitude, "lon": o.longitude, "name": o_icao},
                       {"lat": d.latitude, "lon": d.longitude, "name": d_icao}])

layers = [
    pdk.Layer("LineLayer", arc_df, get_source_position=["s_lon", "s_lat"],
              get_target_position=["t_lon", "t_lat"], get_color=["r", "g", "b"],
              get_width="w", width_min_pixels=2),
    pdk.Layer("ScatterplotLayer", pts_df, get_position=["lon", "lat"],
              get_fill_color=[255, 200, 0], get_radius=60000, pickable=True),
    pdk.Layer("TextLayer", pts_df, get_position=["lon", "lat"], get_text="name",
              get_size=16, get_color=[255, 255, 255], get_pixel_offset=[0, -18]),
]
mid_lat = (o.latitude + d.latitude) / 2; mid_lon = (o.longitude + d.longitude) / 2
view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=2.2, pitch=30)

c1, c2 = st.columns([3, 2])
with c1:
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view,
                             map_style="dark", tooltip={"text": "{name}"}))
with c2:
    best = rows[best_i]
    st.subheader(f"Onerilen: {key_label(best)}")
    m1, m2 = st.columns(2)
    m1.metric("Yakit", f"{best['fuel_kg']:,} kg")
    m2.metric("CO₂", f"{best['co2_kg']:,} kg")
    m1.metric("Sure", f"{best['time_min']:.0f} dk")
    m2.metric("Maliyet", f"${best['cost_usd']:,}")
    st.caption(f"Mesafe ~{best['dist_km']:,} km · {ac}")

st.markdown("### Aday rota karsilastirmasi")
tbl = pd.DataFrame([{"Secenek": key_label(r), "Mesafe (km)": r["dist_km"], "Yakit (kg)": r["fuel_kg"],
                    "CO₂ (kg)": r["co2_kg"], "Sure (dk)": r["time_min"], "Maliyet ($)": r["cost_usd"],
                    "MCDA": r["mcda"], "": r["secim"]} for r in rows])
st.dataframe(tbl, use_container_width=True, hide_index=True)
st.bar_chart(tbl.set_index("Secenek")[["Yakit (kg)"]])

with st.expander("⚠️ Model sinirlari (durustluk notu)"):
    st.markdown(
        "- Skorlar **OpenAP fizik modeli** ile hesaplanir; hipotetik (henuz uculmamis) rotada gercek telemetri yoktur.\n"
        "- Kutle **tahminidir** (yuk faktoru 0.75 varsayimi) — yakitin en buyuk surucusu, gercek operasyonel yukle degisir.\n"
        "- Lateral 'sapma' rotalari **illustratiftir** (gercek ATC/ruzgar-optimal routing degil).\n"
        "- PRC verisiyle dogrulanan ML modeli *uculmus* trajektorilerde yakit tahmin eder; bu arac onun **ekstrapolasyonudur**.\n"
        "- Egitim dagilimindan (orta-uzun menzil ticari jetler) uzak secimlerde guven duser.")
