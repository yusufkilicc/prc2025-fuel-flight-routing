"""
Faz 4 — rota karar-destek motoru.
Hipotetik A->B rotalari icin telemetri YOK -> OpenAP fizigi omurga (Faz 0 durustluk ilkesi).
Bir rotayi great-circle segmentlerine bolup climb/cruise/descent profili kurar,
her segmenti OpenAP FuelFlow ile entegre eder -> yakit / CO2 / sure / ~maliyet.
Aday rotalar = farkli cruise irtifa profilleri (cok-kriter kiyas).
"""
import os
import numpy as np
from openap import FuelFlow, prop, aero

CO2_PER_KG_FUEL = 3.16   # jet-A1 yanma faktoru
LOAD_FACTOR = 0.62       # tek kaynak: takeoff kutle yuk faktoru (her iki arayuz ayni kullanir)
FUEL_PRICE_USD_KG = 0.9  # ~ kaba; degistirilebilir
TIME_COST_USD_MIN = 50.0 # ~ blok-saat maliyeti temsili

# uçak-sınıfı bazlı kalibrasyon (C2): OpenAP sapmasi govde tipine gore degisir
CLASS_CALIB = {"regional": 0.96, "narrow": 0.94, "wide": 0.90}
_WIDE = {"A332","A333","A338","A339","A359","A35K","A388","B772","B773","B77L","B77W",
         "B788","B789","B78X","A306","B742","B744","B748","B763","B764","MD11"}
_REGIONAL = {"E170","E175","E190","E195","CRJ7","CRJ9","AT72","DH8D"}

def _class_calib(t):
    t = t.upper()
    if t in _WIDE:
        return CLASS_CALIB["wide"]
    if t in _REGIONAL:
        return CLASS_CALIB["regional"]
    return CLASS_CALIB["narrow"]

FALLBACK = {"A306": "A332", "B77L": "B77W", "MD11": "B744", "B763": "B752"}
_SUP = set(a.upper() for a in prop.available_aircraft())
_cache = {}

def _ac(t):
    t = t.upper(); t = t if t in _SUP else FALLBACK.get(t, "A320")
    if t not in _cache:
        p = prop.aircraft(t)
        _cache[t] = dict(ff=FuelFlow(t, use_synonym=True),
                         oew=p["limits"]["OEW"], mtow=p["limits"]["MTOW"],
                         mach=p.get("cruise", {}).get("mach", 0.78),
                         crz_ft=p.get("cruise", {}).get("height", 11000) / aero.ft,
                         ceiling=p["limits"].get("ceiling", 12500) / aero.ft * aero.ft)  # m
    return _cache[t]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def great_circle(lat1, lon1, lat2, lon2, n=64):
    """n ara nokta (harita + mesafe icin)."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    d = 2*np.arcsin(np.sqrt(np.sin((lat2r-lat1r)/2)**2 +
        np.cos(lat1r)*np.cos(lat2r)*np.sin((lon2r-lon1r)/2)**2))
    if d == 0:
        return np.array([[lat1, lon1], [lat2, lon2]])
    f = np.linspace(0, 1, n)
    A = np.sin((1-f)*d)/np.sin(d); B = np.sin(f*d)/np.sin(d)
    x = A*np.cos(lat1r)*np.cos(lon1r) + B*np.cos(lat2r)*np.cos(lon2r)
    y = A*np.cos(lat1r)*np.sin(lon1r) + B*np.cos(lat2r)*np.sin(lon2r)
    z = A*np.sin(lat1r) + B*np.sin(lat2r)
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2+y**2)))
    lon = np.degrees(np.arctan2(y, x))
    return np.column_stack([lat, lon])

def _altitude_profile(dist_km, n, cruise_ft):
    """climb -> cruise -> descent irtifa profili (ft) ve dikey hiz (ft/min)."""
    climb_km = min(250.0, 0.30 * dist_km)
    desc_km = min(280.0, 0.30 * dist_km)
    cum = np.linspace(0, dist_km, n)
    alt = np.empty(n)
    for i, x in enumerate(cum):
        if x < climb_km:
            alt[i] = cruise_ft * (x / climb_km)
        elif x > dist_km - desc_km:
            alt[i] = cruise_ft * max(0.0, (dist_km - x) / desc_km)
        else:
            alt[i] = cruise_ft
    alt = np.clip(alt, 0, cruise_ft)
    return alt, cum

def deviate(pts, offset_km):
    """great-circle noktalarini orta noktada dik yonde kaydirir (cana-bell profil).
    Illustratif ATC/hava sapmasi -> haritada ayrik rota + dogal olarak daha uzun mesafe."""
    if offset_km == 0:
        return pts
    n = len(pts)
    f = np.linspace(0, 1, n)
    bell = np.sin(np.pi * f)  # uçlarda 0, ortada 1
    # her noktada yerel yon (bearing) -> dik offset
    lat = pts[:, 0].copy(); lon = pts[:, 1].copy()
    out = pts.copy().astype(float)
    for i in range(1, n - 1):
        coslat = max(0.2, np.cos(np.radians(lat[i])))  # boylam dereceleri ×cos(enlem)
        dlat = pts[i+1, 0] - pts[i-1, 0]
        dlon = (pts[i+1, 1] - pts[i-1, 1]) * coslat   # yon vektorunu metrik uzaya esitle
        norm = np.hypot(dlat, dlon) or 1.0
        # dik birim vektor (sol), metrik uzayda
        perp_lat, perp_lon = -dlon / norm, dlat / norm
        off_deg = offset_km / 111.0 * bell[i]
        out[i, 0] = lat[i] + perp_lat * off_deg
        out[i, 1] = lon[i] + perp_lon * off_deg / coslat  # boylama geri dönerken /cos(enlem)
    return out

def score_path(pts, ac_type, cruise_ft=None, load_factor=LOAD_FACTOR, headwind_kt=0.0,
               label=None, dist_factor=1.0, subtitle=None):
    """Verilen polyline (pts: Nx2 lat,lon) rotasini skorla.
    dist_factor: skorlanan mesafeyi olcekler (illustratif ATC/ruzgar rotalari icin);
    gosterilen path kozmetik kalir, fizik bu olcekli mesafeyle entegre edilir."""
    a = _ac(ac_type)
    calib = _class_calib(ac_type)  # uçak-sınıfı bazlı ampirik kalibrasyon (C2)
    if cruise_ft is None:
        cruise_ft = a["crz_ft"]
    cruise_ft = min(cruise_ft, a["ceiling"]/aero.ft)
    seg_km = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1]) * dist_factor
    dist_km = float(seg_km.sum())
    n = len(pts)
    alt, cum = _altitude_profile(dist_km, n, cruise_ft)
    alt_mid = (alt[:-1] + alt[1:]) / 2

    # --- segment geometrisi/hizi/suresi kutleden bagimsiz: bir kez hesapla ---
    am_arr = alt_mid
    frac_alt = am_arr / max(cruise_ft, 1)
    mach_arr = a["mach"] * (0.7 + 0.3 * frac_alt)
    tas_arr = np.array([aero.mach2tas(m, h * aero.ft) / aero.kts for m, h in zip(mach_arr, am_arr)])
    gs_arr = np.maximum(100.0, tas_arr - headwind_kt)
    segt_arr = seg_km / (gs_arr * 1.852) * 3600.0
    vs_arr = np.where(segt_arr > 0, (alt[1:] - alt[:-1]) / (segt_arr / 60.0), 0.0)
    time_total_s = float(segt_arr.sum())

    def _burn(m0):
        """Bir kalkis kutlesinden yola cikip yakit cikararak entegre et."""
        mass = m0; total = 0.0
        for k in range(len(seg_km)):
            try:
                ff = a["ff"].enroute(mass=mass, tas=tas_arr[k], alt=am_arr[k], vs=vs_arr[k])
            except Exception:
                ff = np.nan
            if not np.isfinite(ff) or ff < 0:
                ff = 0.0
            if vs_arr[k] < -200:      # iniste ~rolanti
                ff *= 0.28
            fkg = ff * segt_arr[k] * calib
            total += fkg
            mass = max(a["oew"], mass - fkg)
        return total

    # --- mesafe-duyarli kutle tahmini (2 gecis) ---
    # Gecis 1: nominal kutleyle trip yakitini tahmin et.
    fuel_est = _burn(a["oew"] + 0.50 * (a["mtow"] - a["oew"]))
    # payload = yuk faktoru × kalan tasima kapasitesi; m0 = OEW + payload + trip yakiti (MTOW ile sinirli)
    payload = load_factor * max(0.0, a["mtow"] - a["oew"] - fuel_est)
    m0 = min(a["mtow"], a["oew"] + payload + fuel_est)
    fuel_total = _burn(m0)

    co2 = fuel_total * CO2_PER_KG_FUEL
    cost = fuel_total * FUEL_PRICE_USD_KG + (time_total_s/60.0) * TIME_COST_USD_MIN
    return dict(aircraft=ac_type, label=label, subtitle=subtitle, cruise_ft=round(cruise_ft),
                dist_km=round(dist_km), fuel_kg=round(fuel_total), co2_kg=round(co2),
                time_min=round(time_total_s/60, 1), cost_usd=round(cost),
                path=pts)

def score_route(lat1, lon1, lat2, lon2, ac_type, cruise_ft=None, n=80, **kw):
    pts = great_circle(lat1, lon1, lat2, lon2, n)
    return score_path(pts, ac_type, cruise_ft=cruise_ft, **kw)

def compare_altitudes(lat1, lon1, lat2, lon2, ac_type, levels=(31000, 35000, 39000), **kw):
    rows = [score_route(lat1, lon1, lat2, lon2, ac_type, cruise_ft=fl, **kw) for fl in levels]
    best = min(rows, key=lambda r: r["fuel_kg"])
    for r in rows:
        r["is_best_fuel"] = (r["cruise_ft"] == best["cruise_ft"])
    return rows

def scenario_routes(lat1, lon1, lat2, lon2, ac_type, base_headwind_kt=0.0, n=96, load_factor=LOAD_FACTOR):
    """Referans tasarimin uc senaryosu (illustratif):
       1) Great-circle (en kisa)        — direkt, standart cruise
       2) Wind/fuel-optimal             — step-climb (+2000ft) + tailwind, hafif uzun
       3) Published ATC (hava sahasi)   — dogleg, +mesafe, biraz dusuk cruise
    """
    base = great_circle(lat1, lon1, lat2, lon2, n)
    a = _ac(ac_type); crz = min(a["crz_ft"], a["ceiling"]/aero.ft)
    specs = [
        dict(label="Great-circle", subtitle="Shortest theoretical path",
             pts=base, cruise_ft=crz, dist_factor=1.00, headwind_kt=base_headwind_kt),
        dict(label="Wind / fuel-optimal", subtitle="Tailwind + step-climb profile",
             pts=deviate(base, 320.0), cruise_ft=min(crz + 2000, a["ceiling"]/aero.ft),
             dist_factor=1.04, headwind_kt=base_headwind_kt - 28.0),
        dict(label="Published ATC route", subtitle="Airspace-constrained real track",
             pts=deviate(base, 720.0), cruise_ft=max(crz - 2000, 28000),
             dist_factor=1.083, headwind_kt=base_headwind_kt),
    ]
    rows = []
    for s in specs:
        r = score_path(s["pts"], ac_type, cruise_ft=s["cruise_ft"], load_factor=load_factor,
                       headwind_kt=s["headwind_kt"], label=s["label"],
                       subtitle=s["subtitle"], dist_factor=s["dist_factor"])
        # belirsizlik bandi (C3): kutle en buyuk surucudur -> yuk faktoru suvurmesi
        lo = score_path(s["pts"], ac_type, cruise_ft=s["cruise_ft"], load_factor=0.50,
                        headwind_kt=s["headwind_kt"], dist_factor=s["dist_factor"])["fuel_kg"]
        hi = score_path(s["pts"], ac_type, cruise_ft=s["cruise_ft"], load_factor=0.72,
                        headwind_kt=s["headwind_kt"], dist_factor=s["dist_factor"])["fuel_kg"]
        r["fuel_lo"], r["fuel_hi"] = int(min(lo, hi)), int(max(lo, hi))
        rows.append(r)
    best = min(rows, key=lambda r: r["fuel_kg"])
    for r in rows:
        r["recommended"] = (r["label"] == best["label"])
    return rows

def compare_routes(lat1, lon1, lat2, lon2, ac_type, cruise_ft=None, n=80, **kw):
    """Lateral alternatifler: direkt great-circle + kuzey/guney illustratif sapma.
    Her biri en iyi (tipik) cruise irtifasinda skorlanir. Haritada ayrik gorunur."""
    base = great_circle(lat1, lon1, lat2, lon2, n)
    options = [("Direkt (great-circle)", 0.0), ("Kuzey sapma", 220.0), ("Guney sapma", -220.0)]
    rows = []
    for name, off in options:
        pts = deviate(base, off)
        rows.append(score_path(pts, ac_type, cruise_ft=cruise_ft, label=name, **kw))
    best = min(rows, key=lambda r: r["fuel_kg"])
    for r in rows:
        r["is_best_fuel"] = (r["label"] == best["label"])
    return rows


if __name__ == "__main__":
    import sys, io, pandas as pd
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    apt = pd.read_parquet(os.path.join(os.environ.get("PRC_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data"),"apt.parquet")).set_index("icao")
    tests = [("EHAM", "KJFK", "A359"), ("MSLP", "KIAD", "A20N"), ("WMKK", "EHAM", "B789")]
    for o, d, ac in tests:
        if o not in apt.index or d not in apt.index:
            continue
        o_, d_ = apt.loc[o], apt.loc[d]
        print(f"\n=== {o} -> {d}  ({ac}) ===")
        rows = compare_altitudes(o_.latitude, o_.longitude, d_.latitude, d_.longitude, ac)
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "path"} for r in rows])
        print(df.to_string(index=False))
