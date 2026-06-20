"""
Faz 2 — OpenAP fizik bazli yakit tahmini + kutle propagasyonu.
features_<split>.parquet uzerinde calisir (trajektori yeniden okunmaz).
Cikti: ayni tabloya 'fuel_openap' ve 'mass_est' kolonlari eklenir.

Mantik (per ucus, interval'lar idx sirasinda):
  m0 (takeoff) = OEW + 0.75*(MTOW-OEW)
  her interval: tas <- mach (varsa) / gs / tipik cruise mach
                vs  <- vr_mean (karanlik cruise'da 0)
                fuel = FuelFlow.enroute(mass, tas, alt, vs) * dur_s
                mass propagasyonu: bir sonraki interval'in kutlesi = mass - fuel
ML bu fizik bazinin ARTIGINI ogrenecek (Faz 2 train).
"""
import os, sys, io, warnings, math
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")

from openap import FuelFlow, prop, aero

# desteklenmeyen tipleri en yakina esle
FALLBACK = {"A306": "A332", "B77L": "B77W", "MD11": "B744", "B763": "B752"}
_SUPPORTED = set(a.upper() for a in prop.available_aircraft())

_FF, _LIM = {}, {}
def get_ff(t):
    t = t.upper()
    t = t if t in _SUPPORTED else FALLBACK.get(t, "A320")
    if t not in _FF:
        _FF[t] = FuelFlow(t, use_synonym=True)
        p = prop.aircraft(t)
        cr = p.get("cruise", {})
        crz_ft = cr.get("height", 11000) / aero.ft   # m -> ft
        _LIM[t] = (p["limits"]["OEW"], p["limits"]["MTOW"],
                   cr.get("mach", 0.78), crz_ft)
    return _FF[t], _LIM[t]

KTS = aero.kts  # m/s per kt

def interval_fuel(ff, mass, alt, vr, mach, gs, cruise_mach, dur, crz_ft=None, frac=None):
    imputed = 0
    if not np.isfinite(alt):
        # irtifa yok: ucus ortasiysa cruise irtifasi empoze et (Faz 2 kapsam genisletme)
        if crz_ft is not None and frac is not None and 0.1 < frac < 0.9:
            alt = crz_ft
            if not np.isfinite(vr):
                vr = 0.0
            if not np.isfinite(mach) and not np.isfinite(gs):
                mach = cruise_mach
            imputed = 1
        else:
            return np.nan, 0
    # TAS (kt)
    if np.isfinite(mach):
        tas = aero.mach2tas(mach, alt * aero.ft) / KTS
    elif np.isfinite(gs):
        tas = gs
    else:
        tas = aero.mach2tas(cruise_mach, alt * aero.ft) / KTS
    vs = vr if np.isfinite(vr) else 0.0
    try:
        ff_kgps = ff.enroute(mass=mass, tas=tas, alt=alt, vs=vs)
    except Exception:
        return np.nan, imputed
    if not np.isfinite(ff_kgps) or ff_kgps < 0:
        return np.nan, imputed
    return float(ff_kgps) * dur, imputed

def process_split(split, sample=None):
    d = pd.read_parquet(os.path.join(ROOT, f"features_{split}.parquet"))
    d = d.sort_values(["flight_id", "idx"]).reset_index(drop=True)
    if sample:
        keep = pd.Series(d["flight_id"].unique()).sample(sample, random_state=1)
        d = d[d["flight_id"].isin(keep)].reset_index(drop=True)

    fuel_op = np.full(len(d), np.nan)
    mass_est = np.full(len(d), np.nan)
    op_imputed = np.zeros(len(d), dtype=int)

    for fid, g in d.groupby("flight_id", sort=False):
        at = g["aircraft_type"].iloc[0]
        if at is None or (isinstance(at, float) and math.isnan(at)):
            continue
        ff, (oew, mtow, cm, crz) = get_ff(str(at))
        mass = oew + 0.75 * (mtow - oew)
        for i in g.index:
            alt = d.at[i, "alt_mean"]; vr = d.at[i, "vr_mean"]
            mach = d.at[i, "mach_mean"]; gs = d.at[i, "gs_mean"]
            dur = d.at[i, "dur_s"]; frac = d.at[i, "frac_elapsed"]
            fkg, imp = interval_fuel(ff, mass, alt, vr, mach, gs, cm, dur, crz_ft=crz, frac=frac)
            mass_est[i] = mass
            fuel_op[i] = fkg
            op_imputed[i] = imp
            if np.isfinite(fkg):
                mass = max(oew, mass - fkg)   # propagasyon

    d["fuel_openap"] = fuel_op
    d["mass_est"] = mass_est
    d["openap_imputed"] = op_imputed
    return d

def evaluate(d):
    m = d["fuel_openap"].notna() & d["fuel_kg"].notna()
    a = d.loc[m, "fuel_kg"].to_numpy(); p = d.loc[m, "fuel_openap"].to_numpy()
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(a, p))
    corr = np.corrcoef(np.log(a.clip(1)), np.log(p.clip(1)))[0, 1]
    ratio = p / a
    print(f"  kapsanan interval: {m.sum():,} / {len(d):,}  (openap NaN: {(~d['fuel_openap'].notna()).sum():,})")
    print(f"  OpenAP ham RMSE(kg) = {rmse:8.1f}   log-korelasyon = {corr:.3f}")
    print(f"  oran (openap/gercek): medyan={np.median(ratio):.2f}  p25={np.percentile(ratio,25):.2f}  p75={np.percentile(ratio,75):.2f}")
    print("  -> korelasyon yuksek + oran ~sabit ise ML residual'i kolay duzeltir")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("ORNEKLEM (500 ucus) OpenAP fizik bazi vs gercek:")
        d = process_split("train", sample=500)
        evaluate(d)
    else:
        split = sys.argv[1] if len(sys.argv) > 1 else "train"
        d = process_split(split)
        out = os.path.join(ROOT, f"features_{split}_openap.parquet")
        d.to_parquet(out)
        print(f"[{split}] -> {out}  shape={d.shape}")
        if "fuel_kg" in d and d["fuel_kg"].notna().any():
            evaluate(d)
