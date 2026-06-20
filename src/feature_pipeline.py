"""
Faz 1 feature pipeline — interval basina tek satir 'set A' (zengin/gozlenmis) feature tablosu.
Faz 0 bulgularina gore: ADS-B binleme + ACARS mach + gap'lerde irtifa interpolasyonu fuzyonu.
Cok-surecli; her ucus dosyasi bir kez okunur.

Kullanim:
    python feature_pipeline.py train   -> features_train.parquet (label'li)
    python feature_pipeline.py rank    -> features_rank.parquet  (label'siz, submission icin)
    python feature_pipeline.py final   -> features_final.parquet
"""
import os, sys, glob, math
import numpy as np
import pandas as pd
from multiprocessing import Pool

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")
SPLITS = {
    "train": ("fuel_train.parquet",            "flights_train/flights_train", "flightlist_train.parquet"),
    "rank":  ("fuel_rank_submission.parquet",  "flights_rank/flights_rank",   "flightlist_rank.parquet"),
    "final": ("fuel_final_submission.parquet", "flights_final/flights_final", "flightlist_final.parquet"),
}

# apt: ICAO -> (lon, lat) ; flightlist meta global yuklenir (worker'larda)
_APT = None
_META = None
_TRAJ_DIR = None
_FUEL = None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _safe(v):
    return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else np.nan


def process_flight(fid):
    """Bir ucusun tum interval'lari icin feature satirlari dondurur."""
    try:
        tdf = pd.read_parquet(os.path.join(_TRAJ_DIR, f"{fid}.parquet"))
    except Exception:
        return []
    tdf = tdf.sort_values("timestamp")
    adsb = tdf[tdf["source"] == "adsb"]
    acars = tdf[tdf["source"] == "acars"]

    ac_type = _META.get(fid, {}).get("aircraft_type", None)
    origin = _META.get(fid, {}).get("origin_icao", None)
    dest = _META.get(fid, {}).get("destination_icao", None)
    # origin->dest great-circle
    gc_od = np.nan
    if origin in _APT and dest in _APT:
        o, d = _APT[origin], _APT[dest]
        gc_od = haversine_km(o[1], o[0], d[1], d[0])

    t0 = tdf["timestamp"].min()
    t1 = tdf["timestamp"].max()
    span = (t1 - t0).total_seconds() or 1.0

    ints = _FUEL[_FUEL["flight_id"] == fid]
    n_int = len(ints)
    out = []
    for _, r in ints.iterrows():
        start, end = r["start"], r["end"]
        dur = (end - start).total_seconds()
        seg = adsb[(adsb["timestamp"] >= start) & (adsb["timestamp"] < end)]
        seg_ac = acars[(acars["timestamp"] >= start) & (acars["timestamp"] < end)]
        n_ad, n_ac = len(seg), len(seg_ac)

        f = {
            "flight_id": fid, "idx": int(r["idx"]),
            "dur_s": dur, "n_adsb": n_ad, "n_acars": n_ac, "has_adsb": int(n_ad > 0),
            "aircraft_type": ac_type,
            "gc_dist_od_km": gc_od,
            "flight_total_s": span,
            "n_intervals_flight": n_int,
            "frac_elapsed": ((start - t0).total_seconds() + dur / 2) / span,
            "t_since_takeoff_s": (start - t0).total_seconds(),
        }

        # --- irtifa ---
        if n_ad > 0:
            alt = seg["altitude"].to_numpy()
            f["alt_mean"] = np.nanmean(alt); f["alt_max"] = np.nanmax(alt); f["alt_min"] = np.nanmin(alt)
            f["alt_start"] = _safe(alt[0]); f["alt_end"] = _safe(alt[-1])
            f["alt_delta"] = f["alt_end"] - f["alt_start"]
            vr = seg["vertical_rate"].to_numpy()
            f["vr_mean"] = np.nanmean(vr); f["vr_std"] = np.nanstd(vr); f["vr_absmean"] = np.nanmean(np.abs(vr))
            gs = seg["groundspeed"].to_numpy()
            f["gs_mean"] = np.nanmean(gs); f["gs_std"] = np.nanstd(gs)
            # interval ici kat edilen mesafe (ardisik great-circle)
            lat = seg["latitude"].to_numpy(); lon = seg["longitude"].to_numpy()
            if n_ad > 1:
                dseg = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
                f["dist_flown_km"] = float(np.nansum(dseg))
            else:
                f["dist_flown_km"] = np.nan
            f["alt_interp"] = 0
        else:
            # GAP: komsu ADS-B'den irtifa interpolasyonu (Faz 0: %59 mumkun)
            before = adsb[adsb["timestamp"] < start].tail(1)
            after = adsb[adsb["timestamp"] >= end].head(1)
            if len(before) and len(after):
                a0 = before["altitude"].values[0]; a1 = after["altitude"].values[0]
                f["alt_mean"] = (a0 + a1) / 2; f["alt_start"] = a0; f["alt_end"] = a1
                f["alt_delta"] = a1 - a0; f["alt_max"] = max(a0, a1); f["alt_min"] = min(a0, a1)
                # gap mesafesi: sinir noktalari arasi great-circle (oran)
                f["dist_flown_km"] = haversine_km(before["latitude"].values[0], before["longitude"].values[0],
                                                  after["latitude"].values[0], after["longitude"].values[0])
                f["alt_interp"] = 1
            else:
                for k in ["alt_mean", "alt_start", "alt_end", "alt_delta", "alt_max", "alt_min", "dist_flown_km"]:
                    f[k] = np.nan
                f["alt_interp"] = 0
            f["vr_mean"] = f["vr_std"] = f["vr_absmean"] = np.nan
            f["gs_mean"] = f["gs_std"] = np.nan

        # --- mach (ACARS + adsb mach kolonu) ---
        mvals = pd.concat([seg["mach"], seg_ac["mach"]]).dropna()
        f["mach_mean"] = float(mvals.mean()) if len(mvals) else np.nan

        # --- faz ---
        vr_m = f["vr_mean"]
        if pd.isna(vr_m):
            f["phase"] = "unknown"
        elif vr_m > 300:
            f["phase"] = "climb"
        elif vr_m < -300:
            f["phase"] = "descent"
        else:
            f["phase"] = "cruise"

        if "fuel_kg" in r and r["fuel_kg"] is not None:
            try:
                f["fuel_kg"] = float(r["fuel_kg"])
            except (TypeError, ValueError):
                f["fuel_kg"] = np.nan
        out.append(f)
    return out


def _init(traj_dir, meta, apt, fuel):
    global _TRAJ_DIR, _META, _APT, _FUEL
    _TRAJ_DIR, _META, _APT, _FUEL = traj_dir, meta, apt, fuel


def main(split):
    fuel_file, traj_dir, fl_file = SPLITS[split]
    traj_dir = os.path.join(ROOT, traj_dir)
    fuel = pd.read_parquet(os.path.join(ROOT, fuel_file))
    fl = pd.read_parquet(os.path.join(ROOT, fl_file))
    apt_df = pd.read_parquet(os.path.join(ROOT, "apt.parquet"))
    apt = {r.icao: (r.longitude, r.latitude) for r in apt_df.itertuples()}
    meta = fl.set_index("flight_id")[["aircraft_type", "origin_icao", "destination_icao"]].to_dict("index")

    flights = fuel["flight_id"].unique().tolist()
    print(f"[{split}] {len(flights):,} ucus, {len(fuel):,} interval islenecek...")

    results = []
    with Pool(processes=(os.cpu_count() or 4), initializer=_init, initargs=(traj_dir, meta, apt, fuel)) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_flight, flights, chunksize=32)):
            results.extend(rows)
            if (i + 1) % 2000 == 0:
                print(f"  ... {i+1:,}/{len(flights):,} ucus")

    df = pd.DataFrame(results)
    df["aircraft_type"] = df["aircraft_type"].astype("category")
    df["phase"] = df["phase"].astype("category")
    out_path = os.path.join(ROOT, f"features_{split}.parquet")
    df.to_parquet(out_path)
    print(f"[{split}] -> {out_path}  shape={df.shape}")
    print(df[["dur_s", "n_adsb", "alt_mean", "mach_mean", "dist_flown_km", "frac_elapsed"]].describe().round(2).to_string())


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(split)
