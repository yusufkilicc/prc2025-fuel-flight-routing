"""
Faz 0 EDA — PRC Data Challenge 2025
Hedef: veri butunlugu, interval<->ADS-B eslesme, hedef dagilimi, ucus fazi ayrismasi.
Buyuk veri (11k dosya) oldugu icin per-interval analizler ORNEKLEM uzerinden yapilir.
"""
import os, glob, sys, io, random
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
random.seed(42)
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")
TRAJ_DIR = os.path.join(ROOT, "flights_train", "flights_train")


def section(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------------------------------------------------------------------
section("1. VERI BUTUNLUGU: trajektori dosyasi her flight_id icin var mi?")
fuel = pd.read_parquet(os.path.join(ROOT, "fuel_train.parquet"))
fl = pd.read_parquet(os.path.join(ROOT, "flightlist_train.parquet"))
fuel_flights = fuel["flight_id"].unique()
fl_flights = fl["flight_id"].unique()
print(f"fuel_train interval sayisi      : {len(fuel):,}")
print(f"fuel_train benzersiz ucus        : {len(fuel_flights):,}")
print(f"flightlist_train ucus            : {len(fl_flights):,}")

traj_files = set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(TRAJ_DIR, "*.parquet")))
print(f"trajektori dosyasi (disk)        : {len(traj_files):,}")
missing_traj = [f for f in fuel_flights if f not in traj_files]
print(f"fuel'de olup trajektorisi EKSIK  : {len(missing_traj)}")
no_meta = set(fuel_flights) - set(fl_flights)
print(f"fuel'de olup flightlist'te yok   : {len(no_meta)}")

# ---------------------------------------------------------------------------
section("2. HEDEF DAGILIMI: fuel_kg")
fk = fuel["fuel_kg"].astype(float)
print(fk.describe(percentiles=[.01, .25, .5, .75, .95, .99]).to_string())
print(f"\nfuel_kg == 0            : {(fk == 0).sum()}")
print(f"fuel_kg < 1 kg          : {(fk < 1).sum()}")
print(f"fuel_kg > 20.000 kg     : {(fk > 20000).sum()}  (tek interval icin supheli)")
# log-normal mi?
pos = fk[fk > 0]
from scipy import stats
sk_raw = stats.skew(pos)
sk_log = stats.skew(np.log(pos))
print(f"carpiklik (skew) ham    : {sk_raw:.2f}")
print(f"carpiklik (skew) log    : {sk_log:.2f}   -> 0'a yakinsa log-normal")

fuel["dur_s"] = (fuel["end"] - fuel["start"]).dt.total_seconds()
print(f"\ninterval suresi (s): mean={fuel['dur_s'].mean():.0f} median={fuel['dur_s'].median():.0f} "
      f"min={fuel['dur_s'].min():.1f} max={fuel['dur_s'].max():.0f}")
print(f"interval basina yakit/sn (kg/s): {(fk / fuel['dur_s']).describe(percentiles=[.5,.95]).to_string()}")

# ---------------------------------------------------------------------------
section("3. ORNEKLEM: interval <-> ADS-B nokta eslesmesi + faz ayrismasi")
SAMPLE_N = 400
sample_flights = random.sample(list(set(fuel_flights) & traj_files), SAMPLE_N)

rows = []          # interval-level kayitlar
zero_pt = 0
total_int = 0
for fid in sample_flights:
    tdf = pd.read_parquet(os.path.join(TRAJ_DIR, f"{fid}.parquet"),
                          columns=["timestamp", "altitude", "vertical_rate", "groundspeed", "source"])
    adsb = tdf[tdf["source"] == "adsb"].sort_values("timestamp")
    ints = fuel[fuel["flight_id"] == fid]
    for _, r in ints.iterrows():
        total_int += 1
        m = (adsb["timestamp"] >= r["start"]) & (adsb["timestamp"] < r["end"])
        seg = adsb[m]
        npts = len(seg)
        if npts == 0:
            zero_pt += 1
        alt_mean = seg["altitude"].mean() if npts else np.nan
        vr_mean = seg["vertical_rate"].mean() if npts else np.nan
        # faz: dikey hiza gore
        if npts == 0 or np.isnan(vr_mean):
            phase = "unknown"
        elif vr_mean > 300:
            phase = "climb"
        elif vr_mean < -300:
            phase = "descent"
        else:
            phase = "cruise"
        rows.append(dict(flight_id=fid, fuel_kg=float(r["fuel_kg"]), dur_s=(r["end"]-r["start"]).total_seconds(),
                         npts=npts, alt_mean=alt_mean, vr_mean=vr_mean, phase=phase))

s = pd.DataFrame(rows)
print(f"orneklem ucus      : {SAMPLE_N}")
print(f"orneklem interval  : {total_int:,}")
print(f"0 ADS-B noktali interval: {zero_pt}  (%{100*zero_pt/total_int:.1f})  -> OpenAP bazi burada kurtarir")
print("\ninterval basina ADS-B nokta sayisi:")
print(s["npts"].describe(percentiles=[.05, .5, .95]).to_string())

# ---------------------------------------------------------------------------
section("4. FAZ BAZLI YAKIT AYRISMASI (climb cok daha fazla yakar mi?)")
ph = s[s["npts"] > 0].groupby("phase").agg(
    n=("fuel_kg", "size"),
    fuel_mean=("fuel_kg", "mean"),
    fuel_median=("fuel_kg", "median"),
    kg_per_s_mean=("fuel_kg", lambda x: (x / s.loc[x.index, "dur_s"]).mean()),
    alt_mean=("alt_mean", "mean"),
).round(2)
print(ph.to_string())

# ---------------------------------------------------------------------------
section("5. UCAK TIPI x YAKIT (orneklem ucuslar)")
meta = fl.set_index("flight_id")["aircraft_type"]
s["aircraft_type"] = s["flight_id"].map(meta)
at = s.groupby("aircraft_type").agg(
    n_interval=("fuel_kg", "size"),
    fuel_mean=("fuel_kg", "mean"),
    kg_per_s=("fuel_kg", lambda x: (x / s.loc[x.index, "dur_s"]).mean()),
).round(2).sort_values("n_interval", ascending=False).head(12)
print(at.to_string())

# ---------------------------------------------------------------------------
section("6. OZET BULGULAR")
print(f"- Eksik trajektori: {len(missing_traj)} | eksik meta: {len(no_meta)}")
print(f"- Hedef log-normal'a yakin mi? log-skew={sk_log:.2f} (ham={sk_raw:.2f}) -> log-hedef egitimi dene")
print(f"- 0-ADS-B interval orani: %{100*zero_pt/total_int:.1f} -> hibrit (OpenAP) gerekli")
print(f"- Faz ayrismasi yukaridaki tabloda (climb kg/s en yuksek beklenir)")
s.to_parquet(os.path.join(ROOT, "_eda_sample.parquet"))
print("\norneklem -> _eda_sample.parquet kaydedildi")
