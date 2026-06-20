"""
Submission uretici — HIBRIT model (OpenAP + LightGBM, feature-augmented).
Tum train'de egit -> hedef split'i tahmin et -> orijinal sema ile yaz.

Onkosul (her split icin sirayla):
    python src/feature_pipeline.py {train,<split>}
    python src/openap_baseline.py  {train,<split>}     # features_*_openap.parquet uretir

Kullanim: python src/submit.py rank   |   python src/submit.py final
"""
import os, sys, io, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")
split = sys.argv[1] if len(sys.argv) > 1 else "rank"

TR_PATH = os.path.join(ROOT, "features_train_openap.parquet")
TE_PATH = os.path.join(ROOT, f"features_{split}_openap.parquet")
for p, hint in [(TR_PATH, "openap_baseline.py train"), (TE_PATH, f"openap_baseline.py {split}")]:
    if not os.path.exists(p):
        sys.exit(f"[HATA] {os.path.basename(p)} yok. Once calistir:  python src/{hint}")

CAT = ["aircraft_type", "phase"]
DROP = ["flight_id", "idx", "fuel_kg", "fuel_openap"]   # log_openap + mass_est feature olarak kalir

tr = pd.read_parquet(TR_PATH)
te = pd.read_parquet(TE_PATH)
for df in (tr, te):
    df["log_openap"] = np.log(df["fuel_openap"].clip(lower=0.1))

features = [c for c in tr.columns if c not in DROP]
for c in CAT:
    tr[c] = tr[c].astype("category")
    te[c] = pd.Categorical(te[c], categories=tr[c].cat.categories)
for c in features:                      # test'te eksik feature varsa hizala
    if c not in te.columns:
        te[c] = np.nan

params = dict(objective="regression", metric="rmse", n_estimators=600, learning_rate=0.03,
              num_leaves=128, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1)
m = lgb.LGBMRegressor(**params)
m.fit(tr[features], np.log(tr["fuel_kg"]), categorical_feature=CAT)
pred = np.clip(np.exp(m.predict(te[features])), 0.1, None)

# orijinal submission semasi ile birlestir (idx hizasi)
sub = pd.read_parquet(os.path.join(ROOT, f"fuel_{split}_submission.parquet"))
pmap = dict(zip(zip(te["flight_id"], te["idx"]), pred))
sub["fuel_kg"] = [pmap.get((f, i), np.nan) for f, i in zip(sub["flight_id"], sub["idx"])]

out = os.path.join(ROOT, f"submission_{split}_hybrid.parquet")
sub.to_parquet(out)
print(f"[{split}] HIBRIT submission -> {out}")
print(f"  satir={len(sub):,}  bos tahmin={sub['fuel_kg'].isna().sum()}  (0 olmali)")
print(f"  tahmin fuel_kg: mean={sub['fuel_kg'].mean():.1f} median={sub['fuel_kg'].median():.1f} "
      f"min={sub['fuel_kg'].min():.1f} max={sub['fuel_kg'].max():.1f}")
print(f"  sema: {list(sub.columns)}  dtype={sub['fuel_kg'].dtype}")
