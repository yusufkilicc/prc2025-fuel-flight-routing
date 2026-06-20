"""
Faz 1 baseline — LightGBM, log-hedef, GroupKFold(flight_id).
OpenAP YOK (o Faz 2). Amac: gercekci zemin skoru + segment bazli (ADS-B'li vs karanlik) hata.
"""
import os, sys, io, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")

d = pd.read_parquet(os.path.join(ROOT, "features_train.parquet"))
print(f"veri: {d.shape}")

DROP = ["flight_id", "idx", "fuel_kg"]
CAT = ["aircraft_type", "phase"]
features = [c for c in d.columns if c not in DROP]
for c in CAT:
    d[c] = d[c].astype("category")

X = d[features]
y = np.log(d["fuel_kg"].to_numpy())          # log-hedef (Faz 0: log-normal)
groups = d["flight_id"]

params = dict(
    objective="regression", metric="rmse", n_estimators=2000, learning_rate=0.03,
    num_leaves=128, min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1,
)

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(d))
imp = np.zeros(len(features))
for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
    m = lgb.LGBMRegressor(**params)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
          categorical_feature=CAT, callbacks=[lgb.early_stopping(80, verbose=False)])
    oof[va] = m.predict(X.iloc[va])
    imp += m.feature_importances_ / 5
    fold_rmse = np.sqrt(mean_squared_error(np.exp(y[va]), np.exp(oof[va])))
    print(f"  fold {k}: best_iter={m.best_iteration_:4d}  RMSE(kg)={fold_rmse:8.1f}")

pred_kg = np.exp(oof)
true_kg = d["fuel_kg"].to_numpy()
rmse = np.sqrt(mean_squared_error(true_kg, pred_kg))
mae = mean_absolute_error(true_kg, pred_kg)
print("\n" + "=" * 60)
print(f"GENEL  OOF RMSE(kg) = {rmse:8.1f}   MAE(kg) = {mae:7.1f}")
print(f"       (referans: hedef ortalama={true_kg.mean():.0f}, medyan={np.median(true_kg):.0f} kg)")
print(f"       naif-medyan baseline RMSE = {np.sqrt(mean_squared_error(true_kg, np.full_like(true_kg, np.median(true_kg)))):.1f}")

# segment bazli
print("\n--- segment bazli RMSE (OpenAP'in nereyi kurtaracagini gosterir) ---")
for name, mask in [
    ("ADS-B'li interval", d["has_adsb"] == 1),
    ("KARANLIK (0 ADS-B)", d["has_adsb"] == 0),
    ("  -> gap interp'li", (d["has_adsb"] == 0) & (d["alt_interp"] == 1)),
    ("  -> tam kor",       (d["has_adsb"] == 0) & (d["alt_interp"] == 0)),
]:
    mm = mask.to_numpy()
    r = np.sqrt(mean_squared_error(true_kg[mm], pred_kg[mm]))
    print(f"  {name:22s} n={mm.sum():6d}  RMSE(kg)={r:8.1f}  (ort.gercek={true_kg[mm].mean():.0f})")

print("\n--- faz bazli RMSE ---")
for ph in d["phase"].cat.categories:
    mm = (d["phase"] == ph).to_numpy()
    r = np.sqrt(mean_squared_error(true_kg[mm], pred_kg[mm]))
    print(f"  {ph:10s} n={mm.sum():6d}  RMSE(kg)={r:8.1f}")

print("\n--- feature importance (top 15) ---")
fi = pd.Series(imp, index=features).sort_values(ascending=False)
print(fi.head(15).round(0).to_string())

# OOF kaydet (Faz 2'de OpenAP residual kiyasi icin)
pd.DataFrame({"flight_id": d["flight_id"], "idx": d["idx"],
              "fuel_kg": true_kg, "pred_baseline": pred_kg}).to_parquet(os.path.join(ROOT, "_oof_baseline.parquet"))
print("\nOOF -> _oof_baseline.parquet")
