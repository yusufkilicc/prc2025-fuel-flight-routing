"""
Faz 2 hibrit — OpenAP fizik bazi + LightGBM.
Iki strateji kiyaslanir:
  (A) feature-augmented: log(fuel_kg) tahmin et, fuel_openap'i feature olarak ekle
  (B) residual: openap olan satirlarda log(fuel_kg)-log(fuel_openap) ogren, fallback (A)
GroupKFold, ayni segment kirilimi ile Faz 1 (258 kg) kiyasi.
"""
import os, sys, io, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")

d = pd.read_parquet(os.path.join(ROOT, "features_train_openap.parquet")).reset_index(drop=True)
d["log_openap"] = np.log(d["fuel_openap"].clip(lower=0.1))
print(f"veri: {d.shape}  | openap kapsam: {d['fuel_openap'].notna().mean()*100:.1f}%")

CAT = ["aircraft_type", "phase"]
DROP = ["flight_id", "idx", "fuel_kg", "fuel_openap"]   # fuel_openap'i ham birakma; log_openap kullan
features = [c for c in d.columns if c not in DROP]
for c in CAT:
    d[c] = d[c].astype("category")

X = d[features]
y = np.log(d["fuel_kg"].to_numpy())
log_op = d["log_openap"].to_numpy()
has_op = d["fuel_openap"].notna().to_numpy() & np.isfinite(log_op)
groups = d["flight_id"]
true_kg = d["fuel_kg"].to_numpy()

params = dict(objective="regression", metric="rmse", n_estimators=2000, learning_rate=0.03,
              num_leaves=128, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1)

def run(mode):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(d))
    imp = np.zeros(len(features))
    for tr, va in gkf.split(X, y, groups):
        if mode == "A":
            ytr = y[tr]
        else:  # residual: openap olan yerde residual, olmayan yerde mutlak log
            ytr = np.where(has_op[tr], y[tr] - log_op[tr], y[tr])
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr], ytr, eval_set=[(X.iloc[va], (y[va] if mode == "A"
              else np.where(has_op[va], y[va]-log_op[va], y[va])))],
              categorical_feature=CAT, callbacks=[lgb.early_stopping(80, verbose=False)])
        pv = m.predict(X.iloc[va])
        if mode == "B":
            pv = np.where(has_op[va], pv + log_op[va], pv)
        oof[va] = pv
        imp += m.feature_importances_ / 5
    pred_kg = np.exp(oof)
    rmse = np.sqrt(mean_squared_error(true_kg, pred_kg))
    mae = mean_absolute_error(true_kg, pred_kg)
    return pred_kg, rmse, mae, pd.Series(imp, index=features).sort_values(ascending=False)

print("\n" + "=" * 60)
for mode, name in [("A", "feature-augmented"), ("B", "residual")]:
    pred_kg, rmse, mae, fi = run(mode)
    print(f"\n[{name}]  GENEL OOF RMSE(kg) = {rmse:8.1f}   MAE = {mae:6.1f}   (Faz1 baseline=258.1)")
    for nm, mask in [
        ("ADS-B'li", d["has_adsb"] == 1),
        ("KARANLIK", d["has_adsb"] == 0),
        ("  gap interp", (d["has_adsb"] == 0) & (d["alt_interp"] == 1)),
        ("  tam kor", (d["has_adsb"] == 0) & (d["alt_interp"] == 0)),
    ]:
        mm = mask.to_numpy()
        r = np.sqrt(mean_squared_error(true_kg[mm], pred_kg[mm]))
        print(f"     {nm:14s} n={mm.sum():6d}  RMSE(kg)={r:8.1f}")
    if mode == "A":
        print("   top feature:", ", ".join(fi.head(6).index))
        best = (pred_kg, rmse)

# en iyi modeli kaydet (residual genelde daha iyi; ikisini de gosterdik)
