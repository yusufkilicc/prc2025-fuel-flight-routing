"""
Faz 3 (kisa) — yorumlanabilirlik (SHAP) + belirsizlik bandi (quantile regression).
Hibrit feature-augmented modeli baz alir. Cikti:
  - shap_summary.png  (yakiti ne suruyor)
  - quantile P10/P50/P90 + ampirik kapsama
  - kaydedilen modeller (Faz 4 icin)
"""
import os, sys, io, warnings, json
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")

d = pd.read_parquet(os.path.join(ROOT, "features_train_openap.parquet")).reset_index(drop=True)
d["log_openap"] = np.log(d["fuel_openap"].clip(lower=0.1))
CAT = ["aircraft_type", "phase"]
DROP = ["flight_id", "idx", "fuel_kg", "fuel_openap"]
features = [c for c in d.columns if c not in DROP]
for c in CAT:
    d[c] = d[c].astype("category")
X = d[features]; y = np.log(d["fuel_kg"].to_numpy()); groups = d["flight_id"]
true_kg = d["fuel_kg"].to_numpy()

# grup-bilincli tek bolme (hizli)
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
print(f"train={len(tr):,}  valid={len(va):,}")

base = dict(num_leaves=128, min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1,
            n_estimators=1500, learning_rate=0.03)

# ---- P50 (ana model) ----
m50 = lgb.LGBMRegressor(objective="regression", metric="rmse", **base)
m50.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], categorical_feature=CAT,
        callbacks=[lgb.early_stopping(80, verbose=False)])
p50 = np.exp(m50.predict(X.iloc[va]))
rmse = np.sqrt(mean_squared_error(true_kg[va], p50))
print(f"P50 valid RMSE(kg) = {rmse:.1f}")

# ---- quantile P10 / P90 ----
def quant(alpha):
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **base)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], categorical_feature=CAT,
          callbacks=[lgb.early_stopping(80, verbose=False)])
    return m
m10, m90 = quant(0.1), quant(0.9)
p10 = np.exp(m10.predict(X.iloc[va])); p90 = np.exp(m90.predict(X.iloc[va]))
p10, p90 = np.minimum(p10, p90), np.maximum(p10, p90)  # monotonluk garanti
cov = ((true_kg[va] >= p10) & (true_kg[va] <= p90)).mean()
width = np.median(p90 - p10)
print(f"[P10,P90] ampirik kapsama = {cov*100:.1f}%  (hedef ~80%)   medyan bant genisligi = {width:.0f} kg")

# ---- SHAP ----
print("SHAP hesaplaniyor (5000 ornek)...")
samp = np.random.RandomState(0).choice(va, size=min(5000, len(va)), replace=False)
expl = shap.TreeExplainer(m50)
sv = expl.shap_values(X.iloc[samp])
mean_abs = np.abs(sv).mean(0)
order = np.argsort(mean_abs)[::-1]
print("\n--- SHAP global onem (log-yakit etkisi, top 12) ---")
for i in order[:12]:
    print(f"  {features[i]:20s} mean|SHAP|={mean_abs[i]:.3f}")

plt.figure()
shap.summary_plot(sv, X.iloc[samp], feature_names=features, show=False, max_display=15)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "shap_summary.png"), dpi=110, bbox_inches="tight")
print("-> shap_summary.png kaydedildi")

# ---- modelleri kaydet (Faz 4) ----
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
m50.booster_.save_model(os.path.join(ROOT, "models", "lgb_p50.txt"))
m10.booster_.save_model(os.path.join(ROOT, "models", "lgb_p10.txt"))
m90.booster_.save_model(os.path.join(ROOT, "models", "lgb_p90.txt"))
json.dump({"features": features, "cat": CAT,
           "aircraft_types": [str(x) for x in d["aircraft_type"].cat.categories]},
          open(os.path.join(ROOT, "models", "meta.json"), "w"))
print("-> models/ klasorune P10/P50/P90 + meta.json kaydedildi")

# ---- shap_importance.json (C5): UI "Why this estimate" paneli icin ----
FRIENDLY = {
    "log_openap": "OpenAP physics estimate", "dur_s": "Interval duration",
    "aircraft_type": "Aircraft type", "mass_est": "Estimated mass",
    "gc_dist_od_km": "Route distance", "alt_delta": "Altitude change",
    "mach_mean": "Mach number", "alt_mean": "Cruise altitude",
    "gs_mean": "Ground speed", "frac_elapsed": "Position in flight",
}
drivers = [{"feature": FRIENDLY.get(features[i], features[i]), "value": round(float(mean_abs[i]), 3)}
           for i in order[:7]]
json.dump({"title": "What drives fuel burn",
           "subtitle": "Global SHAP impact from the PRC-trained hybrid model",
           "drivers": drivers},
          open(os.path.join(ROOT, "models", "shap_importance.json"), "w"), indent=2)
print("-> shap_importance.json (Faz 4 paneli) guncellendi  (pakete kopyalayin: models/)")
