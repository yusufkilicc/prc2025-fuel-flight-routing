"""
Modeling rigor (P2): Optuna hyperparameter tuning + a chronological holdout
on the OpenAP-hybrid features. Reports tuned CV RMSE vs the 252 kg baseline and
an honest time-ordered holdout score, then saves best params.

Requires the full training data (PRC_DATA_DIR). Run:
    pip install optuna
    PRC_DATA_DIR=... python src/tune.py [n_trials]
"""
import os, sys, io, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PRC_DATA_DIR") or os.path.join(_PKG, "data")
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 25

d = pd.read_parquet(os.path.join(ROOT, "features_train_openap.parquet")).reset_index(drop=True)
d["log_openap"] = np.log(d["fuel_openap"].clip(lower=0.1))
CAT = ["aircraft_type", "phase"]
DROP = ["flight_id", "idx", "fuel_kg", "fuel_openap"]
features = [c for c in d.columns if c not in DROP]
for c in CAT:
    d[c] = d[c].astype("category")
X, y = d[features], np.log(d["fuel_kg"].to_numpy())
groups, true_kg = d["flight_id"], d["fuel_kg"].to_numpy()

# --- chronological holdout (honest temporal check) ---
fl = pd.read_parquet(os.path.join(ROOT, "flightlist_train.parquet"))[["flight_id", "takeoff"]]
order = fl.sort_values("takeoff")["flight_id"].tolist()
cut = set(order[int(0.8 * len(order)):])          # latest 20% of flights by takeoff
te_mask = d["flight_id"].isin(cut).to_numpy()
print(f"veri {d.shape} | kronolojik holdout: train={(~te_mask).sum():,} test={te_mask.sum():,}")


def cv_rmse(params, n_splits=3):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(d))
    for tr, va in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], categorical_feature=CAT,
              callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[va] = m.predict(X.iloc[va])
    return np.sqrt(mean_squared_error(true_kg, np.exp(oof)))


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(t):
        params = dict(
            objective="regression", metric="rmse", n_estimators=3000, n_jobs=-1, verbosity=-1,
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
            num_leaves=t.suggest_int("num_leaves", 31, 320),
            min_child_samples=t.suggest_int("min_child_samples", 20, 200),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        return cv_rmse(params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = dict(objective="regression", metric="rmse", n_estimators=3000, n_jobs=-1,
                verbosity=-1, **study.best_params)
    print(f"\nEN IYI CV RMSE = {study.best_value:.1f} kg   (baseline hibrit 252)")
    print("best params:", json.dumps(study.best_params, indent=2))

    # kronolojik holdout skoru (en iyi params)
    m = lgb.LGBMRegressor(**best)
    m.fit(X[~te_mask], y[~te_mask], categorical_feature=CAT)
    ph = np.exp(m.predict(X[te_mask]))
    rmse_t = np.sqrt(mean_squared_error(true_kg[te_mask], ph))
    print(f"KRONOLOJIK HOLDOUT RMSE = {rmse_t:.1f} kg  (zaman kaymasina dayaniklilik)")

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    json.dump({"best_params": study.best_params, "cv_rmse": round(study.best_value, 1),
               "holdout_rmse": round(float(rmse_t), 1), "n_trials": N_TRIALS},
              open(os.path.join(ROOT, "models", "tuning_result.json"), "w"), indent=2)
    print("-> models/tuning_result.json kaydedildi")


if __name__ == "__main__":
    main()
