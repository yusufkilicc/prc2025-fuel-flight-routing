# Code Review, UI Evaluation & Roadmap — prc2025-fuel-routing

End-to-end review of the package: correctness bugs, robustness/portability gaps,
completeness, a detailed UI evaluation, and a prioritized development roadmap.

**Overall grade:** strong portfolio project (clean architecture, honest framing,
working product). Not production-grade yet. The findings below are mostly
medium/low severity — there are no data-loss or security-critical bugs.

---

## ✅ Resolution status (this iteration)

**All P0 + most P1 items are fixed and tested (11 passing tests):**

| ID | Item | Status |
|----|------|--------|
| A1 | Submission uses hybrid (OpenAP) model, not baseline | ✅ fixed (`submit.py` → `submission_*_hybrid.parquet`) |
| A2 | UI fuel inconsistency (0.6 vs 0.75) | ✅ fixed (shared `LOAD_FACTOR=0.62`) |
| A3 | `import os` above docstring | ✅ fixed |
| A4 | Antimeridian map-zoom bug | ✅ fixed (longitude unwrap) |
| A5 | Lateral deviation not cos-lat corrected | ✅ fixed |
| B1 | Hardcoded 16 cores | ✅ fixed (`-1` / `os.cpu_count()`) |
| B3 | Hardcoded host/port | ✅ fixed (`PORT`/`HOST` env) |
| B4 | No error feedback on bad input | ✅ fixed (inline error + AbortController) |
| C1 | No tests / CI | ✅ fixed (pytest suite + GitHub Actions) |
| C2 | Global calibration fudge | ✅ improved (per-class wide/narrow/regional) |
| C3 | Uncertainty band unused in UI | ✅ added (mass-sensitivity band on KPI + cards) |
| C5 | Static SHAP json | ✅ fixed (`train_phase3.py` regenerates it) |
| D1.1 | No request cancellation | ✅ added (AbortController) |
| D1.3 | Accessibility | ✅ improved (aria-labels, focus-visible, reduced-motion) |
| D1.4 | No map legend | ✅ added |
| D1.6 | Muddled headwind slider | ✅ clarified ("Extra headwind", tooltip) |

**P2 — also now addressed:**

| ID | Item | Status |
|----|------|--------|
| — | Distance-aware mass estimation (the "biggest accuracy lever") | ✅ done (2-pass: estimate trip fuel → takeoff mass = OEW + payload + fuel, MTOW-capped) |
| D1.7 | Shareable URL state | ✅ done (`?o=&d=&ac=&w=`, restored on load) |
| C6 | Dockerfile / `.python-version` | ✅ done |
| — | One-command setup/run + browser launch | ✅ done (`run.bat`/`run.sh`/`start_app.py`) |
| — | Optuna tuning + chronological holdout | ✅ tooling added (`src/tune.py`); run on full data |

**Still open (deliberate / low ROI):** B2 full offline asset vendoring (map tiles make
true offline impractical; the Tailwind Play-CDN warning is cosmetic); mobile QA (D1.8).
The Streamlit app (`app_streamlit.py`) is kept as a secondary prototype (now consistent
with the web app on mass via the shared `LOAD_FACTOR`).

---

---

## A. Correctness bugs (prioritized)

### A1 — Submission uses the *weaker* baseline model, not the hybrid ⚠️ (high value)
`src/submit.py` trains on `features_train.parquet` (Faz-1 baseline, set A).
The project's "best" model is the **OpenAP hybrid** (`features_train_openap.parquet`,
RMSE 252 vs 258) and the tuned Phase-3 model. So the official submission silently
uses the weakest model. Also `openap_baseline.py` only generates OpenAP features for
`train`, never for `rank`/`final`, so a hybrid submission isn't even possible without
a pipeline change.
**Fix:** add `openap_baseline.py {rank,final}` to the pipeline and point `submit.py`
at the hybrid features (or load the saved `models/lgb_p50.txt`).

### A2 — The two UIs disagree on fuel for the same route ⚠️
`web/web_app.py` → `scenario_routes(load_factor=0.6)`.
`web/app_streamlit.py` → `compare_altitudes/compare_routes` → `score_path(load_factor=0.75)`.
Same origin/dest/aircraft yields **different fuel numbers** in the two interfaces
(0.6 vs 0.75 takeoff-mass assumption ≈ 5–8 % gap).
**Fix:** make `load_factor` a single shared default (or a config constant) used by both.

### A3 — `import os` sits above the module docstring (packaging artifact)
`src/route_decision.py` line 1 is `import os` followed by the `"""docstring"""`,
so the module loses its `__doc__` and the import ordering is ugly.
**Fix:** move the import below the docstring.

### A4 — Map zoom breaks on antimeridian-crossing routes
`drawMap` computes `fitBounds` from raw min/max lon. A trans-Pacific route
(e.g. lon +179 → −179) produces a bbox spanning the whole globe, so the map
zooms out to the entire world.
**Fix:** detect antimeridian crossing and split/normalize longitudes before bounds.

### A5 — Lateral deviation isn't longitude-corrected
`deviate()` offsets points by `offset_km / 111.0` degrees in **both** lat and lon.
Longitude degrees shrink with latitude (×cos φ), so deviated paths are geometrically
distorted at higher latitudes. Cosmetic (illustrative routes only) but incorrect.
**Fix:** divide the longitude component by `111.0 * cos(lat)`.

---

## B. Robustness & portability

### B1 — Hardcoded 16 cores everywhere
`n_jobs=16` (4 train/submit files) and `Pool(processes=16)` (feature_pipeline).
On a 4-core laptop this oversubscribes; on a 32-core box it underuses.
**Fix:** `n_jobs=-1` for LightGBM, `Pool(processes=os.cpu_count())`.

### B2 — "Self-contained" for data, but NOT offline
The frontend loads Tailwind, MapLibre, Google Fonts and CARTO tiles from CDNs.
With no internet the page renders blank even though the data is bundled. The
Tailwind **Play CDN** also prints a "not for production" console warning.
**Fix (if offline matters):** vendor MapLibre + a built Tailwind CSS locally and
serve them as static files; optionally bundle a local basemap style.

### B3 — Hardcoded host/port
`uvicorn.run(host="127.0.0.1", port=8600)`. No `PORT`/`HOST` env override → awkward
to deploy or run two instances.
**Fix:** read `os.environ.get("PORT", 8600)`.

### B4 — No user feedback on bad input
Invalid/identical airports → API returns 400 and the JS does `if(data.error) return;`
(silent). The user sees stale results with no message.
**Fix:** surface a small inline error/toast.

### B5 — `frac_elapsed` and interval values can fall outside [0,1]
Some fuel intervals extend slightly beyond the trajectory span (observed min −0.75,
max 3.09 in the rank split). Not clamped. Trees tolerate it, but it's a silent
data-quality smell worth guarding.

---

## C. Completeness & quality

- **C1 — No tests, no CI.** The sister Istanbul project has pytest + pre-commit; this
  package has neither. A minimal smoke test (route_decision returns 3 scenarios with
  sane magnitudes; `/api/score` returns 200) would catch regressions cheaply.
- **C2 — Calibration is a global fudge.** `CALIB=0.92` + idle-descent `×0.28` are
  single constants applied to all aircraft/phases. Defensible as "illustrative" but a
  per-class (narrow/wide/regional) calibration table would be more credible — and
  should be named in the README, not just code comments.
- **C3 — Built uncertainty is unused in the UI.** `models/lgb_p10/p50/p90` exist, but
  neither UI shows the P10–P90 band. The decision tool would be far stronger with a
  visible "± range" on fuel.
- **C4 — `recommended` = min-fuel only.** The web app ignores time/cost when picking the
  recommended scenario; the Streamlit app *does* use MCDA weights. Inconsistent decision
  logic between the two front-ends.
- **C5 — `shap_importance.json` is a static snapshot.** Hardcoded values; a retrain won't
  update the "Why this estimate" panel. `train_phase3.py` could dump it.
- **C6 — No Dockerfile / `.python-version`.** Reproducibility relies on the reader
  matching versions by hand.

---

## D. Detailed UI evaluation

### D1 — Web app (`web_app.py`) — the flagship

**Strengths (genuinely high quality):**
- Coherent design system (Inter, slate/blue tokens, 16px-radius cards, soft shadows).
- 3D globe (MapLibre v5) with light CARTO basemap, dashed recommended arc + faint
  alternatives, green/red endpoints with labels, route pill — visually excellent.
- KPI band, scenario cards with "Recommended" ring, SHAP driver panel, honesty box.
- Responsive grid (md/lg breakpoints), tasteful hover/transition micro-interactions.

**Weaknesses / opportunities:**
1. **No loading or empty/error state.** Only a single KPI "pulse." Rapid dropdown
   changes can race (no request cancellation/debounce) → a slow response can overwrite
   a newer one. Add an AbortController + a skeleton/spinner, and an inline error.
2. **Uncertainty not shown** (see C3) — the biggest substantive UI gap. A P10–P90 band
   on the fuel KPI/cards would elevate it from "a number" to "a calibrated estimate."
3. **Accessibility:** information is encoded by color alone (green/red markers, blue =
   recommended). No ARIA labels, no visible keyboard focus styling, `slate-400` captions
   are borderline on contrast. Add text/shape cues + `aria-label`s.
4. **No map legend.** Dashed-grey = alternatives vs blue = recommended is only learnable
   by inference; a 2-line legend would clarify.
5. **Antimeridian zoom bug** (A4) is a visible functional UI defect on Pacific routes.
6. **Headwind slider semantics are muddled:** it shifts all scenarios equally, while the
   "wind-optimal" scenario *also* hardcodes a −28 kt tailwind. Users can't tell what the
   slider represents. Either label it clearly or fold it into the scenario model.
7. **No shareable state.** The selection isn't in the URL, so a specific comparison can't
   be linked/bookmarked.
8. **Mobile:** breakpoints exist but the fixed 460px map + 3 cards + SHAP column get
   cramped < 400px; not verified on a real phone.
9. **Tailwind Play CDN** warning (B2) is visible in console — not production-clean.

**UI grade: 8.5/10** — already portfolio-excellent; the gaps are polish + the
uncertainty-band opportunity, not structural.

### D2 — Streamlit app (`app_streamlit.py`) — the prototype

**Strengths:** fast, includes the **MCDA weighting** the web app lacks, pydeck map,
honesty box, clean enough.
**Weaknesses:**
1. Default Streamlit chrome (dark theme, default widgets) is visibly less polished than
   the web app — fine as a "prototype," but the two diverge in look *and* numbers (A2).
2. `st.bar_chart` is default-styled (no token theming) unlike the Istanbul project's
   Plotly theme helper.
3. Lateral deviations differ by only ~0.3 % distance, so the alternative routes look
   almost identical on the map — low information value.

**Recommendation:** treat the Streamlit app as deprecated/secondary, or delete it to
avoid the A2 inconsistency and a second thing to maintain. The web app supersedes it.

---

## E. Roadmap (prioritized)

### P0 — Quick, high-value (a few hours)
- [ ] A1: generate OpenAP features for rank/final and switch `submit.py` to the hybrid
      (or saved P50) model — makes the headline submission match the "best model" claim.
- [ ] A2 + A3: unify `load_factor` across both UIs; fix the `import os`/docstring order.
- [ ] B1: replace hardcoded `16` with `-1` / `os.cpu_count()`.
- [ ] B3 + B4: `PORT` env; inline error toast on bad input.
- [ ] C1: add a tiny pytest smoke test + a GitHub Actions workflow.

### P1 — Substance & credibility (1–2 days)
- [ ] C3 + D1.2: surface the P10–P90 uncertainty band in the web UI.
- [ ] A4: antimeridian-safe map bounds.
- [ ] C2: per-class calibration table, documented in README.
- [ ] C4: align "recommended" logic (MCDA) across both front-ends, or drop the Streamlit app.
- [ ] D1.1: loading/skeleton + request cancellation.

### P2 — Rigor & deployment (when time allows)
- [ ] Time-based validation split + Optuna tuning (the promised modeling rigor) +
      a results section in REPORT.md.
- [ ] B2: vendor frontend assets for offline use.
- [ ] C6: Dockerfile + `.python-version` for one-command reproducibility.
- [ ] A5: cos-latitude correction in `deviate()`.
- [ ] Distance-aware / iterative mass estimation (the single biggest accuracy lever).

---

*No security-critical or data-loss issues were found. The credentials/data exclusions
in `.gitignore` are correct. The bundled demo data lets the web app run offline-of-data
but still requires internet for frontend CDNs (B2).*
