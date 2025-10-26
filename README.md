# Forgetting Mechanics — Auto Base (toy ↔ bench)

Template to run the C‑series pipeline from GitHub Actions with a one‑button
switch between `toy` and `bench`.

- `MODE=toy` (default): tiny, deterministic pipeline, runs in seconds
- `MODE=bench`: placeholder/stub (connect Colab/real code later)

## Quick start
1. Commit/push this template.
2. Go to **Actions** → workflow `Auto Measure` → **Run workflow** → MODE: `toy`.
3. Download artifacts: `outputs/.../metrics.json` and `figs/...`.

Files:
- `run_common.py` : reads MODE and dispatches to toy/bench
- `scripts/bp_extractor_min.py` : minimal BP extractor for toy
- `requirements.txt` : numpy, matplotlib, networkx, pandas
- `.github/workflows/auto.yml` : GitHub Actions workflow
