[![Auto Measure](https://github.com/HIDEKI-SQ/forgetting-mechanics-auto-base/actions/workflows/auto.yml/badge.svg)](https://github.com/HIDEKI-SQ/forgetting-mechanics-auto-base/actions/workflows/auto.yml)

# Forgetting Mechanics — Auto Base (toy ↔ bench)

Template to run the C-series pipeline from GitHub Actions with a one-button
switch between `toy`, `bench`, and `n_r / stress` experimental modes.

---

## Quick start
1. Commit/push this repository.
2. Go to **Actions** → workflow `Auto Measure` → **Run workflow**.
3. Choose MODE (`toy` / `n_r` / `stress`) and press **Run workflow**.
4. Download artifacts (`outputs/.../metrics.json`, `figs/...`).

Files:
- `run_common.py` : dispatches each MODE
- `scripts/bp_extractor_min.py` : minimal BP extractor for toy
- `.github/workflows/auto.yml` : GitHub Actions workflow
- `requirements.txt` : numpy, matplotlib, networkx, pandas

---

## MODE の使い分け
| MODE | 説明 |
|------|------|
| **toy** | CI確認用の極小パイプライン。BP抽出→CR/SPを簡易算出。 |
| **n_r** | 価値強度 κ をスイープして SP / VS 地形を観測（C3-A）。 |
| **stress** | 過圧縮ストレステスト（C3-D）。p=0.95 で SP 崩壊を確認。 |
| **bench** | 実データ用プレースホルダー。Colab／実測コードを接続予定。 |

---

## Run examples
### κ-sweep (C3-A)
- mode: `n_r`
- target id: `lari_001`
- kappas: `0.0,0.2,0.4,0.6,0.8,1.0`
- artifact: `outputs/n_r/lari_001/metrics.json`

### over-compression stress (C3-D)
- mode: `stress`
- target id: `lari_min`
- artifact: `outputs/stress/lari_min/metrics.json`

---

## Artifacts / Outputs
| 種別 | ファイル |
|------|-----------|
| κ-sweep | `outputs/n_r/lari_001/metrics.json` |
| stress test | `outputs/stress/lari_min/metrics.json` |
| reproducibility | `outputs/repro/metrics_rs_st.json` |

---

## Reproducibility (C3-C)
- 5 runs under identical conditions  
- std ≤ 0.05 （このパイプラインは決定論的設計のため **std=0.00**）  
- file: `outputs/repro/metrics_rs_st.json`

---

## Figures (to be produced in Colab)
- **Fig.C3-A1**：κ vs (SP, VS) 折れ線。VS帯域 [0.30, 0.70] を淡色で塗る  
  - κ: [0.0,0.2,0.4,0.6,0.8,1.0]  
  - SP: [1.00,0.83,0.83,0.83,0.83,0.83]  
  - VS: [0.08,0.20,0.32,0.44,0.56,0.68]  
- **Fig.C3-D1**：p vs (CR, SP)。p=0.95 で CR=0.95, SP≈0 を注記  
  - p: [0.0,0.80,0.95], CR: [0.00,0.80,0.95], SP: [1.00,0.67,0.00]  
- **Fig.C3-C1**：mean±std（std=0）再現性グラフ  

---

## C3 Summary (A+D+C)
- κスイープ：SP≥0.60 & VS∈[0.30,0.70] の最適帯域を確認。  
- 過圧縮テスト：p=0.95 で CR=0.95, SP=0.00（構造崩壊）を観測。  
- 安定性：5回再測定で std=0.00（RS/ST 基準内）。  
=> **C3 Core Completed (100%)**

---

## Next phase
- `docs/index.md` にこのサマリを追加  
- Colab で図（Fig.C3-A1, D1, C1）を生成  
- Preprint **“C3｜測定系──構造を測るという倫理”** 執筆開始
