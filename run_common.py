import os, json, pathlib, numpy as np
from scripts.bp_extractor_min import bp_from_texts, r_struct_from_texts, sp_between

OUT = pathlib.Path("outputs"); OUT.mkdir(exist_ok=True)
(OUT/"toy").mkdir(exist_ok=True, parents=True)
(OUT/"bench").mkdir(exist_ok=True, parents=True)
(pathlib.Path("figs")/"toy").mkdir(exist_ok=True, parents=True)
(pathlib.Path("figs")/"bench").mkdir(exist_ok=True, parents=True)

MODE = os.environ.get("MODE","toy").lower()

def magnitude_prune(W, p):
    flat = np.abs(W).flatten()
    thr = np.quantile(flat, p) if len(flat) else 0.0
    M = W.copy(); M[np.abs(W) <= thr] = 0.0
    return M

def run_toy():
    texts = ["texts/doc1.txt","texts/doc2.txt"]
    r_struct = r_struct_from_texts(texts)
    rng = np.random.default_rng(42)
    W0 = rng.normal(0,1,(64,64))

    def extract_bp(W,p):
        bp = bp_from_texts(texts)
        # prune effect (approx): reduce claim count by p
        k = int(len(bp["C"]) * (1 - 0.9*p))
        bp["C"] = bp["C"][:max(1,k)]
        return bp

    bp0 = extract_bp(W0, 0.0)
    results=[]
    for p in [0.0,0.4,0.8]:
        Wp = magnitude_prune(W0, p)
        bp = extract_bp(Wp,p)
        cr = round(1 - ( (len(bp["C"])+len(bp["phi"])+3) / max(1,r_struct) ), 2)
        sp = round(sp_between(bp0,bp), 2)
        results.append({"p":p,"CR":cr,"SP":sp,"BP_size":len(bp["C"])})
    out_path = OUT/"toy"/"metrics.json"
    json.dump({"meta":{"mode":"toy"}, "results":results}, open(out_path,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Saved:", out_path)

def run_bench():
    # placeholder: create empty benchmark outputs – connect Colab later
    out_path = OUT/"bench"/"metrics.json"
    json.dump({"meta":{"mode":"bench"}, "results":[]}, open(out_path,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Saved:", out_path)

if MODE == "toy":
    run_toy()
elif MODE == "bench":
    run_bench()
elif MODE == "n_r":
    # ---- N-R (価値ゲート toy) ----
    import numpy as np, json, pathlib
    from scripts.bp_extractor_min import bp_from_texts, sp_between
    texts = ["texts/doc1.txt","texts/doc2.txt"]
    bp_ref = bp_from_texts(texts)

    results = []
    for kappa in [0.2, 0.6]:
        # 価値ゲート：κによってClaims数をゆるく圧縮
        bp_mod = bp_from_texts(texts)
        k = int(len(bp_mod["C"]) * (1 - 0.3 * kappa))
        bp_mod["C"] = bp_mod["C"][:max(1, k)]

        sp = round(sp_between(bp_ref, bp_mod), 2)
        vs = round(1 - sp, 2)
        results.append({"kappa": kappa, "SP": sp, "VS": vs})

    OUT_N = pathlib.Path("outputs/n_r"); OUT_N.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"meta": {"mode": "n_r"}, "results": results},
        open(OUT_N / "metrics.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False
    )
    print("Saved: outputs/n_r/metrics.json")
else:
    print(f"Unknown MODE={MODE}")
