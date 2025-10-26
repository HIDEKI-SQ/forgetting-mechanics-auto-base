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

if MODE=="toy":
    run_toy()
else:
    run_bench()
