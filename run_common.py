# -*- coding: utf-8 -*-
import os, json, pathlib, re, itertools, argparse
import numpy as np
import networkx as nx

# -------------------------
# Minimal BP extractor bits
# -------------------------
CAUSE_WORDS = ("because","therefore","so","hence","thus","ので","だから","ゆえに")
TIME_WORDS  = ("today","now","later","before","after","昨日","今日","明日")

def split_sentences(txt: str):
    return [s.strip() for s in re.split(r'[。.!?]', txt) if s.strip()]

def extract_triplets(sent: str):
    toks = re.findall(r'[A-Za-z0-9一-龥ぁ-んァ-ン]+', sent)
    trip = []
    for i in range(len(toks)-2):
        a,b,c = toks[i:i+3]
        if re.match(r'.*(する|した|なる|be|is|are|do|make|cause).*', b):
            trip.append([a,"rel",c])
    return trip or ([[toks[0],"rel",toks[-1]]] if len(toks)>=2 else [])

def build_claims(texts):
    claims=[]
    for tpath in texts:
        with open(tpath, encoding='utf-8') as fh:
            txt = fh.read()
        for s in split_sentences(txt):
            for tri in extract_triplets(s):
                sflat = " ".join(tri)
                if any(w in sflat for w in CAUSE_WORDS):
                    tri[1] = "causes"
                elif "含" in sflat or "include" in sflat:
                    tri[1] = "includes"
                else:
                    tri[1] = "precedes"
                claims.append({"form": tri, "conf": 1.0})
    for i, c in enumerate(claims):
        c["id"] = f"c_{i}"
    return claims

def axes_from_texts(texts):
    txt = " ".join(open(t, encoding='utf-8').read() for t in texts)
    toks = re.findall(r'[A-Za-z0-9一-龥ぁ-んァ-ン]+', txt)
    uniq, total = len(set(toks)), max(1, len(toks))
    abstractness = min(1.0, (uniq/total)*1.5)
    causal_density = 1.0 if any(w in txt for w in CAUSE_WORDS) else 0.3
    timescale = "mid" if any(w in txt for w in TIME_WORDS) else "long"
    return {"abstractness": abstractness, "causal_density": causal_density, "timescale": timescale}

def constraints_from_claims(claims):
    cons=[]
    for rel in ("causes","includes","precedes"):
        edges=[(c["form"][0], c["form"][2]) for c in claims if c["form"][1]==rel]
        G=nx.DiGraph(); G.add_edges_from(edges)
        ok = nx.is_directed_acyclic_graph(G)
        cons.append({"rule": f"{rel}_acyclic", "value": 1 if ok else 0})
    return cons

def bp_from_texts(texts):
    A = axes_from_texts(texts)
    C = build_claims(texts)
    phi = constraints_from_claims(C)
    return {"A": A, "C": C, "phi": phi}

def sp_between(bp0, bp1):
    E0 = set(tuple(c["form"]) for c in bp0["C"])
    E1 = set(tuple(c["form"]) for c in bp1["C"])
    j_edge = len(E0 & E1)/len(E0 | E1) if (E0 or E1) else 1.0
    P0 = set((a,b,c) for (a,_,b) in E0 for (bb,_,c) in E0 if b==bb)
    P1 = set((a,b,c) for (a,_,b) in E1 for (bb,_,c) in E1 if b==bb)
    f1 = (2*len(P0 & P1)/(len(P0)+len(P1))) if (P0 or P1) else 1.0
    return 0.5*j_edge + 0.5*f1

# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default=os.environ.get("MODE","toy"))
    p.add_argument("--target_id", default=os.environ.get("TARGET_ID",""))
    # 単体 κ 指定（ワークフローで for ループする方式に対応）
    p.add_argument("--kappa", type=float, default=None)
    # 複数 κ 指定（カンマ区切り）
    p.add_argument("--kappa_list", default=os.environ.get("KAPPA_LIST",""))
    return p.parse_args()

# -------------------------
# Experiment modes
# -------------------------
ROOT = pathlib.Path(".")
OUT  = ROOT/"outputs"; OUT.mkdir(exist_ok=True)
(OUT/"toy").mkdir(parents=True, exist_ok=True)
(OUT/"bench").mkdir(parents=True, exist_ok=True)
(OUT/"n_r").mkdir(parents=True, exist_ok=True)

(FIGS := (ROOT/"figs")).mkdir(exist_ok=True)
(FIGS/"toy").mkdir(parents=True, exist_ok=True)
(FIGS/"bench").mkdir(parents=True, exist_ok=True)

def magnitude_prune(W: np.ndarray, p: float) -> np.ndarray:
    flat = np.abs(W).flatten()
    thr = np.quantile(flat, p) if len(flat) else 0.0
    M = W.copy()
    M[np.abs(M) <= thr] = 0.0
    return M

def run_toy(target_id:str=""):
    texts = ["texts/doc1.txt","texts/doc2.txt"]
    rng = np.random.default_rng(42)
    W0 = rng.normal(0,1,(64,64))
    # 基準BP（p=0）
    bp0 = bp_from_texts(texts)
    r_struct = len(bp0["C"]) + len(bp0["phi"]) + 3
    results=[]
    for p in [0.0, 0.4, 0.8]:
        Wp = magnitude_prune(W0, p)
        bp = bp_from_texts(texts)
        keep = int(len(bp["C"]) * (1 - 0.9*p))
        bp["C"] = bp["C"][:max(1, keep)]
        sp = round(sp_between(bp0, bp), 2)
        sz = len(bp["C"]) + len(bp["phi"]) + 3
        cr = round(max(0.0, min(1.0, 1 - sz / max(1, r_struct))), 2)
        results.append({"p": p, "CR": cr, "SP": sp, "BP_size": len(bp["C"])})
    out_dir = OUT/"toy"
    if target_id: out_dir = out_dir/target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/"metrics.json"
    json.dump({"meta": {"mode": "toy", "target_id": target_id}, "results": results},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

def run_bench(target_id:str=""):
    out_dir = OUT/"bench"
    if target_id: out_dir = out_dir/target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/"metrics.json"
    json.dump({"meta": {"mode": "bench", "target_id": target_id}, "results": []},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

def parse_kappas(args) -> list[float]:
    # 優先順位：--kappa（単体） > --kappa_list（配列） > 既定
    if args.kappa is not None:
        return [float(args.kappa)]
    if args.kappa_list:
        try:
            return [float(x) for x in re.split(r'[,\s]+', args.kappa_list.strip()) if x!='']
        except Exception:
            pass
    return [0.2, 0.6]  # 既定

def run_n_r(kappas:list[float], target_id:str=""):
    texts = ["texts/doc1.txt","texts/doc2.txt"]
    bp_ref = bp_from_texts(texts)
    results = []
    for kappa in kappas:
        bp_mod = bp_from_texts(texts)
        keep = int(len(bp_mod["C"]) * (1 - 0.3 * float(kappa)))
        bp_mod["C"] = bp_mod["C"][:max(1, keep)]
        sp = round(sp_between(bp_ref, bp_mod), 2)
        vs = round(1 - sp, 2)
        results.append({"kappa": float(kappa), "SP": sp, "VS": vs})
    out_dir = OUT/"n_r"
    if target_id: out_dir = out_dir/target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/"metrics.json"
    meta = {"mode": "n_r", "target_id": target_id, "kappas": kappas}
    json.dump({"meta": meta, "results": results},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    mode = (args.mode or os.environ.get("MODE","toy")).lower()

    if mode == "toy":
        run_toy(target_id=args.target_id)
    elif mode == "bench":
        run_bench(target_id=args.target_id)
    elif mode == "n_r":
        kappas = parse_kappas(args)
        run_n_r(kappas, target_id=args.target_id)
    else:
        print(f"Unknown MODE={mode}")
