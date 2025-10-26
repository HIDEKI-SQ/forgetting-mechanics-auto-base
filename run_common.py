import os, json, pathlib, re, itertools
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
# Experiment modes
# -------------------------
OUT = pathlib.Path("outputs"); OUT.mkdir(exist_ok=True)
(OUT/"toy").mkdir(parents=True, exist_ok=True)
(OUT/"bench").mkdir(parents=True, exist_ok=True)
(OUT/"n_r").mkdir(parents=True, exist_ok=True)
(pathlib.Path("figs")/"toy").mkdir(parents=True, exist_ok=True)
(pathlib.Path("figs")/"bench").mkdir(parents=True, exist_ok=True)

MODE = os.environ.get("MODE", "toy").lower()

def magnitude_prune(W: np.ndarray, p: float) -> np.ndarray:
    flat = np.abs(W).flatten()
    thr = np.quantile(flat, p) if len(flat) else 0.0
    M = W.copy()
    M[np.abs(M) <= thr] = 0.0
    return M

def run_toy():
    texts = ["texts/doc1.txt","texts/doc2.txt"]
    rng = np.random.default_rng(42)
    W0 = rng.normal(0,1,(64,64))

    # 基準BP（p=0）を先に求める
    bp0 = bp_from_texts(texts)

    # ② |R|struct を基準BPから定義（A=3要素分を加算して同一単位化）
    r_struct = len(bp0["C"]) + len(bp0["phi"]) + 3

    results=[]
    for p in [0.0, 0.4, 0.8]:
        Wp = magnitude_prune(W0, p)
        # 簡易的に：剪定で命題を間引く近似
        bp = bp_from_texts(texts)
        keep = int(len(bp["C"]) * (1 - 0.9*p))
        bp["C"] = bp["C"][:max(1, keep)]

        # SP（構造保存）
        sp = round(sp_between(bp0, bp), 2)

        # ① CR のクリップ（0〜1の範囲に整合）
        sz = len(bp["C"]) + len(bp["phi"]) + 3
        cr = round(max(0.0, min(1.0, 1 - sz / max(1, r_struct))), 2)

        results.append({"p": p, "CR": cr, "SP": sp, "BP_size": len(bp["C"])})

    out_path = OUT/"toy"/"metrics.json"
    json.dump({"meta": {"mode": "toy"}, "results": results}, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

def run_bench():
    # 将来：Colabや本計測を接続。今はプレースホルダー。
    out_path = OUT/"bench"/"metrics.json"
    json.dump({"meta": {"mode": "bench"}, "results": []}, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

def run_n_r():
    # N-R: 価値ゲート κ の toy 実験
    texts = ["texts/doc1.txt","texts/doc2.txt"]
    bp_ref = bp_from_texts(texts)
    results = []
    for kappa in [0.2, 0.6]:
        bp_mod = bp_from_texts(texts)
        # κに応じて少しだけ命題数を縮退（価値依存の軽微な再構成の近似）
        keep = int(len(bp_mod["C"]) * (1 - 0.3 * kappa))
        bp_mod["C"] = bp_mod["C"][:max(1, keep)]
        sp = round(sp_between(bp_ref, bp_mod), 2)
        vs = round(1 - sp, 2)
        results.append({"kappa": kappa, "SP": sp, "VS": vs})
    out_path = OUT/"n_r"/"metrics.json"
    json.dump({"meta": {"mode": "n_r"}, "results": results}, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    if MODE == "toy":
        run_toy()
    elif MODE == "bench":
        run_bench()
    elif MODE == "n_r":
        run_n_r()
    else:
        print(f"Unknown MODE={MODE}")
