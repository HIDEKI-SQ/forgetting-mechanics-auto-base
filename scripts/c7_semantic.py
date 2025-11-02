# -*- coding: utf-8 -*-
"""
C7: 意味救済（Semantic Salvage）
 - 目的: C6で形式救済されたテキストに対して、意味的Claims救済を導入し SP″ を算出する。
 - 方法: pivot Claims と原文文の意味類似（SBERT多言語）を用いて Claims スコアを計算。
 - 併せて VS を (ref_basic vs ref_briefing) で算出し、GEN′ を出力。
 - 完全決定論: 乱数・温度・外部APIなし（モデルは固定名で取得）。
入出力:
  inputs/c5/<case>/{ref_basic.md, ref_briefing.md}, inputs/c5/brand_larica/bp.yaml
  outputs/c7_semantic/<case>/metrics.json, outputs/c7_semantic/gen_prime.json
"""

from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import yaml

# ====== 外部モデル（SBERT多言語） ======
# Actions ランナーで自動DLされます
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CASES: List[str] = ["about_us", "brand_larica", "pearl_story", "travel_essay"]

# ===== 基本処理 =====
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    return t.strip()

def split_sentences(text: str) -> List[str]:
    # 「。」と改行で区切る。空要素は除外
    parts = re.split(r"[。.\n]", normalize(text))
    return [p.strip() for p in parts if p.strip()]

# ===== pivot BP（LARICA） =====
def load_pivot_bp() -> Dict[str, Any]:
    bp_path = Path("inputs/c5/brand_larica/bp.yaml")
    bp = yaml.safe_load(bp_path.read_text(encoding="utf-8"))
    # H（存在と順序）はC6ですでに救済済みの前提 → C7ではH=1.0, order=1.0 とみなす
    H = {
        "H1": ((bp.get("headings") or {}).get("H1") or {}).get("title",""),
        "H2": ((bp.get("headings") or {}).get("H2") or {}).get("title",""),
        "H3": ((bp.get("headings") or {}).get("H3") or {}).get("title",""),
    }
    claims = [(c or {}).get("claim","") for c in (bp.get("blueprint") or {}).get("C", []) or []]
    return {"H": H, "C": claims[:3]}  # C1..C3 まで採用

# ===== VS（C4/C5方式の簡易プロキシ） =====
EMO = ("感じ","思い","心","願い","不安","揺れ","喜び","景色","物語","沖縄","海","空","旅")
LOG = ("目的","機能","設計","仕様","戦略","評価","指標","構造","仮説","測定","実装","方針","運用","理念")
SING= ("私","僕","自分")
ORG = ("私たち","我々","当社","ブランド","組織","顧客","チーム","ステークホルダー")
NARR= ("ある日","昨日","今日","明日","物語","記憶","出会い","歩み")
BRIF= ("理念","価値命題","KPI","方針","戦略","運用","評価","測定","再現性")

def ratio_posneg(s: str, pos: Tuple[str, ...], neg: Tuple[str, ...]) -> float:
    import re
    p = sum(len(re.findall(t, s)) for t in pos)
    n = sum(len(re.findall(t, s)) for t in neg)
    return p / max(1, (p + n))

def vs_proxy(text_basic: str, text_briefing: str) -> float:
    tone  = abs(ratio_posneg(text_basic, EMO, LOG) - ratio_posneg(text_briefing, EMO, LOG))
    focus = abs(ratio_posneg(text_basic, SING, ORG) - ratio_posneg(text_briefing, SING, ORG))
    ctx   = abs(ratio_posneg(text_basic, NARR, BRIF) - ratio_posneg(text_briefing, NARR, BRIF))
    return round(min(0.70, (tone + focus + ctx) / 3.0), 2)

# ===== BERT類似度（cosine） =====
_model: SentenceTransformer|None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed(texts: List[str]) -> np.ndarray:
    model = get_model()
    emb = model.encode(texts, normalize_embeddings=False)  # 後で自分で正規化
    emb = np.array(emb, dtype=np.float32)
    # L2正規化
    denom = np.maximum(norm(emb, axis=1, keepdims=True), 1e-8)
    return emb / denom

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(a @ b, -1.0, 1.0))

# ===== SP″（意味救済後）の計算 =====
def compute_sp_dd(text: str, ref_brief: str, pivot: Dict[str,Any],
                  theta_C: float = 0.50) -> Dict[str, Any]:
    """
    SP″ = mean(H=1.0, order=1.0, Claims_score)
      Claims_score: pivot C1..C3 と原文文の最大類似（SBERT）のうち、閾値を超えたヒット数/3
    VS: ref_basic vs ref_briefing （C4/C5プロキシ）
    """
    sents = split_sentences(text)
    claims = pivot["C"]  # C1..C3
    # 文章・Claims を埋め込み
    if not sents:
        claims_score = 0.0
        hits_detail = []
    else:
        emb_sents = embed(sents)
        emb_claims = embed(claims)
        hits = 0
        hits_detail = []
        for i, c_emb in enumerate(emb_claims):
            sims = emb_sents @ c_emb  # cos類似（正規化済）
            j = int(np.argmax(sims))
            sim = float(sims[j])
            pass_flag = sim >= theta_C
            if pass_flag:
                hits += 1
            hits_detail.append({"claim": claims[i], "best_sent": sents[j], "sim": round(sim, 3), "pass": pass_flag})
        # ヒット数→スコア
        if hits >= 3: claims_score = 1.0
        elif hits == 2: claims_score = 0.67
        elif hits == 1: claims_score = 0.33
        else: claims_score = 0.0

    H_score = 1.0   # C6のpresenceゲートにより既に形式救済済み
    order_score = 1.0
    SP_dd = round((H_score + order_score + claims_score)/3.0, 2)
    return {"SP_dd": SP_dd, "claims_score": claims_score, "hits": hits_detail}

# ===== ケース実行 =====
def run_case(case_id: str) -> Dict[str,Any]:
    base = Path("inputs/c5") / case_id
    ref_basic = read_text(base / "ref_basic.md")
    ref_brief = read_text(base / "ref_briefing.md")  # C5で生成済み
    pivot = load_pivot_bp()

    # SP″（意味救済）
    sem = compute_sp_dd(ref_basic, ref_brief, pivot, theta_C=0.50)

    # VS（プロキシ）
    VS = vs_proxy(ref_basic, ref_brief) if (ref_basic and ref_brief) else None

    # 出力
    out = {
        "meta": {"mode":"c7_semantic", "case_id": case_id, "model": MODEL_NAME},
        "semantic": sem,   # {SP_dd, claims_score, hits:[...]}
        "VS": VS
    }
    out_dir = Path("outputs/c7_semantic") / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def run_all() -> None:
    results = {c: run_case(c) for c in CASES}

    # GEN′: 全ペア（C5の式をSP″に置換）
    pairs = []
    keys = list(results.keys())
    def extract_spdd(k: str) -> float:
        return float(results[k]["semantic"]["SP_dd"])
    def extract_vs(k: str) -> float:
        v = results[k]["VS"]
        return float(v) if v is not None else 0.0

    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            A, B = keys[i], keys[j]
            sp_a, vs_a = extract_spdd(A), extract_vs(A)
            sp_b, vs_b = extract_spdd(B), extract_vs(B)
            genp = round(1.0 - 0.5*(abs(sp_a-sp_b) + abs(vs_a-vs_b)), 2)
            pairs.append({"src": A, "tgt": B, "SP_dd_A": sp_a, "SP_dd_B": sp_b,
                          "VS_A": vs_a, "VS_B": vs_b, "GEN_prime": genp})

    summary = {
        "GEN_prime_mean": round(sum(p["GEN_prime"] for p in pairs)/max(1,len(pairs)), 2) if pairs else None,
        "GEN_prime_min": min((p["GEN_prime"] for p in pairs), default=None),
        "GEN_prime_max": max((p["GEN_prime"] for p in pairs), default=None)
    }
    out = {"meta":{"mode":"c7_semantic","cases":keys},
           "pairs": pairs, "summary": summary}
    out_path = Path("outputs/c7_semantic/gen_prime.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[C7] Saved: {out_path}")

if __name__ == "__main__":
    run_all()
