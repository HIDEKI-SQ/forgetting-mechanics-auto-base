# -*- coding: utf-8 -*-
"""
C6: 構造救済（salvage）— L2非対応文の見出し推定によるSP回復（score-based 版）

改訂点:
- 擬似Hの選定を「位置（先頭/中央最長/末尾）」から
  「pivot Hごとに coverage/Jaccard が最大の文を順序制約つきで選ぶ」方式に変更
- H一致判定は coverage を主、Jaccard を従（ログ保存）
"""

from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml  # pip install pyyaml

CASES: List[str] = ["about_us", "brand_larica", "pearl_story", "travel_essay"]

# ============== 文字処理 ==============
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower()
    # 英数・日本語・空白・中点のみ許容
    return re.sub(r"[^0-9a-zぁ-んァ-ン一-龥・\s]", "", t)

def tokenize(text: str) -> set:
    t = normalize(text)
    toks = set()
    for i in range(len(t) - 1):
        toks.add(t[i:i+2])             # 文字bi-gram
    for w in re.findall(r"[a-z]{2,}", t):
        toks.add(w)                    # 英単語(2+)
    return toks

def jaccard(X: set, Y: set) -> float:
    if not X or not Y:
        return 0.0
    inter = len(X & Y)
    union = len(X | Y)
    return inter / union if union else 0.0

def coverage_asym(pivot_title: str, sent: str) -> float:
    """
    非対称 coverage：pivot の核語が sent にどれだけ含まれるか（0..1）
    """
    P = tokenize(pivot_title)
    S = tokenize(sent)
    if not P: return 0.0
    hit = sum(1 for tok in P if tok in S)
    return hit / len(P)

# ============== L2基礎 SP（C3/C4準拠） ==============
def coverage_hit(title: str, sent: str, thr: float) -> bool:
    if not title or not sent: return False
    if title in sent: return True
    P, S = tokenize(title), tokenize(sent)
    if not P: return False
    cov = sum(1 for tok in P if tok in S) / len(P)
    return cov >= thr

def first_hit_index(title: str, text: str, thr_cov: float) -> int:
    sents = [s for s in re.split(r"[。.\n!?]", text or "") if s.strip()]
    for i, s in enumerate(sents):
        if title and (title in s or coverage_hit(title, s, thr_cov)):
            return i
    return -1

def sp_l2_baseline(bp: Dict[str, Any], text: str, thr_cov: float = 0.70) -> float:
    H = (bp.get("headings") or {})
    H1 = (H.get("H1") or {}).get("title", "")
    H2 = (H.get("H2") or {}).get("title", "")
    H3 = (H.get("H3") or {}).get("title", "")
    sents = [s for s in re.split(r"[。.\n!?]", text or "") if s.strip()]

    # H構造一致
    h_hits = 0
    for h in (H1, H2, H3):
        if h and any((h in s) or coverage_hit(h, s, thr_cov) for s in sents):
            h_hits += 1
    h_score = 1.0 if h_hits == 3 else 0.5 if h_hits >= 1 else 0.0

    # 順序一致
    idx1 = first_hit_index(H1, text, thr_cov) if H1 else -1
    idx2 = first_hit_index(H2, text, thr_cov) if H2 else -1
    idx3 = first_hit_index(H3, text, thr_cov) if H3 else -1
    if all(i >= 0 for i in (idx1, idx2, idx3)):
        order_score = 1.0 if (idx1 <= idx2 <= idx3) else 0.5
    elif any(i >= 0 for i in (idx1, idx2, idx3)):
        order_score = 0.5
    else:
        order_score = 0.0

    # Claims一致（C6では baseline のまま）
    claims = (bp.get("blueprint") or {}).get("C", []) or []
    def claim_hit(cl: str) -> bool:
        if not cl: return False
        if cl in text: return True
        for s in sents:
            if coverage_hit(cl, s, 0.60): return True
        return False
    c_hits = sum(1 for c in claims if claim_hit((c or {}).get("claim", "")))
    claims_score = 1.0 if c_hits >= 3 else 0.67 if c_hits == 2 else 0.33 if c_hits == 1 else 0.0

    return round((h_score + order_score + claims_score)/3.0, 2)

# ============== ピボットBP（LARICA） ==============
def load_pivot_bp() -> Dict[str, Any]:
    bp_path = Path("inputs/c5/brand_larica/bp.yaml")
    return yaml.safe_load(bp_path.read_text(encoding="utf-8"))

# ============== coverage sweep ==============
def coverage_sweep(bp_pivot: Dict[str, Any], text: str,
                   thresholds: Tuple[float, ...]) -> List[Dict[str, Any]]:
    return [{"thr": float(t), "SP": float(sp_l2_baseline(bp_pivot, text, thr_cov=t))}
            for t in thresholds]

# ============== 擬似H（score-based選定） ==============
def select_pseudo_h_by_match(bp_pivot: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    pivot H1/H2/H3 それぞれに対して、coverage が最大の文を
    順序制約（H1 <= H2 <= H3 の位置）を保ちながら選ぶ。
    coverage が同点の場合は Jaccard が大きい方を優先。
    """
    sents = [s for s in re.split(r"[。.\n]", text or "") if s.strip()]
    if not sents:
        return {"H1":"", "H2":"", "H3":""}
    N = len(sents)
    pivot = {
        "H1": ((bp_pivot.get("headings") or {}).get("H1") or {}).get("title",""),
        "H2": ((bp_pivot.get("headings") or {}).get("H2") or {}).get("title",""),
        "H3": ((bp_pivot.get("headings") or {}).get("H3") or {}).get("title",""),
    }
    # H1: 全域から coverage 最大
    best1 = max(range(N), key=lambda i: (coverage_asym(pivot["H1"], sents[i]),
                                         jaccard(tokenize(pivot["H1"]), tokenize(sents[i]))))
    # H2: best1 以降から最大
    rng2 = range(best1, N) if best1 < N else range(N)
    best2 = max(rng2, key=lambda i: (coverage_asym(pivot["H2"], sents[i]),
                                     jaccard(tokenize(pivot["H2"]), tokenize(sents[i]))))
    # H3: best2 以降から最大
    rng3 = range(best2, N) if best2 < N else range(N)
    best3 = max(rng3, key=lambda i: (coverage_asym(pivot["H3"], sents[i]),
                                     jaccard(tokenize(pivot["H3"]), tokenize(sents[i]))))
    return {"H1": sents[best1].strip(), "H2": sents[best2].strip(), "H3": sents[best3].strip()}

# ============== H推定 + 判定 + ログ出力 ==============
def h_estimate_score_based(bp_pivot: Dict[str, Any], text: str,
                           theta_H: float = 0.40, theta_lo: float = 0.30) -> Tuple[float, Dict[str, Any]]:
    """
    擬似Hは score-based 選定。判定は coverage を主、Jaccard はログに保存。
    H_score: 3本の coverage >= theta_H なら1.0、2本>=theta_H かつ残り>=theta_lo で0.5、他0.0
    """
    pseudo_h = select_pseudo_h_by_match(bp_pivot, text)
    pivot_h = {
        "H1": ((bp_pivot.get("headings") or {}).get("H1") or {}).get("title",""),
        "H2": ((bp_pivot.get("headings") or {}).get("H2") or {}).get("title",""),
        "H3": ((bp_pivot.get("headings") or {}).get("H3") or {}).get("title",""),
    }
    cov = {k: round(coverage_asym(pivot_h[k], pseudo_h[k]), 3) for k in ("H1","H2","H3")}
    jac = {k: round(jaccard(tokenize(pivot_h[k]), tokenize(pseudo_h[k])), 3) for k in ("H1","H2","H3")}

    passed = [k for k in ("H1","H2","H3") if cov[k] >= theta_H]
    if len(passed) == 3:
        H_score = 1.0
    elif len(passed) >= 2 and min(cov.values()) >= theta_lo:
        H_score = 0.5
    else:
        H_score = 0.0

    order_score = 1.0  # 選定が順序制約付きなので常に昇順
    claims_score = 0.0 # C6では未実装（C7で救済予定）
    SP_h = round((H_score + order_score + claims_score)/3.0, 2)

    h_log = {
        "pseudo_h": pseudo_h,
        "pivot_h": pivot_h,
        "coverage": cov,
        "jaccard": jac,
        "scores": {"H": H_score, "order": order_score, "claims": claims_score},
        "thresholds": {"theta_H": theta_H, "theta_lo": theta_lo}
    }
    return SP_h, h_log

# ============== L1（簡易） ==============
def sp_l1_simple(bp_pivot: Dict[str, Any], text: str, sim_thr: float = 0.55) -> float:
    claims = (bp_pivot.get("blueprint") or {}).get("C", []) or []
    T = tokenize(text or "")
    hits = 0
    for c in claims:
        cl = (c or {}).get("claim","")
        j = jaccard(tokenize(cl), T)
        if j >= sim_thr:
            hits += 1
    return round(hits / max(1, len(claims)), 2)

# ============== ケース実行 ==============
def run_case(case_id: str,
             in_root: Path = Path("inputs/c5"),
             out_root: Path = Path("outputs/c6_salvage"),
             thresholds: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75),
             sim_thr: float = 0.55) -> Dict[str, Any]:
    base = in_root / case_id
    text_path = base / "ref_basic.md"
    if not text_path.exists():
        alt = in_root / case_id.replace("_day1", "") / "ref_basic.md"
        text_path = alt if alt.exists() else text_path
    text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
    bp_pivot = load_pivot_bp()

    SP_base = float(sp_l2_baseline(bp_pivot, text, thr_cov=0.70))
    sweep   = [{"thr": float(t), "SP": float(sp_l2_baseline(bp_pivot, text, thr_cov=t))} for t in thresholds]

    SP_h, h_log = h_estimate_score_based(bp_pivot, text, theta_H=0.40, theta_lo=0.30)
    SP_L1 = float(sp_l1_simple(bp_pivot, text, sim_thr=sim_thr))
    SP_prime = round(max(SP_base, SP_h, SP_L1), 2)

    out = {
        "meta": {"mode": "c6_salvage", "case_id": case_id},
        "baseline": {"SP_base": SP_base},
        "coverage_sweep": sweep,
        "h_estimate": {"SP_h": SP_h, "log": h_log},
        "sp_l1": {"SP_L1": SP_L1},
        "summary": {"SP_prime": SP_prime, "GEN_prime": None}
    }
    out_dir = out_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def run_all(thresholds: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75),
            sim_thr: float = 0.55) -> None:
    results: Dict[str, Any] = {}
    for c in CASES:
        results[c] = run_case(c, thresholds=thresholds, sim_thr=sim_thr)
    summary = {c: results[c]["summary"] for c in CASES}
    out_path = Path("outputs/c6_salvage/summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[C6] summary saved → {out_path}")

if __name__ == "__main__":
    run_all()
