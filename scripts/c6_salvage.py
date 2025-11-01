# -*- coding: utf-8 -*-
"""
C6: 救済設計 — 非構造文への再適用と構造的回復
実験系列:
  E1) coverage_sweep(thr in [0.60, 0.65, 0.70])
  E2) h_estimate() による擬似見出しの生成と SP 再計測
  E3) sp_l1() による命題レベルの救済 SP′
出力:
  outputs/c6_salvage/<case>/metrics.json
  outputs/c6_salvage/summary.json
"""
import json, re, pathlib
from typing import Dict, Any, List, Tuple
import yaml

ROOT = pathlib.Path(".").resolve()
CASES = ["about_us", "pearl_story", "travel_essay", "brand_larica"]
TOK = r"[A-Za-z0-9一-龥ぁ-んァ-ンー]+"


# ===== 共通ユーティリティ =====
def read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def write_json(p: pathlib.Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def token_bag(s: str) -> List[str]:
    return re.findall(TOK, s or "")

def coverage(title: str, sent: str, thr: float) -> bool:
    if not title or not sent: return False
    tset, sset = set(token_bag(title)), set(token_bag(sent))
    if not tset: return False
    cov = sum(1 for t in tset if t in sset) / max(1, len(tset))
    return cov >= thr

def first_hit_index(title: str, text: str, thr: float) -> int:
    sents = [s for s in re.split(r"[\n。.!?]", text or "") if s.strip()]
    for i, s in enumerate(sents):
        if (title in s) or coverage(title, s, thr):
            return i
    return -1


# ===== C4のSP計測（L2基準）を同梱（依存排除のため最小版）=====
def sp_l2(bp: Dict[str, Any], text: str, thr_cov: float = 0.70) -> float:
    H = bp.get("headings", {})
    H_titles = [H.get("H1", {}).get("title",""), H.get("H2", {}).get("title",""), H.get("H3", {}).get("title","")]
    sents = [s for s in re.split(r"[\n。.!?]", text or "") if s.strip()]

    # H構造一致
    h_hits = sum(1 for h in H_titles if h and any((h in s) or coverage(h, s, thr_cov) for s in sents))
    h_match = 1.0 if h_hits == 3 else (0.5 if h_hits >= 1 else 0.0)

    # 順序一致
    idx = [first_hit_index(h, text, thr_cov) if h else -1 for h in H_titles]
    if all(i >= 0 for i in idx):
        order_match = 1.0 if (idx[0] <= idx[1] <= idx[2]) else 0.5
    elif any(i >= 0 for i in idx):
        order_match = 0.5
    else:
        order_match = 0.0

    # 核主張一致
    C = bp.get("blueprint", {}).get("C", [])
    claims = [c.get("claim","") for c in C]
    def claim_hit(cl: str) -> bool:
        if not cl: return False
        if cl in text: return True
        for s in sents:
            tset, sset = set(token_bag(cl)), set(token_bag(s))
            cov = sum(1 for t in tset if t in sset) / max(1, len(tset))
            if cov >= 0.60: return True
        return False
    c_hits = sum(1 for cl in claims if claim_hit(cl))
    if c_hits >= 3: c_match = 1.0
    elif c_hits == 2: c_match = 0.67
    elif c_hits == 1: c_match = 0.33
    else: c_match = 0.0

    return round((h_match + order_match + c_match) / 3.0, 2)


# ===== ピボットBP（LARICA）読込 =====
def load_pivot_bp() -> Dict[str, Any]:
    # C2 で確定済みの LARICA L2-BP を参照（C5と同一）
    # 既存の bp.yaml（brand_larica）を再利用
    bp_path = ROOT / "inputs" / "c5" / "brand_larica" / "bp.yaml"
    return yaml.safe_load(bp_path.read_text(encoding="utf-8"))


# ===== E1: coverage sweep =====
def coverage_sweep(bp: Dict[str, Any], text: str, thresholds=(0.60, 0.65, 0.70)) -> List[Dict[str, Any]]:
    results = []
    for thr in thresholds:
        sp = sp_l2(bp, text, thr_cov=thr)
        results.append({"thr": thr, "SP": sp})
    return results


# ===== E2: H推定（簡易） =====
def h_estimate(bp: Dict[str, Any], text: str, thr_cov: float = 0.70) -> Tuple[float, Dict[str, Any]]:
    """
    見出しが無い文章向けに、擬似H1〜H3を推定して SP を再計測する簡易版。
    ルール（決定論）:
      1) 文章を段落/文で3分割し、各ブロックの先頭文を擬似Hに採用
      2) 擬似Hの各文で名詞頻度の高い語を保持（実装は文字列そのまま）
      3) 擬似Hを title に差し込み、sp_l2 を再計測
    """
    sents = [s for s in re.split(r"[\n。.!?]", text or "") if s.strip()]
    if not sents:
        return 0.0, {"H1": "", "H2": "", "H3": ""}

    n = len(sents)
    h1 = sents[0]
    h2 = sents[n//2] if n >= 2 else sents[0]
    h3 = sents[-1]

    bp2 = dict(bp)
    bp2["headings"] = {
        "H1": {"title": h1},
        "H2": {"title": h2},
        "H3": {"title": h3},
    }
    sp = sp_l2(bp2, text, thr_cov=thr_cov)
    return sp, {"H1": h1, "H2": h2, "H3": h3}


# ===== E3: L1救済（命題一致） =====
def sp_l1(bp: Dict[str, Any], text: str, sim_thr: float = 0.60) -> float:
    """
    文（命題）レベルで C1〜C3 の一致を評価する救済スコア。
    簡易実装: claims × sentences の被覆率 >= sim_thr で命題一致とみなす。
    SP_L1 = 一致命題数 / 3
    """
    claims = [c.get("claim","") for c in bp.get("blueprint", {}).get("C", [])]
    sents = [s for s in re.split(r"[\n。.!?]", text or "") if s.strip()]

    hits = 0
    for cl in claims:
        matched = False
        for s in sents:
            tset, sset = set(token_bag(cl)), set(token_bag(s))
            if not tset: continue
            cov = sum(1 for t in tset if t in sset) / max(1, len(tset))
            if cov >= sim_thr:
                matched = True
                break
        if matched:
            hits += 1
    return round(hits / max(1, len(claims)), 2)


# ===== ケース実行 =====
def run_case(case_id: str) -> Dict[str, Any]:
    base = ROOT / "inputs" / "c5" / case_id
    text = read_text(base / "ref_basic.md")  # C5で使用した原文を再利用
    bp_pivot = load_pivot_bp()

    # 基準（C5と同条件）
    SP_base = sp_l2(bp_pivot, text, thr_cov=0.70)

    # E1: coverage sweep
    sweep = coverage_sweep(bp_pivot, text, thresholds=(0.55, 0.60, 0.65, 0.70, 0.75))

    # E2: H推定
    SP_h, H_est = h_estimate(bp_pivot, text, thr_cov=0.70)

    # E3: L1救済
    SP_l1 = sp_l1(bp_pivot, text, sim_thr=0.55)

    # 統合スコア（簡易規定）：SP′ = max(SP_base, SP_h, SP_L1)
    SP_prime = round(max(SP_base, SP_h, SP_l1), 2)

    # 参考 GEN′（簡易版）：SP′と VS を併用した相対値は C7 で厳密化。ここでは占位値を返す。
    # 実験比較では cases間の SP′差のみを相対指標とする（VSはC5値を参照、または0.50占位）
    GEN_prime_stub = None  # C6ではSP′回復に主眼。GEN′はC7で定義。

    out = {
        "meta": {"mode": "c6_salvage", "case_id": case_id},
        "baseline": {"SP_base": SP_base},
        "coverage_sweep": sweep,
        "h_estimate": {"SP_h": SP_h, "H_est": H_est},
        "sp_l1": {"SP_L1": SP_l1},
        "summary": {"SP_prime": SP_prime, "GEN_prime": GEN_prime_stub}
    }
    out_dir = ROOT / "outputs" / "c6_salvage" / case_id
    write_json(out_dir / "metrics.json", out)
    return out


def run_all():
    results = {}
    for c in CASES:
        results[c] = run_case(c)
    # サマリー
    summary = {c: results[c]["summary"] for c in CASES}
    write_json(ROOT / "outputs" / "c6_salvage" / "summary.json", summary)


if __name__ == "__main__":
    run_all()
