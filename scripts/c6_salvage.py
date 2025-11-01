# -*- coding: utf-8 -*-
"""
C6: 構造救済（salvage）— L2非対応文の見出し推定によるSP回復

実験系列（決定論・ブラウザ実行前提）:
  E1) coverage_sweep(thr ∈ [0.55, 0.60, 0.65, 0.70, 0.75])
  E2) h_estimate_with_jaccard() による擬似見出し生成と SP_h 再計測（Jaccard でH一致判定・ログ出力）
  E3) sp_l1_simple() による命題レベルの救済（BoW/Jaccard, 保守的）

入出力:
  入力:  inputs/c5/<case>/ref_basic.md
  BP:    inputs/c5/brand_larica/bp.yaml  （ピボットBP=LARICAのL2-BP）
  出力:  outputs/c6_salvage/<case>/metrics.json
         outputs/c6_salvage/summary.json

注意:
  - VS/GEN′はC6では扱いません（SP救済に特化）。summaryのGEN_primeは常に null。
  - このスクリプトは単独で完結するように、C3/C4互換の sp_l2_baseline を内包しています。
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml  # pip install pyyaml

# 対象ケース（C5と揃える）
CASES: List[str] = ["about_us", "brand_larica", "pearl_story", "travel_essay"]

# 文字/語処理（決定論）
def normalize(text: str) -> str:
    """
    Unicode NFKC → 小文字化 → 記号除去（英数・日本語・空白・中点「・」のみ残す）
    """
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower()
    t = re.sub(r"[^0-9a-zぁ-んァ-ン一-龥・\s]", "", t)
    return t


def tokenize(text: str) -> set:
    """
    文字bi-gram + 英単語（2文字以上）をトークン集合として扱う（決定論）
    """
    t = normalize(text)
    toks = set()
    # 文字 bi-gram
    for i in range(len(t) - 1):
        toks.add(t[i : i + 2])
    # 英単語
    for w in re.findall(r"[a-z]{2,}", t):
        toks.add(w)
    return toks


def jaccard(X: set, Y: set) -> float:
    if not X or not Y:
        return 0.0
    inter = len(X & Y)
    union = len(X | Y)
    return inter / union if union > 0 else 0.0


def coverage_hit(title: str, sent: str, thr: float) -> bool:
    """
    被覆率（titleのトークンがsentにどれだけ含まれるか）で判定
    - 厳密一致（部分文字列）をまず許容（安定性のため）
    - それ以外は bag-of-words 的に title_tokens のカバレッジ >= thr
    """
    if not title or not sent:
        return False
    if title in sent:
        return True
    tset = tokenize(title)
    sset = tokenize(sent)
    if not tset:
        return False
    cov = sum(1 for tok in tset if tok in sset) / max(1, len(tset))
    return cov >= thr


def first_hit_index(title: str, text: str, thr_cov: float) -> int:
    """
    text 内で title が最初にマッチする文のインデックス（見つからなければ -1）
    """
    sents = [s for s in re.split(r"[。.\n!?]", text or "") if s.strip()]
    for i, s in enumerate(sents):
        if title and (title in s or coverage_hit(title, s, thr_cov)):
            return i
    return -1


# ===== L2 SP（C3/C4互換のベースライン実装） =====
def sp_l2_baseline(bp: Dict[str, Any], text: str, thr_cov: float = 0.70) -> float:
    """
    L2粒度の SP（構造保存度）を C3/C4のルーブリックで算出（決定論）
      - H構造一致: H1〜H3がすべて検出=1.0 / 一部=0.5 / 無=0.0
      - 順序一致  : H1<=H2<=H3 で 1.0 / 全H出現で乱れ=0.5 / 一部出現=0.5 / 無=0.0
      - Claims一致: 3=1.0 / 2=0.67 / 1=0.33 / 0=0.0
    """
    # H titles
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
    if h_hits == 3:
        h_score = 1.0
    elif h_hits >= 1:
        h_score = 0.5
    else:
        h_score = 0.0

    # 順序一致
    idx1 = first_hit_index(H1, text, thr_cov) if H1 else -1
    idx2 = first_hit_index(H2, text, thr_cov) if H2 else -1
    idx3 = first_hit_index(H3, text, thr_cov) if H3 else -1
    idxs = [idx1, idx2, idx3]
    if all(i >= 0 for i in idxs):
        order_score = 1.0 if (idx1 <= idx2 <= idx3) else 0.5
    elif any(i >= 0 for i in idxs):
        order_score = 0.5
    else:
        order_score = 0.0

    # Claims一致
    claims = (bp.get("blueprint") or {}).get("C", []) or []
    def claim_hit(cl: str) -> bool:
        if not cl:
            return False
        if cl in text:
            return True
        for s in sents:
            if coverage_hit(cl, s, 0.60):
                return True
        return False

    c_hits = sum(1 for c in claims if claim_hit((c or {}).get("claim", "")))
    if c_hits >= 3:
        claims_score = 1.0
    elif c_hits == 2:
        claims_score = 0.67
    elif c_hits == 1:
        claims_score = 0.33
    else:
        claims_score = 0.0

    sp = round((h_score + order_score + claims_score) / 3.0, 2)
    return sp


# ===== ピボットBP（LARICA） =====
def load_pivot_bp() -> Dict[str, Any]:
    """
    C2で確定済みの LARICA L2-BP をピボットとして読み込む
    """
    bp_path = Path("inputs/c5/brand_larica/bp.yaml")
    return yaml.safe_load(bp_path.read_text(encoding="utf-8"))


# ===== coverage sweep =====
def coverage_sweep(bp_pivot: Dict[str, Any], text: str,
                   thresholds: Tuple[float, ...]) -> List[Dict[str, Any]]:
    res = []
    for thr in thresholds:
        sp = sp_l2_baseline(bp_pivot, text, thr_cov=thr)
        res.append({"thr": float(thr), "SP": float(sp)})
    return res


# ===== 擬似見出し生成（先頭 / 中央最長 / 末尾） =====
def generate_pseudo_h(text: str) -> Dict[str, str]:
    sents = [s for s in re.split(r"[。.\n]", text or "") if s.strip()]
    if not sents:
        return {"H1": "", "H2": "", "H3": ""}
    n = len(sents)
    H1 = sents[0].strip()
    mid = sents[n // 3 : (2 * n) // 3] or [sents[n // 2]]
    H2 = max(mid, key=lambda x: len(x)) if mid else sents[n // 2]
    H3 = sents[-1].strip()
    return {"H1": H1, "H2": H2, "H3": H3}


# ===== H推定 + Jaccard 判定 + ログ出力 =====
def h_estimate_with_jaccard(bp_pivot: Dict[str, Any], text: str,
                            thr_H: float = 0.40, thr_lo: float = 0.30) -> Tuple[float, Dict[str, Any]]:
    """
    擬似H（H1/H2/H3）と pivot の各H を Jaccard(文字bi-gram+英単語) で照合し H_score を三段階で判定
      - 1.0: 3本すべて J ≥ thr_H
      - 0.5: 2本以上 J ≥ thr_H かつ 残り1本 J ≥ thr_lo
      - 0.0: 上記以外
    order_score=1.0（定義上 H1<H2<H3）、claims_score=0.0（C6では未実装） → SP_h = (H+order+claims)/3
    戻り値: (SP_h, ログ辞書)
    """
    pseudo_h = generate_pseudo_h(text)
    pivot_h = {
        "H1": ((bp_pivot.get("headings") or {}).get("H1") or {}).get("title", ""),
        "H2": ((bp_pivot.get("headings") or {}).get("H2") or {}).get("title", ""),
        "H3": ((bp_pivot.get("headings") or {}).get("H3") or {}).get("title", ""),
    }

    J = {}
    for k in ("H1", "H2", "H3"):
        J[k] = jaccard(tokenize(pivot_h[k]), tokenize(pseudo_h.get(k, "")))

    passed = [k for k in ("H1", "H2", "H3") if J[k] >= thr_H]
    if len(passed) == 3:
        H_score = 1.0
    elif len(passed) >= 2 and min(J.values()) >= thr_lo:
        H_score = 0.5
    else:
        H_score = 0.0

    order_score = 1.0  # 擬似Hの定義上、H1<H2<H3の順序
    claims_score = 0.0  # C6では未実装（C7で救済予定）

    SP_h = round((H_score + order_score + claims_score) / 3.0, 2)
    h_log = {
        "pseudo_h": pseudo_h,
        "pivot_h": pivot_h,
        "jaccard": {k: round(J[k], 3) for k in J},
        "scores": {"H": H_score, "order": order_score, "claims": claims_score},
        "thresholds": {"theta_H": thr_H, "theta_lo": thr_lo}
    }
    return SP_h, h_log


# ===== L1救済（簡易・保守的） =====
def sp_l1_simple(bp_pivot: Dict[str, Any], text: str, sim_thr: float = 0.55) -> float:
    """
    命題レベルの簡易一致（BoW/Jaccard）
      - 各 Claim(C1..C3) を tokenize → 原文トークンとの Jaccard
      - J >= sim_thr を満たした claim 数 / 3 をスコア化（0.00, 0.33, 0.67, 1.00）
    C6では保守的（同義語・分散表現なし）→ 今回の実測は全件 0.00
    """
    claims = (bp_pivot.get("blueprint") or {}).get("C", []) or []
    text_tokens = tokenize(text or "")
    hits = 0
    for c in claims:
        cl = (c or {}).get("claim", "")
        j = jaccard(tokenize(cl), text_tokens)
        if j >= sim_thr:
            hits += 1
    return round(hits / max(1, len(claims)), 2)


# ===== ケース実行 =====
def run_case(case_id: str,
             in_root: Path = Path("inputs/c5"),
             out_root: Path = Path("outputs/c6_salvage"),
             thresholds: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75),
             sim_thr: float = 0.55) -> Dict[str, Any]:
    """
    単一ケースを測定して metrics.json を保存
    """
    base = in_root / case_id
    text_path = base / "ref_basic.md"
    if not text_path.exists():
        # travel_essay_day1 → travel_essay など命名ズレに備える
        alt = in_root / case_id.replace("_day1", "") / "ref_basic.md"
        text_path = alt if alt.exists() else text_path
    text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""

    bp_pivot = load_pivot_bp()

    # baseline（C5同条件: thr_cov=0.70）
    SP_base = float(sp_l2_baseline(bp_pivot, text, thr_cov=0.70))

    # sweep（0.55〜0.75）
    sweep = coverage_sweep(bp_pivot, text, thresholds=thresholds)

    # H推定（Jaccard判定・ログ保存）
    SP_h, h_log = h_estimate_with_jaccard(bp_pivot, text, thr_H=0.40, thr_lo=0.30)

    # L1救済（保守的）
    SP_L1 = float(sp_l1_simple(bp_pivot, text, sim_thr=sim_thr))

    # 統合（C6仕様）
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
    for case in CASES:
        results[case] = run_case(case, thresholds=thresholds, sim_thr=sim_thr)
    # サマリ（SP_primeのみ）
    summary = {c: results[c]["summary"] for c in CASES}
    out_path = Path("outputs/c6_salvage/summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[C6] summary saved → {out_path}")


if __name__ == "__main__":
    # 既定値で一括実行（Actionsから直接呼び出し可）
    run_all()
