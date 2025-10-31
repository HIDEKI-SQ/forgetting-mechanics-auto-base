# -*- coding: utf-8 -*-
"""
C5: 複数ケースの計測（CR / SP / VS）＋ GEN の集計
inputs/c5/{case}/ に ref_basic.md / ref_briefing.md / bp.yaml がある前提
出力:
  outputs/c5_cases/{case}/metrics.json
  outputs/c5_cases/gen_scores.json
"""
import os, re, json, yaml, pathlib
from typing import Dict, Any, List, Optional

ROOT = pathlib.Path(".").resolve()
TOK = r"[A-Za-z0-9一-龥ぁ-んァ-ンー]+"

def read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def write_json(p: pathlib.Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def token_bag(s: str) -> List[str]:
    return re.findall(TOK, s or "")

def coverage(title: str, sent: str, thr: float = 0.70) -> bool:
    if not title or not sent: return False
    tset, sset = set(token_bag(title)), set(token_bag(sent))
    if not tset: return False
    cov = sum(1 for t in tset if t in sset) / max(1, len(tset))
    return cov >= thr

def first_hit_index(title: str, text: str, thr: float = 0.70) -> Optional[int]:
    sents = [s for s in re.split(r"[\n。.!?]", text) if s.strip()]
    for i, s in enumerate(sents):
        if title in s or coverage(title, s, thr=thr):
            return i
    return None

def sp_l2(bp: Dict[str, Any], text: str) -> float:
    """C3ルーブリック準拠（H構造/順序/核主張）"""
    H = bp.get("headings", {})
    H_titles = [H.get("H1", {}).get("title",""), H.get("H2", {}).get("title",""), H.get("H3", {}).get("title","")]
    sents = [s for s in re.split(r"[\n。.!?]", text or "") if s.strip()]

    # H構造一致
    h_hits = sum(1 for h in H_titles if h and any((h in s) or coverage(h, s, 0.70) for s in sents))
    h_match = 1.0 if h_hits == 3 else (0.5 if h_hits >= 1 else 0.0)

    # 順序一致
    idx = [first_hit_index(h, text, 0.70) if h else None for h in H_titles]
    if all(i is not None for i in idx):
        order_match = 1.0 if (idx[0] <= idx[1] <= idx[2]) else 0.5
    elif any(i is not None for i in idx):
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
    elif c_hits == 2: c_match = 0.67  # ★ 2/3 = 0.67
    elif c_hits == 1: c_match = 0.33
    else: c_match = 0.0

    return round((h_match + order_match + c_match) / 3.0, 2)

# VS（Fi差分）: 比率差分の連続値（clip 0.70）
EMO = ("感じ","思い","心","願い","不安","揺れ","喜び","景色","物語","沖縄","海","空","旅")
LOG = ("目的","機能","設計","仕様","戦略","評価","指標","構造","仮説","測定","実装","方針","運用","理念")
SING= ("私","僕","自分")
ORG = ("私たち","我々","当社","ブランド","組織","顧客","チーム","ステークホルダー")
NARR= ("ある日","昨日","今日","明日","物語","記憶","出会い","歩み")
BRIF= ("理念","価値命題","KPI","方針","戦略","運用","評価","測定","再現性")

def ratio_posneg(s: str, pos: tuple, neg: tuple) -> float:
    p = sum(len(re.findall(t, s or "")) for t in pos)
    n = sum(len(re.findall(t, s or "")) for t in neg)
    return p / max(1, (p + n))

def vs_proxy(text_basic: str, text_briefing: str) -> float:
    tone  = abs(ratio_posneg(text_basic, EMO, LOG) - ratio_posneg(text_briefing, EMO, LOG))
    focus = abs(ratio_posneg(text_basic, SING, ORG) - ratio_posneg(text_briefing, SING, ORG))
    ctx   = abs(ratio_posneg(text_basic, NARR, BRIF) - ratio_posneg(text_briefing, NARR, BRIF))
    return round(min(0.70, (tone + focus + ctx) / 3.0), 2)

def compute_cr(bp: Dict[str,Any], ref_basic: str) -> float:
    """文字ベースCR。Hタイトル＋C主張の長さをBP文字数近似として用いる（C2互換方針）。"""
    H = bp.get("headings", {})
    hchars = sum(len(H.get(k,{}).get("title","")) for k in ("H1","H2","H3"))
    claims = bp.get("blueprint",{}).get("C",[])
    cchars = sum(len(c.get("claim","")) for c in claims)
    bp_chars = hchars + cchars
    ref_chars = len(ref_basic or "")
    if ref_chars == 0: return 0.0
    return round(1 - (bp_chars / ref_chars), 2)

def run_case(case_id: str) -> Dict[str,Any]:
    base = ROOT / "inputs" / "c5" / case_id
    bp_file = base / "bp.yaml"
    ref_basic = read_text(base / "ref_basic.md")
    ref_brief = read_text(base / "ref_briefing.md")
    bp = yaml.safe_load(bp_file.read_text(encoding="utf-8"))
    # 計測
    CR = compute_cr(bp, ref_basic)
    SP = sp_l2(bp, ref_brief)
    VS = vs_proxy(ref_basic, ref_brief)

    out_dir = ROOT / "outputs" / "c5_cases" / case_id
    out = {
        "meta": {"mode":"c5_cases", "case_id": case_id},
        "ref_basic": {"chars": len(ref_basic)},
        "bp": {"chars_approx": sum(len(bp.get("headings",{}).get(k,{}).get("title","")) for k in ("H1","H2","H3"))
                           + sum(len(c.get("claim","")) for c in bp.get("blueprint",{}).get("C",[]))},
        "metrics": {"CR": CR, "SP": SP, "VS": VS}
    }
    write_json(out_dir / "metrics.json", out)
    return out

def load_cases() -> List[str]:
    c5 = ROOT / "inputs" / "c5"
    return sorted([p.name for p in c5.iterdir() if p.is_dir()])

def run_all():
    cases = load_cases()
    results = {c: run_case(c) for c in cases}

    # GEN: ペアごとのSP/VS差分から算出
    pairs = []
    keys = list(results.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            A, B = keys[i], keys[j]
            sp_a, vs_a = results[A]["metrics"]["SP"], results[A]["metrics"]["VS"]
            sp_b, vs_b = results[B]["metrics"]["SP"], results[B]["metrics"]["VS"]
            gen_ab = round(1 - 0.5 * (abs(sp_a - sp_b) + abs(vs_a - vs_b)), 2)
            pairs.append({"src": A, "tgt": B, "SP_AB": round(abs(sp_a - sp_b),2),
                          "VS_AB": round(abs(vs_a - vs_b),2), "GEN_AB": gen_ab})

    gen_mean = round(sum(p["GEN_AB"] for p in pairs) / max(1,len(pairs)), 2) if pairs else None
    summary = {"GEN_mean": gen_mean,
               "GEN_min": min((p["GEN_AB"] for p in pairs), default=None),
               "GEN_max": max((p["GEN_AB"] for p in pairs), default=None)}
    write_json(ROOT / "outputs" / "c5_cases" / "gen_scores.json",
               {"meta":{"mode":"c5_cases","cases":keys}, "pairs":pairs, "summary":summary})

if __name__ == "__main__":
    run_all()
