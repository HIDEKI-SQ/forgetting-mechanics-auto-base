# -*- coding: utf-8 -*-
"""
C4 Implementation (ref/auto/delta)
- ref（人手）と auto（半自動テンプレ）で R′ を2条件（basic/briefing）生成
- L2ルーブリックで SP/VS を測定し、差分（delta）を出力
- 目的は「工程の機械化可能性」の検証であり、優劣比較ではない
"""
import os, json, pathlib, re
from typing import List, Dict, Any, Optional

import yaml  # requirements.txt は pyyaml でOK

TOK = r"[A-Za-z0-9一-龥ぁ-んァ-ンー]+"

# -------------------------
# text utils
# -------------------------
def read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def write_text(p: pathlib.Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def token_bag(s: str) -> List[str]:
    return re.findall(TOK, s)

def coverage(title: str, sent: str, thr: float = 0.70) -> bool:
    """タイトル準厳密一致：完全一致 or トークン被覆率>=thr"""
    if not title or not sent:
        return False
    tset = set(token_bag(title))
    if not tset:
        return False
    sset = set(token_bag(sent))
    cov = sum(1 for t in tset if t in sset) / max(1, len(tset))
    return cov >= thr

def first_hit_index(title: str, text: str, thr: float = 0.70) -> Optional[int]:
    sents = [s for s in re.split(r"[\n。.!?]", text) if s.strip()]
    for i, s in enumerate(sents):
        if title in s or coverage(title, s, thr=thr):
            return i
    return None

# -------------------------
# SP (L2 rubric)
# -------------------------
def sp_l2(bp: Dict[str, Any], text: str) -> float:
    H = bp.get("headings", {})
    H_titles = [H.get("H1", {}).get("title", ""),
                H.get("H2", {}).get("title", ""),
                H.get("H3", {}).get("title", "")]
    # H構造一致（準厳密：被覆率>=0.70）
    h_hits = 0
    sents = [s for s in re.split(r"[\n。.!?]", text) if s.strip()]
    for h in H_titles:
        if not h:
            continue
        if any((h in s) or coverage(h, s, thr=0.70) for s in sents):
            h_hits += 1
    h_match = 1.0 if h_hits == 3 else (0.5 if h_hits >= 1 else 0.0)

    # 順序一致
    idx = [first_hit_index(h, text, thr=0.70) if h else None for h in H_titles]
    if all(i is not None for i in idx):
        order_match = 1.0 if (idx[0] <= idx[1] <= idx[2]) else 0.5
    elif any(i is not None for i in idx):
        order_match = 0.5
    else:
        order_match = 0.0

    # 核主張一致（3/3=1.0, 2/3=0.67, 1/3=0.33, 0/3=0.0）
    C = bp.get("blueprint", {}).get("C", [])
    claims = [c.get("claim", "") for c in C]
    def claim_hit(cl: str) -> bool:
        if not cl:
            return False
        if cl in text:
            return True
        for s in sents:
            tset = set(token_bag(cl)); sset = set(token_bag(s))
            cov = sum(1 for t in tset if t in sset) / max(1, len(tset))
            if cov >= 0.60:
                return True
        return False
    c_hits = sum(1 for cl in claims if claim_hit(cl))
    if c_hits >= 3:
        c_match = 1.0
    elif c_hits == 2:
        c_match = 0.67
    elif c_hits == 1:
        c_match = 0.33
    else:
        c_match = 0.0

    return round((h_match + order_match + c_match) / 3.0, 2)

# -------------------------
# VS proxy（tone/focus/context）
# -------------------------
EMO = ("感じ","思い","心","願い","不安","揺れ","喜び","景色","物語","沖縄","海","空","旅")
LOG = ("目的","機能","設計","仕様","戦略","評価","指標","構造","仮説","測定","実装","方針","運用","理念")
SING= ("私","僕","自分")
ORG = ("私たち","我々","当社","ブランド","組織","顧客","チーム","ステークホルダー")
NARR= ("ある日","昨日","今日","明日","物語","記憶","出会い","歩み")
BRIF= ("理念","価値命題","KPI","方針","戦略","運用","評価","測定","再現性")

def ratio_posneg(s: str, pos: tuple, neg: tuple) -> float:
    p = sum(len(re.findall(t, s)) for t in pos)
    n = sum(len(re.findall(t, s)) for t in neg)
    return p / max(1, (p + n))

def vs_proxy(text_basic: str, text_briefing: str) -> float:
    tone = abs(ratio_posneg(text_basic, EMO, LOG) - ratio_posneg(text_briefing, EMO, LOG))
    focus= abs(ratio_posneg(text_basic, SING, ORG) - ratio_posneg(text_briefing, SING, ORG))
    ctx  = abs(ratio_posneg(text_basic, NARR, BRIF) - ratio_posneg(text_briefing, NARR, BRIF))
    return round(min(0.70, (tone + focus + ctx) / 3.0), 2)

# -------------------------
# auto renderer（H1にだけC1、H2/H3はパラフレーズ）
# -------------------------
def render_auto_paragraphs(bp: Dict[str, Any], style: str) -> str:
    H = bp.get("headings", {})
    titles = [H.get("H1", {}).get("title",""),
              H.get("H2", {}).get("title",""),
              H.get("H3", {}).get("title","")]
    C = [c.get("claim","") for c in bp.get("blueprint",{}).get("C",[])]

    if style == "basic":
        seg1 = f"{titles[0]}。{C[0] if len(C)>0 else ''}。心の揺れと願いを抱えながら、私は今日の景色に出会い直す。"
        seg2 = f"{titles[1]}。反復の底で少しずつ変わる自分を、昨日と明日のあいだに感じている。"
        seg3 = f"{titles[2]}。歩みの記憶を束ね、物語の軸を保つと決めた。"
        return " ".join([seg1, seg2, seg3])
    else:
        seg1 = f"{titles[0]}—{('再出発の核としてLarimarを定義する') if len(C)>0 else ''}。我々は価値命題と方針を明確にし、評価指標を共有する。"
        seg2 = f"{titles[1]}—反復は学習螺旋として運用する。組織・顧客・チーム間の再現性を高めるKPIを設定する。"
        seg3 = f"{titles[2]}—価値×構造をブランド体験に統合し、理念の転写率を測定する。"
        return " ".join([seg1, seg2, seg3])

# -------------------------
# public API: run(case_id, bp_path)  ← ここが run_common.py から呼ばれる
# -------------------------
def run(case_id: str, bp_path: str, out_root: str = "outputs/c4_impl") -> pathlib.Path:
    case = case_id or "brand_larica"
    bp_guess = bp_path or f"outputs/c2_manual/bp_{case}.yaml"

    bp_file = pathlib.Path(bp_guess)
    if not bp_file.exists():
        raise FileNotFoundError(f"BP not found: {bp_file}")

    bp = yaml.safe_load(bp_file.read_text(encoding="utf-8"))

    in_dir = pathlib.Path(f"inputs/c4/{case}")
    ref_basic_p = in_dir / "ref_basic.md"
    ref_brief_p = in_dir / "ref_briefing.md"

    # ref（存在しなければ delta は None のまま）
    ref_basic_txt = read_text(ref_basic_p)
    ref_brief_txt = read_text(ref_brief_p)

    # auto（決定論テンプレ）
    auto_basic_txt = render_auto_paragraphs(bp, "basic")
    auto_brief_txt = render_auto_paragraphs(bp, "briefing")

    out_dir = pathlib.Path(out_root) / case
    write_text(out_dir / "regen_auto_basic.md", auto_basic_txt)
    write_text(out_dir / "regen_auto_briefing.md", auto_brief_txt)
    if ref_basic_txt: write_text(out_dir / "regen_ref_basic.md", ref_basic_txt)
    if ref_brief_txt: write_text(out_dir / "regen_ref_briefing.md", ref_brief_txt)

    # 測定
    sp_auto = round((sp_l2(bp, auto_basic_txt)+sp_l2(bp, auto_brief_txt))/2, 2)
    vs_auto = vs_proxy(auto_basic_txt, auto_brief_txt)

    SP_ref = round((sp_l2(bp, ref_basic_txt)+sp_l2(bp, ref_brief_txt))/2, 2) if (ref_basic_txt and ref_brief_txt) else None
    VS_ref = vs_proxy(ref_basic_txt, ref_brief_txt) if (ref_basic_txt and ref_brief_txt) else None

    metrics = {
        "meta":  {"mode":"c4_impl","case_id":case,"bp_path":str(bp_file)},
        "ref":   {"SP": SP_ref, "VS": VS_ref},
        "auto":  {"SP": sp_auto, "VS": vs_auto},
        "delta": {"dSP": None if SP_ref is None else round(abs(sp_auto - SP_ref),2),
                  "dVS": None if VS_ref is None else round(abs(vs_auto - VS_ref),2)}
    }
    out_path = out_dir / "metrics.json"
    write_text(out_path, json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")
    return out_path

# 直接実行にも対応（Actions では run_common から import される）
if __name__ == "__main__":
    CASE = os.environ.get("TARGET_ID","brand_larica")
    BP_PATH = os.environ.get("BP_PATH","")
    run(CASE, BP_PATH)
