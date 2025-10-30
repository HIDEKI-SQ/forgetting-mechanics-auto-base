# -*- coding: utf-8 -*-
"""
C4 Implementation (Minimal Core)
- ref（人手）と auto（半自動テンプレ）で R′ を2条件（basic/briefing）生成
- SP/VS を L2ルーブリックで測定し、差分（delta）を出力
- 目的は「工程の機械化可能性」の検証であり、AIの優劣比較ではない
"""
import os, json, pathlib, re
from typing import List, Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

# -------------------------
# Text utilities
# -------------------------
TOK = r"[A-Za-z0-9一-龥ぁ-んァ-ンー]+"  # 日本語+英数の簡易トークン

def read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def write_text(p: pathlib.Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def token_bag(s: str) -> List[str]:
    return re.findall(TOK, s)

def coverage(title: str, sent: str, thr: float = 0.70) -> bool:
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
    # H一致
    h_hits = 0
    for h in H_titles:
        if not h:
            continue
        if h in text:
            h_hits += 1
            continue
        # 準厳密一致（被覆≥0.70）
        sents = [s for s in re.split(r"[\n。.!?]", text) if s.strip()]
        if any(coverage(h, s, thr=0.70) for s in sents):
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
    sents = [s for s in re.split(r"[\n。.!?]", text) if s.strip()]
    def claim_hit(cl: str) -> bool:
        if not cl:
            return False
        # 完全一致 or 被覆≥0.60 でヒット判定
        if cl in text:
            return True
        tset = set(token_bag(cl))
        if not tset:
            return False
        for s in sents:
            sset = set(token_bag(s))
            cov = sum(1 for t in tset if t in sset) / max(1, len(tset))
            if cov >= 0.60:
                return True
        return False
    c_hits = sum(1 for cl in claims if claim_hit(cl))
    if c_hits >= 3:
        c_match = 1.0
    elif c_hits == 2:
        c_match = 0.67  # C3整合：2/3
    elif c_hits == 1:
        c_match = 0.33
    else:
        c_match = 0.0

    sp = round((h_match + order_match + c_match) / 3.0, 2)
    return sp

# -------------------------
# VS (proxy: tone/focus/context 差の連続値)
# -------------------------
EMO = ("感じ", "思い", "心", "願い", "悩み", "不安", "喜び", "景色", "物語", "旅", "沖縄", "海", "空")
LOG = ("目的", "機能", "設計", "仕様", "戦略", "評価", "指標", "構造", "仮説", "測定", "実装", "プロトコル", "CI", "DOI")
SING = ("私", "僕", "自分")
ORG  = ("私たち", "我々", "当社", "ブランド", "チーム", "組織", "顧客", "ステークホルダー")
NARR = ("物語", "出会い", "風景", "記憶", "歩み", "螺旋")
BRIF = ("理念", "価値命題", "方針", "KPI", "運用", "評価", "測定")

def count_terms(s: str, terms: tuple) -> int:
    return sum(len(re.findall(t, s)) for t in terms)

def ratio_posneg(s: str, pos: tuple, neg: tuple) -> float:
    p = count_terms(s, pos)
    n = count_terms(s, neg)
    return p / max(1, (p + n))

def diff01(a: float, b: float) -> float:
    return min(1.0, abs(a - b))

def vs_proxy(text_basic: str, text_briefing: str) -> float:
    # tone: EMO vs LOG の“感情比率”の差
    tone_b = ratio_posneg(text_basic, EMO, LOG)
    tone_r = ratio_posneg(text_briefing, EMO, LOG)
    tone = diff01(tone_b, tone_r)
    # focus: SING vs ORG の“主体比率”の差
    foc_b = ratio_posneg(text_basic, SING, ORG)
    foc_r = ratio_posneg(text_briefing, SING, ORG)
    focus = diff01(foc_b, foc_r)
    # context: NARR vs BRIF の“文脈比率”の差
    ctx_b = ratio_posneg(text_basic, NARR, BRIF)
    ctx_r = ratio_posneg(text_briefing, NARR, BRIF)
    context = diff01(ctx_b, ctx_r)
    vs = round(min(0.70, (tone + focus + context) / 3.0), 2)
    return vs

# -------------------------
# auto renderer (決定論テンプレ／段落化)
# -------------------------
def render_auto_paragraphs(bp: Dict[str, Any], style: str) -> str:
    H = bp.get("headings", {})
    titles = [H.get("H1", {}).get("title", ""),
              H.get("H2", {}).get("title", ""),
              H.get("H3", {}).get("title", "")]
    C = [c.get("claim", "") for c in bp.get("blueprint", {}).get("C", [])]

    if style == "basic":
        # 情緒寄り（1人称の叙述で H を必ず含める）
        t = []
        if titles[0]:
            t.append(f"{titles[0]}。{C[0] if len(C)>0 else ''}。")
        if titles[1]:
            t.append(f"{titles[1]}。{C[1] if len(C)>1 else ''}。")
        if titles[2]:
            t.append(f"{titles[2]}。{C[2] if len(C)>2 else ''}。")
        text = "私はその流れの中で歩みを確かめる。"+ " ".join(t)
        return text
    else:
        # briefing：理念・方針寄り（3人称／命題化、Hタイトルを含める）
        t = []
        if titles[0]:
            t.append(f"{titles[0]}—{C[0] if len(C)>0 else ''}。")
        if titles[1]:
            t.append(f"{titles[1]}—{C[1] if len(C)>1 else ''}。")
        if titles[2]:
            t.append(f"{titles[2]}—{C[2] if len(C)>2 else ''}。")
        text = "本方針は次の骨格を保持する。"+ " ".join(t)
        return text

# -------------------------
# main run
# -------------------------
def run(case_id: str, bp_path: str, out_root: str = "outputs/c4_impl") -> pathlib.Path:
    if yaml is None:
        raise RuntimeError("PyYAML が見つかりません。requirements.txt に pyyaml を追加してください。")

    case = case_id or "brand_larica"
    bp_guess = bp_path or f"outputs/c2_manual/bp_{case}.yaml"

    bp_file = pathlib.Path(bp_guess)
    if not bp_file.exists():
        raise FileNotFoundError(f"BP not found: {bp_file}")

    bp = yaml.safe_load(bp_file.read_text(encoding="utf-8"))

    in_dir = pathlib.Path(f"inputs/c4/{case}")
    ref_basic_p = in_dir / "ref_basic.md"
    ref_brief_p = in_dir / "ref_briefing.md"

    # ref（存在しなければ空→SPが下がるので人手投入推奨）
    ref_basic_txt = read_text(ref_basic_p)
    ref_brief_txt = read_text(ref_brief_p)

    # auto（決定論テンプレ）
    auto_basic_txt = render_auto_paragraphs(bp, "basic")
    auto_brief_txt = render_auto_paragraphs(bp, "briefing")

    out_dir = pathlib.Path(out_root) / case
    write_text(out_dir / "regen_auto_basic.md", auto_basic_txt)
    write_text(out_dir / "regen_auto_briefing.md", auto_brief_txt)
    if ref_basic_txt:
        write_text(out_dir / "regen_ref_basic.md", ref_basic_txt)
    if ref_brief_txt:
        write_text(out_dir / "regen_ref_briefing.md", ref_brief_txt)

    # SP（BP↔各R′）、VS（basic vs briefing）
    sp_ref_vals, sp_auto_vals = [], []
    if ref_basic_txt:
        sp_ref_vals.append(sp_l2(bp, ref_basic_txt))
    if ref_brief_txt:
        sp_ref_vals.append(sp_l2(bp, ref_brief_txt))
    sp_auto_vals = [sp_l2(bp, auto_basic_txt), sp_l2(bp, auto_brief_txt)]

    SP_ref = round(sum(sp_ref_vals)/len(sp_ref_vals), 2) if sp_ref_vals else None
    SP_auto = round(sum(sp_auto_vals)/len(sp_auto_vals), 2)

    VS_ref = vs_proxy(ref_basic_txt, ref_brief_txt) if (ref_basic_txt and ref_brief_txt) else None
    VS_auto = vs_proxy(auto_basic_txt, auto_brief_txt)

    result = {
        "meta": {
            "mode": "c4_impl",
            "case_id": case,
            "bp_path": str(bp_file)
        },
        "ref":  {"SP": SP_ref,  "VS": VS_ref},
        "auto": {"SP": SP_auto, "VS": VS_auto},
        "delta": {
            "dSP": (None if SP_ref is None else round(abs(SP_auto - SP_ref), 2)),
            "dVS": (None if VS_ref is None else round(abs(VS_auto - VS_ref), 2))
        }
    }

    out_path = out_dir / "metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return out_path

if __name__ == "__main__":
    # 環境変数から受け取る（Actions から）
    MODE = os.environ.get("MODE", "c4_impl")
    CASE = os.environ.get("TARGET_ID", "brand_larica")
    BP_PATH = os.environ.get("BP_PATH", "")
    run(CASE, BP_PATH)
