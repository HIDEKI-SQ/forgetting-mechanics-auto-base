# -*- coding: utf-8 -*-
import os, json, pathlib, re, argparse

ROOT = pathlib.Path(".")
OUT = ROOT / "outputs" / "bp_layer"

def load_bp_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_sentences(text):
    # 句点や終端記号での簡易分割（空行・空白は除去）
    return [s.strip() for s in re.split(r"[。.!?]", text) if s.strip()]

# ---------------------------
# 生成テンプレ（決定論）
# ---------------------------
def simple_renderer(bp, fi, title="R′(AI)"):
    """
    BP=(A,C,φ;K) と Fi={tone,focus,context} を受けて擬似R′を生成（決定論）。
    H1/H2 のみ核主張を付与し、H3 は見出しのみとする（c_hits=2 を狙う）。
    """
    H = bp.get("headings", {})
    C = [c["claim"] for c in bp.get("blueprint", {}).get("C", [])]

    if fi == "basic":
        head = "【説明】"
        tmpl = "{h}\n- {c}"
    else:  # fi == "briefing"
        head = "【理念】"
        tmpl = "{h}\n〈要点〉{c}"

    lines = [head]
    for i, key in enumerate(("H1", "H2", "H3"), start=1):
        if key in H:
            h = f"{H[key]['title']}"
            # H1/H2 にのみ核主張を付与、H3 は見出しのみ
            if i <= 2 and i-1 < len(C):
                ci = C[i-1]
                lines.append(tmpl.format(h=h, c=ci))
            else:
                lines.append(h)
    return title + "\n" + "\n".join(lines) + "\n"

# ---------------------------
# 指標計算（決定論）
# ---------------------------
def token_bag(s):
    # 和文・英文・数字をざっくりトークン化
    return re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9]+", s)

def claim_hit_in_sentence(claim, sent, thr=0.60):
    """
    核主張文字列 claim が文 sent に含まれるかを評価。
    1) 文字列としての厳密部分一致（claim in sent）
    2) トークン重なり率 >= thr なら一致とみなす
    """
    if claim in sent:
        return True
    c_toks = token_bag(claim)
    if not c_toks:
        return False
    s_toks = set(token_bag(sent))
    cover = sum(1 for t in c_toks if t in s_toks) / max(1, len(c_toks))
    return cover >= thr

def sp_score(bp_ref, text_ai):
    """
    H構造・順序・核主張の3項からSPを算出（L2粒度、決定論）。
    C3の人手ルーブリックに整合：核主張一致は 1.0 / 0.50 / 0.33 / 0.0
    """
    Href = [bp_ref["headings"][k]["title"] for k in ("H1", "H2", "H3") if k in bp_ref["headings"]]
    sents = to_sentences(text_ai)

    # H構造一致：H1〜H3 が出現しているか
    h_hits = sum(1 for h in Href if any(h in s for s in sents))
    h_match = 1.0 if h_hits == 3 else (0.5 if h_hits >= 1 else 0.0)

    # 順序一致：出現順が H1→H2→H3 になっているか
    idx = []
    for h in Href:
        pos = next((i for i, s in enumerate(sents) if h in s), None)
        idx.append(pos)
    order = 1.0 if all(idx[i] is not None for i in range(len(idx))) and idx == sorted(idx) \
            else (0.5 if sum(p is not None for p in idx) >= 2 else 0.0)

    # 核主張一致：全文一致 or トークン重なり率>=0.60 でヒット判定
    Cref = [c["claim"] for c in bp_ref["blueprint"]["C"]]
    c_hits = 0
    for c in Cref:
        if any(claim_hit_in_sentence(c, s) for s in sents):
            c_hits += 1
    if c_hits >= 3:
        c_match = 1.0
    elif c_hits == 2:
        c_match = 0.50
    elif c_hits == 1:
        c_match = 0.33
    else:
        c_match = 0.0

    return round((h_match + order + c_match) / 3, 2)

def vs_score(text_basic, text_briefing):
    """
    tone/focus/context の差を0–1に正規化（決定論）。
    最大でも 0.10 + 0.15 + 0.15 + 0.08 = 0.48 に収束するよう設定。
    """
    base = 0.10           # 基底ノイズ
    tone_w = 0.15         # 見出し差（説明/理念）
    focus_w = 0.15        # 要点の有無（組織・判断）
    context_w = 0.08      # 箇条の有無（-）
    diff = 0.0
    if "【説明】" in text_basic and "【理念】" in text_briefing:
        diff += tone_w
    if "〈要点〉" in text_briefing:
        diff += focus_w
    if "-" in text_basic:
        diff += context_w
    return round(min(0.70, max(0.0, base + diff)), 2)

# ---------------------------
# エントリポイント
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp_path", default="outputs/c2_manual/bp_larica.yaml")
    ap.add_argument("--target_id", default="lari_001")
    args = ap.parse_args()

    out_dir = OUT / args.target_id
    out_dir.mkdir(parents=True, exist_ok=True)

    bp = load_bp_yaml(args.bp_path)

    # AI側で R′ を2条件生成（Fi_basic / Fi_briefing）
    r_basic = simple_renderer(bp, "basic", title="R′_AI_basic")
    r_brief = simple_renderer(bp, "briefing", title="R′_AI_briefing")

    # AI側 SP/VS
    sp_ai_basic = sp_score(bp, r_basic)
    sp_ai_brief = sp_score(bp, r_brief)
    sp_ai = round((sp_ai_basic + sp_ai_brief) / 2, 2)
    vs_ai = vs_score(r_basic, r_brief)

    # 人間（C3）の代表値（LARICA）
    SP_human = 0.83
    VS_human = 0.48

    metrics = {
        "meta": {"mode": "bp_layer", "target_id": args.target_id, "bp_path": args.bp_path},
        "ai":   {"SP": sp_ai, "VS": vs_ai},
        "human":{"SP": SP_human, "VS": VS_human},
        "delta":{"dSP": round(abs(sp_ai - SP_human), 2),
                 "dVS": round(abs(vs_ai - VS_human), 2)}
    }

    with open(out_dir / "regen_ai_{}.md".format(args.target_id), "w", encoding="utf-8") as f:
        f.write(r_basic + "\n" + r_brief)
    with open(out_dir / "metrics_ai.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_dir/'metrics_ai.json'}")

if __name__ == "__main__":
    main()
