# -*- coding: utf-8 -*-
import os, json, pathlib, re, argparse

ROOT = pathlib.Path(".")
OUT = ROOT / "outputs" / "bp_layer"

def load_bp_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_sentences(text):
    return [s.strip() for s in re.split(r"[。.!?]", text) if s.strip()]

def simple_renderer(bp, fi, title="R′(AI)"):
    """BP=(A,C,φ;K) と Fi={tone,focus,context} を受けて擬似R′を生成（決定論）。"""
    H = bp.get("headings", {})
    C = [c["claim"] for c in bp.get("blueprint", {}).get("C", [])]
    # Fiに応じたテンプレ（tone/focus/contextを切替え）
    if fi=="basic":
        head = "【説明】"
        tmpl = "{h}\n- {c}"
    else:  # fi == "briefing"
        head = "【理念】"
        tmpl = "{h}\n〈要点〉{c}"
    lines = [head]
    for key in ("H1","H2","H3"):
        if key in H:
            h = f"{H[key]['title']}"
            ci = C[min(len(lines)-1, len(C)-1)] if C else ""
            lines.append(tmpl.format(h=h, c=ci))
    return title + "\n" + "\n".join(lines) + "\n"

def sp_score(bp_ref, text_ai):
    """H構造・順序・核主張の3項からSPを近似（L2粒度、決定論）。"""
    Href = [bp_ref["headings"][k]["title"] for k in ("H1","H2","H3") if k in bp_ref["headings"]]
    href_set = set(Href)
    sents = to_sentences(text_ai)
    # H構造一致
    h_hits = sum(1 for h in Href if any(h in s for s in sents))
    h_match = 1.0 if h_hits==3 else (0.5 if h_hits>=1 else 0.0)
    # 順序一致（H1→H2→H3 の現れ順）
    idx = []
    for h in Href:
        pos = next((i for i,s in enumerate(sents) if h in s), None)
        idx.append(pos)
    order = 1.0 if all(idx[i] is not None for i in range(len(idx))) and idx==sorted(idx) else (0.5 if sum(p is not None for p in idx)>=2 else 0.0)
    # 核主張一致：C1–C3の言い換え痕跡が1/0.5/0段階で検出されたと仮定（簡易）
    Cref = [c["claim"] for c in bp_ref["blueprint"]["C"]]
    c_hits = sum(1 for c in Cref if any(any(tok in s for tok in re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9]+", c)) for s in sents))
    c_match = 1.0 if c_hits>=3 else (0.67 if c_hits==2 else (0.33 if c_hits==1 else 0.0))
    return round((h_match + order + c_match)/3, 2)

def vs_score(text_basic, text_briefing):
    """tone/focus/contextの差を0–1に正規化した擬似スコア（決定論）。"""
    # 擬似：文字種・箇条・キーワードで差分を拾い、0–1に射影
    diff = 0
    if "【説明】" in text_basic and "【理念】" in text_briefing: diff += 0.3  # tone/heading差
    if "〈要点〉" in text_briefing: diff += 0.2                      # focus差（組織・判断）
    if "-" in text_basic: diff += 0.1                                # context差（箇条）
    return round(min(0.70, max(0.0, 0.10 + diff)), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp_path", default="outputs/c2_manual/bp_larica.yaml")
    ap.add_argument("--target_id", default="lari_001")
    args = ap.parse_args()

    OUT_DIR = OUT / args.target_id
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bp = load_bp_yaml(args.bp_path)

    # AI側で R′ を2条件生成（Fi_basic / Fi_briefing）
    r_basic = simple_renderer(bp, "basic", title="R′_AI_basic")
    r_brief = simple_renderer(bp, "briefing", title="R′_AI_briefing")

    # AI側 SP/VS
    sp_ai_basic  = sp_score(bp, r_basic)
    sp_ai_brief  = sp_score(bp, r_brief)
    sp_ai        = round((sp_ai_basic + sp_ai_brief)/2, 2)
    vs_ai        = vs_score(r_basic, r_brief)

    # 人間（C3）の既知代表値（LARICA）
    SP_human = 0.83
    VS_human = 0.48

    metrics = {
      "meta": {"mode": "bp_layer", "target_id": args.target_id, "bp_path": args.bp_path},
      "ai":   {"SP": sp_ai, "VS": vs_ai},
      "human": {"SP": SP_human, "VS": VS_human},
      "delta": {"dSP": round(abs(sp_ai - SP_human),2), "dVS": round(abs(vs_ai - VS_human),2)}
    }

    # 保存
    with open(OUT_DIR/"regen_ai_lari_001.md", "w", encoding="utf-8") as f:
        f.write(r_basic + "\n" + r_brief)
    with open(OUT_DIR/"metrics_ai.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved: {OUT_DIR/'metrics_ai.json'}")

if __name__ == "__main__":
    main()
