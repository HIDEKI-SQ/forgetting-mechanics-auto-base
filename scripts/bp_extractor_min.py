# Minimal BP extractor (toy): from tiny texts to BP and SP
import re, networkx as nx

CAUSE_WORDS = ("because","therefore","so","hence","thus","ので","だから","ゆえに")
TIME_WORDS  = ("today","now","later","before","after","昨日","今日","明日")

def split_sentences(txt):
    return [s.strip() for s in re.split(r'[。.!?]', txt) if s.strip()]

def extract_triplets(sent):
    toks = re.findall(r'[A-Za-z0-9一-龥ぁ-んァ-ン]+', sent)
    trip = []
    for i in range(len(toks)-2):
        a,b,c = toks[i:i+3]
        if re.match(r'.*(する|した|なる|be|is|are|do|make|cause).*', b):
            trip.append([a,"rel",c])
    return trip or ([[toks[0],"rel",toks[-1]]] if len(toks)>=2 else [])

def build_claims(texts):
    claims=[]
    for path in texts:
        with open(path,encoding='utf-8') as f:
            t=f.read()
        for s in split_sentences(t):
            for tri in extract_triplets(s):
                s_join = " ".join(tri)
                if any(w in s_join for w in CAUSE_WORDS): tri[1]="causes"
                elif "含" in s_join or "include" in s_join: tri[1]="includes"
                else: tri[1]="precedes"
                claims.append({"form":tri,"conf":1.0})
    for i,c in enumerate(claims): c["id"]=f"c_{i}"
    return claims

def axes_from_texts(texts):
    with open(texts[0],encoding='utf-8') as f: t0=f.read()
    abstractness = 0.6
    causal_density = 1.0 if any(w in t0 for w in CAUSE_WORDS) else 0.3
    timescale = "mid" if any(w in t0 for w in TIME_WORDS) else "long"
    return {"abstractness":abstractness,"causal_density":causal_density,"timescale":timescale}

def constraints_from_claims(claims):
    cons=[]
    for rel in ("causes","includes","precedes"):
        edges=[(c["form"][0],c["form"][2]) for c in claims if c["form"][1]==rel]
        G=nx.DiGraph(); G.add_edges_from(edges)
        ok = nx.is_directed_acyclic_graph(G)
        cons.append({"rule":f"{rel}_acyclic","value":1 if ok else 0})
    return cons

def bp_from_texts(texts):
    A=axes_from_texts(texts)
    C=build_claims(texts)
    phi=constraints_from_claims(C)
    return {"A":A,"C":C,"phi":phi}

def r_struct_from_texts(texts):
    total=0
    for path in texts:
        with open(path,encoding='utf-8') as f: t=f.read()
        for s in split_sentences(t):
            total += len(extract_triplets(s))
    return total or 1

def sp_between(bp0,bp1):
    E0=set((tuple(c["form"]) for c in bp0["C"]))
    E1=set((tuple(c["form"]) for c in bp1["C"]))
    j_edge = len(E0 & E1)/len(E0 | E1) if (E0|E1) else 1.0
    P0=set((a,b,c) for (a,_,b) in E0 for (bb,_,c) in E0 if b==bb)
    P1=set((a,b,c) for (a,_,b) in E1 for (bb,_,c) in E1 if b==bb)
    f1 = (2*len(P0 & P1)/(len(P0)+len(P1))) if (P0 or P1) else 1.0
    return 0.5*j_edge + 0.5*f1
