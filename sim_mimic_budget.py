
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, math, argparse, numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__author__ = 'kqureshi'


_tok = re.compile(r"\w+")
def toks(x): return _tok.findall(x.lower()) if isinstance(x,str) else []
def ngrams(xs,n): return [tuple(xs[i:i+n]) for i in range(len(xs)-n+1)] if len(xs)>=n else []
def rouge_f1(pred, ref, n=1):
    pt,rt = toks(pred), toks(ref); P,R = ngrams(pt,n), ngrams(rt,n)
    if not P or not R: return 0.0
    from collections import Counter
    Pc,Rc = Counter(P), Counter(R); inter = sum(min(Pc[g],Rc[g]) for g in Pc.keys() & Rc.keys())
    if inter==0: return 0.0
    prec = inter/sum(Pc.values()); rec = inter/sum(Rc.values())
    return 2*prec*rec/(prec+rec)
def token_f1(pred, ref):
    pt,rt = toks(pred), toks(ref)
    if not pt or not rt: return 0.0
    from collections import Counter
    Pc,Rc = Counter(pt),Counter(rt); inter = sum(min(Pc[w],Rc[w]) for w in Pc if w in Rc)
    if inter==0: return 0.0
    prec = inter/len(pt); rec = inter/len(rt); return 2*prec*rec/(prec+rec)
def wc(x): return len(x.split()) if isinstance(x,str) else 0

_sent = re.compile(r'(?<=[.?!])\s+')
def split_sentences(x): return [s.strip() for s in _sent.split(x) if s.strip()]

def lead_selector(units, B):
    out=[]; L=0
    for u in units:
        lu=wc(u)
        if L+lu<=B: out.append(u); L+=lu
        else: break
    return " ".join(out)

def tfidf_embed(units, ref):
    vect = TfidfVectorizer(min_df=1)
    M = vect.fit_transform(units + [ref])
    return M[:-1], M[-1], vect

def compress_to_budget(text, B_rem, vect=None):
    if wc(text) <= B_rem: return text
    words = text.split()
    if vect is None: return " ".join(words[:B_rem])
    vocab = vect.vocabulary_; idf = getattr(vect, "idf_", None)
    weights=[]
    for w in words:
        idx = vocab.get(w.lower(), None)
        wgt = (idf[idx] if idf is not None and idx is not None else 1.0)
        weights.append(wgt)
    order = np.argsort(-np.array(weights))
    keep_idx = set(order[:max(1,B_rem)].tolist())
    out=[]; cnt=0
    for i,w in enumerate(words):
        if i in keep_idx and cnt < B_rem:
            out.append(w); cnt+=1
        if cnt>=B_rem: break
    if cnt < B_rem: out = words[:B_rem]
    return " ".join(out)

def post_trim_dedup(selected_units, sim_thresh=0.92):
    if not selected_units: return selected_units
    vect = TfidfVectorizer(min_df=1)
    M = vect.fit_transform(selected_units)
    S = cosine_similarity(M)
    keep=[0]
    for i in range(1, len(selected_units)):
        if max(S[i,keep]) < sim_thresh:
            keep.append(i)
    return [selected_units[i] for i in keep]

def mmr_selector_tfidf(units, B, query_text, lambda_div=0.5, compress=True):
    if not units: return ""
    U, q, vect = tfidf_embed(units, query_text)
    rel = cosine_similarity(U, q)[:,0]
    S   = cosine_similarity(U, U)
    sel=[]; used=0; rem=set(range(len(units)))
    while rem:
        best=None; sc=-1e9
        for i in list(rem):
            lu = wc(units[i])
            if used and used+lu>B: continue
            red = 0.0 if not sel else float(S[i, sel].max())
            val = float(rel[i]) - lambda_div*red
            if val>sc: sc=val; best=i
        if best is None: break
        if used+wc(units[best])>B and not sel:
            if compress: units[best] = compress_to_budget(units[best], B-used, vect)
            else:        units[best] = " ".join(units[best].split()[:max(1,B-used)])
        sel.append(best); used += wc(units[best]); rem.remove(best)
        if used>=B: break
    selected_units = [units[i] for i in sorted(sel)]
    selected_units = post_trim_dedup(selected_units, sim_thresh=0.92)
    return " ".join(selected_units)

def facility_location_selector(units, B, query_text, alpha_cov=1.0, beta_rel=0.8, lambda_div=0.15, pos_decay=0.03):
    if not units: return ""
    U, q, vect = tfidf_embed(units, query_text)
    U = U.tocsr()
    sim = cosine_similarity(U, U)
    rel = cosine_similarity(U, q)[:,0]
    n = sim.shape[0]
    pos = np.arange(n); w = np.exp(-pos_decay*pos)
    cov_w = w * (0.5 + 0.5*(rel - rel.min())/(rel.max()-rel.min()+1e-8))
    best_cover = np.zeros(n)
    selected=[]; used=0; rem=set(range(n))
    while rem:
        best=None; best_gain_per_cost=0.0
        for i in list(rem):
            cost = wc(units[i])
            if cost==0 or (used and used+cost>B): 
                continue
            new_cover = np.maximum(best_cover, sim[i])
            cov_gain = float(np.sum(cov_w * (new_cover - best_cover)))
            red_pen  = 0.0 if not selected else float(sim[i, selected].sum())
            gain = beta_rel*float(rel[i]) + alpha_cov*cov_gain - lambda_div*red_pen
            gpc = gain / max(1,cost)
            if gpc > best_gain_per_cost:
                best_gain_per_cost = gpc; best=i
        if best is None: break
        selected.append(best); used += wc(units[best])
        best_cover = np.maximum(best_cover, sim[best])
        rem.remove(best)
        if used>=B: break
    final_units = [units[i] for i in sorted(selected)]
    final_units = post_trim_dedup(final_units, sim_thresh=0.9)
    return " ".join(final_units)

def run(args):
    df = pd.read_csv(args.input, compression="infer").dropna(subset=[args.text_col, args.ref_col]).reset_index(drop=True)
    if args.max_docs and len(df)>args.max_docs:
        df = df.sample(args.max_docs, random_state=7).reset_index(drop=True)
    budgets = [int(b) for b in args.budgets.split(",")]
    rows=[]
    for _, ex in tqdm(df.iterrows(), total=len(df), desc="Docs"):
        text, ref = str(ex[args.text_col]), str(ex[args.ref_col])
        units = split_sentences(text)
        for B in budgets:
            ctx = lead_selector(units, B)
            rows.append({"dataset":args.tag,"unit":"sentence","selector":"lead","budget":B,
                         "r1":rouge_f1(ctx,ref,1),"r2":rouge_f1(ctx,ref,2),"tF1":token_f1(ctx,ref)})
            ctx = mmr_selector_tfidf(units[:], B, ref, 0.5, True)
            rows.append({"dataset":args.tag,"unit":"sentence","selector":"mmr","budget":B,
                         "r1":rouge_f1(ctx,ref,1),"r2":rouge_f1(ctx,ref,2),"tF1":token_f1(ctx,ref)})
            ctx = facility_location_selector(units[:], B, ref, 1.0, 0.8, 0.15, 0.03)
            rows.append({"dataset":args.tag,"unit":"sentence","selector":"facility","budget":B,
                         "r1":rouge_f1(ctx,ref,1),"r2":rouge_f1(ctx,ref,2),"tF1":token_f1(ctx,ref)})
    res = pd.DataFrame(rows)
    cur = res.groupby(["dataset","unit","selector","budget"], as_index=False).mean(numeric_only=True)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out,"results_per_doc.csv"), index=False)
    cur.to_csv(os.path.join(args.out,"results_curves.csv"), index=False)
    print("Wrote:", os.path.join(args.out,"results_per_doc.csv"), os.path.join(args.out,"results_curves.csv"))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default='mimic-iv-bhc.csv')
    ap.add_argument("--text-col", type=str, default="input")
    ap.add_argument("--ref-col", type=str, default="target")
    ap.add_argument("--out", type=str, default="mimic_sig_outputs_small")
    ap.add_argument("--budgets", type=str, default="64,128,256,512,1024,2048")
    ap.add_argument("--max-docs", type=int, default=400)
    ap.add_argument("--tag", type=str, default="mimic")
    run(ap.parse_args())
