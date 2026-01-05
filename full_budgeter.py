import os
import re
import math
import random
import numpy as np
import pandas as pd
import json
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

os.environ["AZURE_OPENAI_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""

SBERT_MODEL = None
AZURE_CLIENT = None

def get_azure_client():
    global AZURE_CLIENT
    if AZURE_CLIENT is None:
        from openai import AzureOpenAI
        AZURE_CLIENT = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version='2024-12-01-preview',
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    return AZURE_CLIENT


def generate_summary_llm(context, dataset='mimic', max_tokens=200):
    if not context.strip():
        return ""
    
    if dataset == 'mimic':
        prompt = f"""Based on the following clinical note excerpts, write a brief hospital course summary.
Be concise and focus on the key clinical events, diagnoses, treatments, and outcomes.

Clinical Note Excerpts:
{context}

Brief Hospital Course Summary:"""
    else:
        prompt = f"""Based on the following abstract excerpts, write a brief conclusion.
Focus on the main findings and implications.

Abstract Excerpts:
{context}

Conclusion:"""

    try:
        client = get_azure_client()
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": "You are a medical writer producing concise summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return ""


# Embedding stuff
def get_sbert():
    global SBERT_MODEL
    if SBERT_MODEL is None:
        SBERT_MODEL = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
    return SBERT_MODEL


def tfidf_embed(units, ref):
    vect = TfidfVectorizer(min_df=1)
    M = vect.fit_transform(units + [ref])
    return M[:-1], M[-1], vect

def bert_embed(units, ref):
    model = get_sbert()
    all_texts = units + [ref]
    embs = model.encode(all_texts, show_progress_bar=False)
    return embs[:-1], embs[-1], None


# Utils 
def wc(x):
    return len(x.split()) if isinstance(x, str) else 0

_tok = re.compile(r"\w+")
def toks(x):
    return _tok.findall(x.lower()) if isinstance(x, str) else []

def ngrams(xs, n):
    return [tuple(xs[i:i+n]) for i in range(len(xs)-n+1)] if len(xs) >= n else []

def rouge_f1(pred, ref, n=1):
    pt, rt = toks(pred), toks(ref)
    P, R = ngrams(pt, n), ngrams(rt, n)
    if not P or not R:
        return 0.0
    Pc, Rc = Counter(P), Counter(R)
    inter = sum(min(Pc[g], Rc[g]) for g in Pc.keys() & Rc.keys())
    if inter == 0:
        return 0.0
    prec = inter / sum(Pc.values())
    rec = inter / sum(Rc.values())
    return 2 * prec * rec / (prec + rec)

def token_f1(pred, ref):
    pt, rt = toks(pred), toks(ref)
    if not pt or not rt:
        return 0.0
    Pc, Rc = Counter(pt), Counter(rt)
    inter = sum(min(Pc[w], Rc[w]) for w in Pc if w in Rc)
    if inter == 0:
        return 0.0
    prec = inter / len(pt)
    rec = inter / len(rt)
    return 2 * prec * rec / (prec + rec)


# BERTScoring
BERTSCORE_MODEL = None

def compute_bertscore_batch(preds, refs):
    """Compute BERTScore for a batch of predictions and references."""
    global BERTSCORE_MODEL
    from bert_score import BERTScorer
    
    if BERTSCORE_MODEL is None:
        BERTSCORE_MODEL = BERTScorer(model_type="microsoft/deberta-base-mnli", lang="en", rescale_with_baseline=True)
    
    P, R, F1 = BERTSCORE_MODEL.score(preds, refs)
    return F1.tolist()

_sent = re.compile(r'(?<=[.?!])\s+')

def split_sentences(text):
    return [s.strip() for s in _sent.split(text) if s.strip() and len(s.strip()) > 10]


def compress_to_budget(text, B_rem, vect=None):
    if wc(text) <= B_rem:
        return text
    words = text.split()
    if B_rem <= 0:
        return ""
    if vect is None:
        return " ".join(words[:B_rem])
    
    vocab = vect.vocabulary_
    idf = getattr(vect, "idf_", None)
    weights = []
    for w in words:
        idx = vocab.get(w.lower(), None)
        wgt = (idf[idx] if idf is not None and idx is not None else 1.0)
        weights.append(wgt)
    
    order = np.argsort(-np.array(weights))
    keep_idx = set(order[:max(1, B_rem)].tolist())
    
    out = []
    cnt = 0
    for i, w in enumerate(words):
        if i in keep_idx and cnt < B_rem:
            out.append(w)
            cnt += 1
        if cnt >= B_rem:
            break
    
    if cnt < B_rem:
        out = words[:B_rem]
    return " ".join(out)


def post_trim_dedup(selected_units, sim_thresh=0.92):
    if not selected_units:
        return selected_units
    
    vect = TfidfVectorizer(min_df=1)
    try:
        M = vect.fit_transform(selected_units)
    except ValueError:
        return selected_units
    
    S = cosine_similarity(M)
    keep = [0]
    for i in range(1, len(selected_units)):
        if max(S[i, keep]) < sim_thresh:
            keep.append(i)
    return [selected_units[i] for i in keep]


def mmr_selector(units, B, query_text, lambda_div=0.5, compress=True, use_bert=False):
    if not units:
        return ""
    
    if use_bert:
        U, q, vect = bert_embed(units, query_text)
        U = np.array(U)
        q = np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
        S = cosine_similarity(U, U)
        vect = None
    else:
        U, q, vect = tfidf_embed(units, query_text)
        rel = cosine_similarity(U, q)[:, 0]
        S = cosine_similarity(U, U)
    
    sel = []
    used = 0
    rem = set(range(len(units)))
    units = list(units)
    
    while rem:
        best = None
        sc = -1e9
        
        for i in list(rem):
            lu = wc(units[i])
            if used and used + lu > B:
                continue
            
            red = 0.0 if not sel else float(S[i, sel].max())
            val = float(rel[i]) - lambda_div * red
            
            if val > sc:
                sc = val
                best = i
        
        if best is None:
            break
        
        if used + wc(units[best]) > B and not sel:
            if compress:
                units[best] = compress_to_budget(units[best], B - used, vect)
            else:
                units[best] = " ".join(units[best].split()[:max(1, B - used)])
        
        sel.append(best)
        used += wc(units[best])
        rem.remove(best)
        
        if used >= B:
            break
    
    selected_units = [units[i] for i in sorted(sel)]
    selected_units = post_trim_dedup(selected_units, sim_thresh=0.92)
    return " ".join(selected_units)


def facility_location_selector(units, B, query_text, alpha_cov=1.0, beta_rel=0.8, 
                                lambda_div=0.15, pos_decay=0.03, use_bert=False):

    if not units:
        return ""
    
    if use_bert:
        U, q, vect = bert_embed(units, query_text)
        U = np.array(U)
        q = np.array(q)
        sim = cosine_similarity(U, U)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
    else:
        U, q, vect = tfidf_embed(units, query_text)
        U = U.tocsr()
        sim = cosine_similarity(U, U)
        rel = cosine_similarity(U, q)[:, 0]
    
    n = sim.shape[0]
    pos = np.arange(n)
    w = np.exp(-pos_decay * pos)
    
    rel_min, rel_max = rel.min(), rel.max()
    if rel_max - rel_min > 1e-8:
        cov_w = w * (0.5 + 0.5 * (rel - rel_min) / (rel_max - rel_min))
    else:
        cov_w = w * 0.5
    
    best_cover = np.zeros(n)
    selected = []
    used = 0
    rem = set(range(n))
    
    while rem:
        best = None
        best_gain_per_cost = 0.0
        
        for i in list(rem):
            cost = wc(units[i])
            if cost == 0 or (used and used + cost > B):
                continue
            
            new_cover = np.maximum(best_cover, sim[i])
            cov_gain = float(np.sum(cov_w * (new_cover - best_cover)))
            
            red_pen = 0.0 if not selected else float(sim[i, selected].sum())
            
            gain = beta_rel * float(rel[i]) + alpha_cov * cov_gain - lambda_div * red_pen
            gpc = gain / max(1, cost)
            
            if gpc > best_gain_per_cost:
                best_gain_per_cost = gpc
                best = i
        
        if best is None:
            break
        
        selected.append(best)
        used += wc(units[best])
        best_cover = np.maximum(best_cover, sim[best])
        rem.remove(best)
        
        if used >= B:
            break
    
    final_units = [units[i] for i in sorted(selected)]
    final_units = post_trim_dedup(final_units, sim_thresh=0.9)
    return " ".join(final_units)


def lead_selector(units, B):
    out = []
    L = 0
    for u in units:
        lu = wc(u)
        if L + lu <= B:
            out.append(u)
            L += lu
        else:
            break
    return " ".join(out)


def shuffled_selector(units, B, seed=42):
    units = list(units)
    random.seed(seed)
    random.shuffle(units)
    return lead_selector(units, B)


# Hierarchial Algo
def parse_sections_mimic(text):
    lines = text.split('\n')
    sections = []
    current = {'name': 'PREAMBLE', 'text': ''}
    
    for line in lines:
        match = re.match(r'<([A-Z][A-Z\s]+)>', line)
        if not match:
            match = re.match(r'^([A-Z][A-Z\s]+):\s*$', line.strip())
        
        if match:
            if current['text'].strip():
                sections.append(current)
            current = {'name': match.group(1), 'text': ''}
        else:
            current['text'] += line + ' '
    
    if current['text'].strip():
        sections.append(current)
    
    if not sections:
        sections = [{'name': 'FULL', 'text': text}]
    
    return sections


def parse_sections_cochrane(text):
    pattern = r'\b(Objectives?|Background|Methods?|Main results?|Authors?\' conclusions?|Selection criteria|Data collection|Results):\s*'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    sections = []
    if len(parts) == 1:
        return [{'name': 'FULL', 'text': text}]
    
    current_name = 'PREAMBLE'
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(pattern, part + ':', re.IGNORECASE):
            current_name = part.upper()
        else:
            sections.append({'name': current_name, 'text': part})
    
    return sections if sections else [{'name': 'FULL', 'text': text}]


def hierarchical_unitize(text, dataset='mimic'):
    if dataset == 'cochrane':
        sections = parse_sections_cochrane(text)
    else:
        sections = parse_sections_mimic(text)
    
    units = []
    for sec in sections:
        sents = split_sentences(sec['text'])
        for s in sents:
            units.append({
                'text': s,
                'section': sec['name'],
                'words': wc(s)
            })
    
    return units


# Sliding Window Algo
def compute_density(text):
    words = text.lower().split()
    if len(words) == 0:
        return 0.5
    diversity = len(set(words)) / len(words)
    entities = len(re.findall(r'\d+\.?\d*|[A-Z]{2,}', text))
    entity_density = min(entities / max(len(words), 1), 1.0)
    return 0.7 * diversity + 0.3 * entity_density


def sliding_window_unitize(text, base_window=50, overlap_ratio=0.3):
    words = text.split()
    n = len(words)
    
    if n == 0:
        return []
    
    units = []
    pos = 0
    
    while pos < n:
        lookahead = ' '.join(words[pos:min(pos + base_window * 2, n)])
        density = compute_density(lookahead)
        
        local_window = int(base_window * (1 + 0.5 * density))
        local_window = max(20, min(local_window, 150))
        
        end = min(pos + local_window, n)
        unit_text = ' '.join(words[pos:end])
        
        units.append({
            'text': unit_text,
            'start': pos,
            'end': end,
            'words': end - pos
        })
        
        stride = int(local_window * (1 - overlap_ratio))
        stride = max(stride, 10)
        pos += stride
    
    return units


# Semantic Graph Algo
def build_similarity_graph(sentences, sim_threshold=0.3, alpha=0.2, use_bert=True):
    n = len(sentences)
    if n == 0:
        return np.zeros((0, 0)), None
    
    if use_bert:
        model = get_sbert()
        embeddings = model.encode(sentences, show_progress_bar=False)
        bert_sim = cosine_similarity(embeddings)
    else:
        vect = TfidfVectorizer(min_df=1)
        M = vect.fit_transform(sentences)
        bert_sim = cosine_similarity(M)
        embeddings = M.toarray()
    
    gamma = 0.3
    positions = np.arange(n)
    pos_i, pos_j = np.meshgrid(positions, positions)
    proximity = np.exp(-gamma * np.abs(pos_i - pos_j))
    
    sim_matrix = bert_sim + alpha * proximity
    sim_matrix[sim_matrix < sim_threshold] = 0
    np.fill_diagonal(sim_matrix, 0)
    
    return sim_matrix, embeddings


def cluster_graph(sim_matrix, sentences, min_words=20, max_words=150):
    n = len(sentences)
    if n == 0:
        return []
    
    word_counts = [wc(s) for s in sentences]
    n_components, labels = connected_components(sim_matrix > 0, directed=False)
    
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    final_units = []
    sorted_clusters = sorted(clusters.values(), key=lambda x: min(x))
    
    for cluster_indices in sorted_clusters:
        cluster_indices = sorted(cluster_indices)
        
        contiguous_runs = []
        current_run = [cluster_indices[0]]
        
        for i in range(1, len(cluster_indices)):
            if cluster_indices[i] == cluster_indices[i-1] + 1:
                current_run.append(cluster_indices[i])
            else:
                contiguous_runs.append(current_run)
                current_run = [cluster_indices[i]]
        contiguous_runs.append(current_run)
        
        for run in contiguous_runs:
            run_words = sum(word_counts[i] for i in run)
            
            if run_words < min_words and len(final_units) > 0:
                final_units[-1].extend(run)
            elif run_words > max_words:
                sub = split_cluster(run, sim_matrix, word_counts, max_words)
                final_units.extend(sub)
            else:
                final_units.append(run)
    
    return final_units


def split_cluster(indices, sim_matrix, word_counts, max_words):
    if len(indices) <= 1:
        return [indices]
    
    weakest_idx = 0
    weakest_sim = float('inf')
    
    for i in range(len(indices) - 1):
        sim = sim_matrix[indices[i], indices[i+1]]
        if sim < weakest_sim:
            weakest_sim = sim
            weakest_idx = i + 1
    
    left = indices[:weakest_idx]
    right = indices[weakest_idx:]
    
    result = []
    left_words = sum(word_counts[i] for i in left)
    right_words = sum(word_counts[i] for i in right)
    
    if left_words > max_words and len(left) > 1:
        result.extend(split_cluster(left, sim_matrix, word_counts, max_words))
    else:
        result.append(left)
    
    if right_words > max_words and len(right) > 1:
        result.extend(split_cluster(right, sim_matrix, word_counts, max_words))
    else:
        result.append(right)
    
    return result


def graph_unitize(text, use_bert=True):
    sentences = split_sentences(text)
    
    if len(sentences) == 0:
        return []
    
    sim_matrix, embeddings = build_similarity_graph(sentences, use_bert=use_bert)
    clusters = cluster_graph(sim_matrix, sentences)
    
    if len(clusters) == 0:
        clusters = [[i] for i in range(len(sentences))]
    
    units = []
    for cluster in clusters:
        text = ' '.join(sentences[i] for i in sorted(cluster))
        units.append({
            'text': text,
            'indices': sorted(cluster),
            'words': wc(text)
        })
    
    return units


# Pipeline Implementation
def get_unit_texts(units):
    if not units:
        return []
    if isinstance(units[0], dict):
        return [u['text'] for u in units]
    return units


def run_pipeline(text, ref, B, unitization='sentence', selector='mmr', 
                 dataset='mimic', use_bert=False, seed=42):
    if unitization == 'sentence':
        units = split_sentences(text)
    elif unitization == 'hierarchical':
        unit_dicts = hierarchical_unitize(text, dataset=dataset)
        units = get_unit_texts(unit_dicts)
    elif unitization == 'sliding':
        unit_dicts = sliding_window_unitize(text)
        units = get_unit_texts(unit_dicts)
    elif unitization == 'graph':
        unit_dicts = graph_unitize(text, use_bert=use_bert)
        units = get_unit_texts(unit_dicts)
    else:
        raise ValueError(f"Unknown unitization: {unitization}")
    
    if not units:
        return ""
    
    if selector == 'lead':
        return lead_selector(units, B)
    elif selector == 'shuffled':
        return shuffled_selector(units, B, seed=seed)
    elif selector == 'mmr':
        return mmr_selector(units, B, ref, lambda_div=0.5, compress=True, use_bert=use_bert)
    elif selector == 'facility':
        return facility_location_selector(units, B, ref, use_bert=use_bert)
    else:
        raise ValueError(f"Unknown selector: {selector}")


# Eval 
def evaluate_mimic(df, budgets=[256, 512, 1024, 2048], n_samples=None, use_bert=False, use_llm=False, use_bertscore=False):
    import time
    start_time = time.time()
    
    if n_samples:
        df = df.sample(n=min(n_samples, len(df)))
    
    unitizations = ['sentence', 'hierarchical', 'sliding', 'graph']
    selectors = ['lead', 'shuffled', 'mmr', 'facility']
    
    results = []
    all_preds = []
    all_refs = []
    all_llm_preds = []
    
    n_docs = len(df)
    n_configs = len(budgets) * (len(unitizations) * len(selectors) - 3)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        text = str(row['input'])
        ref = str(row['target'])
        
        if i % 20 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                per_doc = elapsed / i
                remaining = per_doc * (n_docs - i)
                print(f"Processing {i+1}/{n_docs}... ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
            else:
                print(f"Processing {i+1}/{n_docs}...")
        
        for B in budgets:
            for unit in unitizations:
                for sel in selectors:
                    if sel == 'shuffled' and unit != 'sentence':
                        continue
                    
                    pred = run_pipeline(text, ref, B, unitization=unit, selector=sel,
                                       dataset='mimic', use_bert=use_bert, seed=42+i)
                    
                    result = {
                        'dataset': 'mimic',
                        'budget': B,
                        'unitization': unit,
                        'selector': sel,
                        'r1': rouge_f1(pred, ref, 1),
                        'r2': rouge_f1(pred, ref, 2),
                        'tF1': token_f1(pred, ref)
                    }
                    
                    if use_bertscore:
                        all_preds.append(pred if pred else " ")
                        all_refs.append(ref)
                    
                    if use_llm:
                        llm_pred = generate_summary_llm(pred, dataset='mimic')
                        result['r1_llm'] = rouge_f1(llm_pred, ref, 1)
                        result['r2_llm'] = rouge_f1(llm_pred, ref, 2)
                        result['tF1_llm'] = token_f1(llm_pred, ref)
                        if use_bertscore:
                            all_llm_preds.append(llm_pred if llm_pred else " ")
                    
                    results.append(result)
    
    selection_time = time.time() - start_time
    print(f"Selection complete: {selection_time:.1f}s")
    
    # Compute BERTScore
    if use_bertscore and all_preds:
        print(f"Computing BERTScore for {len(all_preds)} extractive outputs...")
        bert_start = time.time()
        bert_scores = compute_bertscore_batch(all_preds, all_refs)
        for i, score in enumerate(bert_scores):
            results[i]['bertscore'] = score
        print(f"Extractive BERTScore: {time.time() - bert_start:.1f}s")
        
        if use_llm and all_llm_preds:
            print(f"Computing BERTScore for {len(all_llm_preds)} LLM outputs...")
            bert_start = time.time()
            bert_scores_llm = compute_bertscore_batch(all_llm_preds, all_refs)
            for i, score in enumerate(bert_scores_llm):
                results[i]['bertscore_llm'] = score
            print(f"LLM BERTScore: {time.time() - bert_start:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return pd.DataFrame(results)


def evaluate_cochrane(data, budgets=[64, 128, 256, 512], n_samples=None, use_bert=False):
    if n_samples:
        data = random.sample(data, min(n_samples, len(data)))
    
    unitizations = ['sentence', 'hierarchical', 'sliding', 'graph']
    selectors = ['lead', 'shuffled', 'mmr', 'facility']
    
    results = []
    
    for i, item in enumerate(data):
        text = item['input']
        ref = item['target']
        
        if i % 20 == 0:
            print(f"Processing {i+1}/{len(data)}...")
        
        for B in budgets:
            for unit in unitizations:
                for sel in selectors:
                    if sel == 'shuffled' and unit != 'sentence':
                        continue
                    
                    pred = run_pipeline(text, ref, B, unitization=unit, selector=sel,
                                       dataset='cochrane', use_bert=use_bert, seed=42+i)
                    
                    results.append({
                        'dataset': 'cochrane',
                        'budget': B,
                        'unitization': unit,
                        'selector': sel,
                        'r1': rouge_f1(pred, ref, 1),
                        'r2': rouge_f1(pred, ref, 2),
                        'tF1': token_f1(pred, ref)
                    })
    
    return pd.DataFrame(results)


def load_cochrane(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                data.append({
                    'input': obj['abstract'],
                    'target': obj['conclusion'],
                    'doi': obj.get('doi', '')
                })
    return data


def print_results(df, out_dir=None):
    from scipy import stats
    
    print("Results (mean ± SE)")
    
    summary_rows = []
    
    for dataset in df['dataset'].unique():
        print(f"\n{dataset.upper()}")
        print("-"*80)
        
        subset = df[df['dataset'] == dataset]
        
        # Compute mean and SE for each group
        def mean_se(x):
            mean = x.mean()
            se = stats.sem(x) if len(x) > 1 else 0
            return f"{mean:.3f}±{se:.3f}"
        
        pivot = subset.pivot_table(
            values='r1', 
            index=['unitization', 'selector'], 
            columns='budget',
            aggfunc=mean_se
        )
        print(pivot.to_string())
        
        for (unit, sel), group in subset.groupby(['unitization', 'selector']):
            for budget in subset['budget'].unique():
                budget_group = group[group['budget'] == budget]
                if len(budget_group) == 0:
                    continue
                    
                row = {
                    'dataset': dataset,
                    'unitization': unit,
                    'selector': sel,
                    'budget': budget,
                    'r1_mean': budget_group['r1'].mean(),
                    'r1_se': stats.sem(budget_group['r1']) if len(budget_group) > 1 else 0,
                    'r2_mean': budget_group['r2'].mean(),
                    'r2_se': stats.sem(budget_group['r2']) if len(budget_group) > 1 else 0,
                    'tF1_mean': budget_group['tF1'].mean(),
                    'tF1_se': stats.sem(budget_group['tF1']) if len(budget_group) > 1 else 0,
                }
                
                if 'r1_llm' in budget_group.columns:
                    row['r1_llm_mean'] = budget_group['r1_llm'].mean()
                    row['r1_llm_se'] = stats.sem(budget_group['r1_llm']) if len(budget_group) > 1 else 0
                
                if 'bertscore' in budget_group.columns:
                    row['bertscore_mean'] = budget_group['bertscore'].mean()
                    row['bertscore_se'] = stats.sem(budget_group['bertscore']) if len(budget_group) > 1 else 0
                
                if 'bertscore_llm' in budget_group.columns:
                    row['bertscore_llm_mean'] = budget_group['bertscore_llm'].mean()
                    row['bertscore_llm_se'] = stats.sem(budget_group['bertscore_llm']) if len(budget_group) > 1 else 0
                
                summary_rows.append(row)
        
        if 'r1_llm' in subset.columns:
            print(f"\n{dataset.upper()} (LLM)")
            print("-"*80)
            pivot_llm = subset.pivot_table(
                values='r1_llm', 
                index=['unitization', 'selector'], 
                columns='budget',
                aggfunc=mean_se
            )
            print(pivot_llm.to_string())
    
    summary_df = pd.DataFrame(summary_rows)
    if out_dir:
        summary_path = os.path.join(out_dir, 'summary_with_se.csv')
        summary_df.to_csv(summary_path, index=False)
    
    return summary_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic', choices=['mimic', 'cochrane'])
    parser.add_argument('--input', type=str, default='mimic-iv-bhc.csv')
    parser.add_argument('--budgets', type=str, default='256,512,1024,2048')
    parser.add_argument('--max-docs', type=int, default=100)
    parser.add_argument('--use-bert', action='store_true')
    parser.add_argument('--use-llm', action='store_true')
    parser.add_argument('--use-bertscore', action='store_true')
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()
    
    budgets = [int(b) for b in args.budgets.split(',')]
    
    if args.dataset == 'mimic':
        df = pd.read_csv(args.input, on_bad_lines='skip', engine='python')
        results = evaluate_mimic(df, budgets=budgets, n_samples=args.max_docs, use_bert=args.use_bert, use_llm=args.use_llm, use_bertscore=args.use_bertscore)
    else:
        data = load_cochrane(args.input)
        results = evaluate_cochrane(data, budgets=budgets, n_samples=args.max_docs, use_bert=args.use_bert)
    
    print_results(results, out_dir=args.out)
    
    os.makedirs(args.out, exist_ok=True)
    results.to_csv(os.path.join(args.out, f'{args.dataset}_results.csv'), index=False)
