import os, re, math, random, json, time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

os.environ["AZURE_OPENAI_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""

SBERT_MODEL = None
AZURE_CLIENT = None
BERTSCORE_MODEL = None
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
def classify_with_llm(context, question, max_tokens=50, retries=3):
    if not context.strip():
        return "maybe"

    prompt = f"""Based on the following biomedical abstract excerpts, answer the research question.
You must respond with exactly one word: yes, no, or maybe.

Context:
{context}

Question: {question}

Answer (yes/no/maybe):"""

    for attempt in range(retries):
        try:
            client = get_azure_client()
            resp = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": "You are a biomedical research assistant. Answer the question based only on the provided context. Respond with exactly one word: yes, no, or maybe."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens, temperature=0.0
            )
            answer = resp.choices[0].message.content.strip().lower()
            for label in ['yes', 'no', 'maybe']:
                if label in answer:
                    return label
            return "maybe"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"LLM error after {retries} attempts: {e}")
                return "maybe"
def generate_long_answer_llm(context, question, max_tokens=200, retries=3):
    if not context.strip():
        return ""

    prompt = f"""Based on the following biomedical abstract excerpts, provide a detailed answer to the research question.
Focus on the key findings, evidence, and conclusions.

Context:
{context}

Question: {question}

Detailed Answer:"""

    for attempt in range(retries):
        try:
            client = get_azure_client()
            resp = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": "You are a biomedical research assistant. Provide a detailed, evidence-based answer using only the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens, temperature=0.3
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"LLM error: {e}")
                return ""

def get_sbert():
    global SBERT_MODEL
    if SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        SBERT_MODEL = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
    return SBERT_MODEL

def tfidf_embed(units, ref):
    vect = TfidfVectorizer(min_df=1)
    M = vect.fit_transform(units + [ref])
    return M[:-1], M[-1], vect

def bert_embed(units, ref):
    model = get_sbert()
    embs = model.encode(units + [ref], show_progress_bar=False)
    return embs[:-1], embs[-1], None

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

def compute_bertscore_batch(preds, refs):
    global BERTSCORE_MODEL
    from bert_score import BERTScorer
    if BERTSCORE_MODEL is None:
        BERTSCORE_MODEL = BERTScorer(model_type="distilbert-base-uncased", lang="en",
                                     rescale_with_baseline=True)
    P, R, F1 = BERTSCORE_MODEL.score(preds, refs, batch_size=4)
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
    out, cnt = [], 0
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

def lead_selector(units, B):
    out, L = [], 0
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

def _get_relevance(units, query_text, use_bert=False):
    if use_bert:
        U, q, _ = bert_embed(units, query_text)
        U, q = np.array(U), np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
        S = cosine_similarity(U, U)
    else:
        U, q, _ = tfidf_embed(units, query_text)
        rel = cosine_similarity(U, q)[:, 0]
        S = cosine_similarity(U, U)
    return rel, S

def sliding_selector(units, B, query_text, use_bert=False):
    if not units:
        return ""
    if use_bert:
        U, q, _ = bert_embed(units, query_text)
        U, q = np.array(U), np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
    else:
        U, q, _ = tfidf_embed(units, query_text)
        rel = cosine_similarity(U, q)[:, 0]

    costs = [wc(u) for u in units]
    n = len(units)
    best_score, best_start, best_end = -1, 0, 0

    for start in range(n):
        total_cost, total_rel = 0, 0
        for end in range(start, n):
            total_cost += costs[end]
            if total_cost > B:
                break
            total_rel += rel[end]
            if total_rel > best_score:
                best_score = total_rel
                best_start, best_end = start, end + 1

    return " ".join(units[best_start:best_end])

def hierarchical_selector(units, B, query_text, use_bert=False):
    if not units:
        return ""
    if use_bert:
        U, q, _ = bert_embed(units, query_text)
        U, q = np.array(U), np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
    else:
        U, q, _ = tfidf_embed(units, query_text)
        rel = cosine_similarity(U, q)[:, 0]

    costs = np.array([wc(u) for u in units])
    n = len(units)
    selected = set()
    used = 0

    for idx in np.argsort(-rel):
        c = costs[idx]
        if used + c <= B:
            selected.add(idx)
            used += c

    for anchor in sorted(selected):
        for delta in [-1, 1]:
            nb = anchor + delta
            if 0 <= nb < n and nb not in selected:
                c = costs[nb]
                if used + c <= B:
                    selected.add(nb)
                    used += c

    return " ".join(units[i] for i in sorted(selected))

def mmr_selector(units, B, query_text, lambda_div=0.5, compress=True, use_bert=False):
    if not units:
        return ""
    if use_bert:
        U, q, vect = bert_embed(units, query_text)
        U, q = np.array(U), np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
        S = cosine_similarity(U, U)
        vect = None
    else:
        U, q, vect = tfidf_embed(units, query_text)
        rel = cosine_similarity(U, q)[:, 0]
        S = cosine_similarity(U, U)

    sel, used = [], 0
    rem = set(range(len(units)))
    units = list(units)

    while rem:
        best, sc = None, -1e9
        for i in list(rem):
            lu = wc(units[i])
            if used and used + lu > B:
                continue
            red = 0.0 if not sel else float(S[i, sel].max())
            val = float(rel[i]) - lambda_div * red
            if val > sc:
                sc, best = val, i
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

def graph_cluster_selector(units, B, query_text, use_bert=False):
    if not units:
        return ""
    if use_bert:
        U, q, _ = bert_embed(units, query_text)
        U, q = np.array(U), np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
        S = cosine_similarity(U, U)
    else:
        U, q, _ = tfidf_embed(units, query_text)
        rel = cosine_similarity(U, q)[:, 0]
        S = cosine_similarity(U, U)

    n = len(units)
    adj = (S > 0.3).astype(float)
    np.fill_diagonal(adj, 0)
    n_comp, labels = connected_components(adj, directed=False)

    comp_scores = {}
    comp_members = defaultdict(list)
    for i in range(n):
        comp_members[labels[i]].append(i)
        comp_scores[labels[i]] = comp_scores.get(labels[i], 0) + rel[i]

    selected, used = [], 0
    for comp_id in sorted(comp_scores, key=comp_scores.get, reverse=True):
        members = sorted(comp_members[comp_id], key=lambda i: -rel[i])
        for idx in members:
            c = wc(units[idx])
            if used + c <= B:
                selected.append(idx)
                used += c
            if used >= B:
                break
        if used >= B:
            break

    return " ".join(units[i] for i in sorted(selected))

def rcd_selector(units, B, query_text, alpha=0.4, beta=0.4, gamma=0.2,
                 eta=1.0, use_bert=False):
    if not units:
        return ""

    n = len(units)
    if use_bert:
        U, q, _ = bert_embed(units, query_text)
        U, q = np.array(U), np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
        K = cosine_similarity(U, U)
    else:
        U, q, _ = tfidf_embed(units, query_text)
        rel = cosine_similarity(U, q)[:, 0]
        K = cosine_similarity(U, U)

    # ensure PSD
    K = 0.5 * (K + K.T)
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0)
    K = eigvecs @ np.diag(eigvals) @ eigvecs.T
    K += 1e-6 * np.eye(n)

    costs = np.array([wc(u) for u in units])
    selected, selected_indices = [], []
    used = 0
    rem = set(range(n))
    best_cover = np.zeros(n)

    while rem:
        best_idx, best_gpc = None, -1e9

        for i in list(rem):
            if costs[i] == 0 or (used > 0 and used + costs[i] > B):
                continue

            rel_gain = float(rel[i])
            new_cover = np.maximum(best_cover, K[i])
            cov_gain = float(np.sum(new_cover - best_cover))

            if not selected_indices:
                div_gain = math.log(1 + eta * K[i, i])
            else:
                S_idx = np.array(selected_indices)
                M_S = np.eye(len(S_idx)) + eta * K[np.ix_(S_idx, S_idx)]
                try:
                    M_S_inv = np.linalg.inv(M_S)
                    k_iS = eta * K[i, S_idx]
                    schur = 1 + eta * K[i, i] - k_iS @ M_S_inv @ k_iS
                    div_gain = math.log(max(schur, 1e-10))
                except np.linalg.LinAlgError:
                    div_gain = 0.0

            gain = alpha * rel_gain + beta * cov_gain + gamma * div_gain
            gpc = gain / max(1, costs[i])
            if gpc > best_gpc:
                best_gpc, best_idx = gpc, i

        if best_idx is None:
            break

        if used + costs[best_idx] > B and not selected:
            units = list(units)
            units[best_idx] = " ".join(units[best_idx].split()[:max(1, B - used)])
            costs[best_idx] = wc(units[best_idx])

        selected.append(best_idx)
        selected_indices.append(best_idx)
        used += costs[best_idx]
        best_cover = np.maximum(best_cover, K[best_idx])
        rem.remove(best_idx)
        if used >= B:
            break

    final_units = [units[i] for i in sorted(selected)]
    final_units = post_trim_dedup(final_units, sim_thresh=0.9)
    return " ".join(final_units)

def parse_sections_pubmedqa(text):
    pattern = (r'\b(BACKGROUND|OBJECTIVES?|PURPOSE|AIMS?|INTRODUCTION|'
               r'METHODS?|DESIGN|SETTING|PATIENTS?|PARTICIPANTS?|'
               r'INTERVENTIONS?|MAIN OUTCOME MEASURES?|MEASUREMENTS?|'
               r'RESULTS?|FINDINGS?|'
               r'CONCLUSIONS?|INTERPRETATION|IMPLICATIONS?)[\s:.]+')
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    sections = []
    if len(parts) <= 1:
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

def parse_sections_from_labels(contexts, labels):
    return [{'name': lbl.upper() if lbl else 'UNKNOWN', 'text': ctx}
            for ctx, lbl in zip(contexts, labels)]

def hierarchical_unitize(text, contexts=None, labels=None):
    if contexts and labels and len(contexts) == len(labels):
        sections = parse_sections_from_labels(contexts, labels)
    else:
        sections = parse_sections_pubmedqa(text)
    units = []
    for sec in sections:
        for s in split_sentences(sec['text']):
            units.append({'text': s, 'section': sec['name'], 'words': wc(s)})
    return units

def compute_density(text):
    words = text.lower().split()
    if not words:
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
    units, pos = [], 0
    while pos < n:
        lookahead = ' '.join(words[pos:min(pos + base_window * 2, n)])
        density = compute_density(lookahead)
        local_window = int(base_window * (1 + 0.5 * density))
        local_window = max(20, min(local_window, 150))
        end = min(pos + local_window, n)
        units.append({'text': ' '.join(words[pos:end]), 'start': pos, 'end': end, 'words': end - pos})
        stride = max(int(local_window * (1 - overlap_ratio)), 10)
        pos += stride
    return units

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
    positions = np.arange(n)
    pos_i, pos_j = np.meshgrid(positions, positions)
    proximity = np.exp(-0.3 * np.abs(pos_i - pos_j))
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
    for cluster_indices in sorted(clusters.values(), key=lambda x: min(x)):
        cluster_indices = sorted(cluster_indices)
        runs = []
        current = [cluster_indices[0]]
        for i in range(1, len(cluster_indices)):
            if cluster_indices[i] == cluster_indices[i-1] + 1:
                current.append(cluster_indices[i])
            else:
                runs.append(current)
                current = [cluster_indices[i]]
        runs.append(current)
        for run in runs:
            run_wc = sum(word_counts[i] for i in run)
            if run_wc < min_words and final_units:
                final_units[-1].extend(run)
            elif run_wc > max_words:
                final_units.extend(split_cluster(run, sim_matrix, word_counts, max_words))
            else:
                final_units.append(run)
    return final_units

def split_cluster(indices, sim_matrix, word_counts, max_words):
    if len(indices) <= 1:
        return [indices]
    weakest_idx, weakest_sim = 0, float('inf')
    for i in range(len(indices) - 1):
        sim = sim_matrix[indices[i], indices[i+1]]
        if sim < weakest_sim:
            weakest_sim = sim
            weakest_idx = i + 1
    left, right = indices[:weakest_idx], indices[weakest_idx:]
    result = []
    for part in [left, right]:
        if sum(word_counts[i] for i in part) > max_words and len(part) > 1:
            result.extend(split_cluster(part, sim_matrix, word_counts, max_words))
        else:
            result.append(part)
    return result

def graph_unitize(text, use_bert=True):
    sentences = split_sentences(text)
    if not sentences:
        return []
    sim_matrix, _ = build_similarity_graph(sentences, use_bert=use_bert)
    clusters = cluster_graph(sim_matrix, sentences)
    if not clusters:
        clusters = [[i] for i in range(len(sentences))]
    return [{'text': ' '.join(sentences[i] for i in sorted(c)),
             'indices': sorted(c), 'words': sum(wc(sentences[i]) for i in c)}
            for c in clusters]

def compute_front_loading_index(units, query_text, B, use_bert=False):
    """Fraction of total relevance in leading units that fit in B (Eq. 4)."""
    if not units:
        return 0.0
    if use_bert:
        U, q, _ = bert_embed(units, query_text)
        U, q = np.array(U), np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
    else:
        U, q, _ = tfidf_embed(units, query_text)
        rel = cosine_similarity(U, q)[:, 0]

    total_rel = rel.sum()
    if total_rel < 1e-8:
        return 0.0
    budget_rel, used = 0.0, 0
    for i, u in enumerate(units):
        c = wc(u)
        if used + c <= B:
            budget_rel += rel[i]
            used += c
        else:
            break
    return budget_rel / total_rel

def compute_redundancy_index(units, use_bert=False):
    """Mean consecutive cosine similarity (Eq. 5)."""
    if len(units) < 2:
        return 0.0
    if use_bert:
        embs = get_sbert().encode(units, show_progress_bar=False)
        sims = [float(cosine_similarity(embs[i:i+1], embs[i+1:i+2])[0, 0])
                for i in range(len(units) - 1)]
    else:
        vect = TfidfVectorizer(min_df=1)
        try:
            M = vect.fit_transform(units)
            S = cosine_similarity(M)
            sims = [S[i, i+1] for i in range(len(units) - 1)]
        except ValueError:
            return 0.0
    return float(np.mean(sims))

def analyze_relevance_position(item, use_bert=False):
    text, question = item['context'], item['question']
    units = split_sentences(text)
    if not units:
        return {}

    if use_bert:
        U, q, _ = bert_embed(units, question)
        U, q = np.array(U), np.array(q)
        rel = cosine_similarity(U, q.reshape(1, -1))[:, 0]
    else:
        U, q, _ = tfidf_embed(units, question)
        rel = cosine_similarity(U, q)[:, 0]

    n = len(units)
    if n == 0:
        return {}

    max_rel_pos = float(np.argmax(rel)) / max(n - 1, 1)
    positions = np.arange(n) / max(n - 1, 1)
    total_rel = rel.sum()
    weighted_pos = float(np.dot(rel, positions) / total_rel) if total_rel > 0 else 0.5

    mid = n // 2
    first_half_rel = rel[:mid].sum()
    second_half_rel = rel[mid:].sum()

    section_rel = {}
    if item.get('labels') and item.get('contexts'):
        offset = 0
        for ctx, lbl in zip(item['contexts'], item['labels']):
            ctx_sents = split_sentences(ctx)
            n_ctx = len(ctx_sents)
            if n_ctx > 0 and offset + n_ctx <= n:
                section_rel[lbl] = float(rel[offset:offset + n_ctx].mean())
            offset += n_ctx

    return {
        'max_rel_position': max_rel_pos,
        'weighted_rel_position': weighted_pos,
        'first_half_rel': float(first_half_rel),
        'second_half_rel': float(second_half_rel),
        'rel_back_loaded': float(second_half_rel > first_half_rel),
        'section_relevance': section_rel,
    }

def routing_heuristic(units, B, query_text, B1=96, B2=192, use_bert=False):
    if B <= B1:
        return 'lead'
    elif B <= B2:
        return 'mmr'
    else:
        return 'rcd'

def adaptive_router(units, B, query_text, use_bert=False):
    phi = compute_front_loading_index(units, query_text, B, use_bert=use_bert)
    if phi < 0.5:
        return 'mmr' if B <= 128 else 'rcd'
    else:
        return routing_heuristic(units, B, query_text, use_bert=use_bert)

def get_unit_texts(units):
    if not units:
        return []
    if isinstance(units[0], dict):
        return [u['text'] for u in units]
    return units

def run_pipeline(text, query, B, unitization='sentence', selector='mmr',
                 use_bert=False, seed=42, contexts=None, labels=None):
    if unitization == 'sentence':
        units = split_sentences(text)
    elif unitization == 'hierarchical':
        units = get_unit_texts(hierarchical_unitize(text, contexts=contexts, labels=labels))
    elif unitization == 'sliding':
        units = get_unit_texts(sliding_window_unitize(text))
    elif unitization == 'graph':
        units = get_unit_texts(graph_unitize(text, use_bert=use_bert))
    else:
        raise ValueError(f"Unknown unitization: {unitization}")

    if not units:
        return text[:B * 5] if text else ""

    actual_selector = selector
    if selector == 'router':
        actual_selector = routing_heuristic(units, B, query, use_bert=use_bert)
    elif selector == 'adaptive_router':
        actual_selector = adaptive_router(units, B, query, use_bert=use_bert)

    dispatch = {
        'lead': lambda: lead_selector(units, B),
        'shuffled': lambda: shuffled_selector(units, B, seed=seed),
        'sliding': lambda: sliding_selector(units, B, query, use_bert=use_bert),
        'hierarchical_sel': lambda: hierarchical_selector(units, B, query, use_bert=use_bert),
        'graph_cluster': lambda: graph_cluster_selector(units, B, query, use_bert=use_bert),
        'mmr': lambda: mmr_selector(units, B, query, lambda_div=0.5, compress=True, use_bert=use_bert),
        'rcd': lambda: rcd_selector(units, B, query, use_bert=use_bert),
    }
    if actual_selector not in dispatch:
        raise ValueError(f"Unknown selector: {actual_selector}")
    return dispatch[actual_selector]()

def load_pubmedqa(split='pqa_labeled', n_samples=None, seed=42):
    from datasets import load_dataset
    ds = load_dataset('qiaojin/PubMedQA', split)
    data_split = ds['train']
    records = []
    for item in data_split:
        contexts = item.get('context', {})
        if isinstance(contexts, dict):
            ctx_texts = contexts.get('contexts', [])
            ctx_labels = contexts.get('labels', [])
        elif isinstance(contexts, list):
            ctx_texts, ctx_labels = contexts, []
        else:
            ctx_texts, ctx_labels = [str(contexts)], []
        records.append({
            'pubid': str(item.get('pubid', '')),
            'question': item.get('question', ''),
            'context': ' '.join(ctx_texts),
            'contexts': ctx_texts,
            'labels': ctx_labels,
            'long_answer': item.get('long_answer', ''),
            'final_decision': item.get('final_decision', 'maybe'),
        })
    if n_samples and n_samples < len(records):
        random.seed(seed)
        records = random.sample(records, n_samples)
    return records

def load_pubmedqa_from_json(filepath, n_samples=None, seed=42):
    with open(filepath, 'r') as f:
        raw = json.load(f)
    records = []
    for pubid, item in raw.items():
        contexts = item.get('CONTEXTS', [])
        labels_list = item.get('LABELS', [])
        full_context = ' '.join(contexts) if isinstance(contexts, list) else str(contexts)
        records.append({
            'pubid': str(pubid),
            'question': item.get('QUESTION', ''),
            'context': full_context,
            'contexts': contexts if isinstance(contexts, list) else [contexts],
            'labels': labels_list if isinstance(labels_list, list) else [],
            'long_answer': item.get('LONG_ANSWER', ''),
            'final_decision': item.get('final_decision', 'maybe'),
        })
    if n_samples and n_samples < len(records):
        random.seed(seed)
        records = random.sample(records, n_samples)
    return records

def evaluate_pubmedqa(data, budgets, unitizations, selectors,
                      use_bert=False, use_llm=True,
                      use_bertscore=False, use_long_answer=False):
    start_time = time.time()
    n_docs = len(data)

    n_configs = 0
    for unit in unitizations:
        for sel in selectors:
            if sel == 'shuffled' and unit != 'sentence':
                continue
            n_configs += 1
    n_configs *= len(budgets)

    print(f"Docs: {n_docs}, Budgets: {budgets}, Configs/doc: {n_configs}, "
          f"Total evals: {n_docs * n_configs}")

    results = []
    position_analyses = []
    all_long_preds, all_long_refs = [], []
    llm_call_count = 0

    for item in data:
        pa = analyze_relevance_position(item, use_bert=use_bert)
        pa['pubid'] = item['pubid']
        position_analyses.append(pa)

    back_loaded = np.mean([p['rel_back_loaded'] for p in position_analyses])
    
    print(f"  Back-loaded: {back_loaded:.1%}, "
          f"Mean max-rel position: {np.mean([p['max_rel_position'] for p in position_analyses]):.2f}")

    for i, item in enumerate(data):
        text = item['context']
        question = item['question']
        gold_label = item['final_decision']
        long_answer_ref = item.get('long_answer', '')
        full_wc_val = wc(text)

        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"  [{i}/{n_docs}] {elapsed:.0f}s elapsed, {llm_call_count} LLM calls")

        units_for_phi = split_sentences(text)

        for B in budgets:
            phi = compute_front_loading_index(units_for_phi, question, B, use_bert=use_bert)
            for unit in unitizations:
                for sel in selectors:
                    if sel == 'shuffled' and unit != 'sentence':
                        continue

                    selected_context = run_pipeline(
                        text, question, B, unitization=unit, selector=sel,
                        use_bert=use_bert, seed=42 + i,
                        contexts=item.get('contexts'), labels=item.get('labels'),
                    )

                    result = {
                        'pubid': item['pubid'], 'budget': B,
                        'unitization': unit, 'selector': sel,
                        'context_words': wc(selected_context),
                        'full_doc_words': full_wc_val,
                        'budget_ratio': B / max(full_wc_val, 1),
                        'gold_label': gold_label,
                        'front_loading_index': phi,
                    }

                    if long_answer_ref:
                        result['r1_extractive'] = rouge_f1(selected_context, long_answer_ref, 1)
                        result['r2_extractive'] = rouge_f1(selected_context, long_answer_ref, 2)

                    if use_llm:
                        pred_label = classify_with_llm(selected_context, question)
                        llm_call_count += 1
                        result['pred_label'] = pred_label
                        result['correct'] = int(pred_label == gold_label)

                        if use_long_answer and long_answer_ref:
                            long_pred = generate_long_answer_llm(selected_context, question)
                            llm_call_count += 1
                            result['r1_llm'] = rouge_f1(long_pred, long_answer_ref, 1)
                            result['r2_llm'] = rouge_f1(long_pred, long_answer_ref, 2)
                            if use_bertscore:
                                all_long_preds.append(long_pred if long_pred else " ")
                                all_long_refs.append(long_answer_ref)

                    results.append(result)

    # full-context baseline
    if use_llm:
        for item in data:
            pred_label = classify_with_llm(item['context'], item['question'])
            llm_call_count += 1
            results.append({
                'pubid': item['pubid'], 'budget': 99999,
                'unitization': 'full', 'selector': 'full',
                'context_words': wc(item['context']),
                'full_doc_words': wc(item['context']),
                'budget_ratio': 1.0,
                'gold_label': item['final_decision'],
                'front_loading_index': 1.0,
                'pred_label': pred_label,
                'correct': int(pred_label == item['final_decision']),
            })

    if use_bertscore and all_long_preds:
        scores = compute_bertscore_batch(all_long_preds, all_long_refs)
        idx = 0
        for r in results:
            if 'r1_llm' in r:
                r['bertscore_llm'] = scores[idx]
                idx += 1

    total_time = time.time() - start_time

    return pd.DataFrame(results), pd.DataFrame(position_analyses)

def print_full_results(df, pos_df, out_dir=None):
    from scipy import stats

    # position analysis
    if pos_df is not None and len(pos_df) > 0:
        print(f"\nPosition analysis: {pos_df['rel_back_loaded'].mean():.1%} back-loaded, "
              f"max-rel pos {pos_df['max_rel_position'].mean():.2f}, "
              f"weighted pos {pos_df['weighted_rel_position'].mean():.2f}")

        all_sec_rel = defaultdict(list)
        for _, row in pos_df.iterrows():
            sr = row.get('section_relevance', {})
            if isinstance(sr, dict):
                for sec, val in sr.items():
                    all_sec_rel[sec].append(val)
        if all_sec_rel:
            print("Relevance by section:")
            for sec in sorted(all_sec_rel.keys()):
                print(f"  {sec}: {np.mean(all_sec_rel[sec]):.3f} (n={len(all_sec_rel[sec])})")

    if 'pred_label' not in df.columns:
        if 'r1_extractive' in df.columns:
            budgeted = df[df['budget'] != 99999]
            pivot = budgeted.groupby(['budget', 'selector'])['r1_extractive'].mean().unstack()
            print("\nExtractive ROUGE-1:")
            print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, 'pubmedqa_full_results.csv'), index=False)
        return

    # full-context baseline
    full = df[df['budget'] == 99999]
    if len(full) > 0:
        labels = ['yes', 'no', 'maybe']
        acc = full['correct'].mean()
        mf1 = f1_score(full['gold_label'].tolist(), full['pred_label'].tolist(),
                       labels=labels, average='macro', zero_division=0)
        print(f"\nFull-context baseline: acc={acc:.3f}, macro-F1={mf1:.3f}, n={len(full)}")

    budgeted = df[df['budget'] != 99999].copy()

    # Lead vs Shuffled
    lead_data = budgeted[budgeted['selector'] == 'lead']
    shuf_data = budgeted[budgeted['selector'] == 'shuffled']
    if len(lead_data) > 0 and len(shuf_data) > 0:
        print("\nLead vs Shuffled:")
        for B in sorted(budgeted['budget'].unique()):
            l = lead_data[lead_data['budget'] == B]['correct'].mean()
            s = shuf_data[shuf_data['budget'] == B]['correct'].mean()
            print(f"  B={B}: Lead={l:.3f}, Shuffled={s:.3f}, diff={l-s:+.3f}")

    # accuracy by budget x selector
    print("\nAccuracy by budget x selector:")
    pivot_acc = budgeted.groupby(['budget', 'selector'])['correct'].mean()
    print(pivot_acc.unstack(level='selector').to_string(float_format=lambda x: f"{x:.3f}"))

    # macro-F1
    print("\nMacro-F1 by budget x selector:")
    labels = ['yes', 'no', 'maybe']
    f1_rows = []
    for (B, sel), grp in budgeted.groupby(['budget', 'selector']):
        valid = grp.dropna(subset=['pred_label'])
        if len(valid) == 0:
            continue
        mf1 = f1_score(valid['gold_label'].tolist(), valid['pred_label'].tolist(),
                       labels=labels, average='macro', zero_division=0)
        f1_rows.append({'budget': B, 'selector': sel, 'macro_f1': mf1})
    if f1_rows:
        f1_df = pd.DataFrame(f1_rows)
        print(f1_df.pivot(index='budget', columns='selector', values='macro_f1')
              .to_string(float_format=lambda x: f"{x:.3f}"))

    # accuracy by unitization
    print("\nAccuracy by budget x unitization:")
    print(budgeted.groupby(['budget', 'unitization'])['correct'].mean().unstack()
          .to_string(float_format=lambda x: f"{x:.3f}"))

    # extractive ROUGE
    if 'r1_extractive' in budgeted.columns:
        print("\nExtractive ROUGE-1:")
        print(budgeted.groupby(['budget', 'selector'])['r1_extractive'].mean().unstack()
              .to_string(float_format=lambda x: f"{x:.3f}"))

    # front-loading index
    print("\nFront-loading index by budget:")
    for B, phi in budgeted.groupby('budget')['front_loading_index'].mean().items():
        print(f"  B={B}: {phi:.3f}")

    # save
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, 'pubmedqa_full_results.csv'), index=False)
        if pos_df is not None:
            pos_df.to_csv(os.path.join(out_dir, 'pubmedqa_position_analysis.csv'), index=False)

        summary_rows = []
        for (B, unit, sel), grp in budgeted.groupby(['budget', 'unitization', 'selector']):
            valid = grp.dropna(subset=['pred_label']) if 'pred_label' in grp.columns else grp
            if len(valid) == 0:
                continue
            row = {
                'budget': B, 'unitization': unit, 'selector': sel,
                'n': len(valid),
                'accuracy': valid['correct'].mean() if 'correct' in valid.columns else None,
                'front_loading_index': valid['front_loading_index'].mean(),
            }
            if 'pred_label' in valid.columns:
                row['macro_f1'] = f1_score(valid['gold_label'].tolist(),
                                           valid['pred_label'].tolist(),
                                           labels=['yes','no','maybe'],
                                           average='macro', zero_division=0)
            if 'r1_extractive' in valid.columns:
                row['r1_extractive'] = valid['r1_extractive'].mean()
            if 'r1_llm' in valid.columns:
                row['r1_llm'] = valid['r1_llm'].mean()
            summary_rows.append(row)

        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, 'pubmedqa_summary.csv'), index=False)

        if 'correct' in budgeted.columns:
            budgeted.groupby(['budget', 'selector'])['correct'].mean().unstack().to_csv(
                os.path.join(out_dir, 'pubmedqa_accuracy_pivot.csv'))

        print(f"\nSaved to {out_dir}/")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--split', type=str, default='pqa_labeled',
                        choices=['pqa_labeled', 'pqa_artificial'])
    parser.add_argument('--budgets', type=str, default='32,64,96,128,192,256')
    parser.add_argument('--max-docs', type=int, default=500)
    parser.add_argument('--unitizations', type=str,
                        default='sentence,hierarchical,sliding,graph')
    parser.add_argument('--selectors', type=str,
                        default='lead,shuffled,mmr,rcd,router,adaptive_router')
    parser.add_argument('--use-bert', action='store_true')
    parser.add_argument('--use-llm', action='store_true', default=True)
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--use-bertscore', action='store_true')
    parser.add_argument('--use-long-answer', action='store_true')
    parser.add_argument('--out', type=str, default='results_pubmedqa')
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(',')]
    unitizations = args.unitizations.split(',')
    selectors = args.selectors.split(',')
    if args.no_llm:
        args.use_llm = False

    if args.input:
        data = load_pubmedqa_from_json(args.input, n_samples=args.max_docs)
    else:
        data = load_pubmedqa(split=args.split, n_samples=args.max_docs)

    context_lengths = [wc(d['context']) for d in data]
    label_dist = Counter(d['final_decision'] for d in data)
    print(f"{len(data)} instances, median {np.median(context_lengths):.0f} words, "
          f"labels: {dict(label_dist)}")

    results_df, position_df = evaluate_pubmedqa(
        data, budgets=budgets, unitizations=unitizations, selectors=selectors,
        use_bert=args.use_bert, use_llm=args.use_llm,
        use_bertscore=args.use_bertscore, use_long_answer=args.use_long_answer,
    )

    print_full_results(results_df, position_df, out_dir=args.out)
