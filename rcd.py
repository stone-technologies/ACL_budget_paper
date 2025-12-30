from __future__ import annotations
import math, random, numpy as np, re, torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

__author__ = 'kqureshi'


def l2norm(X: np.ndarray, eps: float=1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / nrm

def wc(x: str) -> int:
    return len(x.split()) if isinstance(x,str) else 0


class Embedder:
    def __init__(self, backend: str="tfidf", sbert_model: str="all-MiniLM-L6-v2", bert_model: str="bert-base-uncased", batch_size: int=16):
        self.backend = backend.lower()
        self.sbert_model_name = sbert_model
        self.bert_model_name  = bert_model
        self.batch_size = batch_size
        self._loaded = False

    def _load(self):
        if self._loaded: return
        if self.backend == "tfidf":
            # nothing to preload
            self._loaded = True
            return
        if self.backend == "sbert":
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError("sentence-transformers not installed; pip install sentence-transformers") from e
            self.sbert = SentenceTransformer(self.sbert_model_name, device="cpu")
            self._loaded = True; return
        if self.backend == "bert":
            try:
                from transformers import AutoTokenizer, AutoModel
            except Exception as e:
                raise RuntimeError("you need to install transformers") from e
            self.bert_tok = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.bert = AutoModel.from_pretrained(self.bert_model_name)
            self.bert.eval(); self.bert.to("cpu")
            self._loaded = True; return
        raise ValueError(f"Unknown backend {self.backend}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized sentence embeddings (n x d)"""
        self._load()
        if not texts: return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "tfidf":
            vec = TfidfVectorizer(min_df=1)
            M = vec.fit_transform(texts).astype(np.float32)
            X = M.toarray()
            return l2norm(X).astype(np.float32)
        if self.backend == "sbert":
            X = self.sbert.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            return X.astype(np.float32)
        if self.backend == "bert":
            # Mean-pool last hidden states
            toks = self.bert_tok(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = self.bert(**toks)
                last = out.last_hidden_state  # (b, L, d)
                mask = toks["attention_mask"].unsqueeze(-1)  # (b, L, 1)
                summed = (last * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                X = (summed / counts).cpu().numpy()
            return l2norm(X).astype(np.float32)


@dataclass
class RCDConfig:
    beta: float = 0.7
    alpha: float = 1.0
    eta: float = 0.05
    pos_decay: float = 0.04
    cand_cap: int = 80
    sample_frac: float = 0.4
    seed: int = 7

class LogDetState:
    def __init__(self, L: np.ndarray):
        self.L = L
        self.idx: List[int] = []
        self.Lchol: Optional[np.ndarray] = None
        self.logdet: float = 0.0
    def marginal(self, i: int) -> float:
        if not self.idx:
            return math.log(1.0 + float(self.L[i,i]))
        k = self.L[np.ix_(self.idx, [i])].reshape(-1,1)
        y = np.linalg.solve(self.Lchol, k)
        alpha2 = 1.0 + float(self.L[i,i]) - float((y.T @ y)[0,0])
        return max(0.0, math.log(max(alpha2, 1e-12)))
    def add(self, i: int) -> None:
        if not self.idx:
            a = math.sqrt(1.0 + float(self.L[i,i]))
            self.Lchol = np.array([[a]], dtype=np.float64)
            self.idx=[i]; self.logdet = 2.0*math.log(a); return
        k = self.L[np.ix_(self.idx, [i])].reshape(-1,1)
        y = np.linalg.solve(self.Lchol, k)
        alpha2 = 1.0 + float(self.L[i,i]) - float((y.T @ y)[0,0])
        alpha = math.sqrt(max(alpha2, 1e-12))
        n = self.Lchol.shape[0]
        Lnew = np.zeros((n+1, n+1), dtype=np.float64)
        Lnew[:n,:n] = self.Lchol
        Lnew[n,:n]  = y.T
        Lnew[n,n]   = alpha
        self.Lchol = Lnew
        self.idx.append(i)
        self.logdet += 2.0*math.log(alpha)

def build_kernel(units: List[str], ref: str, embedder: Embedder):
    U = embedder.encode(units).astype(np.float32)
    q = embedder.encode([ref]).astype(np.float32)[0]
    L = (U @ U.T).astype(np.float64)  # Gram (PSD)
    # numeric symmetrize
    L = ((L + L.T) / 2.0).astype(np.float64)
    rel = (U @ q).astype(np.float64)  # cosine (since L2-normalized)
    return U, q, L, rel

def rcd_select(units: List[str], ref: str, B: int, cfg: RCDConfig, embedder: Embedder) -> List[int]:
    U, q, L, rel = build_kernel(units, ref, embedder)
    n = len(units)
    sim = L  # coverage similarity
    pos = np.arange(n); w = np.exp(-cfg.pos_decay * pos)
    cov_w = w * (0.5 + 0.5*(rel - rel.min())/(rel.max()-rel.min()+1e-8))
    lengths = [wc(u) for u in units]
    score = rel + 0.2*w
    order = np.argsort(-score)[:min(cfg.cand_cap, n)].tolist()

    ld = LogDetState(L)
    cover = np.zeros(n, dtype=np.float64)
    chosen: List[int] = []
    rng = random.Random(cfg.seed)
    used = 0
    remaining = set(order)
    while remaining:
        pool = rng.sample(list(remaining), max(1, int(cfg.sample_frac*len(remaining))))
        best=None; best_ratio=-1e9
        for i in pool:
            c = lengths[i]
            if c==0 or used + c > B: 
                continue
            rel_g = float(rel[i])
            cov_g = float(np.sum(cov_w * np.maximum(0.0, sim[i] - cover)))
            log_g = ld.marginal(i)
            gain = cfg.beta*rel_g + cfg.alpha*cov_g + cfg.eta*log_g
            ratio = gain / max(1,c)
            if ratio > best_ratio + 1e-12:
                best_ratio = ratio; best = i
        if best is None: break
        # update state
        ld.add(best)
        cover = np.maximum(cover, sim[best])
        chosen.append(best)
        used += lengths[best]
        remaining.remove(best)
        if used >= B: break
    return sorted(chosen)

def assemble(units: List[str], idxs: List[int], B: int) -> str:
    words=0; out=[]
    for i in sorted(idxs):
        w=len(units[i].split())
        if words+w<=B: out.append(units[i]); words+=w
        else: out.append(" ".join(units[i].split()[:max(1,B-words)])); break
    return " ".join(out)


def mmr_with_embeddings(units: List[str], ref: str, B: int, embedder: Embedder, lam: float=0.5) -> str:
    U = embedder.encode(units).astype(np.float32)
    q = embedder.encode([ref]).astype(np.float32)[0]
    S = (U @ U.T).astype(np.float64)
    rel = (U @ q).astype(np.float64)
    sel=[]; used=0; rem=set(range(len(units)))
    while rem:
        best=None; sc=-1e9
        for i in list(rem):
            l=wc(units[i])
            if used and used+l>B: continue
            red=0.0 if not sel else float(S[i,sel].max())
            val=float(rel[i])-lam*red
            if val>sc: sc=val; best=i
        if best is None: break
        if used+wc(units[best])>B and not sel:
            units[best]=" ".join(units[best].split()[:max(1,B-used)])
        sel.append(best); used+=wc(units[best]); rem.remove(best)
        if used>=B: break
    return assemble(units, sel, B)
