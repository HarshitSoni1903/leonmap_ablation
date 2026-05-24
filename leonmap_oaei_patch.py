"""
Run LeonMap against OAEI Bio-ML.

Two eval modes:
  default : fixed threshold (TASK_THRESHOLDS or DEFAULT_THRESHOLD), eval on refs_equiv/full.tsv
  --sweep : threshold tuned on refs_equiv/train.tsv, eval on refs_equiv/test.tsv

Layout under {LEONMAP_ROOT}/{BIOML_DATA_DIR}/{task}/: <src>.owl, <tgt>.owl,
refs_equiv/{full,train,test}.tsv, refs_equiv/test.cands.tsv.

Fine-tuned SapBERT is fetched from HuggingFace on first run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from huggingface_hub import snapshot_download


# Config

LEONMAP_ROOT = Path(__file__).resolve().parent.parent  # script lives under mapnet/leonmap_ablation/
RECORD_ID = "13119437"
BIOML_DATA_DIR = f"data/{RECORD_ID}"
HF_MODEL_REPO = "harshitsoni1903/sapbert-finetuned-semra"

DEFAULT_THRESHOLD = 0.9
TASK_THRESHOLDS: Dict[str, Optional[float]] = {
    "omim-ordo":           None,
    "ncit-doid":           None,
    "snomed-fma.body":     None,
    "snomed-ncit.pharm":   None,
    "snomed-ncit.neoplas": None,
}

# Mapper runs at this floor so its TSV keeps every candidate we might threshold against later.
# Must be <= min(SWEEP_GRID) <= min(TASK_THRESHOLDS) <= DEFAULT_THRESHOLD.
MAPPER_FLOOR_THRESHOLD = 0.5
SWEEP_GRID: List[float] = [round(0.50 + 0.05 * i, 2) for i in range(10)]

# Semi-supervised pattern/subtype rerank. Only used in --sweep mode (needs train.tsv).
# Does not modify base mapper or rebuild FAISS.
ENABLE_SEMISUPERVISED_RERANK = True
RERANK_TASKS = {"omim-ordo"}
RERANK_TOP_K = 50
RERANK_GRID: List[Tuple[float, float]] = [
    (0.00, 0.00), (0.03, 0.03), (0.05, 0.03), (0.05, 0.05), (0.08, 0.05),
    (0.10, 0.05), (0.10, 0.08), (0.10, 0.10), (0.12, 0.08), (0.15, 0.10), (0.20, 0.10),
]
_RERANK_SPECIFIC_PATTERNS = [
    ("type",     re.compile(r"\btype\s+([0-9]+[a-z]?|[ivx]+[a-z]?)\b", re.I)),
    ("ar",       re.compile(r"\bautosomal\s+recessive\s+([0-9]+[a-z]?)\b", re.I)),
    ("ad",       re.compile(r"\bautosomal\s+dominant\s+([0-9]+[a-z]?)\b", re.I)),
    ("xl",       re.compile(r"\bx[-\s]?linked\s+([0-9]+[a-z]?)\b", re.I)),
    ("familial", re.compile(r"\bfamilial\s*,?\s*([0-9]+[a-z]?)\b", re.I)),
]
_RERANK_TRAILING_TOKEN_RE = re.compile(r"\b([0-9]+[a-z]?|[ivx]+[a-z]?)\s*$", re.I)

# SNOMED label cleanup: strip " (body structure)" / " (disorder)" parens,
# leading "Structure of ", and trailing " structure|part|region|area".
# Requires --rebuild if SNOMED FAISS DBs predate this flag.
STRIP_SNOMED_SUFFIXES = True
_SNOMED_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]+\)\s*$")
_SNOMED_LEADING_RE = re.compile(r"^Structure of\s+", re.IGNORECASE)
_SNOMED_TRAILING_NOISE_RE = re.compile(r"\s+(structure|part|region|area)\s*$", re.IGNORECASE)

# OMIM Bio-ML 2024 artifact: "TYPE" replaced with "  iia" (two spaces + 'iia') in
# 928/9622 labels. Signature is unambiguous (legitimate iiia/iiib/iiic/iiid use
# one space). Requires --rebuild if OMIM FAISS DB predates this flag.
STRIP_OMIM_TYPE_ARTIFACT = True
_OMIM_TYPE_ARTIFACT_RE = re.compile(r"  iia\b")

# Contamination filter.
# Modes:
#   "exact_pair"      : drop eval pairs whose (src_label, tgt_label) appeared in sapbert_all.csv
#   "label_disjoint"  : drop eval pairs if either src_label or tgt_label appeared anywhere in sapbert_all.csv
#   "both"            : require both exact-pair-clean and label-disjoint-clean
ENABLE_CONTAMINATION_FILTER = True
CONTAMINATION_FILTER_MODE = "both"
SAPBERT_CSV_PATH = "data/sapbert_all.csv"

TASKS = [
    "omim-ordo",
    "ncit-doid",
    "snomed-fma.body",
    "snomed-ncit.pharm",
    "snomed-ncit.neoplas",
]

# Populated in __main__ once PROJECT_ROOT is patched.
_LM: Dict = {}


# Label / IRI utilities

def _owl_files(task: str) -> Tuple[str, str]:
    """Derive (src_owl, tgt_owl) from a Bio-ML task name. snomed-fma.body -> snomed.body.owl, fma.body.owl."""
    src_part, tgt_part = task.split("-", 1)
    if "." in tgt_part:
        tgt_short, subdomain = tgt_part.split(".", 1)
        return f"{src_part}.{subdomain}.owl", f"{tgt_short}.{subdomain}.owl"
    return f"{src_part}.owl", f"{tgt_part}.owl"


def _strip_paren_suffix(s: str) -> str:
    if not s:
        return s
    s = _SNOMED_TRAILING_PAREN_RE.sub("", s)
    s = _SNOMED_LEADING_RE.sub("", s)
    s = _SNOMED_TRAILING_NOISE_RE.sub("", s)
    return s.strip()


def _restore_omim_type(s: str) -> str:
    if not s:
        return s
    s = _OMIM_TYPE_ARTIFACT_RE.sub(" type", s)
    return re.sub(r"\s+", " ", s).strip()


def _iri_to_id(iri: str) -> str:
    """LeonMap's canonicalize_id on the IRI tail (with mesh prefix special-case)."""
    canonicalize_id = _LM["canonicalize_id"]
    tail = iri.split("#")[-1].rsplit("/", 1)[-1].strip()
    if "id.nlm.nih.gov/mesh/" in iri or "obo/mesh#" in iri or "purl.obolibrary.org/obo/mesh" in iri:
        return canonicalize_id(f"mesh:{tail}")
    return canonicalize_id(tail)


def _get_ignored_iris(owl_path: Path) -> set:
    """IRIs of classes annotated use_in_alignment=false (locality-module aux classes)."""
    from owlready2 import get_ontology
    onto = get_ontology(Path(owl_path).as_posix()).load()
    ignored: set = set()
    for cls in onto.classes():
        val = getattr(cls, "use_in_alignment", None)
        if (val is False) or (isinstance(val, list) and any(v is False for v in val)):
            ignored.add(str(cls.iri))
    return ignored


# Contamination filter

_SEEN_LABELS: Optional[frozenset] = None
_SEEN_PAIRS: Optional[frozenset] = None


def _norm_label(s: str) -> str:
    return " ".join(str(s).lower().split())


def _clean_seen_label(s: str) -> str:
    """Apply the same lightweight label cleanup used by the ontology loader."""
    s = str(s)
    s = _restore_omim_type(s)
    s = _strip_paren_suffix(s)
    return _norm_label(s)


def _load_seen_labels() -> frozenset:
    """Cached set of normalized labels from sapbert_all.csv."""
    global _SEEN_LABELS, _SEEN_PAIRS
    if _SEEN_LABELS is not None:
        return _SEEN_LABELS

    import pandas as pd

    csv_path = LEONMAP_ROOT / SAPBERT_CSV_PATH
    df = pd.read_csv(csv_path)

    subj = df["subject_label"].dropna().map(_clean_seen_label)
    obj = df["object_label"].dropna().map(_clean_seen_label)

    _SEEN_LABELS = frozenset(pd.concat([subj, obj]).dropna().tolist())

    pairs = []
    for _, row in df.iterrows():
        if pd.isna(row.get("subject_label")) or pd.isna(row.get("object_label")):
            continue
        s = _clean_seen_label(row["subject_label"])
        t = _clean_seen_label(row["object_label"])
        if s and t:
            pairs.append((s, t))
            pairs.append((t, s))

    _SEEN_PAIRS = frozenset(pairs)

    print(f"  [FILTER] Loaded {len(_SEEN_LABELS)} unique seen labels from {csv_path.name}")
    print(f"  [FILTER] Loaded {len(_SEEN_PAIRS)} directional seen label-pairs from {csv_path.name}")
    print(f"  [FILTER] Mode: {CONTAMINATION_FILTER_MODE}")

    return _SEEN_LABELS


def _load_seen_pairs() -> frozenset:
    global _SEEN_PAIRS
    if _SEEN_PAIRS is None:
        _load_seen_labels()
    return _SEEN_PAIRS or frozenset()


def _label_for_iri(iri: str, db) -> str:
    payload = db.get_payload_by_id(_iri_to_id(iri)) or {}
    return payload.get("label", "")


def _clean_label_for_filter(s: str) -> str:
    s = _restore_omim_type(str(s))
    s = _strip_paren_suffix(s)
    return _norm_label(s)


def _label_pair_for_iris(src_iri: str, tgt_iri: str, src_db, tgt_db) -> Tuple[str, str]:
    src_label = _clean_label_for_filter(_label_for_iri(src_iri, src_db))
    tgt_label = _clean_label_for_filter(_label_for_iri(tgt_iri, tgt_db))
    return src_label, tgt_label


def _is_clean_pair(
    src_iri: str,
    tgt_iri: str,
    src_db,
    tgt_db,
    seen_labels: Optional[frozenset],
    seen_pairs: Optional[frozenset] = None,
) -> bool:
    """Return True if a pair passes the selected contamination filter."""
    if seen_labels is None:
        return True

    src_label, tgt_label = _label_pair_for_iris(src_iri, tgt_iri, src_db, tgt_db)
    pair_seen = seen_pairs is not None and (src_label, tgt_label) in seen_pairs
    label_seen = src_label in seen_labels or tgt_label in seen_labels

    if CONTAMINATION_FILTER_MODE == "exact_pair":
        return not pair_seen

    if CONTAMINATION_FILTER_MODE == "label_disjoint":
        return not label_seen

    if CONTAMINATION_FILTER_MODE == "both":
        return (not pair_seen) and (not label_seen)

    raise ValueError(f"Unknown CONTAMINATION_FILTER_MODE: {CONTAMINATION_FILTER_MODE}")


def _filter_refs_set(
    refs: set,
    src_db,
    tgt_db,
    seen_labels: Optional[frozenset],
    seen_pairs: Optional[frozenset],
) -> set:
    if seen_labels is None:
        return refs
    return {
        (s, t)
        for s, t in refs
        if _is_clean_pair(s, t, src_db, tgt_db, seen_labels, seen_pairs)
    }


def _refs_to_src_to_gold(refs: set) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for s, t in refs:
        out[s].add(t)
    return out


# Task registration + mapper invocation

def _register_task(task: str, src_owl_abs: Path, tgt_owl_abs: Path) -> Tuple[str, str, str]:
    """Inject runtime collection/mapping entries into leonmap.config."""
    _cfg = _LM["cfg_mod"]
    src_col = f"oaei_{task}_src"
    tgt_col = f"oaei_{task}_tgt"
    mapping_key = f"oaei_{task}"
    _cfg.COLLECTIONS[src_col] = {"source": "owl", "model": "ft", "owl_path": str(src_owl_abs), "id_prefixes": []}
    _cfg.COLLECTIONS[tgt_col] = {"source": "owl", "model": "ft", "owl_path": str(tgt_owl_abs), "id_prefixes": []}
    _cfg.MAPPINGS[mapping_key] = {
        "src_collection": src_col, "tgt_collection": tgt_col,
        "threshold": MAPPER_FLOOR_THRESHOLD, "top_k": 1, "reverse": False,
    }
    return src_col, tgt_col, mapping_key


def _score_candidates(src_id: str, candidate_ids: List[str], src_db, tgt_db) -> List[Tuple[str, float, str]]:
    """Score a fixed candidate list against src; no FAISS, reconstruct vectors + cosine. Boost still applies."""
    rank_pool = _LM["rank_pool"]
    if src_id not in src_db.id2pos:
        return []
    sv = src_db.index.reconstruct(src_db.id2pos[src_id]).astype("float32")
    pool: List[Tuple[str, float]] = []
    for cid in candidate_ids:
        pos = tgt_db.id2pos.get(cid)
        if pos is None:
            pool.append((cid, 0.0))
            continue
        tv = tgt_db.index.reconstruct(pos).astype("float32")
        pool.append((cid, float(sv @ tv)))
    src_label = (src_db.get_payload_by_id(src_id) or {}).get("label", "")
    return rank_pool(pool, tgt_db, src_label, threshold=0.0, enable_boost=True)


def _run(entry_main, cli_name: str, argv: List[str]) -> None:
    """Invoke a LeonMap CLI's main() with a synthesized argv."""
    old = sys.argv
    sys.argv = [cli_name] + argv
    try:
        entry_main()
    finally:
        sys.argv = old


# I/O: predictions and refs

def _load_mapper_predictions(mapper_tsv: Path, src_db, tgt_db, ignored: set) -> List[Tuple[str, str, float]]:
    """Read mapper TSV (canonical-id cols) -> (src_iri, tgt_iri, score) tuples, ignored classes filtered."""
    preds: List[Tuple[str, str, float]] = []
    with open(mapper_tsv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            src_id = row.get("src_id", "").strip()
            tgt_id = row.get("tgt_id", "").strip()
            score_str = row.get("score", "").strip()
            if not (src_id and tgt_id and score_str):
                continue
            try:
                score = float(score_str)
            except ValueError:
                continue
            src_iri = (src_db.get_payload_by_id(src_id) or {}).get("iri", "")
            tgt_iri = (tgt_db.get_payload_by_id(tgt_id) or {}).get("iri", "")
            if not (src_iri and tgt_iri):
                continue
            if src_iri in ignored or tgt_iri in ignored:
                continue
            preds.append((src_iri, tgt_iri, score))
    return preds


def _write_match_result(preds: List[Tuple[str, str, float]], threshold: float, out_path: Path) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["SrcEntity", "TgtEntity", "Score"])
        for src_iri, tgt_iri, score in preds:
            if score >= threshold:
                w.writerow([src_iri, tgt_iri, score])
                n += 1
    return n


def _load_refs(refs_path: Path) -> set:
    pairs: set = set()
    with open(refs_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row.get("SrcEntity", "").strip()
            t = row.get("TgtEntity", "").strip()
            if s and t:
                pairs.add((s, t))
    return pairs


def _load_refs_by_src(refs_path: Path) -> Tuple[Dict[str, Set[str]], set]:
    src_to_gold: Dict[str, Set[str]] = defaultdict(set)
    pairs: set = set()
    with open(refs_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row.get("SrcEntity", "").strip()
            t = row.get("TgtEntity", "").strip()
            if s and t:
                src_to_gold[s].add(t)
                pairs.add((s, t))
    return src_to_gold, pairs


def _f1(preds: set, refs: set) -> Tuple[float, float, float, int]:
    tp = len(preds & refs)
    p = tp / len(preds) if preds else 0.0
    r = tp / len(refs) if refs else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp


# Semi-supervised pattern rerank

def _rerank_signatures(label: str) -> Set[str]:
    """Extract subtype/pattern tokens. 'multiple endocrine neoplasia, type 2b' -> {'2b', 'type:2b'}."""
    if not label:
        return set()
    s = label.lower().strip()
    out: Set[str] = set()
    for prefix, pattern in _RERANK_SPECIFIC_PATTERNS:
        for m in pattern.finditer(s):
            token = m.group(1).lower()
            out.add(token)
            out.add(f"{prefix}:{token}")
    m = _RERANK_TRAILING_TOKEN_RE.search(s)
    if m:
        out.add(m.group(1).lower())
    return out


def _build_topk_boosted_cache(
    src_iris: Set[str], src_db, tgt_db, top_k: int,
) -> Dict[str, Tuple[str, List[Tuple[str, float]]]]:
    """Re-query top-K then apply LeonMap's lexical boost. No DB rebuild."""
    rank_pool = _LM["rank_pool"]
    tgt_pos2id: Dict[int, str] = {pos: tid for tid, pos in tgt_db.id2pos.items()}
    cache: Dict[str, Tuple[str, List[Tuple[str, float]]]] = {}
    missing = 0
    for src_iri in sorted(src_iris):
        src_id = _iri_to_id(src_iri)
        if src_id not in src_db.id2pos:
            missing += 1
            continue
        src_vec = src_db.index.reconstruct(src_db.id2pos[src_id]).astype("float32").reshape(1, -1)
        distances, indices = tgt_db.index.search(src_vec, top_k)
        pool: List[Tuple[str, float]] = []
        for pos, score in zip(indices[0], distances[0]):
            if pos < 0:
                continue
            tgt_id = tgt_pos2id.get(int(pos))
            if tgt_id is not None:
                pool.append((tgt_id, float(score)))
        src_label = (src_db.get_payload_by_id(src_id) or {}).get("label", "")
        boosted = rank_pool(pool, tgt_db, src_label, threshold=0.0, enable_boost=True)
        cache[src_iri] = (src_label, [(r[0], float(r[1])) for r in boosted])
    if missing:
        print(f"  [RERANK] Missing source concepts from FAISS DB: {missing}")
    return cache


def _apply_pattern_rerank(
    pool: List[Tuple[str, float]], src_label: str, tgt_db, alpha: float, beta: float,
) -> List[Tuple[str, float]]:
    """+alpha to sig-matching candidates, -beta to others. No-op if src has no sig or no candidate matches."""
    if not pool or (alpha == 0.0 and beta == 0.0):
        return list(pool)
    src_sigs = _rerank_signatures(src_label)
    if not src_sigs:
        return list(pool)
    cand_sigs: Dict[str, Set[str]] = {}
    for tgt_id, _ in pool:
        payload = tgt_db.get_payload_by_id(tgt_id) or {}
        labels = [payload.get("label", "")] + list(payload.get("synonyms") or [])
        sigs: Set[str] = set()
        for label in labels:
            sigs |= _rerank_signatures(label)
        cand_sigs[tgt_id] = sigs
    if not any(src_sigs & sigs for sigs in cand_sigs.values()):
        return list(pool)
    adjusted: List[Tuple[str, float]] = []
    for tgt_id, score in pool:
        if src_sigs & cand_sigs[tgt_id]:
            adjusted.append((tgt_id, min(1.0, score + alpha)))
        else:
            adjusted.append((tgt_id, max(0.0, score - beta)))
    adjusted.sort(key=lambda x: -x[1])
    return adjusted


def _tgt_id_to_iri_map(tgt_db) -> Dict[str, str]:
    return {tid: (tgt_db.get_payload_by_id(tid) or {}).get("iri", "")
            for tid in tgt_db.id2pos
            if (tgt_db.get_payload_by_id(tid) or {}).get("iri", "")}


def _evaluate_rerank_cache(
    cache: Dict[str, Tuple[str, List[Tuple[str, float]]]],
    tgt_db, tgt_id_to_iri: Dict[str, str], refs: set,
    alpha: float, beta: float, threshold: float,
) -> Tuple[float, float, float, int, int, Dict[str, Tuple[str, float]]]:
    preds: set = set()
    top1_by_src: Dict[str, Tuple[str, float]] = {}
    for src_iri, (src_label, pool) in cache.items():
        reranked = _apply_pattern_rerank(pool, src_label, tgt_db, alpha, beta)
        if not reranked:
            continue
        tgt_id, score = reranked[0]
        top1_by_src[src_iri] = (tgt_id, score)
        if score < threshold:
            continue
        tgt_iri = tgt_id_to_iri.get(tgt_id)
        if tgt_iri:
            preds.add((src_iri, tgt_iri))
    p, r, f1, tp = _f1(preds, refs)
    return p, r, f1, tp, len(preds), top1_by_src


def _flip_analysis(
    baseline_top1: Dict[str, Tuple[str, float]],
    rerank_top1: Dict[str, Tuple[str, float]],
    src_to_gold: Dict[str, Set[str]],
) -> Tuple[int, int, int]:
    """Diagnostic: count good / bad / neutral flips between baseline and reranked top1."""
    good = bad = neutral = 0
    for src_iri, (new_id, _) in rerank_top1.items():
        if src_iri not in baseline_top1:
            continue
        old_id = baseline_top1[src_iri][0]
        if old_id == new_id:
            continue
        gold_ids = {_iri_to_id(g) for g in src_to_gold.get(src_iri, set())}
        old_correct = old_id in gold_ids
        new_correct = new_id in gold_ids
        if not old_correct and new_correct:
            good += 1
        elif old_correct and not new_correct:
            bad += 1
        else:
            neutral += 1
    return good, bad, neutral


def _semisupervised_pattern_rerank(
    task: str,
    src_db,
    tgt_db,
    refs_train: Path,
    refs_test: Path,
    *,
    seen_labels: Optional[frozenset] = None,
    seen_pairs: Optional[frozenset] = None,
) -> Tuple[List[Tuple[str, str, float]], float, Dict]:
    """Tune (alpha, beta, threshold) on train.tsv, apply to test.tsv.
    If contamination filtering is enabled, train/tune only on clean train pairs."""
    train_src_to_gold_raw, train_refs_raw = _load_refs_by_src(refs_train)
    test_src_to_gold_raw, test_refs_raw = _load_refs_by_src(refs_test)

    if seen_labels is not None:
        train_refs = _filter_refs_set(train_refs_raw, src_db, tgt_db, seen_labels, seen_pairs)
        test_refs = _filter_refs_set(test_refs_raw, src_db, tgt_db, seen_labels, seen_pairs)
        train_src_to_gold = _refs_to_src_to_gold(train_refs)
        test_src_to_gold = _refs_to_src_to_gold(test_refs)
    else:
        train_refs = train_refs_raw
        test_refs = test_refs_raw
        train_src_to_gold = train_src_to_gold_raw
        test_src_to_gold = test_src_to_gold_raw

    tgt_id_to_iri = _tgt_id_to_iri_map(tgt_db)

    print(f"  [RERANK] {task}: top_k={RERANK_TOP_K} "
          f"train_src={len(train_src_to_gold)} test_src={len(test_src_to_gold)}")
    if seen_labels is not None:
        print(f"  [RERANK] train refs {len(train_refs_raw)} -> {len(train_refs)} after contamination filter")
        print(f"  [RERANK] test refs  {len(test_refs_raw)} -> {len(test_refs)} after contamination filter")

    train_cache = _build_topk_boosted_cache(set(train_src_to_gold), src_db, tgt_db, RERANK_TOP_K)
    test_cache = _build_topk_boosted_cache(set(test_src_to_gold), src_db, tgt_db, RERANK_TOP_K)

    _, _, _, _, _, baseline_train_top1 = _evaluate_rerank_cache(
        train_cache, tgt_db, tgt_id_to_iri, train_refs, 0.0, 0.0, 0.75)

    best = {
        "alpha": 0.0,
        "beta": 0.0,
        "threshold": SWEEP_GRID[0],
        "train_F1": -1.0,
        "train_P": 0.0,
        "train_R": 0.0,
        "train_tp": 0,
        "train_n_preds": 0,
    }

    sweep_log: List[Dict] = []

    for alpha, beta in RERANK_GRID:
        for threshold in SWEEP_GRID:
            p, r, f1, tp, n_preds, top1 = _evaluate_rerank_cache(
                train_cache, tgt_db, tgt_id_to_iri, train_refs, alpha, beta, threshold)
            good, bad, neutral = _flip_analysis(baseline_train_top1, top1, train_src_to_gold)
            sweep_log.append({
                "alpha": alpha,
                "beta": beta,
                "threshold": threshold,
                "P": p,
                "R": r,
                "F1": f1,
                "tp": tp,
                "n_preds": n_preds,
                "good_flips": good,
                "bad_flips": bad,
                "neutral_flips": neutral,
                "net_flips": good - bad,
            })
            if f1 > best["train_F1"]:
                best.update({
                    "alpha": alpha,
                    "beta": beta,
                    "threshold": threshold,
                    "train_F1": f1,
                    "train_P": p,
                    "train_R": r,
                    "train_tp": tp,
                    "train_n_preds": n_preds,
                })

    alpha = float(best["alpha"])
    beta = float(best["beta"])
    threshold = float(best["threshold"])

    print(f"  [RERANK] best train: alpha={alpha:.2f} beta={beta:.2f} thr={threshold:.2f} F1={best['train_F1']:.4f}")

    base_p, base_r, base_f1, base_tp, base_n, baseline_test_top1 = _evaluate_rerank_cache(
        test_cache, tgt_db, tgt_id_to_iri, test_refs, 0.0, 0.0, 0.75)
    test_p, test_r, test_f1, test_tp, test_n, test_top1 = _evaluate_rerank_cache(
        test_cache, tgt_db, tgt_id_to_iri, test_refs, alpha, beta, threshold)

    good, bad, neutral = _flip_analysis(baseline_test_top1, test_top1, test_src_to_gold)

    print(f"  [RERANK] baseline test @0.75: P={base_p:.4f} R={base_r:.4f} F1={base_f1:.4f}")
    print(f"  [RERANK] reranked test:      P={test_p:.4f} R={test_r:.4f} F1={test_f1:.4f}")
    print(f"  [RERANK] test flips: good={good} bad={bad} neutral={neutral} net={good - bad}")

    test_preds: List[Tuple[str, str, float]] = []
    for src_iri, (tgt_id, score) in test_top1.items():
        tgt_iri = tgt_id_to_iri.get(tgt_id)
        if tgt_iri:
            test_preds.append((src_iri, tgt_iri, float(score)))

    rerank_info = {
        "rerank_enabled": True,
        "rerank_task": task,
        "rerank_top_k": RERANK_TOP_K,
        "rerank_alpha": alpha,
        "rerank_beta": beta,
        "rerank_threshold": threshold,
        "rerank_train_P": best["train_P"],
        "rerank_train_R": best["train_R"],
        "rerank_train_F1": best["train_F1"],
        "rerank_train_tp": best["train_tp"],
        "rerank_train_n_preds": best["train_n_preds"],
        "rerank_baseline_test_P_at_075": base_p,
        "rerank_baseline_test_R_at_075": base_r,
        "rerank_baseline_test_F1_at_075": base_f1,
        "rerank_test_P": test_p,
        "rerank_test_R": test_r,
        "rerank_test_F1": test_f1,
        "rerank_test_tp": test_tp,
        "rerank_test_n_preds": test_n,
        "rerank_good_flips_test": good,
        "rerank_bad_flips_test": bad,
        "rerank_neutral_flips_test": neutral,
        "rerank_net_flips_test": good - bad,
        "rerank_sweep_log": sweep_log,
        "rerank_train_refs_pre_filter": len(train_refs_raw),
        "rerank_train_refs_post_filter": len(train_refs),
        "rerank_test_refs_pre_filter": len(test_refs_raw),
        "rerank_test_refs_post_filter": len(test_refs),
    }

    return test_preds, threshold, rerank_info


# Threshold sweep on train

def _sweep_threshold(
    preds: List[Tuple[str, str, float]],
    train_refs_path: Path,
    grid: List[float],
    *,
    seen_labels: Optional[frozenset] = None,
    seen_pairs: Optional[frozenset] = None,
    src_db=None,
    tgt_db=None,
) -> Dict:
    """Pick threshold maximizing F1 on train.tsv. If contamination filtering is enabled,
    tune only on clean train references and clean train predictions."""
    train_refs_raw = _load_refs(train_refs_path)

    if seen_labels is not None and src_db is not None and tgt_db is not None:
        train_refs = _filter_refs_set(train_refs_raw, src_db, tgt_db, seen_labels, seen_pairs)
        train_preds_all = [
            (s, t, sc)
            for s, t, sc in preds
            if _is_clean_pair(s, t, src_db, tgt_db, seen_labels, seen_pairs)
        ]
    else:
        train_refs = train_refs_raw
        train_preds_all = preds

    train_src = {s for s, _ in train_refs}
    train_preds = [(s, t, sc) for s, t, sc in train_preds_all if s in train_src]

    sweep_log: List[Dict] = []
    best = {"threshold": grid[0], "F1": -1.0, "P": 0.0, "R": 0.0, "tp": 0, "n_preds": 0}

    for t in grid:
        preds_at_t = {(s, tg) for s, tg, sc in train_preds if sc >= t}
        p, r, f1, tp = _f1(preds_at_t, train_refs)
        rec = {"threshold": t, "P": p, "R": r, "F1": f1, "tp": tp, "n_preds": len(preds_at_t)}
        sweep_log.append(rec)
        if f1 > best["F1"]:
            best = rec

    return {
        "best": best,
        "sweep": sweep_log,
        "n_train_refs": len(train_refs),
        "n_train_refs_pre_filter": len(train_refs_raw),
    }


# Local ranking (threshold-independent)

def _run_local_ranking(test_cands_path: Path, src_db, tgt_db, out_path: Path) -> Tuple[int, int]:
    n_written = n_src_missing = 0
    with open(test_cands_path, "r", encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8", newline="") as g:
        r = csv.DictReader(f, delimiter="\t")
        w = csv.writer(g, delimiter="\t")
        w.writerow(["SrcEntity", "TgtEntity", "TgtCandidates"])
        for row in r:
            src_iri = row["SrcEntity"].strip()
            tgt_iri = row["TgtEntity"].strip()
            cand_iris: List[str] = list(ast.literal_eval(row["TgtCandidates"]))
            src_id = _iri_to_id(src_iri)
            cand_ids = [_iri_to_id(c) for c in cand_iris]
            if src_id not in src_db.id2pos:
                n_src_missing += 1
            ranked = _score_candidates(src_id, cand_ids, src_db, tgt_db)
            score_map: Dict[str, float] = {cid: float(s) for cid, s, _ in ranked}
            scored: List[Tuple[str, float]] = [(iri, score_map.get(cid, 0.0))
                                               for iri, cid in zip(cand_iris, cand_ids)]
            w.writerow([src_iri, tgt_iri, repr(scored)])
            n_written += 1
    return n_written, n_src_missing


def _eval_ranking(
    rank_path: Path,
    ks: Tuple[int, ...] = (1, 5, 10),
    *,
    seen_labels: Optional[frozenset] = None,
    seen_pairs: Optional[frozenset] = None,
    src_db=None,
    tgt_db=None,
) -> Dict:
    """MRR / Hits@K. If filtering is enabled, skip rows whose source-gold pair is contaminated."""
    mrr_sum = 0.0
    hits = {k: 0 for k in ks}
    n = n_pre = 0
    filter_on = seen_labels is not None and src_db is not None and tgt_db is not None

    with open(rank_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            n_pre += 1
            src_iri = row["SrcEntity"].strip()
            gold = row["TgtEntity"].strip()

            if filter_on and not _is_clean_pair(src_iri, gold, src_db, tgt_db, seen_labels, seen_pairs):
                continue

            scored = list(ast.literal_eval(row["TgtCandidates"]))
            scored.sort(key=lambda x: -float(x[1]))

            rank = next((i + 1 for i, (iri, _s) in enumerate(scored) if iri == gold), None)
            if rank is not None:
                mrr_sum += 1.0 / rank
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
            n += 1

    out: Dict = {"MRR": mrr_sum / n if n else 0.0, "n_test": n, "n_test_pre_filter": n_pre}
    for k in ks:
        out[f"Hits@{k}"] = hits[k] / n if n else 0.0
    return out


def _eval_global_match_file(
    match_path: Path,
    refs_path: Path,
    restrict_to_refs_src: bool,
    *,
    seen_labels: Optional[frozenset] = None,
    seen_pairs: Optional[frozenset] = None,
    src_db=None,
    tgt_db=None,
) -> Dict:
    """Eval match.result.tsv against refs. If filtering is enabled, drop contaminated refs and preds."""
    raw_refs = _load_refs(refs_path)
    raw_refs_src = {s for s, _ in raw_refs}

    raw_preds: set = set()
    with open(match_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row["SrcEntity"].strip()
            t = row["TgtEntity"].strip()
            if restrict_to_refs_src and s not in raw_refs_src:
                continue
            raw_preds.add((s, t))

    n_refs_pre, n_preds_pre = len(raw_refs), len(raw_preds)

    if seen_labels is not None and src_db is not None and tgt_db is not None:
        refs = _filter_refs_set(raw_refs, src_db, tgt_db, seen_labels, seen_pairs)
        preds = {
            p
            for p in raw_preds
            if _is_clean_pair(p[0], p[1], src_db, tgt_db, seen_labels, seen_pairs)
        }
    else:
        refs, preds = raw_refs, raw_preds

    p, r_, f1, tp = _f1(preds, refs)

    return {
        "P": p,
        "R": r_,
        "F1": f1,
        "tp": tp,
        "n_preds": len(preds),
        "n_refs": len(refs),
        "n_preds_pre_filter": n_preds_pre,
        "n_refs_pre_filter": n_refs_pre,
    }


# Per-task driver

def run_task(task: str, year_dir: Path, out_dir: Path, sweep: bool, rebuild: bool) -> Optional[Dict]:
    task_dir = year_dir / task
    if not task_dir.is_dir():
        print(f"[SKIP] Task data not found: {task_dir}")
        return None

    src_owl_name, tgt_owl_name = _owl_files(task)
    src_owl = task_dir / src_owl_name
    tgt_owl = task_dir / tgt_owl_name
    if not (src_owl.exists() and tgt_owl.exists()):
        print(f"[SKIP] OWL files missing for {task}: expected {src_owl.name}, {tgt_owl.name}")
        return None

    refs_full = task_dir / "refs_equiv" / "full.tsv"
    refs_train = task_dir / "refs_equiv" / "train.tsv"
    refs_test = task_dir / "refs_equiv" / "test.tsv"
    test_cands = task_dir / "refs_equiv" / "test.cands.tsv"
    required = [refs_full, test_cands] + ([refs_train, refs_test] if sweep else [])
    for p in required:
        if not p.exists():
            print(f"[SKIP] Missing reference file: {p}")
            return None

    print(f"\n=== Task: {task}  (mode: {'sweep' if sweep else 'fixed'}) ===")
    print(f"  src OWL: {src_owl}")
    print(f"  tgt OWL: {tgt_owl}")

    src_col, tgt_col, mapping_key = _register_task(task, src_owl, tgt_owl)

    # Build FAISS (no-op if already built unless --rebuild) + run mapper at floor threshold.
    build_argv = ["--collections", src_col, tgt_col] + (["--rebuild"] if rebuild else [])
    _run(_LM["build_main"], "leonmap-build", build_argv)
    _run(_LM["mapper_main"], "leonmap-map",
         ["--study", mapping_key, "--threshold", str(MAPPER_FLOOR_THRESHOLD)])

    project_root: Path = _LM["cfg_mod"].PROJECT_ROOT
    run_dirs = sorted((project_root / "mapper_results" / mapping_key).glob("run_*"), key=lambda p: p.name)
    if not run_dirs:
        print(f"[ERROR] No mapper run output for {mapping_key}")
        return None
    mapper_tsv = run_dirs[-1] / f"{src_col}_to_{tgt_col}.tsv"
    if not mapper_tsv.exists():
        print(f"[ERROR] Mapper output missing: {mapper_tsv}")
        return None

    BuildConfig = _LM["BuildConfig"]
    load_collection = _LM["load_collection"]
    cfg = BuildConfig()
    src_db = load_collection(cfg, src_col)
    tgt_db = load_collection(cfg, tgt_col)

    ignored = _get_ignored_iris(src_owl) | _get_ignored_iris(tgt_owl)
    print(f"  {len(ignored)} classes annotated use_in_alignment=false")

    out_dir.mkdir(parents=True, exist_ok=True)
    match_path = out_dir / "match.result.tsv"
    rank_path = out_dir / "rank.result.tsv"
    metrics_path = out_dir / "metrics.json"

    raw_preds = _load_mapper_predictions(mapper_tsv, src_db, tgt_db, ignored)
    print(f"  Raw predictions (>= {MAPPER_FLOOR_THRESHOLD}): {len(raw_preds)}")

    seen_labels = _load_seen_labels() if ENABLE_CONTAMINATION_FILTER else None
    seen_pairs = _load_seen_pairs() if ENABLE_CONTAMINATION_FILTER else None

    sweep_info: Optional[Dict] = None
    rerank_info: Optional[Dict] = None
    preds_for_output = raw_preds

    if sweep:
        eval_refs, eval_target, restrict_src = refs_test, "test.tsv", True

        if ENABLE_SEMISUPERVISED_RERANK and task in RERANK_TASKS:
            preds_for_output, chosen_threshold, rerank_info = _semisupervised_pattern_rerank(
                task=task,
                src_db=src_db,
                tgt_db=tgt_db,
                refs_train=refs_train,
                refs_test=refs_test,
                seen_labels=seen_labels,
                seen_pairs=seen_pairs,
            )
            sweep_info = {
                "best": {"threshold": chosen_threshold,
                         "P": rerank_info["rerank_train_P"], "R": rerank_info["rerank_train_R"],
                         "F1": rerank_info["rerank_train_F1"], "tp": rerank_info["rerank_train_tp"],
                         "n_preds": rerank_info["rerank_train_n_preds"]},
                "sweep": rerank_info["rerank_sweep_log"],
                "n_train_refs": rerank_info["rerank_train_refs_post_filter"],
                "n_train_refs_pre_filter": rerank_info["rerank_train_refs_pre_filter"],
            }
        else:
            sweep_info = _sweep_threshold(
                raw_preds,
                refs_train,
                SWEEP_GRID,
                seen_labels=seen_labels,
                seen_pairs=seen_pairs,
                src_db=src_db,
                tgt_db=tgt_db,
            )
            chosen_threshold = float(sweep_info["best"]["threshold"])
            test_src = {s for s, _ in _filter_refs_set(
                _load_refs(refs_test), src_db, tgt_db, seen_labels, seen_pairs
            )}
            preds_for_output = [
                (s, t, sc)
                for s, t, sc in raw_preds
                if s in test_src and _is_clean_pair(s, t, src_db, tgt_db, seen_labels, seen_pairs)
            ]

        print(f"  Sweep best: thr={chosen_threshold:.2f} "
              f"trainF1={sweep_info['best']['F1']:.4f} "
              f"trainP={sweep_info['best']['P']:.4f} "
              f"trainR={sweep_info['best']['R']:.4f}")
    else:
        chosen_threshold = TASK_THRESHOLDS.get(task) or DEFAULT_THRESHOLD
        eval_refs, eval_target, restrict_src = refs_full, "full.tsv", False
        print(f"  Fixed threshold: {chosen_threshold}")

    n_match = _write_match_result(preds_for_output, chosen_threshold, match_path)
    print(f"  Global matching: {n_match} predictions @ thr={chosen_threshold} -> {eval_target}")

    global_metrics = _eval_global_match_file(
        match_path,
        eval_refs,
        restrict_to_refs_src=restrict_src,
        seen_labels=seen_labels,
        seen_pairs=seen_pairs,
        src_db=src_db,
        tgt_db=tgt_db,
    )

    n_rank, n_src_missing = _run_local_ranking(test_cands, src_db, tgt_db, rank_path)
    print(f"  Local ranking: {n_rank} test rows scored ({n_src_missing} src not in DB)")
    ranking_metrics = _eval_ranking(
        rank_path,
        seen_labels=seen_labels,
        seen_pairs=seen_pairs,
        src_db=src_db,
        tgt_db=tgt_db,
    )

    if ENABLE_CONTAMINATION_FILTER:
        print(f"  [FILTER] refs  {global_metrics['n_refs_pre_filter']:>6} -> {global_metrics['n_refs']:>6}")
        print(f"  [FILTER] preds {global_metrics['n_preds_pre_filter']:>6} -> {global_metrics['n_preds']:>6}")
        print(f"  [FILTER] rank  {ranking_metrics['n_test_pre_filter']:>6} -> {ranking_metrics['n_test']:>6}")

    metrics = {
        "task": task, "mode": "sweep" if sweep else "fixed",
        "threshold": chosen_threshold, "eval_on": eval_target,
        "contamination_filter": bool(ENABLE_CONTAMINATION_FILTER),
        "contamination_filter_mode": CONTAMINATION_FILTER_MODE,
        **global_metrics, **ranking_metrics, "src_missing": n_src_missing,
    }
    if sweep_info is not None:
        metrics["sweep_log"] = sweep_info["sweep"]
        metrics["n_train_refs"] = sweep_info["n_train_refs"]
        if "n_train_refs_pre_filter" in sweep_info:
            metrics["n_train_refs_pre_filter"] = sweep_info["n_train_refs_pre_filter"]
    if rerank_info is not None:
        metrics.update(rerank_info)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"  P={global_metrics['P']:.4f}  R={global_metrics['R']:.4f}  F1={global_metrics['F1']:.4f}")
    print(f"  MRR={ranking_metrics['MRR']:.4f}  "
          f"H@1={ranking_metrics['Hits@1']:.4f}  "
          f"H@5={ranking_metrics['Hits@5']:.4f}  "
          f"H@10={ranking_metrics['Hits@10']:.4f}")
    return metrics


# Summary writer (shared by orchestrator + single-task modes)

_SUMMARY_COLS = ["task", "mode", "threshold", "eval_on", "contamination_filter", "contamination_filter_mode",
                 "P", "R", "F1", "MRR", "Hits@1", "Hits@5", "Hits@10",
                 "n_preds", "n_refs", "tp", "n_test", "src_missing",
                 "n_preds_pre_filter", "n_refs_pre_filter", "n_test_pre_filter",
                 "n_train_refs", "n_train_refs_pre_filter"]


def _write_summary(metrics_list: List[Dict], out_root: Path) -> None:
    if not metrics_list:
        return
    summary_path = out_root / "results_summary.tsv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_COLS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for m in metrics_list:
            w.writerow(m)
    print(f"\nSummary -> {summary_path}\n")

    hdr = (f"{'task':<24} {'thr':>5} {'on':<10} "
           f"{'P':>6} {'R':>6} {'F1':>6} {'MRR':>6} {'H@1':>6} {'H@5':>6} {'H@10':>6}")
    print(hdr)
    print("-" * len(hdr))
    for m in metrics_list:
        print(f"{m['task']:<24} {m['threshold']:>5.2f} {m['eval_on']:<10} "
              f"{m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} "
              f"{m['MRR']:>6.3f} {m['Hits@1']:>6.3f} {m['Hits@5']:>6.3f} {m['Hits@10']:>6.3f}")


# Entry point

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--task", default=None, help=f"Run a single task. One of: {TASKS}")
    ap.add_argument("--sweep", action="store_true",
                    help="Tune threshold on train.tsv, eval on test.tsv. Default: fixed threshold, eval on full.tsv.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Force build_vdb to overwrite cached FAISS DBs. Needed after enabling STRIP_* flags.")
    args = ap.parse_args()
    root = Path(LEONMAP_ROOT)

    # Multi-task mode: shell out per task so each gets a fresh interpreter.
    # LeonMap's build_vdb/mapper carry process-level state (model cache, owlready2 world,
    # faiss threads) that doesn't reset reliably in-process. Subprocess per task is the
    # difference between neoplas F1=0.528 (stale) and 0.798 (clean).
    if args.task is None:
        suffix = "sweep" if args.sweep else "fixed"
        out_root = root / "oaei_results" / RECORD_ID / suffix
        out_root.mkdir(parents=True, exist_ok=True)

        print(f"\n[ORCHESTRATOR] Running {len(TASKS)} tasks as subprocesses.")
        for task in TASKS:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--task", task]
            if args.sweep:
                cmd.append("--sweep")
            if args.rebuild:
                cmd.append("--rebuild")
            print(f"\n[ORCHESTRATOR] -> {' '.join(cmd)}")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"[ORCHESTRATOR] Task {task} exited with code {rc}; continuing.")

        merged: List[Dict] = []
        for task in TASKS:
            mj = out_root / task / "metrics.json"
            if mj.exists():
                merged.append(json.loads(mj.read_text(encoding="utf-8")))
        _write_summary(merged, out_root)
        print("\n[ORCHESTRATOR] Done.")
        sys.exit(0)

    # Single-task mode.

    # Patch PROJECT_ROOT before anything imports from leonmap.
    import leonmap.config as _cfg
    _cfg.PROJECT_ROOT = root

    from leonmap.config import BuildConfig, COLLECTIONS, MAPPINGS, resolve_path
    from leonmap.utils import load_collection, rank_pool, canonicalize_id

    # Disable the build_vdb interactive preview ("Proceed? [y/n]") by injecting
    # monitor_samples=0 into BuildConfig defaults. Explicit overrides still win.
    _orig_buildcfg_init = BuildConfig.__init__

    def _patched_buildcfg_init(self, *a, **kw):
        kw.setdefault("monitor_samples", 0)
        _orig_buildcfg_init(self, *a, **kw)

    BuildConfig.__init__ = _patched_buildcfg_init

    # Patch load_owl_concepts on build_vdb's bound name (not leonmap.utils) since
    # build_vdb does `from leonmap.utils import load_owl_concepts` at import time.
    from leonmap.build_vdb import main as build_main
    from leonmap.mapper import main as mapper_main
    import leonmap.build_vdb as _build_vdb
    _orig_load_owl_concepts = _build_vdb.load_owl_concepts

    def _patched_load_owl_concepts(owl_path, id_prefixes=None):
        concepts = _orig_load_owl_concepts(owl_path, id_prefixes=id_prefixes)
        fname = Path(owl_path).name.lower()
        if STRIP_SNOMED_SUFFIXES and "snomed" in fname:
            n_lbl = n_syn = n_syn_total = 0
            for c in concepts:
                lbl = c.get("label", "")
                cleaned = _strip_paren_suffix(lbl)
                if cleaned != lbl:
                    c["label"] = cleaned; n_lbl += 1
                syns = c.get("synonyms", []) or []
                new_syns = []
                for s in syns:
                    n_syn_total += 1
                    s2 = _strip_paren_suffix(s)
                    if s2 != s:
                        n_syn += 1
                    new_syns.append(s2)
                c["synonyms"] = new_syns
            print(f"  [SNOMED] stripped {n_lbl}/{len(concepts)} labels, {n_syn}/{n_syn_total} synonyms in {fname}")
        if STRIP_OMIM_TYPE_ARTIFACT and "omim" in fname:
            n_lbl = n_syn = n_syn_total = 0
            for c in concepts:
                lbl = c.get("label", "")
                cleaned = _restore_omim_type(lbl)
                if cleaned != lbl:
                    c["label"] = cleaned; n_lbl += 1
                syns = c.get("synonyms", []) or []
                new_syns = []
                for s in syns:
                    n_syn_total += 1
                    s2 = _restore_omim_type(s)
                    if s2 != s:
                        n_syn += 1
                    new_syns.append(s2)
                c["synonyms"] = new_syns
            print(f"  [OMIM] restored 'type' in {n_lbl}/{len(concepts)} labels, {n_syn}/{n_syn_total} synonyms in {fname}")
        return concepts

    _build_vdb.load_owl_concepts = _patched_load_owl_concepts

    _LM.update({
        "cfg_mod": _cfg, "BuildConfig": BuildConfig,
        "load_collection": load_collection, "rank_pool": rank_pool,
        "canonicalize_id": canonicalize_id,
        "build_main": build_main, "mapper_main": mapper_main,
    })

    # Fetch FT SapBERT if missing.
    cfg = BuildConfig()
    model_dir = resolve_path(cfg.ft_model_path)
    if not model_dir.exists():
        print(f"Model not found locally, downloading from HF: {HF_MODEL_REPO}")
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(model_dir))
        print(f"Model -> {model_dir}")

    year_dir = root / BIOML_DATA_DIR
    if not year_dir.is_dir():
        raise SystemExit(f"Bio-ML data directory not found: {year_dir}")

    if not args.rebuild and (STRIP_SNOMED_SUFFIXES or STRIP_OMIM_TYPE_ARTIFACT):
        print("[NOTE] STRIP_* flags on. Pass --rebuild if affected FAISS DBs predate them.")

    if args.task and args.task not in TASKS:
        raise SystemExit(f"Unknown task: {args.task}. One of: {TASKS}")
    tasks_to_run = [args.task] if args.task else list(TASKS)

    suffix = "sweep" if args.sweep else "fixed"
    out_root = root / "oaei_results" / RECORD_ID / suffix
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict] = []
    for task in tasks_to_run:
        m = run_task(task, year_dir, out_root / task, sweep=args.sweep, rebuild=args.rebuild)
        if m:
            all_metrics.append(m)

    _write_summary(all_metrics, out_root)
    print("\nDone.")