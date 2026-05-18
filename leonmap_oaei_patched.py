"""
Run LeonMap against the OAEI Bio-ML benchmark.

Evaluates LeonMap on the five Bio-ML pairs (OMIM-ORDO, NCIT-DOID, and three UMLS
pairs) using OAEI's own protocol: global matching (P/R/F1) and local ranking
(MRR / Hits@K). Mapper is invoked once per task at a low floor threshold so the
raw scored predictions can be filtered post-hoc without re-embedding.

Two evaluation modes:
  - default: fixed threshold per task (TASK_THRESHOLDS, falling back to
    DEFAULT_THRESHOLD), evaluated against refs_equiv/full.tsv. Unsupervised.
  - --sweep: tune threshold on refs_equiv/train.tsv, evaluate against
    refs_equiv/test.tsv. Semi-supervised (OAEI-comparable).

Expected layout (placed manually):

    {LEONMAP_ROOT}/{BIOML_DATA_DIR}/{task}/
        <src>.owl
        <tgt>.owl
        refs_equiv/{full,test,train}.tsv
        refs_equiv/test.cands.tsv

Usage:
    python leonmap_oaei.py                                # all tasks, fixed thresholds
    python leonmap_oaei.py --sweep                        # all tasks, tuned on train
    python leonmap_oaei.py --task ncit-doid --sweep       # one task, sweep mode

The fine-tuned SapBERT model is pulled from HuggingFace on first run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np

from huggingface_hub import snapshot_download


# Config

# Anchor everything to the script's parent dir, so cwd doesn't matter.
LEONMAP_ROOT = Path(__file__).resolve().parent.parent       # leonmap_ablation/ -> mapnet/
RECORD_ID = "13119437"                                      # Zenodo record id; doubles as folder name
BIOML_DATA_DIR = f"data/{RECORD_ID}"                        # joined with LEONMAP_ROOT
HF_MODEL_REPO = "harshitsoni1903/sapbert-finetuned-semra"

# Threshold used when NOT sweeping. Single source of truth.
DEFAULT_THRESHOLD = 0.9

# Per-task override (None -> DEFAULT_THRESHOLD). Lets us pin per-pair values
# after looking at score distributions, without rerunning the encoder.
TASK_THRESHOLDS: Dict[str, Optional[float]] = {
    "omim-ordo":           None,
    "ncit-doid":           None,
    "snomed-fma.body":     None,
    "snomed-ncit.pharm":   None,
    "snomed-ncit.neoplas": None,
}

# Sweep grid (used only with --sweep). Coarse 0.05 steps from 0.50 to 0.95.
SWEEP_GRID: List[float] = [round(0.50 + 0.05 * i, 2) for i in range(10)]

# Mapper runs at this floor so its TSV contains every post-boost candidate
# we might ever want to threshold above. Must be <= min(SWEEP_GRID) and <=
# min(TASK_THRESHOLDS.values()) and <= DEFAULT_THRESHOLD.
MAPPER_FLOOR_THRESHOLD = 0.5

# Semi-supervised pattern/subtype reranking.
# Used ONLY in --sweep mode, where train.tsv is available.
# This does not modify LeonMap's base mapper and does not rebuild FAISS DBs.
ENABLE_SEMISUPERVISED_RERANK = True
RERANK_TASKS = {"omim-ordo"}
RERANK_TOP_K = 50

RERANK_GRID: List[Tuple[float, float]] = [
    (0.00, 0.00),
    (0.03, 0.03),
    (0.05, 0.03),
    (0.05, 0.05),
    (0.08, 0.05),
    (0.10, 0.05),
    (0.10, 0.08),
    (0.10, 0.10),
    (0.12, 0.08),
    (0.15, 0.10),
    (0.20, 0.10),
]

_RERANK_SPECIFIC_PATTERNS = [
    ("type", re.compile(r"\btype\s+([0-9]+[a-z]?|[ivx]+[a-z]?)\b", re.I)),
    ("ar", re.compile(r"\bautosomal\s+recessive\s+([0-9]+[a-z]?)\b", re.I)),
    ("ad", re.compile(r"\bautosomal\s+dominant\s+([0-9]+[a-z]?)\b", re.I)),
    ("xl", re.compile(r"\bx[-\s]?linked\s+([0-9]+[a-z]?)\b", re.I)),
    ("familial", re.compile(r"\bfamilial\s*,?\s*([0-9]+[a-z]?)\b", re.I)),
]

_RERANK_TRAILING_TOKEN_RE = re.compile(
    r"\b([0-9]+[a-z]?|[ivx]+[a-z]?)\s*$",
    re.I,
)

# SNOMED labels in Bio-ML come wrapped with semantic-type markers and
# structural noise that defeat LeonMap's lexical boost on what would otherwise
# be exact label matches. We strip, in order:
#   1. trailing parenthetical: " (body structure)", " (disorder)", etc.
#   2. leading "Structure of " (8k+ of 34k SNOMED Body labels start this way)
#   3. trailing " structure" / " part" / " region" / " area" left over after
#      paren removal (e.g. "Lateral meniscus structure" -> "Lateral meniscus")
# Applied to labels and synonyms of any collection whose OWL filename starts
# with "snomed". Requires --rebuild on first run to refresh cached FAISS DBs.
STRIP_SNOMED_SUFFIXES = True
_SNOMED_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]+\)\s*$")
_SNOMED_LEADING_RE = re.compile(r"^Structure of\s+", re.IGNORECASE)
_SNOMED_TRAILING_NOISE_RE = re.compile(r"\s+(structure|part|region|area)\s*$", re.IGNORECASE)

# OMIM labels in Bio-ML 2024 OMIM-ORDO have a data-prep artifact: the literal
# token "TYPE" has been replaced with "  iia" (two spaces plus 'iia') in both
# rdfs:label and skos:exactMatch synonyms. Affects 928/9622 OMIM concepts
# (10%). Examples:
#   "ACROFACIAL DYSOSTOSIS, CATANIA TYPE"  -> "acrofacial dysostosis, catania  iia"
#   "NEUROFIBROMATOSIS, TYPE 2"            -> "neurofibromatosis,  iia 2"
#   "AMELOGENESIS IMPERFECTA, TYPE 1B"     -> "amelogenesis imperfecta,  iia 1b"
# Without correction, SapBERT embeds the corrupted string and the lexical boost
# never fires against ORDO (which has clean labels). Test F1 on the affected
# slice is 0.29 vs 0.62 on the clean slice.
#
# Signature is unambiguous: every double-space in the OMIM OWL is part of
# "  iia"; every legitimate subtype suffix ("iiia", "iiib", "iiic", "iiid",
# real single-space "iia" in CDG-IIa) uses one space and is left alone.
# Applied to labels and synonyms of any collection whose OWL filename starts
# with "omim". Requires --rebuild on first run to refresh cached FAISS DBs.
STRIP_OMIM_TYPE_ARTIFACT = True
_OMIM_TYPE_ARTIFACT_RE = re.compile(r"  iia\b")  # exactly two spaces, exactly 'iia'

TASKS = [
    "omim-ordo",
    "ncit-doid",
    "snomed-fma.body",
    "snomed-ncit.pharm",
    "snomed-ncit.neoplas",
]

# Populated in __main__ once PROJECT_ROOT is patched.
_LM: Dict = {}


# Helpers

def _owl_files(task: str) -> Tuple[str, str]:
    """
    Derive (src_owl, tgt_owl) filenames from a Bio-ML task name.

    Examples:
        "ncit-doid"            -> ("ncit.owl",          "doid.owl")
        "snomed-fma.body"      -> ("snomed.body.owl",   "fma.body.owl")
        "snomed-ncit.neoplas"  -> ("snomed.neoplas.owl","ncit.neoplas.owl")
    """
    src_part, tgt_part = task.split("-", 1)
    if "." in tgt_part:
        tgt_short, subdomain = tgt_part.split(".", 1)
        return f"{src_part}.{subdomain}.owl", f"{tgt_short}.{subdomain}.owl"
    return f"{src_part}.owl", f"{tgt_part}.owl"


def _strip_paren_suffix(s: str) -> str:
    """
    Clean a SNOMED label/synonym in three passes:
      "Structure of base of lung (body structure)"  -> "base of lung"
      "Lateral meniscus structure (body structure)" -> "Lateral meniscus"
      "Medulla oblongata part (body structure)"     -> "Medulla oblongata"
      "Acquired pericardial cyst (disorder)"        -> "Acquired pericardial cyst"
    """
    if not s:
        return s
    s = _SNOMED_TRAILING_PAREN_RE.sub("", s)
    s = _SNOMED_LEADING_RE.sub("", s)
    s = _SNOMED_TRAILING_NOISE_RE.sub("", s)
    return s.strip()


def _restore_omim_type(s: str) -> str:
    """
    Reverse the Bio-ML 2024 OMIM 'TYPE' -> '  iia' substitution. Matches the
    artifact strictly (exactly two spaces, exactly the token 'iia'), so
    legitimate subtype tokens with one space ('iiia', 'iiib', 'iiic', 'iiid',
    real ' iia' in CDG-IIa) are left alone.
      "acrofacial dysostosis, catania  iia"   -> "acrofacial dysostosis, catania type"
      "neurofibromatosis,  iia 2"             -> "neurofibromatosis, type 2"
      "congenital disorder of glycosylation,  iia iia"
                                              -> "congenital disorder of glycosylation, type iia"
      "glycogen storage disease iiia"         -> unchanged
    """
    if not s:
        return s
    s = _OMIM_TYPE_ARTIFACT_RE.sub(" type", s)
    return re.sub(r"\s+", " ", s).strip()


def _iri_to_id(iri: str) -> str:
    """Re-use LeonMap's own canonicalize_id on the IRI tail. Same rule as build-time."""
    canonicalize_id = _LM["canonicalize_id"]
    tail = iri.split("#")[-1].rsplit("/", 1)[-1].strip()
    if "id.nlm.nih.gov/mesh/" in iri or "obo/mesh#" in iri or "purl.obolibrary.org/obo/mesh" in iri:
        return canonicalize_id(f"mesh:{tail}")
    return canonicalize_id(tail)


def _get_ignored_iris(owl_path: Path) -> set:
    """
    IRIs of classes annotated `use_in_alignment=false` (locality-module
    auxiliary classes). Predictions involving these are dropped before eval.
    """
    from owlready2 import get_ontology

    onto = get_ontology(Path(owl_path).as_posix()).load()
    ignored: set = set()
    for cls in onto.classes():
        val = getattr(cls, "use_in_alignment", None)
        is_false = (val is False) or (isinstance(val, list) and any(v is False for v in val))
        if is_false:
            ignored.add(str(cls.iri))
    return ignored


def _register_task(task: str, src_owl_abs: Path, tgt_owl_abs: Path) -> Tuple[str, str, str]:
    """
    Inject runtime entries into leonmap.config so build_vdb and mapper can find
    them via --collections / --study. Mapper threshold is set to the floor so
    the resulting TSV preserves every candidate we might want to keep later.
    """
    _cfg = _LM["cfg_mod"]
    src_col = f"oaei_{task}_src"
    tgt_col = f"oaei_{task}_tgt"
    mapping_key = f"oaei_{task}"

    _cfg.COLLECTIONS[src_col] = {
        "source": "owl",
        "model": "ft",
        "owl_path": str(src_owl_abs),
        "id_prefixes": [],
    }
    _cfg.COLLECTIONS[tgt_col] = {
        "source": "owl",
        "model": "ft",
        "owl_path": str(tgt_owl_abs),
        "id_prefixes": [],
    }
    _cfg.MAPPINGS[mapping_key] = {
        "src_collection": src_col,
        "tgt_collection": tgt_col,
        "threshold": MAPPER_FLOOR_THRESHOLD,
        "top_k": 1,
        "reverse": False,
    }
    return src_col, tgt_col, mapping_key


def _score_candidates(src_id: str, candidate_ids: List[str], src_db, tgt_db) -> List[Tuple[str, float, str]]:
    """
    Score a fixed candidate list against a source concept. No FAISS retrieval:
    reconstruct vectors directly, cosine, hand to rank_pool at threshold 0 so
    nothing is dropped. Boost still applies.
    """
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

    src_payload = src_db.get_payload_by_id(src_id) or {}
    src_label = src_payload.get("label", "")
    return rank_pool(pool, tgt_db, src_label, threshold=0.0, enable_boost=True)


def _run(entry_main, cli_name: str, argv: List[str]) -> None:
    """Invoke a LeonMap script's main() with a synthesized argv."""
    old = sys.argv
    sys.argv = [cli_name] + argv
    try:
        entry_main()
    finally:
        sys.argv = old


# I/O for predictions and references

def _load_mapper_predictions(
    mapper_tsv: Path, src_db, tgt_db, ignored: set,
) -> List[Tuple[str, str, float]]:
    """
    Read mapper.py's output TSV (canonical-id columns) and return a list of
    (src_iri, tgt_iri, score) tuples with ignored classes already filtered out.
    This is the raw post-boost prediction set we threshold against later.
    """
    preds: List[Tuple[str, str, float]] = []
    with open(mapper_tsv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            src_id = row.get("src_id", "").strip()
            tgt_id = row.get("tgt_id", "").strip()
            score_str = row.get("score", "").strip()
            if not src_id or not tgt_id or not score_str:
                continue
            try:
                score = float(score_str)
            except ValueError:
                continue
            src_iri = (src_db.get_payload_by_id(src_id) or {}).get("iri", "")
            tgt_iri = (tgt_db.get_payload_by_id(tgt_id) or {}).get("iri", "")
            if not src_iri or not tgt_iri:
                continue
            if src_iri in ignored or tgt_iri in ignored:
                continue
            preds.append((src_iri, tgt_iri, score))
    return preds


def _write_match_result(preds: List[Tuple[str, str, float]], threshold: float, out_path: Path) -> int:
    """Filter raw predictions at threshold and write the OAEI match.result.tsv."""
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
    """Load (src_iri, tgt_iri) gold pairs from a Bio-ML reference TSV."""
    pairs: set = set()
    with open(refs_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row.get("SrcEntity", "").strip()
            t = row.get("TgtEntity", "").strip()
            if s and t:
                pairs.add((s, t))
    return pairs


def _f1(preds: set, refs: set) -> Tuple[float, float, float, int]:
    tp = len(preds & refs)
    p = tp / len(preds) if preds else 0.0
    r = tp / len(refs) if refs else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp


# Semi-supervised top-K reranking helpers

def _rerank_signatures(label: str) -> Set[str]:
    """
    Extract lightweight subtype/pattern signatures from a label.

    Examples:
      "multiple endocrine neoplasia, type 2b" -> {"2b", "type:2b"}
      "spinocerebellar ataxia, autosomal recessive 7" -> {"7", "ar:7"}
      "charcot-marie-tooth disease, axonal, type 2d" -> {"2d", "type:2d"}
    """
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
    src_iris: Set[str],
    src_db,
    tgt_db,
    top_k: int,
) -> Dict[str, Tuple[str, List[Tuple[str, float]]]]:
    """
    Re-query FAISS top-K for source IRIs, then apply LeonMap's existing
    lexical boost via rank_pool. No DB rebuild and no base-code modification.
    """
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

        src_payload = src_db.get_payload_by_id(src_id) or {}
        src_label = src_payload.get("label", "")

        boosted = rank_pool(pool, tgt_db, src_label, threshold=0.0, enable_boost=True)
        boosted_pairs = [(r[0], float(r[1])) for r in boosted]
        cache[src_iri] = (src_label, boosted_pairs)

    if missing:
        print(f"  [RERANK] Missing source concepts from FAISS DB: {missing}")

    return cache


def _apply_pattern_rerank(
    pool: List[Tuple[str, float]],
    src_label: str,
    tgt_db,
    alpha: float,
    beta: float,
) -> List[Tuple[str, float]]:
    """
    Conservative pattern/subtype rerank.

    If the source has no recognizable signature, do nothing.
    If no candidate in the pool shares a source signature, do nothing.
    Otherwise, matching candidates get +alpha and nonmatching candidates get -beta.
    """
    if not pool:
        return []

    if alpha == 0.0 and beta == 0.0:
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

    has_matching_candidate = any(src_sigs & sigs for sigs in cand_sigs.values())

    if not has_matching_candidate:
        return list(pool)

    adjusted: List[Tuple[str, float]] = []

    for tgt_id, score in pool:
        if src_sigs & cand_sigs[tgt_id]:
            new_score = min(1.0, score + alpha)
        else:
            new_score = max(0.0, score - beta)

        adjusted.append((tgt_id, new_score))

    adjusted.sort(key=lambda x: -x[1])
    return adjusted


def _tgt_id_to_iri_map(tgt_db) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for tgt_id in tgt_db.id2pos:
        payload = tgt_db.get_payload_by_id(tgt_id) or {}
        iri = payload.get("iri", "")
        if iri:
            out[tgt_id] = iri

    return out


def _evaluate_rerank_cache(
    cache: Dict[str, Tuple[str, List[Tuple[str, float]]]],
    tgt_db,
    tgt_id_to_iri: Dict[str, str],
    refs: set,
    alpha: float,
    beta: float,
    threshold: float,
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
    good = 0
    bad = 0
    neutral = 0

    for src_iri, new_pair in rerank_top1.items():
        if src_iri not in baseline_top1:
            continue

        old_id = baseline_top1[src_iri][0]
        new_id = new_pair[0]

        if old_id == new_id:
            continue

        gold_ids = {_iri_to_id(gold_iri) for gold_iri in src_to_gold.get(src_iri, set())}

        old_correct = old_id in gold_ids
        new_correct = new_id in gold_ids

        if not old_correct and new_correct:
            good += 1
        elif old_correct and not new_correct:
            bad += 1
        else:
            neutral += 1

    return good, bad, neutral


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


def _semisupervised_pattern_rerank(
    task: str,
    src_db,
    tgt_db,
    refs_train: Path,
    refs_test: Path,
) -> Tuple[List[Tuple[str, str, float]], float, Dict]:
    """
    Tune pattern reranker on train.tsv, then apply selected config to test.tsv.

    Returns:
      - test raw predictions as (src_iri, tgt_iri, score), one per test source
      - chosen threshold
      - rerank metadata for metrics.json
    """
    train_src_to_gold, train_refs = _load_refs_by_src(refs_train)
    test_src_to_gold, test_refs = _load_refs_by_src(refs_test)

    tgt_id_to_iri = _tgt_id_to_iri_map(tgt_db)

    print(f"  [RERANK] Enabled for {task} in semi-supervised mode")
    print(f"  [RERANK] top_k={RERANK_TOP_K}")
    print(f"  [RERANK] Train sources={len(train_src_to_gold)}, refs={len(train_refs)}")
    print(f"  [RERANK] Test sources={len(test_src_to_gold)}, refs={len(test_refs)}")

    print("  [RERANK] Caching train top-K pools...")
    train_cache = _build_topk_boosted_cache(set(train_src_to_gold.keys()), src_db, tgt_db, RERANK_TOP_K)

    print("  [RERANK] Caching test top-K pools...")
    test_cache = _build_topk_boosted_cache(set(test_src_to_gold.keys()), src_db, tgt_db, RERANK_TOP_K)

    _, _, _, _, _, baseline_train_top1 = _evaluate_rerank_cache(
        train_cache, tgt_db, tgt_id_to_iri, train_refs, 0.0, 0.0, 0.75
    )

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

    print("  [RERANK] Tuning alpha/beta/threshold on train...")

    for alpha, beta in RERANK_GRID:
        for threshold in SWEEP_GRID:
            p, r, f1, tp, n_preds, top1 = _evaluate_rerank_cache(
                train_cache, tgt_db, tgt_id_to_iri, train_refs, alpha, beta, threshold
            )

            good, bad, neutral = _flip_analysis(baseline_train_top1, top1, train_src_to_gold)

            rec = {
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
            }
            sweep_log.append(rec)

            if f1 > best["train_F1"]:
                best.update(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "threshold": threshold,
                        "train_F1": f1,
                        "train_P": p,
                        "train_R": r,
                        "train_tp": tp,
                        "train_n_preds": n_preds,
                    }
                )

    alpha = float(best["alpha"])
    beta = float(best["beta"])
    threshold = float(best["threshold"])

    print(
        f"  [RERANK] Best train config: "
        f"alpha={alpha:.2f}, beta={beta:.2f}, threshold={threshold:.2f}, "
        f"trainF1={best['train_F1']:.4f}"
    )

    base_p, base_r, base_f1, base_tp, base_n, baseline_test_top1 = _evaluate_rerank_cache(
        test_cache, tgt_db, tgt_id_to_iri, test_refs, 0.0, 0.0, 0.75
    )

    test_p, test_r, test_f1, test_tp, test_n, test_top1 = _evaluate_rerank_cache(
        test_cache, tgt_db, tgt_id_to_iri, test_refs, alpha, beta, threshold
    )

    good, bad, neutral = _flip_analysis(baseline_test_top1, test_top1, test_src_to_gold)

    print(
        f"  [RERANK] Baseline test @0.75: "
        f"P={base_p:.4f} R={base_r:.4f} F1={base_f1:.4f} "
        f"tp={base_tp} preds={base_n}"
    )
    print(
        f"  [RERANK] Reranked test: "
        f"P={test_p:.4f} R={test_r:.4f} F1={test_f1:.4f} "
        f"tp={test_tp} preds={test_n}"
    )
    print(f"  [RERANK] Test flips: good={good}, bad={bad}, neutral={neutral}, net={good - bad}")

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
    }

    return test_preds, threshold, rerank_info


# Threshold selection: sweep on train

def _sweep_threshold(
    preds: List[Tuple[str, str, float]], train_refs_path: Path, grid: List[float],
) -> Dict:
    """
    Pick the threshold that maximizes F1 on train.tsv. Predictions are filtered
    to those whose src_iri appears in train (OAEI convention: only score
    predictions for source concepts that have a gold mapping in the slice
    you're evaluating against).
    """
    train_refs = _load_refs(train_refs_path)
    train_src = {s for s, _ in train_refs}
    train_preds = [(s, t, sc) for s, t, sc in preds if s in train_src]

    sweep_log: List[Dict] = []
    best = {"threshold": grid[0], "F1": -1.0, "P": 0.0, "R": 0.0, "tp": 0, "n_preds": 0}
    for t in grid:
        preds_at_t = {(s, tg) for s, tg, sc in train_preds if sc >= t}
        p, r, f1, tp = _f1(preds_at_t, train_refs)
        rec = {"threshold": t, "P": p, "R": r, "F1": f1, "tp": tp, "n_preds": len(preds_at_t)}
        sweep_log.append(rec)
        if f1 > best["F1"]:
            best = rec
    return {"best": best, "sweep": sweep_log, "n_train_refs": len(train_refs)}


# Local ranking (unchanged by threshold; output is candidate scores)

def _run_local_ranking(test_cands_path: Path, src_db, tgt_db, out_path: Path) -> Tuple[int, int]:
    n_written = 0
    n_src_missing = 0
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

            scored: List[Tuple[str, float]] = [
                (iri, score_map.get(cid, 0.0))
                for iri, cid in zip(cand_iris, cand_ids)
            ]

            w.writerow([src_iri, tgt_iri, repr(scored)])
            n_written += 1
    return n_written, n_src_missing


def _eval_ranking(rank_path: Path, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict:
    mrr_sum = 0.0
    hits = {k: 0 for k in ks}
    n = 0
    with open(rank_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            gold = row["TgtEntity"].strip()
            scored = list(ast.literal_eval(row["TgtCandidates"]))
            scored.sort(key=lambda x: -float(x[1]))
            rank = next((i + 1 for i, (iri, _s) in enumerate(scored) if iri == gold), None)
            if rank is not None:
                mrr_sum += 1.0 / rank
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
            n += 1
    out: Dict = {"MRR": mrr_sum / n if n else 0.0, "n_test": n}
    for k in ks:
        out[f"Hits@{k}"] = hits[k] / n if n else 0.0
    return out


# Global eval: from a written match.result.tsv against a refs file

def _eval_global_match_file(match_path: Path, refs_path: Path, restrict_to_refs_src: bool) -> Dict:
    refs = _load_refs(refs_path)
    refs_src = {s for s, _ in refs}
    preds: set = set()
    with open(match_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row["SrcEntity"].strip()
            t = row["TgtEntity"].strip()
            if restrict_to_refs_src and s not in refs_src:
                continue
            preds.add((s, t))
    p, r_, f1, tp = _f1(preds, refs)
    return {"P": p, "R": r_, "F1": f1, "tp": tp, "n_preds": len(preds), "n_refs": len(refs)}


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

    # Build FAISS collections (no-op if already built, unless --rebuild).
    build_argv = ["--collections", src_col, tgt_col]
    if rebuild:
        build_argv.append("--rebuild")
    _run(_LM["build_main"], "leonmap-build", build_argv)

    # Mapper at the floor threshold so we keep every candidate.
    _run(_LM["mapper_main"], "leonmap-map",
         ["--study", mapping_key, "--threshold", str(MAPPER_FLOOR_THRESHOLD)])

    project_root: Path = _LM["cfg_mod"].PROJECT_ROOT
    mapper_runs_dir = project_root / "mapper_results" / mapping_key
    run_dirs = sorted(mapper_runs_dir.glob("run_*"), key=lambda p: p.name)
    if not run_dirs:
        print(f"[ERROR] No mapper run output found under {mapper_runs_dir}")
        return None
    latest = run_dirs[-1]
    mapper_tsv = latest / f"{src_col}_to_{tgt_col}.tsv"
    if not mapper_tsv.exists():
        print(f"[ERROR] Mapper output missing: {mapper_tsv}")
        return None

    # Load collections for IRI lookups + ranking.
    BuildConfig = _LM["BuildConfig"]
    load_collection = _LM["load_collection"]
    cfg = BuildConfig()
    src_db = load_collection(cfg, src_col)
    tgt_db = load_collection(cfg, tgt_col)

    print("  Reading locality-module annotations...")
    ignored = _get_ignored_iris(src_owl) | _get_ignored_iris(tgt_owl)
    print(f"  {len(ignored)} classes annotated use_in_alignment=false")

    out_dir.mkdir(parents=True, exist_ok=True)
    match_path = out_dir / "match.result.tsv"
    rank_path = out_dir / "rank.result.tsv"
    metrics_path = out_dir / "metrics.json"

    # Raw post-boost predictions (filtered for ignored). One row per source.
    raw_preds = _load_mapper_predictions(mapper_tsv, src_db, tgt_db, ignored)
    print(f"  Raw predictions (>= {MAPPER_FLOOR_THRESHOLD}): {len(raw_preds)}")

    # Pick threshold + decide which refs we evaluate against.
    sweep_info: Optional[Dict] = None
    rerank_info: Optional[Dict] = None
    preds_for_output = raw_preds

    if sweep:
        eval_refs = refs_test
        eval_target = "test.tsv"
        restrict_src = True

        if ENABLE_SEMISUPERVISED_RERANK and task in RERANK_TASKS:
            preds_for_output, chosen_threshold, rerank_info = _semisupervised_pattern_rerank(
                task=task,
                src_db=src_db,
                tgt_db=tgt_db,
                refs_train=refs_train,
                refs_test=refs_test,
            )
            sweep_info = {
                "best": {
                    "threshold": chosen_threshold,
                    "P": rerank_info["rerank_train_P"],
                    "R": rerank_info["rerank_train_R"],
                    "F1": rerank_info["rerank_train_F1"],
                    "tp": rerank_info["rerank_train_tp"],
                    "n_preds": rerank_info["rerank_train_n_preds"],
                },
                "sweep": rerank_info["rerank_sweep_log"],
                "n_train_refs": len(_load_refs(refs_train)),
            }
            print(
                f"  Sweep+rerank best: threshold={chosen_threshold:.2f}  "
                f"trainF1={rerank_info['rerank_train_F1']:.4f}  "
                f"trainP={rerank_info['rerank_train_P']:.4f}  "
                f"trainR={rerank_info['rerank_train_R']:.4f}"
            )
        else:
            sweep_info = _sweep_threshold(raw_preds, refs_train, SWEEP_GRID)
            chosen_threshold = float(sweep_info["best"]["threshold"])

            test_src = {s for s, _ in _load_refs(refs_test)}
            preds_for_output = [
                (s, t, sc)
                for s, t, sc in raw_preds
                if s in test_src
            ]

            print(
                f"  Sweep best: threshold={chosen_threshold:.2f}  "
                f"trainF1={sweep_info['best']['F1']:.4f}  "
                f"trainP={sweep_info['best']['P']:.4f}  "
                f"trainR={sweep_info['best']['R']:.4f}"
            )
    else:
        chosen_threshold = TASK_THRESHOLDS.get(task) or DEFAULT_THRESHOLD
        eval_refs = refs_full
        eval_target = "full.tsv"
        restrict_src = False
        print(f"  Fixed threshold: {chosen_threshold}")

    # Write match.result.tsv at the chosen threshold and evaluate.
    n_match = _write_match_result(preds_for_output, chosen_threshold, match_path)
    print(f"  Global matching: {n_match} predictions @ threshold={chosen_threshold} -> {eval_target}")
    global_metrics = _eval_global_match_file(match_path, eval_refs, restrict_to_refs_src=restrict_src)

    # Local ranking is threshold-independent.
    n_rank, n_src_missing = _run_local_ranking(test_cands, src_db, tgt_db, rank_path)
    print(f"  Local ranking: {n_rank} test rows scored ({n_src_missing} src not in DB)")
    ranking_metrics = _eval_ranking(rank_path)

    metrics = {
        "task": task,
        "mode": "sweep" if sweep else "fixed",
        "threshold": chosen_threshold,
        "eval_on": eval_target,
        **global_metrics,
        **ranking_metrics,
        "src_missing": n_src_missing,
    }
    if sweep_info is not None:
        metrics["sweep_log"] = sweep_info["sweep"]
        metrics["n_train_refs"] = sweep_info["n_train_refs"]

    if rerank_info is not None:
        metrics.update(rerank_info)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"  P={global_metrics['P']:.4f}  R={global_metrics['R']:.4f}  F1={global_metrics['F1']:.4f}")
    print(f"  MRR={ranking_metrics['MRR']:.4f}  "
          f"H@1={ranking_metrics['Hits@1']:.4f}  "
          f"H@5={ranking_metrics['Hits@5']:.4f}  "
          f"H@10={ranking_metrics['Hits@10']:.4f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--task", default=None, help=f"Run a single task. One of: {TASKS}")
    ap.add_argument("--sweep", action="store_true",
                    help="Tune threshold per task on refs_equiv/train.tsv, evaluate on test.tsv. "
                         "Default: fixed threshold per TASK_THRESHOLDS (or DEFAULT_THRESHOLD), eval on full.tsv.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Pass --rebuild to build_vdb so existing FAISS DBs are overwritten. "
                         "Needed after enabling STRIP_SNOMED_SUFFIXES if SNOMED collections were built earlier.")
    args = ap.parse_args()

    root = Path(LEONMAP_ROOT)

    # Multi-task mode: shell out per task to give each one a fresh interpreter.
    # LeonMap's build_vdb.main / mapper.main carry process-level state (model
    # cache, owlready2 default world, faiss thread state) that doesn't reliably
    # reset between in-process calls. Running each task in its own subprocess
    # is the difference between (e.g.) neoplas F1=0.528 (stale state) and
    # F1=0.798 (clean state). Single-task mode below does the actual work.
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

        # Stitch per-task metrics.json into a single summary table.
        merged: List[Dict] = []
        for task in TASKS:
            mj = out_root / task / "metrics.json"
            if mj.exists():
                merged.append(json.loads(mj.read_text(encoding="utf-8")))
        if merged:
            summary_path = out_root / "results_summary.tsv"
            cols = ["task", "mode", "threshold", "eval_on",
                    "P", "R", "F1", "MRR", "Hits@1", "Hits@5", "Hits@10",
                    "n_preds", "n_refs", "tp", "n_test", "src_missing"]
            with open(summary_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
                w.writeheader()
                for m in merged:
                    w.writerow(m)
            print(f"\nSummary -> {summary_path}\n")

            hdr = (f"{'task':<24} {'thr':>5} {'on':<10} "
                   f"{'P':>6} {'R':>6} {'F1':>6} {'MRR':>6} {'H@1':>6} {'H@5':>6} {'H@10':>6}")
            print(hdr)
            print("-" * len(hdr))
            for m in merged:
                print(f"{m['task']:<24} {m['threshold']:>5.2f} {m['eval_on']:<10} "
                      f"{m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} "
                      f"{m['MRR']:>6.3f} {m['Hits@1']:>6.3f} {m['Hits@5']:>6.3f} {m['Hits@10']:>6.3f}")
        print("\n[ORCHESTRATOR] Done.")
        sys.exit(0)

    # ----------------------------------------------------------------------
    # Single-task mode below. Runs exactly one task in this interpreter.
    # ----------------------------------------------------------------------

    # Patch PROJECT_ROOT before importing anything else from leonmap.
    import leonmap.config as _cfg
    _cfg.PROJECT_ROOT = root

    from leonmap.config import BuildConfig, COLLECTIONS, MAPPINGS, resolve_path
    from leonmap.utils import load_collection, rank_pool, canonicalize_id

    # Disable the interactive build preview ("Proceed with building all? [y/n]")
    # for our runs without touching LeonMap's config.py. build_vdb.main reads
    # cfg.monitor_samples from a freshly-instantiated BuildConfig, so we wrap
    # BuildConfig.__init__ to inject monitor_samples=0 by default. Explicit
    # overrides (BuildConfig(monitor_samples=5)) still win.
    _orig_buildcfg_init = BuildConfig.__init__

    def _patched_buildcfg_init(self, *a, **kw):
        kw.setdefault("monitor_samples", 0)
        _orig_buildcfg_init(self, *a, **kw)

    BuildConfig.__init__ = _patched_buildcfg_init

    # Monkey-patch load_owl_concepts to strip SNOMED label noise. This has to
    # patch build_vdb's already-bound reference, NOT leonmap.utils. build_vdb
    # does `from leonmap.utils import load_owl_concepts` at import time, so the
    # name in build_vdb's namespace points to the original function object;
    # patching leonmap.utils.load_owl_concepts after that doesn't reach the
    # caller. We import build_vdb here, then overwrite its bound name.
    from leonmap.build_vdb import main as build_main
    from leonmap.mapper import main as mapper_main
    import leonmap.build_vdb as _build_vdb
    _orig_load_owl_concepts = _build_vdb.load_owl_concepts

    def _patched_load_owl_concepts(owl_path, id_prefixes=None):
        concepts = _orig_load_owl_concepts(owl_path, id_prefixes=id_prefixes)
        if STRIP_SNOMED_SUFFIXES and "snomed" in Path(owl_path).name.lower():
            n_label_changed = 0
            n_syn_changed = 0
            n_syns_total = 0
            for c in concepts:
                lbl = c.get("label", "")
                cleaned = _strip_paren_suffix(lbl)
                if cleaned != lbl:
                    c["label"] = cleaned
                    n_label_changed += 1
                syns = c.get("synonyms", []) or []
                new_syns = []
                for s in syns:
                    n_syns_total += 1
                    s2 = _strip_paren_suffix(s)
                    if s2 != s:
                        n_syn_changed += 1
                    new_syns.append(s2)
                c["synonyms"] = new_syns
            print(f"  [SNOMED] Stripped suffixes from {n_label_changed}/{len(concepts)} labels "
                  f"and {n_syn_changed}/{n_syns_total} synonyms in {Path(owl_path).name}")
        if STRIP_OMIM_TYPE_ARTIFACT and "omim" in Path(owl_path).name.lower():
            n_label_changed = 0
            n_syn_changed = 0
            n_syns_total = 0
            for c in concepts:
                lbl = c.get("label", "")
                cleaned = _restore_omim_type(lbl)
                if cleaned != lbl:
                    c["label"] = cleaned
                    n_label_changed += 1
                syns = c.get("synonyms", []) or []
                new_syns = []
                for s in syns:
                    n_syns_total += 1
                    s2 = _restore_omim_type(s)
                    if s2 != s:
                        n_syn_changed += 1
                    new_syns.append(s2)
                c["synonyms"] = new_syns
            print(f"  [OMIM] Restored 'type' in {n_label_changed}/{len(concepts)} labels "
                  f"and {n_syn_changed}/{n_syns_total} synonyms in {Path(owl_path).name}")
        return concepts

    _build_vdb.load_owl_concepts = _patched_load_owl_concepts

    _LM.update({
        "cfg_mod": _cfg,
        "BuildConfig": BuildConfig,
        "load_collection": load_collection,
        "rank_pool": rank_pool,
        "canonicalize_id": canonicalize_id,
        "build_main": build_main,
        "mapper_main": mapper_main,
    })

    cfg = BuildConfig()
    model_dir = resolve_path(cfg.ft_model_path)
    if not model_dir.exists():
        print(f"Model not found locally, downloading from HF: {HF_MODEL_REPO}")
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(model_dir))
        print(f"Model -> {model_dir}")

    year_dir = root / BIOML_DATA_DIR
    if not year_dir.is_dir():
        raise SystemExit(f"Bio-ML data directory not found: {year_dir}")

    if STRIP_SNOMED_SUFFIXES and not args.rebuild:
        print("[NOTE] STRIP_SNOMED_SUFFIXES is ON. If SNOMED FAISS DBs were built before this "
              "feature was enabled, pass --rebuild on this run to refresh them.")
    if STRIP_OMIM_TYPE_ARTIFACT and not args.rebuild:
        print("[NOTE] STRIP_OMIM_TYPE_ARTIFACT is ON. If OMIM FAISS DB was built before this "
              "feature was enabled, pass --rebuild on this run to refresh it.")

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

    if all_metrics:
        summary_path = out_root / "results_summary.tsv"
        cols = ["task", "mode", "threshold", "eval_on",
                "P", "R", "F1", "MRR", "Hits@1", "Hits@5", "Hits@10",
                "n_preds", "n_refs", "tp", "n_test", "src_missing"]
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for m in all_metrics:
                w.writerow(m)
        print(f"\nSummary -> {summary_path}\n")

        hdr = (f"{'task':<24} {'thr':>5} {'on':<10} "
               f"{'P':>6} {'R':>6} {'F1':>6} {'MRR':>6} {'H@1':>6} {'H@5':>6} {'H@10':>6}")
        print(hdr)
        print("-" * len(hdr))
        for m in all_metrics:
            print(f"{m['task']:<24} {m['threshold']:>5.2f} {m['eval_on']:<10} "
                  f"{m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} "
                  f"{m['MRR']:>6.3f} {m['Hits@1']:>6.3f} {m['Hits@5']:>6.3f} {m['Hits@10']:>6.3f}")

    print("\nDone.")