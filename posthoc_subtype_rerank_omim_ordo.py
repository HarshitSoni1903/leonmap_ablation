from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


LEONMAP_ROOT = Path(__file__).resolve().parent.parent
RECORD_ID = "13119437"
BIOML_DATA_DIR = f"data/{RECORD_ID}"
TASK = "omim-ordo"

SRC_COL = f"oaei_{TASK}_src"
TGT_COL = f"oaei_{TASK}_tgt"
MAPPING_KEY = f"oaei_{TASK}"

ALPHA_BETA_GRID: List[Tuple[float, float]] = [
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

THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85]

SPECIFIC_PATTERNS = [
    ("type", re.compile(r"\btype\s+([0-9]+[a-z]?|[ivx]+[a-z]?)\b", re.I)),
    ("ar", re.compile(r"\bautosomal\s+recessive\s+([0-9]+[a-z]?)\b", re.I)),
    ("ad", re.compile(r"\bautosomal\s+dominant\s+([0-9]+[a-z]?)\b", re.I)),
    ("xl", re.compile(r"\bx[-\s]?linked\s+([0-9]+[a-z]?)\b", re.I)),
    ("familial", re.compile(r"\bfamilial\s*,?\s*([0-9]+[a-z]?)\b", re.I)),
]

TRAILING_TOKEN_RE = re.compile(r"\b([0-9]+[a-z]?|[ivx]+[a-z]?)\s*$", re.I)


def subtype_signatures(label: str) -> Set[str]:
    if not label:
        return set()

    s = label.lower().strip()
    out: Set[str] = set()

    for prefix, pattern in SPECIFIC_PATTERNS:
        for m in pattern.finditer(s):
            token = m.group(1).lower()
            out.add(token)
            out.add(f"{prefix}:{token}")

    m = TRAILING_TOKEN_RE.search(s)
    if m:
        out.add(m.group(1).lower())

    return out


def iri_to_id(canonicalize_id, iri: str) -> str:
    tail = iri.split("#")[-1].rsplit("/", 1)[-1].strip()
    if (
        "id.nlm.nih.gov/mesh/" in iri
        or "obo/mesh#" in iri
        or "purl.obolibrary.org/obo/mesh" in iri
    ):
        return canonicalize_id(f"mesh:{tail}")
    return canonicalize_id(tail)


def load_refs(path: Path) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]]]:
    src_to_gold: Dict[str, Set[str]] = defaultdict(set)
    pairs: Set[Tuple[str, str]] = set()

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row["SrcEntity"].strip()
            t = row["TgtEntity"].strip()
            if s and t:
                src_to_gold[s].add(t)
                pairs.add((s, t))

    return src_to_gold, pairs


def f1_score(
    preds: Set[Tuple[str, str]],
    refs: Set[Tuple[str, str]],
) -> Tuple[float, float, float, int]:
    tp = len(preds & refs)
    p = tp / len(preds) if preds else 0.0
    r = tp / len(refs) if refs else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, tp


def find_latest_mapper_tsv(project_root: Path) -> Path:
    runs = sorted((project_root / "mapper_results" / MAPPING_KEY).glob("run_*"))
    if not runs:
        raise SystemExit(f"No mapper runs found under mapper_results/{MAPPING_KEY}")

    path = runs[-1] / f"{SRC_COL}_to_{TGT_COL}.tsv"
    if not path.exists():
        raise SystemExit(f"Latest mapper TSV not found: {path}")

    return path


def apply_subtype_rerank(
    pool: List[Tuple[str, float]],
    src_label: str,
    tgt_db,
    alpha: float,
    beta: float,
) -> List[Tuple[str, float]]:
    if not pool:
        return []

    if alpha == 0.0 and beta == 0.0:
        return list(pool)

    src_sigs = subtype_signatures(src_label)
    if not src_sigs:
        return list(pool)

    cand_sigs: Dict[str, Set[str]] = {}

    for tgt_id, _ in pool:
        payload = tgt_db.get_payload_by_id(tgt_id) or {}
        labels = [payload.get("label", "")] + list(payload.get("synonyms") or [])

        sigs: Set[str] = set()
        for label in labels:
            sigs |= subtype_signatures(label)

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


def build_topk_cache(
    src_iris: Set[str],
    src_db,
    tgt_db,
    tgt_pos2id: Dict[int, str],
    canonicalize_id,
    rank_pool,
    top_k: int,
) -> Dict[str, Tuple[str, List[Tuple[str, float]]]]:
    cache: Dict[str, Tuple[str, List[Tuple[str, float]]]] = {}
    missing = 0

    for src_iri in sorted(src_iris):
        src_id = iri_to_id(canonicalize_id, src_iri)

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
        print(f"Missing source concepts from FAISS DB: {missing}")

    return cache


def evaluate_config(
    cache: Dict[str, Tuple[str, List[Tuple[str, float]]]],
    tgt_db,
    tgt_id_to_iri: Dict[str, str],
    gold_pairs: Set[Tuple[str, str]],
    alpha: float,
    beta: float,
    threshold: float,
) -> Tuple[float, float, float, int, int, Dict[str, Tuple[str, float]]]:
    preds: Set[Tuple[str, str]] = set()
    top1_by_src: Dict[str, Tuple[str, float]] = {}

    for src_iri, (src_label, pool) in cache.items():
        reranked = apply_subtype_rerank(pool, src_label, tgt_db, alpha, beta)

        if not reranked:
            continue

        tgt_id, score = reranked[0]
        top1_by_src[src_iri] = (tgt_id, score)

        if score < threshold:
            continue

        tgt_iri = tgt_id_to_iri.get(tgt_id)

        if tgt_iri:
            preds.add((src_iri, tgt_iri))

    p, r, f1, tp = f1_score(preds, gold_pairs)
    return p, r, f1, tp, len(preds), top1_by_src


def flip_analysis(
    baseline_top1: Dict[str, Tuple[str, float]],
    rerank_top1: Dict[str, Tuple[str, float]],
    src_to_gold: Dict[str, Set[str]],
    canonicalize_id,
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

        gold_ids = {
            iri_to_id(canonicalize_id, iri)
            for iri in src_to_gold.get(src_iri, set())
        }

        old_correct = old_id in gold_ids
        new_correct = new_id in gold_ids

        if not old_correct and new_correct:
            good += 1
        elif old_correct and not new_correct:
            bad += 1
        else:
            neutral += 1

    return good, bad, neutral


def write_predictions(
    out_path: Path,
    top1_by_src: Dict[str, Tuple[str, float]],
    tgt_id_to_iri: Dict[str, str],
    threshold: float,
) -> int:
    n = 0

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["SrcEntity", "TgtEntity", "Score"])

        for src_iri, (tgt_id, score) in sorted(top1_by_src.items()):
            if score < threshold:
                continue

            tgt_iri = tgt_id_to_iri.get(tgt_id)

            if not tgt_iri:
                continue

            w.writerow([src_iri, tgt_iri, f"{score:.6f}"])
            n += 1

    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--mode", default="sweep", choices=["sweep", "fixed"])
    ap.add_argument("--write-best", action="store_true")
    ap.add_argument("--selection-threshold", type=float, default=None)
    args = ap.parse_args()

    import leonmap.config as _cfg

    _cfg.PROJECT_ROOT = LEONMAP_ROOT

    from leonmap.config import BuildConfig
    from leonmap.utils import canonicalize_id, load_collection, rank_pool

    _cfg.COLLECTIONS[SRC_COL] = {
        "source": "owl",
        "model": "ft",
        "owl_path": str(LEONMAP_ROOT / BIOML_DATA_DIR / args.task / "omim.owl"),
        "id_prefixes": [],
    }

    _cfg.COLLECTIONS[TGT_COL] = {
        "source": "owl",
        "model": "ft",
        "owl_path": str(LEONMAP_ROOT / BIOML_DATA_DIR / args.task / "ordo.owl"),
        "id_prefixes": [],
    }

    cfg = BuildConfig()
    src_db = load_collection(cfg, SRC_COL)
    tgt_db = load_collection(cfg, TGT_COL)

    tgt_pos2id: Dict[int, str] = {
        pos: tid for tid, pos in tgt_db.id2pos.items()
    }

    tgt_id_to_iri: Dict[str, str] = {}
    for tgt_id in tgt_db.id2pos:
        payload = tgt_db.get_payload_by_id(tgt_id) or {}
        iri = payload.get("iri", "")
        if iri:
            tgt_id_to_iri[tgt_id] = iri

    task_dir = LEONMAP_ROOT / BIOML_DATA_DIR / args.task
    refs_train = task_dir / "refs_equiv" / "train.tsv"
    refs_test = task_dir / "refs_equiv" / "test.tsv"

    match_path = (
        LEONMAP_ROOT
        / "oaei_results"
        / RECORD_ID
        / args.mode
        / args.task
        / "match.result.tsv"
    )
    mapper_tsv = find_latest_mapper_tsv(LEONMAP_ROOT)

    train_src_to_gold, train_gold_pairs = load_refs(refs_train)
    test_src_to_gold, test_gold_pairs = load_refs(refs_test)

    print(f"Loaded FAISS DBs: src={len(src_db.id2pos)}, tgt={len(tgt_db.id2pos)}")
    print(f"Train sources: {len(train_src_to_gold)}, train gold pairs: {len(train_gold_pairs)}")
    print(f"Test sources:  {len(test_src_to_gold)}, test gold pairs:  {len(test_gold_pairs)}")
    print(f"Current match file: {match_path}")
    print(f"Latest mapper TSV:  {mapper_tsv}")
    print(f"Top-K: {args.top_k}")

    print("\nCaching train top-K pools...")
    train_cache = build_topk_cache(
        set(train_src_to_gold.keys()),
        src_db,
        tgt_db,
        tgt_pos2id,
        canonicalize_id,
        rank_pool,
        args.top_k,
    )

    print("Caching test top-K pools...")
    test_cache = build_topk_cache(
        set(test_src_to_gold.keys()),
        src_db,
        tgt_db,
        tgt_pos2id,
        canonicalize_id,
        rank_pool,
        args.top_k,
    )

    _, _, _, _, _, baseline_train_top1 = evaluate_config(
        train_cache,
        tgt_db,
        tgt_id_to_iri,
        train_gold_pairs,
        0.0,
        0.0,
        0.75,
    )

    _, _, _, _, _, baseline_test_top1 = evaluate_config(
        test_cache,
        tgt_db,
        tgt_id_to_iri,
        test_gold_pairs,
        0.0,
        0.0,
        0.75,
    )

    print("\nTuning alpha/beta/threshold on TRAIN:")
    print(
        f"{'alpha':>6} {'beta':>6} {'thr':>6} | "
        f"{'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'Preds':>7} | "
        f"{'good':>6} {'bad':>6} {'neutral':>8} {'net':>6}"
    )
    print("-" * 96)

    best = {
        "alpha": 0.0,
        "beta": 0.0,
        "threshold": 0.75,
        "train_f1": -1.0,
        "train_top1": None,
    }

    thresholds_to_use = (
        [args.selection_threshold]
        if args.selection_threshold is not None
        else THRESHOLDS
    )

    for alpha, beta in ALPHA_BETA_GRID:
        for threshold in thresholds_to_use:
            p, r, f1, tp, n_preds, top1 = evaluate_config(
                train_cache,
                tgt_db,
                tgt_id_to_iri,
                train_gold_pairs,
                alpha,
                beta,
                threshold,
            )

            good, bad, neutral = flip_analysis(
                baseline_train_top1,
                top1,
                train_src_to_gold,
                canonicalize_id,
            )

            print(
                f"{alpha:6.2f} {beta:6.2f} {threshold:6.2f} | "
                f"{p:8.4f} {r:8.4f} {f1:8.4f} {tp:6d} {n_preds:7d} | "
                f"{good:6d} {bad:6d} {neutral:8d} {good - bad:6d}"
            )

            if f1 > best["train_f1"]:
                best.update(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "threshold": threshold,
                        "train_f1": f1,
                        "train_top1": top1,
                    }
                )

    alpha = float(best["alpha"])
    beta = float(best["beta"])
    threshold = float(best["threshold"])

    print("\nBest TRAIN config:")
    print(f"  alpha     = {alpha:.2f}")
    print(f"  beta      = {beta:.2f}")
    print(f"  threshold = {threshold:.2f}")
    print(f"  train F1  = {best['train_f1']:.4f}")

    print("\nEvaluating selected config on TEST:")

    base_p, base_r, base_f1, base_tp, base_n, base_top1 = evaluate_config(
        test_cache,
        tgt_db,
        tgt_id_to_iri,
        test_gold_pairs,
        0.0,
        0.0,
        0.75,
    )

    test_p, test_r, test_f1, test_tp, test_n, test_top1 = evaluate_config(
        test_cache,
        tgt_db,
        tgt_id_to_iri,
        test_gold_pairs,
        alpha,
        beta,
        threshold,
    )

    good, bad, neutral = flip_analysis(
        base_top1,
        test_top1,
        test_src_to_gold,
        canonicalize_id,
    )

    print("\nBaseline TEST @ threshold 0.75:")
    print(
        f"  P={base_p:.4f} R={base_r:.4f} "
        f"F1={base_f1:.4f} TP={base_tp} Preds={base_n}"
    )

    print("\nReranked TEST using train-selected config:")
    print(
        f"  P={test_p:.4f} R={test_r:.4f} "
        f"F1={test_f1:.4f} TP={test_tp} Preds={test_n}"
    )

    print("\nFlip analysis on TEST:")
    print(f"  miss -> hit good:  {good}")
    print(f"  hit -> miss bad:   {bad}")
    print(f"  wrong -> wrong:    {neutral}")
    print(f"  net correctness:   {good - bad}")

    if args.write_best:
        out_path = match_path.with_name("match.result.rerank.train_selected.tsv")
        n_written = write_predictions(out_path, test_top1, tgt_id_to_iri, threshold)
        print(f"\nWrote reranked test predictions: {out_path}")
        print(f"Rows written: {n_written}")


if __name__ == "__main__":
    main()