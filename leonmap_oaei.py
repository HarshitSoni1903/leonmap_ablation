"""
Run LeonMap against the OAEI Bio-ML benchmark.

Evaluates LeonMap on the five Bio-ML pairs (OMIM-ORDO, NCIT-DOID, and three UMLS
pairs) using OAEI's own protocol: global matching (P/R/F1 vs refs_equiv/full.tsv)
and local ranking (MRR / Hits@K vs refs_equiv/test.cands.tsv). Results land in a
table that slots into the OAEI 2024 leaderboard.

Expected layout (you place the data manually):

    {LEONMAP_ROOT}/{BIOML_DATA_DIR}/{task}/
        <src>.owl
        <tgt>.owl
        refs_equiv/full.tsv
        refs_equiv/test.tsv
        refs_equiv/test.cands.tsv
        refs_equiv/train.tsv

Usage:
    python leonmap_oaei.py                       # run all five tasks
    python leonmap_oaei.py --task ncit-doid      # one task only

Install:
    pip install git+https://github.com/HarshitSoni1903/Weakly-Supervised-Representation-Learning-for-Cross-Ontology-Mapping.git
    pip install owlready2 huggingface_hub numpy

The fine-tuned SapBERT model is pulled from HuggingFace on first run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import snapshot_download


# Config

LEONMAP_ROOT = Path(__file__).resolve().parent.parent
RECORD_ID = "13119437"                       # Zenodo record id; doubles as folder name
BIOML_DATA_DIR = f"data/{RECORD_ID}"  # joined with LEONMAP_ROOT at runtime
THRESHOLD = 0.9
HF_MODEL_REPO = "harshitsoni1903/sapbert-finetuned-semra"

TASKS = [
    "omim-ordo",
    "ncit-doid",
    "snomed-fma.body",
    "snomed-ncit.pharm",
    "snomed-ncit.neoplas",
]

# Populated in __main__ once PROJECT_ROOT is patched.
_LM: Dict = {}


# Helpers (depend on _LM populated in __main__)

def _owl_files(task: str) -> Tuple[str, str]:
    """
    Derive (src_owl, tgt_owl) filenames from a Bio-ML task name.

    Examples:
        "ncit-doid"            -> ("ncit.owl",         "doid.owl")
        "snomed-fma.body"      -> ("snomed.body.owl",  "fma.body.owl")
        "snomed-ncit.neoplas"  -> ("snomed.neoplas.owl","ncit.neoplas.owl")
    """
    src_part, tgt_part = task.split("-", 1)
    if "." in tgt_part:
        tgt_short, subdomain = tgt_part.split(".", 1)
        return f"{src_part}.{subdomain}.owl", f"{tgt_short}.{subdomain}.owl"
    return f"{src_part}.owl", f"{tgt_part}.owl"


def _iri_to_id(iri: str) -> str:
    """
    Same IRI tail extraction LeonMap's _owl_class_id uses, then canonicalize.
    Going through LeonMap's own canonicalize_id guarantees we get the form
    that's actually in the FAISS collection's id2pos.
    """
    canonicalize_id = _LM["canonicalize_id"]
    tail = iri.split("#")[-1].rsplit("/", 1)[-1].strip()
    if "id.nlm.nih.gov/mesh/" in iri or "obo/mesh#" in iri or "purl.obolibrary.org/obo/mesh" in iri:
        return canonicalize_id(f"mesh:{tail}")
    return canonicalize_id(tail)


def _get_ignored_iris(owl_path: Path) -> set:
    """
    Collect IRIs of classes annotated `use_in_alignment=false` (locality-module
    auxiliary classes added by Bio-ML 2023+). Predictions involving these are
    excluded from global matching evaluation.
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
    Inject runtime entries into leonmap.config.COLLECTIONS and MAPPINGS.
    id_prefixes=[] because Bio-ML's pruned OWLs already contain just one
    ontology's classes; no filter needed and it dodges the canonicalize-id
    namespace mismatch (e.g. NCIT classes have no namespace prefix).
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
        "threshold": THRESHOLD,
        "top_k": 1,
        "reverse": False,
    }
    return src_col, tgt_col, mapping_key


def _score_candidates(src_id: str, candidate_ids: List[str], src_db, tgt_db) -> List[Tuple[str, float, str]]:
    """
    Score a fixed candidate list against a source concept. No FAISS retrieval:
    reconstruct vectors directly from each side's index, compute cosine, hand
    the pool to rank_pool with threshold=0 so all candidates stay ranked.
    Boost still applies (unambiguous label match in pool -> 1.0).
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


# Format conversion + evaluation

def _mapper_to_match_result(mapper_tsv: Path, src_db, tgt_db, ignored: set, out_path: Path) -> int:
    """
    Convert mapper.py's TSV (canonical-id columns) into OAEI match.result.tsv
    (IRI columns). Rows referencing locality-module classes are dropped.
    """
    n = 0
    with open(mapper_tsv, "r", encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8", newline="") as g:
        r = csv.DictReader(f, delimiter="\t")
        w = csv.writer(g, delimiter="\t")
        w.writerow(["SrcEntity", "TgtEntity", "Score"])
        for row in r:
            src_id = row.get("src_id", "").strip()
            tgt_id = row.get("tgt_id", "").strip()
            score = row.get("score", "").strip()
            if not src_id or not tgt_id:
                continue
            src_iri = (src_db.get_payload_by_id(src_id) or {}).get("iri", "")
            tgt_iri = (tgt_db.get_payload_by_id(tgt_id) or {}).get("iri", "")
            if not src_iri or not tgt_iri:
                continue
            if src_iri in ignored or tgt_iri in ignored:
                continue
            w.writerow([src_iri, tgt_iri, score])
            n += 1
    return n


def _run_local_ranking(test_cands_path: Path, src_db, tgt_db, out_path: Path) -> Tuple[int, int]:
    """
    For each row in test.cands.tsv (SrcEntity, TgtEntity, TgtCandidates list of
    IRIs), score the candidates and write rank.result.tsv with the same
    columns but TgtCandidates as a list of (iri, score) tuples.

    Returns (rows_written, rows_with_src_not_in_db).
    """
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

            # Preserve input order; eval will sort.
            scored: List[Tuple[str, float]] = [
                (iri, score_map.get(cid, 0.0))
                for iri, cid in zip(cand_iris, cand_ids)
            ]

            w.writerow([src_iri, tgt_iri, repr(scored)])
            n_written += 1
    return n_written, n_src_missing


def _eval_global(match_path: Path, refs_full_path: Path, ignored: set) -> Dict:
    preds: set = set()
    refs: set = set()
    with open(match_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            preds.add((row["SrcEntity"], row["TgtEntity"]))
    with open(refs_full_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row.get("SrcEntity", "")
            t = row.get("TgtEntity", "")
            if s in ignored or t in ignored:
                continue
            refs.add((s, t))
    tp = len(preds & refs)
    p = tp / len(preds) if preds else 0.0
    rec = tp / len(refs) if refs else 0.0
    f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
    return {"P": p, "R": rec, "F1": f1, "n_preds": len(preds), "n_refs": len(refs), "tp": tp}


def _eval_ranking(rank_path: Path, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict:
    mrr_sum = 0.0
    hits = {k: 0 for k in ks}
    n = 0
    with open(rank_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            gold = row["TgtEntity"].strip()
            scored = list(ast.literal_eval(row["TgtCandidates"]))  # list of (iri, score)
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


# Per-task driver

def run_task(task: str, year_dir: Path, out_dir: Path) -> Optional[Dict]:
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
    test_cands = task_dir / "refs_equiv" / "test.cands.tsv"
    for p in (refs_full, test_cands):
        if not p.exists():
            print(f"[SKIP] Missing reference file: {p}")
            return None

    print(f"\n=== Task: {task} ===")
    print(f"  src OWL: {src_owl}")
    print(f"  tgt OWL: {tgt_owl}")

    src_col, tgt_col, mapping_key = _register_task(task, src_owl, tgt_owl)

    # Build FAISS collections (skipped automatically if already present).
    _run(_LM["build_main"], "leonmap-build", ["--collections", src_col, tgt_col])

    # Global matching: produces mapper_results/<study>/run_<stamp>/<src>_to_<tgt>.tsv
    _run(_LM["mapper_main"], "leonmap-map", ["--study", mapping_key])

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

    # Load collections for ranking + IRI lookups.
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

    n_match = _mapper_to_match_result(mapper_tsv, src_db, tgt_db, ignored, match_path)
    print(f"  Global matching: {n_match} predictions written")

    n_rank, n_src_missing = _run_local_ranking(test_cands, src_db, tgt_db, rank_path)
    print(f"  Local ranking: {n_rank} test rows scored ({n_src_missing} src not in DB)")

    global_metrics = _eval_global(match_path, refs_full, ignored)
    ranking_metrics = _eval_ranking(rank_path)

    metrics = {"task": task, **global_metrics, **ranking_metrics, "src_missing": n_src_missing}
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
    args = ap.parse_args()

    root = Path(os.path.abspath(LEONMAP_ROOT))

    # Patch PROJECT_ROOT before importing anything else from leonmap.
    import leonmap.config as _cfg
    _cfg.PROJECT_ROOT = root

    from leonmap.config import BuildConfig, COLLECTIONS, MAPPINGS, resolve_path
    from leonmap.utils import load_collection, rank_pool, canonicalize_id
    from leonmap.build_vdb import main as build_main
    from leonmap.mapper import main as mapper_main

    _LM.update({
        "cfg_mod": _cfg,
        "BuildConfig": BuildConfig,
        "load_collection": load_collection,
        "rank_pool": rank_pool,
        "canonicalize_id": canonicalize_id,
        "build_main": build_main,
        "mapper_main": mapper_main,
    })

    # Pull the fine-tuned SapBERT if it's not already local.
    cfg = BuildConfig()
    model_dir = resolve_path(cfg.ft_model_path)
    if not model_dir.exists():
        print(f"Model not found locally, downloading from HF: {HF_MODEL_REPO}")
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(model_dir))
        print(f"Model -> {model_dir}")

    year_dir = root / BIOML_DATA_DIR
    if not year_dir.is_dir():
        raise SystemExit(f"Bio-ML data directory not found: {year_dir}")

    if args.task and args.task not in TASKS:
        raise SystemExit(f"Unknown task: {args.task}. One of: {TASKS}")
    tasks_to_run = [args.task] if args.task else list(TASKS)

    out_root = root / "oaei_results" / RECORD_ID
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict] = []
    for task in tasks_to_run:
        m = run_task(task, year_dir, out_root / task)
        if m:
            all_metrics.append(m)

    # Summary table
    if all_metrics:
        summary_path = out_root / "results_summary.tsv"
        cols = ["task", "P", "R", "F1", "MRR", "Hits@1", "Hits@5", "Hits@10",
                "n_preds", "n_refs", "tp", "n_test", "src_missing"]
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for m in all_metrics:
                w.writerow(m)
        print(f"\nSummary -> {summary_path}\n")

        # stdout table
        hdr = f"{'task':<24} {'P':>6} {'R':>6} {'F1':>6} {'MRR':>6} {'H@1':>6} {'H@5':>6} {'H@10':>6}"
        print(hdr)
        print("-" * len(hdr))
        for m in all_metrics:
            print(f"{m['task']:<24} "
                  f"{m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} "
                  f"{m['MRR']:>6.3f} {m['Hits@1']:>6.3f} {m['Hits@5']:>6.3f} {m['Hits@10']:>6.3f}")

    print("\nDone.")
