#!/usr/bin/env python3
"""
Conservative post-hoc policy + path-divergence diagnostic for LeonMap OAEI Bio-ML.

What this script does
---------------------
1) Runs the normal LeonMap mapper at the low floor threshold (or reuses the latest run).
2) Reads the raw production mapper TSV and converts canonical IDs -> IRIs.
3) Evaluates three post-hoc policies on the exact same raw production output:
   - raw: no pruning, threshold only
   - mutual_best: one best source per target across raw output
   - safe_mutual_best: preserve score==1.0 hits unchanged, apply mutual-best only to non-exact hits
4) Tunes threshold on train.tsv, reports test/full metrics, and writes the best policy output.
5) Optionally diagnoses path divergence between production output and the separate top-K rerank path.

Place this file next to leonmap_oaei_patched.py and run for one task first:
    python oaei_policy_debug.py --task omim-ordo
    python oaei_policy_debug.py --task omim-ordo --rerun-mapper
    python oaei_policy_debug.py --task omim-ordo --diagnose-topk --probe-label "smith-kingsmore syndrome"

Outputs:
    oaei_results/13119437/policy_debug/<task>/match.result.policy.tsv
    oaei_results/13119437/policy_debug/<task>/metrics_policy.json
    oaei_results/13119437/policy_debug/<task>/policy_summary.tsv
    oaei_results/13119437/policy_debug/<task>/diagnostic_*   (if --diagnose-topk)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from huggingface_hub import snapshot_download

import leonmap_oaei_patched as base

ROOT = Path(base.LEONMAP_ROOT)
RECORD_ID = base.RECORD_ID
TASKS = list(base.TASKS)
DEFAULT_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def bootstrap_leonmap() -> None:
    import leonmap.config as _cfg

    _cfg.PROJECT_ROOT = ROOT

    from leonmap.config import BuildConfig, resolve_path
    from leonmap.utils import load_collection, rank_pool, canonicalize_id
    from leonmap.build_vdb import main as build_main
    from leonmap.mapper import main as mapper_main
    import leonmap.build_vdb as _build_vdb

    orig_buildcfg_init = BuildConfig.__init__

    def patched_buildcfg_init(self, *a, **kw):
        kw.setdefault("monitor_samples", 0)
        orig_buildcfg_init(self, *a, **kw)

    BuildConfig.__init__ = patched_buildcfg_init

    orig_load_owl_concepts = _build_vdb.load_owl_concepts

    def patched_load_owl_concepts(owl_path, id_prefixes=None):
        concepts = orig_load_owl_concepts(owl_path, id_prefixes=id_prefixes)

        if base.STRIP_SNOMED_SUFFIXES and "snomed" in Path(owl_path).name.lower():
            for c in concepts:
                c["label"] = base._strip_paren_suffix(c.get("label", ""))
                syns = c.get("synonyms", []) or []
                c["synonyms"] = [base._strip_paren_suffix(s) for s in syns]

        if base.STRIP_OMIM_TYPE_ARTIFACT and "omim" in Path(owl_path).name.lower():
            for c in concepts:
                c["label"] = base._restore_omim_type(c.get("label", ""))
                syns = c.get("synonyms", []) or []
                c["synonyms"] = [base._restore_omim_type(s) for s in syns]

        return concepts

    _build_vdb.load_owl_concepts = patched_load_owl_concepts

    base._LM.update(
        {
            "cfg_mod": _cfg,
            "BuildConfig": BuildConfig,
            "load_collection": load_collection,
            "rank_pool": rank_pool,
            "canonicalize_id": canonicalize_id,
            "build_main": build_main,
            "mapper_main": mapper_main,
        }
    )

    cfg = BuildConfig()
    model_dir = resolve_path(cfg.ft_model_path)
    if not model_dir.exists():
        print(f"[BOOT] Model not found locally, downloading {base.HF_MODEL_REPO}")
        snapshot_download(repo_id=base.HF_MODEL_REPO, local_dir=str(model_dir))
        print(f"[BOOT] Model -> {model_dir}")


def _run(entry_main, cli_name: str, argv: List[str]) -> None:
    old = sys.argv
    sys.argv = [cli_name] + argv
    try:
        entry_main()
    finally:
        sys.argv = old


def _load_pairs(path: Path) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s = row["SrcEntity"].strip()
            t = row["TgtEntity"].strip()
            if s and t:
                out.add((s, t))
    return out


def _eval(
    preds: Iterable[Tuple[str, str]],
    gold: Set[Tuple[str, str]],
    restrict: Optional[Set[str]] = None,
) -> Dict[str, float]:
    pred_set = {(s, t) for s, t in preds if restrict is None or s in restrict}
    tp = len(pred_set & gold)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"P": p, "R": r, "F1": f1, "tp": tp, "n_preds": len(pred_set), "n_refs": len(gold)}


def _score_is_exact(score: float, exact_score: float, eps: float = 1e-12) -> bool:
    return score >= exact_score - eps


def _split_exact(
    rows: List[Tuple[str, str, float]],
    exact_score: float,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    exact, other = [], []
    for row in rows:
        if _score_is_exact(row[2], exact_score):
            exact.append(row)
        else:
            other.append(row)
    return exact, other


def _mutual_best(
    rows: List[Tuple[str, str, float]],
) -> List[Tuple[str, str, float]]:
    tgt_to_best: Dict[str, Tuple[str, float]] = {}
    for s, t, sc in rows:
        cur = tgt_to_best.get(t)
        if cur is None or sc > cur[1]:
            tgt_to_best[t] = (s, sc)
    return [(s, t, sc) for s, t, sc in rows if tgt_to_best[t][0] == s]


def _safe_mutual_best(
    rows: List[Tuple[str, str, float]],
    exact_score: float,
) -> List[Tuple[str, str, float]]:
    exact_rows, other_rows = _split_exact(rows, exact_score=exact_score)
    exact_targets = {t for _, t, _ in exact_rows}
    other_rows = [r for r in other_rows if r[1] not in exact_targets]
    kept_other = _mutual_best(other_rows)
    return exact_rows + kept_other


def _policy_rows(
    rows: List[Tuple[str, str, float]],
    policy: str,
    exact_score: float,
) -> List[Tuple[str, str, float]]:
    if policy == "raw":
        return list(rows)
    if policy == "mutual_best":
        return _mutual_best(rows)
    if policy == "safe_mutual_best":
        return _safe_mutual_best(rows, exact_score=exact_score)
    raise ValueError(f"Unknown policy: {policy}")


def _threshold_rows(
    rows: List[Tuple[str, str, float]],
    threshold: float,
) -> List[Tuple[str, str, float]]:
    return [(s, t, sc) for s, t, sc in rows if sc >= threshold]


def _sweep_policy(
    rows: List[Tuple[str, str, float]],
    train_pairs: Set[Tuple[str, str]],
    test_pairs: Set[Tuple[str, str]],
    full_pairs: Set[Tuple[str, str]],
    grid: Sequence[float],
) -> Dict:
    train_src = {s for s, _ in train_pairs}
    test_src = {s for s, _ in test_pairs}

    sweep_log: List[Dict] = []
    best: Optional[Dict] = None

    for thr in grid:
        kept = _threshold_rows(rows, thr)
        pred_pairs = {(s, t) for s, t, _ in kept}
        train_metrics = _eval(pred_pairs, train_pairs, restrict=train_src)
        test_metrics = _eval(pred_pairs, test_pairs, restrict=test_src)
        full_metrics = _eval(pred_pairs, full_pairs, restrict=None)

        rec = {
            "threshold": thr,
            "train": train_metrics,
            "test": test_metrics,
            "full": full_metrics,
            "n_rows_after_threshold": len(kept),
        }
        sweep_log.append(rec)

        key = (train_metrics["F1"], train_metrics["P"], train_metrics["R"], -thr)
        if best is None or key > (
            best["train"]["F1"],
            best["train"]["P"],
            best["train"]["R"],
            -best["threshold"],
        ):
            best = rec

    assert best is not None
    return {"best": best, "sweep": sweep_log}


def _indegree_stats(rows: List[Tuple[str, str, float]]) -> Dict[str, float]:
    indeg = Counter(t for _, t, _ in rows)
    if not indeg:
        return {
            "unique_targets": 0,
            "max_indegree": 0,
            "mean_indegree": 0.0,
            "median_indegree": 0.0,
            "p90_indegree": 0.0,
            "targets_ge_2": 0,
            "targets_ge_5": 0,
            "targets_ge_10": 0,
            "targets_ge_20": 0,
            "targets_ge_50": 0,
        }

    vals = sorted(indeg.values())
    p90_idx = min(len(vals) - 1, math.ceil(0.9 * len(vals)) - 1)
    return {
        "unique_targets": len(vals),
        "max_indegree": max(vals),
        "mean_indegree": sum(vals) / len(vals),
        "median_indegree": statistics.median(vals),
        "p90_indegree": vals[p90_idx],
        "targets_ge_2": sum(v >= 2 for v in vals),
        "targets_ge_5": sum(v >= 5 for v in vals),
        "targets_ge_10": sum(v >= 10 for v in vals),
        "targets_ge_20": sum(v >= 20 for v in vals),
        "targets_ge_50": sum(v >= 50 for v in vals),
    }


def _write_match(rows: List[Tuple[str, str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["SrcEntity", "TgtEntity", "Score"])
        for s, t, sc in rows:
            w.writerow([s, t, f"{sc:.6f}"])


def _write_policy_summary(results: List[Dict], out_path: Path) -> None:
    cols = [
        "policy",
        "best_threshold",
        "train_F1",
        "train_P",
        "train_R",
        "test_F1",
        "test_P",
        "test_R",
        "full_F1",
        "full_P",
        "full_R",
        "rows_in",
        "rows_after_policy",
        "rows_after_threshold",
        "n_exact_rows_in",
        "n_exact_rows_after_policy",
        "unique_targets",
        "max_indegree",
        "p90_indegree",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "policy": r["policy"],
                    "best_threshold": r["best_threshold"],
                    "train_F1": r["best"]["train"]["F1"],
                    "train_P": r["best"]["train"]["P"],
                    "train_R": r["best"]["train"]["R"],
                    "test_F1": r["best"]["test"]["F1"],
                    "test_P": r["best"]["test"]["P"],
                    "test_R": r["best"]["test"]["R"],
                    "full_F1": r["best"]["full"]["F1"],
                    "full_P": r["best"]["full"]["P"],
                    "full_R": r["best"]["full"]["R"],
                    "rows_in": r["rows_in"],
                    "rows_after_policy": r["rows_after_policy"],
                    "rows_after_threshold": r["best"]["n_rows_after_threshold"],
                    "n_exact_rows_in": r["n_exact_rows_in"],
                    "n_exact_rows_after_policy": r["n_exact_rows_after_policy"],
                    "unique_targets": r["indegree_after_policy"]["unique_targets"],
                    "max_indegree": r["indegree_after_policy"]["max_indegree"],
                    "p90_indegree": r["indegree_after_policy"]["p90_indegree"],
                }
            )


def _label_to_iris(db) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for cid in db.id2pos:
        payload = db.get_payload_by_id(cid) or {}
        label = (payload.get("label") or "").strip().lower()
        iri = payload.get("iri") or ""
        if label and iri:
            out[label].append(iri)
    return out


def _ensure_task_run(
    task: str,
    rebuild: bool,
    rerun_mapper: bool,
):
    task_dir = ROOT / base.BIOML_DATA_DIR / task
    if not task_dir.is_dir():
        raise SystemExit(f"Task data not found: {task_dir}")

    src_owl_name, tgt_owl_name = base._owl_files(task)
    src_owl = task_dir / src_owl_name
    tgt_owl = task_dir / tgt_owl_name
    if not src_owl.exists() or not tgt_owl.exists():
        raise SystemExit(f"OWL files missing for {task}: {src_owl.name}, {tgt_owl.name}")

    refs_dir = task_dir / "refs_equiv"
    refs_full = refs_dir / "full.tsv"
    refs_train = refs_dir / "train.tsv"
    refs_test = refs_dir / "test.tsv"
    for p in (refs_full, refs_train, refs_test):
        if not p.exists():
            raise SystemExit(f"Missing reference file: {p}")

    src_col, tgt_col, mapping_key = base._register_task(task, src_owl, tgt_owl)

    build_argv = ["--collections", src_col, tgt_col]
    if rebuild:
        build_argv.append("--rebuild")
    print(f"[TASK {task}] build_vdb {' '.join(build_argv)}")
    _run(base._LM["build_main"], "leonmap-build", build_argv)

    if rerun_mapper:
        print(f"[TASK {task}] mapper --study {mapping_key} --threshold {base.MAPPER_FLOOR_THRESHOLD}")
        _run(
            base._LM["mapper_main"],
            "leonmap-map",
            ["--study", mapping_key, "--threshold", str(base.MAPPER_FLOOR_THRESHOLD)],
        )

    project_root: Path = base._LM["cfg_mod"].PROJECT_ROOT
    mapper_runs_dir = project_root / "mapper_results" / mapping_key
    run_dirs = sorted(mapper_runs_dir.glob("run_*"), key=lambda p: p.name)
    if not run_dirs:
        print(f"[TASK {task}] no mapper runs found, running mapper now")
        _run(
            base._LM["mapper_main"],
            "leonmap-map",
            ["--study", mapping_key, "--threshold", str(base.MAPPER_FLOOR_THRESHOLD)],
        )
        run_dirs = sorted(mapper_runs_dir.glob("run_*"), key=lambda p: p.name)

    if not run_dirs:
        raise SystemExit(f"No mapper output found under {mapper_runs_dir}")

    latest = run_dirs[-1]
    mapper_tsv = latest / f"{src_col}_to_{tgt_col}.tsv"
    if not mapper_tsv.exists():
        raise SystemExit(f"Mapper TSV missing: {mapper_tsv}")

    BuildConfig = base._LM["BuildConfig"]
    load_collection = base._LM["load_collection"]
    cfg = BuildConfig()
    src_db = load_collection(cfg, src_col)
    tgt_db = load_collection(cfg, tgt_col)

    ignored = base._get_ignored_iris(src_owl) | base._get_ignored_iris(tgt_owl)
    raw_rows = base._load_mapper_predictions(mapper_tsv, src_db, tgt_db, ignored)

    return {
        "task_dir": task_dir,
        "refs_full": refs_full,
        "refs_train": refs_train,
        "refs_test": refs_test,
        "src_db": src_db,
        "tgt_db": tgt_db,
        "mapper_tsv": mapper_tsv,
        "raw_rows": raw_rows,
    }


def _policy_search(
    task: str,
    raw_rows: List[Tuple[str, str, float]],
    refs_full: Path,
    refs_train: Path,
    refs_test: Path,
    out_dir: Path,
    exact_score: float,
    grid: Sequence[float],
) -> Dict:
    full_pairs = _load_pairs(refs_full)
    train_pairs = _load_pairs(refs_train)
    test_pairs = _load_pairs(refs_test)

    results: List[Dict] = []
    for policy in ("raw", "mutual_best", "safe_mutual_best"):
        policy_rows = _policy_rows(raw_rows, policy=policy, exact_score=exact_score)
        search = _sweep_policy(
            policy_rows,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            full_pairs=full_pairs,
            grid=grid,
        )
        best = search["best"]

        exact_in, _ = _split_exact(raw_rows, exact_score=exact_score)
        exact_after, _ = _split_exact(policy_rows, exact_score=exact_score)

        result = {
            "policy": policy,
            "rows_in": len(raw_rows),
            "rows_after_policy": len(policy_rows),
            "n_exact_rows_in": len(exact_in),
            "n_exact_rows_after_policy": len(exact_after),
            "best_threshold": best["threshold"],
            "best": best,
            "sweep": search["sweep"],
            "indegree_before": _indegree_stats(raw_rows),
            "indegree_after_policy": _indegree_stats(policy_rows),
            "policy_rows": policy_rows,
        }
        results.append(result)

    best_result = max(
        results,
        key=lambda r: (
            r["best"]["train"]["F1"],
            r["best"]["train"]["P"],
            r["best"]["train"]["R"],
            -r["best_threshold"],
            r["best"]["test"]["F1"],
        ),
    )

    final_rows = _threshold_rows(best_result["policy_rows"], best_result["best_threshold"])
    _write_match(final_rows, out_dir / "match.result.policy.tsv")
    _write_policy_summary(results, out_dir / "policy_summary.tsv")

    payload = {
        "task": task,
        "exact_score_protection": exact_score,
        "grid": list(grid),
        "best_policy": best_result["policy"],
        "best_threshold": best_result["best_threshold"],
        "best_metrics": best_result["best"],
        "all_policies": [
            {
                "policy": r["policy"],
                "rows_in": r["rows_in"],
                "rows_after_policy": r["rows_after_policy"],
                "n_exact_rows_in": r["n_exact_rows_in"],
                "n_exact_rows_after_policy": r["n_exact_rows_after_policy"],
                "best_threshold": r["best_threshold"],
                "best": r["best"],
                "indegree_before": r["indegree_before"],
                "indegree_after_policy": r["indegree_after_policy"],
                "sweep": r["sweep"],
            }
            for r in results
        ],
        "final_n_rows": len(final_rows),
    }

    (out_dir / "metrics_policy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _diagnose_topk_divergence(
    raw_rows: List[Tuple[str, str, float]],
    src_db,
    tgt_db,
    out_dir: Path,
    top_k: int,
    exact_score: float,
    max_sources: int,
    probe_srcs: Optional[List[str]] = None,
) -> Dict:
    prod_by_src: Dict[str, Tuple[str, float]] = {s: (t, sc) for s, t, sc in raw_rows}
    exact_srcs = [s for s, _, sc in raw_rows if _score_is_exact(sc, exact_score=exact_score)]

    chosen: List[str] = []
    if probe_srcs:
        chosen.extend([s for s in probe_srcs if s in prod_by_src])

    remaining = [s for s in exact_srcs if s not in set(chosen)]
    if max_sources > 0:
        chosen.extend(sorted(remaining)[:max_sources])

    chosen = sorted(set(chosen))
    if not chosen:
        return {
            "diagnosed_sources": 0,
            "same_target": 0,
            "different_target": 0,
            "missing_topk": 0,
            "exact_prod_sources": len(exact_srcs),
        }

    tgt_id_to_iri = base._tgt_id_to_iri_map(tgt_db)
    cache = base._build_topk_boosted_cache(set(chosen), src_db, tgt_db, top_k)

    diag_rows: List[Dict] = []
    same_target = 0
    different_target = 0
    missing_topk = 0

    for src_iri in chosen:
        prod_tgt, prod_sc = prod_by_src[src_iri]
        src_label, pool = cache.get(src_iri, ("", []))

        if not pool:
            top_tgt = ""
            top_sc = None
            missing_topk += 1
        else:
            top_tgt = tgt_id_to_iri.get(pool[0][0], "")
            top_sc = float(pool[0][1])

        if top_tgt == "":
            relation = "missing_topk"
        elif top_tgt == prod_tgt:
            relation = "same_target"
            same_target += 1
        else:
            relation = "different_target"
            different_target += 1

        diag_rows.append(
            {
                "SrcEntity": src_iri,
                "SrcLabel": src_label,
                "ProdTgt": prod_tgt,
                "ProdScore": prod_sc,
                "TopKTgt": top_tgt,
                "TopKScore": "" if top_sc is None else f"{top_sc:.6f}",
                "Relation": relation,
            }
        )

    with open(out_dir / "diagnostic_topk_vs_production.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["SrcEntity", "SrcLabel", "ProdTgt", "ProdScore", "TopKTgt", "TopKScore", "Relation"],
            delimiter="\t",
        )
        w.writeheader()
        for row in diag_rows:
            w.writerow(row)

    if probe_srcs:
        lines: List[str] = []
        for src_iri in probe_srcs:
            src_label, pool = cache.get(src_iri, ("", []))
            prod = prod_by_src.get(src_iri)
            lines.append(f"SRC\t{src_iri}")
            lines.append(f"LABEL\t{src_label}")
            if prod is None:
                lines.append("PRODUCTION\t<missing from raw_rows>")
                lines.append("")
                continue
            lines.append(f"PRODUCTION\t{prod[0]}\t{prod[1]:.6f}")
            for i, (tgt_id, sc) in enumerate(pool[:5], start=1):
                payload = tgt_db.get_payload_by_id(tgt_id) or {}
                tgt_iri = tgt_id_to_iri.get(tgt_id, "")
                tgt_label = payload.get("label", "")
                lines.append(f"TOP{i}\t{tgt_iri}\t{sc:.6f}\t{tgt_label}")
            lines.append("")

        (out_dir / "diagnostic_probe_top5.txt").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "diagnosed_sources": len(chosen),
        "exact_prod_sources": len(exact_srcs),
        "same_target": same_target,
        "different_target": different_target,
        "missing_topk": missing_topk,
        "top_k": top_k,
    }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--grid", nargs="+", type=float, default=DEFAULT_GRID)
    ap.add_argument("--exact-score", type=float, default=1.0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--rerun-mapper", action="store_true")
    ap.add_argument("--diagnose-topk", action="store_true")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-diagnose", type=int, default=500)
    ap.add_argument("--probe-src", action="append", default=[])
    ap.add_argument("--probe-label", action="append", default=[])
    args = ap.parse_args()

    bootstrap_leonmap()

    task = args.task
    out_dir = ROOT / "oaei_results" / RECORD_ID / "policy_debug" / task
    out_dir.mkdir(parents=True, exist_ok=True)

    run = _ensure_task_run(task=task, rebuild=args.rebuild, rerun_mapper=args.rerun_mapper)
    raw_rows = run["raw_rows"]
    refs_full = run["refs_full"]
    refs_train = run["refs_train"]
    refs_test = run["refs_test"]
    src_db = run["src_db"]
    tgt_db = run["tgt_db"]

    print(f"[TASK {task}] raw rows: {len(raw_rows):,}")
    exact_rows, _ = _split_exact(raw_rows, exact_score=args.exact_score)
    print(f"[TASK {task}] exact-score rows (score >= {args.exact_score}): {len(exact_rows):,}")

    policy_payload = _policy_search(
        task=task,
        raw_rows=raw_rows,
        refs_full=refs_full,
        refs_train=refs_train,
        refs_test=refs_test,
        out_dir=out_dir,
        exact_score=args.exact_score,
        grid=args.grid,
    )

    print()
    print("=" * 88)
    print(f"BEST POLICY FOR {task}")
    print("=" * 88)
    print(f"Policy:    {policy_payload['best_policy']}")
    print(f"Threshold: {policy_payload['best_threshold']:.2f}")
    print(
        f"Train F1:  {policy_payload['best_metrics']['train']['F1']:.4f} | "
        f"Test F1: {policy_payload['best_metrics']['test']['F1']:.4f} | "
        f"Full F1: {policy_payload['best_metrics']['full']['F1']:.4f}"
    )

    label_map = _label_to_iris(src_db)
    probe_srcs = list(args.probe_src)
    for lbl in args.probe_label:
        probe_srcs.extend(label_map.get(lbl.strip().lower(), []))

    if args.diagnose_topk:
        diag = _diagnose_topk_divergence(
            raw_rows=raw_rows,
            src_db=src_db,
            tgt_db=tgt_db,
            out_dir=out_dir,
            top_k=args.top_k,
            exact_score=args.exact_score,
            max_sources=args.max_diagnose,
            probe_srcs=probe_srcs or None,
        )
        print()
        print("=" * 88)
        print("TOP-K PATH DIVERGENCE")
        print("=" * 88)
        print(
            f"Diagnosed: {diag['diagnosed_sources']:,} | "
            f"Same target: {diag['same_target']:,} | "
            f"Different target: {diag['different_target']:,} | "
            f"Missing top-k: {diag['missing_topk']:,}"
        )

    print()
    print(f"Wrote: {out_dir / 'match.result.policy.tsv'}")
    print(f"Wrote: {out_dir / 'metrics_policy.json'}")
    print(f"Wrote: {out_dir / 'policy_summary.tsv'}")
    if args.diagnose_topk:
        print(f"Wrote: {out_dir / 'diagnostic_summary.json'}")
        print(f"Wrote: {out_dir / 'diagnostic_topk_vs_production.tsv'}")
        if probe_srcs:
            print(f"Wrote: {out_dir / 'diagnostic_probe_top5.txt'}")


if __name__ == "__main__":
    main()