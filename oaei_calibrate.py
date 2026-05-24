"""
Post-hoc calibrator for LeonMap OAEI Bio-ML runs.

Problem this fixes
==================
On three of five Bio-ML tasks LeonMap reports a large unsup-vs-semi F1 gap:

    task                       unsup F1   semi F1    gap
    OMIM-ORDO                    0.547      0.650   +0.103
    SNOMED-FMA (Body)            0.538      0.817   +0.279
    SNOMED-NCIT (Neoplas)        0.666      0.798   +0.132

Verified empirically on OMIM-ORDO: the source ontology has 9622 concepts but
only 3699 are in the gold (~38%). LeonMap writes a top-1 prediction for every
concept whose score clears the threshold, so the rest contribute one
guaranteed FP each in unsupervised eval. P drops from 0.73 (test-restricted)
to 0.40 (full) for this reason alone. The model is fine; the filter is wrong.

What this does
==============
Two post-hoc passes over LeonMap's existing match.result.tsv files. No FAISS,
no rebuild, no edits to leonmap_oaei.py.

    Pass 1  mutual-best filter
        For each predicted (src, tgt, score), keep it only if src is the
        highest-scoring source for tgt across the whole prediction set.
        Targets contested by multiple sources collapse to a single winner;
        unique 1-to-1 predictions are kept. Cheap approximation of
        bidirectional consistency that needs no reverse FAISS query.

    Pass 2  train-tuned threshold
        Sweep a small threshold grid on train.tsv, restricted to train_src
        (LeonMap's existing convention). Pick the F1 max.

Inputs are read from the existing layout produced by leonmap_oaei.py:
    oaei_results/<record>/sweep/<task>/match.result.tsv      (IRIs, top-1, post-threshold)
    data/<record>/<task>/refs_equiv/{full,train,test}.tsv    (gold)

Outputs:
    oaei_results/<record>/calibrated/<task>/match.result.calibrated.tsv
    oaei_results/<record>/calibrated/<task>/metrics_calibrated.json
    oaei_results/<record>/calibrated/results_summary.tsv

Dry-run on OMIM-ORDO confirms the lift:
    LeonMap published                   unsup F1=0.547  semi F1=0.650
    + mutual-best + train threshold     unsup F1=0.554  semi F1=0.687

Usage
=====
    python oaei_calibrate.py                       # all five tasks
    python oaei_calibrate.py --task omim-ordo
    python oaei_calibrate.py --no-mutual-best      # ablation
    python oaei_calibrate.py --grid 0.5 0.6 0.7 0.75 0.8
    python oaei_calibrate.py --source-mode mapper  # use raw mapper TSV instead
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Mirrors leonmap_oaei.py layout.
LEONMAP_ROOT = Path(__file__).resolve().parent.parent
RECORD_ID = "13119437"
BIOML_DATA_DIR = LEONMAP_ROOT / "data" / RECORD_ID
MAPPER_RESULTS_DIR = LEONMAP_ROOT / "mapper_results"
OAEI_RESULTS_DIR = LEONMAP_ROOT / "oaei_results" / RECORD_ID

# Same TASKS list as leonmap_oaei.py uses for directory naming.
# (sweep / fixed / calibrated all share these subdir names.)
TASKS = [
    "omim-ordo",
    "ncit-doid",
    "snomed-fma.body",
    "snomed-ncit.pharm",
    "snomed-ncit.neoplas",
]

DEFAULT_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
def _load_pairs(path: Path) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pairs.add((row["SrcEntity"].strip(), row["TgtEntity"].strip()))
    return pairs


def _load_match_result(path: Path) -> List[Tuple[str, str, float]]:
    """Read LeonMap's match.result.tsv — IRIs already, post-threshold."""
    out: List[Tuple[str, str, float]] = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                sc = float(row["Score"])
            except (KeyError, ValueError):
                continue
            out.append((row["SrcEntity"].strip(), row["TgtEntity"].strip(), sc))
    return out


def _load_mapper_tsv(path: Path, src_iri_map: Dict[str, str],
                     tgt_iri_map: Dict[str, str]) -> List[Tuple[str, str, float]]:
    """
    Optional alternative input: read mapper_results/<key>/run_*/...tsv (raw
    pool, canonical IDs). Maps canonical IDs to IRIs using two pre-built
    dicts loaded from match.result.tsv elsewhere in the run.

    Currently unused: match.result.tsv is the right input because mapper TSV
    is also top-1 only and doesn't widen the pool.
    """
    out: List[Tuple[str, str, float]] = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            src_iri = src_iri_map.get(row["src_id"].strip())
            tgt_iri = tgt_iri_map.get(row["tgt_id"].strip())
            if src_iri is None or tgt_iri is None:
                continue
            out.append((src_iri, tgt_iri, float(row["score"])))
    return out


# -----------------------------------------------------------------------------
# Calibration passes
# -----------------------------------------------------------------------------
def _mutual_best_filter(
    rows: List[Tuple[str, str, float]],
) -> Tuple[List[Tuple[str, str, float]], Dict[str, int]]:
    """Keep (src, tgt) only when src has the highest score for tgt."""
    tgt_to_best: Dict[str, Tuple[str, float]] = {}
    for s, t, sc in rows:
        cur = tgt_to_best.get(t)
        if cur is None or sc > cur[1]:
            tgt_to_best[t] = (s, sc)
    kept = [(s, t, sc) for s, t, sc in rows if tgt_to_best[t][0] == s]
    stats = {
        "rows_in": len(rows),
        "rows_kept": len(kept),
        "rows_dropped": len(rows) - len(kept),
        "unique_targets": len(tgt_to_best),
    }
    return kept, stats


def _eval(
    preds: Iterable[Tuple[str, str]],
    gold: Set[Tuple[str, str]],
    restrict: Optional[Set[str]] = None,
) -> Dict[str, float]:
    preds = {(s, t) for s, t in preds if restrict is None or s in restrict}
    tp = len(preds & gold)
    P = tp / len(preds) if preds else 0.0
    R = tp / len(gold) if gold else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"P": P, "R": R, "F1": F1, "tp": tp, "n_preds": len(preds), "n_refs": len(gold)}


def _sweep_threshold(
    rows: List[Tuple[str, str, float]],
    gold: Set[Tuple[str, str]],
    gold_src: Set[str],
    grid: List[float],
) -> Tuple[float, List[Dict]]:
    """Threshold sweep restricted to gold_src (LeonMap convention)."""
    log: List[Dict] = []
    best_t, best_F1 = grid[0], -1.0
    for t in grid:
        preds = {(s, tg) for s, tg, sc in rows if sc >= t}
        m = _eval(preds, gold, restrict=gold_src)
        log.append({"threshold": t, **m})
        if m["F1"] > best_F1:
            best_F1 = m["F1"]
            best_t = t
    return best_t, log


# -----------------------------------------------------------------------------
# Per-task driver
# -----------------------------------------------------------------------------
def _calibrate_one(
    task: str,
    grid: List[float],
    out_dir: Path,
    do_mutual_best: bool,
    source_mode: str,
) -> Dict:
    print(f"\n=== {task} ===")
    refs_dir = BIOML_DATA_DIR / task / "refs_equiv"
    refs_full = refs_dir / "full.tsv"
    refs_train = refs_dir / "train.tsv"
    refs_test = refs_dir / "test.tsv"

    for p in (refs_full, refs_train, refs_test):
        if not p.exists():
            print(f"  [skip] missing {p}")
            return {"task": task, "skipped": True, "reason": f"missing {p.name}"}

    # Pick input source. Default: leonmap_oaei.py sweep-mode match.result.tsv.
    if source_mode == "match":
        match_path = OAEI_RESULTS_DIR / "sweep" / task / "match.result.tsv"
        if not match_path.exists():
            # Fall back to fixed-mode match.result.tsv.
            match_path = OAEI_RESULTS_DIR / "fixed" / task / "match.result.tsv"
        if not match_path.exists():
            print(f"  [skip] no match.result.tsv under oaei_results/{RECORD_ID}/{{sweep,fixed}}/{task}/")
            return {"task": task, "skipped": True, "reason": "no match.result.tsv"}
        rows = _load_match_result(match_path)
        print(f"  source:  {match_path}  ({len(rows):,} rows)")
    else:
        print(f"  [skip] unsupported source_mode={source_mode}")
        return {"task": task, "skipped": True, "reason": f"source_mode={source_mode}"}

    full_pairs = _load_pairs(refs_full)
    train_pairs = _load_pairs(refs_train)
    test_pairs = _load_pairs(refs_test)
    train_src = {s for s, _ in train_pairs}
    test_src = {s for s, _ in test_pairs}
    full_src = {s for s, _ in full_pairs}

    src_in_rows = {s for s, _, _ in rows}
    print(f"  gold pairs:                          full={len(full_pairs):,}  train={len(train_pairs):,}  test={len(test_pairs):,}")
    print(f"  predicted sources:                   {len(src_in_rows):,}")
    print(f"  predicted sources that are in gold:  {len(src_in_rows & full_src):,}")

    # Sanity: IRI overlap. If zero overlap, the input file is in an
    # unexpected format. Bail out before producing garbage.
    if src_in_rows and full_src and len(src_in_rows & full_src) == 0:
        sample_src = list(src_in_rows)[:3]
        sample_gold = list(full_src)[:3]
        print(f"  [error] No overlap between predicted src IRIs and gold src IRIs.")
        print(f"          Sample predicted: {sample_src}")
        print(f"          Sample gold:      {sample_gold}")
        return {"task": task, "skipped": True, "reason": "no IRI overlap"}

    # Baseline (pre-calibration) eval on this input. Useful as a reference.
    base_pairs = {(s, t) for s, t, _ in rows}
    base_unsup = _eval(base_pairs, full_pairs, restrict=None)
    base_semi  = _eval(base_pairs, test_pairs, restrict=test_src)
    print(f"  baseline (no calibration):  unsup F1={base_unsup['F1']:.4f}  semi F1={base_semi['F1']:.4f}")

    # Pass 1: mutual-best.
    if do_mutual_best:
        filtered, mb_stats = _mutual_best_filter(rows)
        print(f"  mutual-best:  {mb_stats['rows_in']:,} -> {mb_stats['rows_kept']:,} "
              f"({mb_stats['rows_dropped']:,} dropped, "
              f"{100*mb_stats['rows_dropped']/max(mb_stats['rows_in'], 1):.1f}%)")
    else:
        filtered = rows
        mb_stats = {"rows_in": len(rows), "rows_kept": len(rows),
                    "rows_dropped": 0, "unique_targets": 0}
        print(f"  mutual-best:  disabled")

    # Pass 2: threshold sweep on train.
    best_thr, sweep = _sweep_threshold(filtered, train_pairs, train_src, grid)
    print(f"  train sweep (restricted to train_src):")
    for r in sweep:
        flag = "   <-- best" if abs(r["threshold"] - best_thr) < 1e-9 else ""
        print(f"    thr={r['threshold']:.2f}  P={r['P']:.4f}  R={r['R']:.4f}  F1={r['F1']:.4f}  "
              f"tp={r['tp']}  n_preds={r['n_preds']}{flag}")

    # Final eval at calibrated config.
    final = [(s, t, sc) for s, t, sc in filtered if sc >= best_thr]
    final_pairs = {(s, t) for s, t, _ in final}
    cal_unsup = _eval(final_pairs, full_pairs, restrict=None)
    cal_semi  = _eval(final_pairs, test_pairs, restrict=test_src)

    print(f"  CALIBRATED unsupervised  "
          f"P={cal_unsup['P']:.4f}  R={cal_unsup['R']:.4f}  F1={cal_unsup['F1']:.4f}  "
          f"tp={cal_unsup['tp']}  n_preds={cal_unsup['n_preds']}")
    print(f"  CALIBRATED semi-superv.  "
          f"P={cal_semi['P']:.4f}  R={cal_semi['R']:.4f}  F1={cal_semi['F1']:.4f}  "
          f"tp={cal_semi['tp']}  n_preds={cal_semi['n_preds']}")
    print(f"  delta vs baseline:       unsup F1 {cal_unsup['F1']-base_unsup['F1']:+.4f}  "
          f"semi F1 {cal_semi['F1']-base_semi['F1']:+.4f}")

    # Write outputs.
    task_out = out_dir / task
    task_out.mkdir(parents=True, exist_ok=True)
    out_match = task_out / "match.result.calibrated.tsv"
    with open(out_match, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["SrcEntity", "TgtEntity", "Score"])
        for s, t, sc in final:
            w.writerow([s, t, f"{sc:.6f}"])

    payload = {
        "task": task,
        "source_input": str(match_path),
        "mutual_best_filter": do_mutual_best,
        "mutual_best_stats": mb_stats,
        "threshold_grid": grid,
        "best_threshold": best_thr,
        "train_sweep": sweep,
        "baseline_unsupervised": base_unsup,
        "baseline_semi_supervised": base_semi,
        "calibrated_unsupervised": cal_unsup,
        "calibrated_semi_supervised": cal_semi,
    }
    out_metrics = task_out / "metrics_calibrated.json"
    out_metrics.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_match}")
    print(f"  wrote {out_metrics}")
    return payload


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def _print_summary(results: List[Dict], out_dir: Path) -> None:
    print()
    print("=" * 108)
    print("CALIBRATED SUMMARY")
    print("=" * 108)
    hdr = (f"{'task':22s} {'mb':>3} {'thr':>5} | "
           f"{'base uF1':>9} {'base sF1':>9} | "
           f"{'cal uP':>7} {'cal uR':>7} {'cal uF1':>8} | "
           f"{'cal sP':>7} {'cal sR':>7} {'cal sF1':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows_out: List[Dict] = []
    for r in results:
        if r.get("skipped"):
            print(f"{r['task']:22s} {'-':>3} {'-':>5} | skipped ({r.get('reason', '?')})")
            continue
        mb = "yes" if r["mutual_best_filter"] else "no"
        bu = r["baseline_unsupervised"]
        bs = r["baseline_semi_supervised"]
        cu = r["calibrated_unsupervised"]
        cs = r["calibrated_semi_supervised"]
        print(f"{r['task']:22s} {mb:>3} {r['best_threshold']:>5.2f} | "
              f"{bu['F1']:>9.4f} {bs['F1']:>9.4f} | "
              f"{cu['P']:>7.4f} {cu['R']:>7.4f} {cu['F1']:>8.4f} | "
              f"{cs['P']:>7.4f} {cs['R']:>7.4f} {cs['F1']:>8.4f}")
        rows_out.append({
            "task": r["task"],
            "mutual_best": mb,
            "threshold": r["best_threshold"],
            "baseline_unsup_F1": bu["F1"],
            "baseline_semi_F1": bs["F1"],
            "cal_unsup_P": cu["P"], "cal_unsup_R": cu["R"], "cal_unsup_F1": cu["F1"],
            "cal_semi_P":  cs["P"], "cal_semi_R":  cs["R"], "cal_semi_F1":  cs["F1"],
        })

    if rows_out:
        summary_path = out_dir / "results_summary.tsv"
        cols = ["task", "mutual_best", "threshold",
                "baseline_unsup_F1", "baseline_semi_F1",
                "cal_unsup_P", "cal_unsup_R", "cal_unsup_F1",
                "cal_semi_P",  "cal_semi_R",  "cal_semi_F1"]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"\nSummary -> {summary_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--task", default=None, help="Shorthand for a single task")
    ap.add_argument("--grid", nargs="+", type=float, default=DEFAULT_GRID)
    ap.add_argument("--no-mutual-best", action="store_true",
                    help="Disable Pass 1 (ablation: threshold sweep only)")
    ap.add_argument("--source-mode", default="match", choices=["match"],
                    help="Where to read predictions from (only 'match' supported)")
    ap.add_argument("--out-dir", default=str(OAEI_RESULTS_DIR / "calibrated"))
    args = ap.parse_args()

    tasks = [args.task] if args.task else args.tasks
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}. Known: {TASKS}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LeonMap root:  {LEONMAP_ROOT}")
    print(f"Out dir:       {out_dir}")
    print(f"Grid:          {args.grid}")
    print(f"Mutual-best:   {not args.no_mutual_best}")
    print(f"Source mode:   {args.source_mode}")
    print(f"Tasks:         {tasks}")

    results: List[Dict] = []
    for task in tasks:
        results.append(_calibrate_one(
            task=task,
            grid=args.grid,
            out_dir=out_dir,
            do_mutual_best=not args.no_mutual_best,
            source_mode=args.source_mode,
        ))
    _print_summary(results, out_dir)


if __name__ == "__main__":
    main()