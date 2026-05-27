"""
Diagnostic: list rows the contamination filter removed from each refs file.

Run AFTER leonmap_oaei_patch.py has materialized at least one filtered variant
(`*_disjoint.tsv` / `*_nopair.tsv` / their `.cands.tsv` counterparts).

For each (task, refs file), emits one row per *removed* entry:
  src_iri, tgt_iri, src_label, tgt_label,
  reason ("exact_pair" | "label_disjoint"),
  matched_csv_subject, matched_csv_object  (one row from sapbert_all.csv that explains the removal)

Reason semantics:
  "exact_pair"     -> (src_label, tgt_label) appeared in sapbert_all.csv directly.
                       Such rows are ALSO removed by label_disjoint.
  "label_disjoint" -> only src_label or tgt_label appeared in CSV (the pair did not).
                       These rows are kept under exact_pair, removed under label_disjoint.

Output: {LEONMAP_ROOT}/oaei_results_diagnostics/{RECORD_ID}/{task}/{stem}_removed.tsv
plus a top-level summary.tsv.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

from leonmap_oaei_patch import (
    LEONMAP_ROOT, RECORD_ID, BIOML_DATA_DIR, SAPBERT_CSV_PATH, TASKS,
    _owl_files, _clean_label, _register_task, LM,
)

REF_FILES = ["full.tsv", "train.tsv", "test.tsv", "test.cands.tsv"]
DIAG_ROOT = LEONMAP_ROOT / "oaei_results_diagnostics" / RECORD_ID


def _bootstrap_leonmap() -> None:
    """Populate LM with leonmap bindings (mirrors leonmap_oaei_patch.py __main__)."""
    if hasattr(LM, "load_collection"):
        return
    import leonmap.config as _cfg_mod
    _cfg_mod.PROJECT_ROOT = LEONMAP_ROOT
    from leonmap.config import BuildConfig
    from leonmap.utils import load_collection, canonicalize_id
    LM.cfg = _cfg_mod
    LM.BuildConfig = BuildConfig
    LM.load_collection = load_collection
    LM.canonicalize_id = canonicalize_id


def _db_iri_to_label(task: str, src_owl: Path, tgt_owl: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load LeonMap FAISS DB labels (same source the patch's filter reads from)."""
    _bootstrap_leonmap()
    src_col, tgt_col, _ = _register_task(task, src_owl, tgt_owl)
    cfg = LM.BuildConfig()
    src_db = LM.load_collection(cfg, src_col)
    tgt_db = LM.load_collection(cfg, tgt_col)

    def collect(db) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for tid in db.id2pos:
            payload = db.get_payload_by_id(tid) or {}
            iri = payload.get("iri", "")
            if iri:
                out[iri] = payload.get("label", "")
        return out

    return collect(src_db), collect(tgt_db)


def _load_csv_index() -> Tuple[frozenset, frozenset,
                               Dict[str, Tuple[str, str]],
                               Dict[Tuple[str, str], Tuple[str, str]]]:
    """Build seen_labels, seen_pairs, label -> first raw CSV row, pair -> raw CSV row."""
    import pandas as pd
    df = pd.read_csv(LEONMAP_ROOT / SAPBERT_CSV_PATH)
    label_row: Dict[str, Tuple[str, str]] = {}
    pair_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    labels: Set[str] = set()
    for _, row in df.iterrows():
        if not (isinstance(row.get("subject_label"), str) and isinstance(row.get("object_label"), str)):
            continue
        s_raw, t_raw = str(row["subject_label"]), str(row["object_label"])
        s, t = _clean_label(s_raw), _clean_label(t_raw)
        if not (s and t):
            continue
        labels.add(s); labels.add(t)
        pair_map.setdefault((s, t), (s_raw, t_raw))
        pair_map.setdefault((t, s), (s_raw, t_raw))
        label_row.setdefault(s, (s_raw, t_raw))
        label_row.setdefault(t, (s_raw, t_raw))
    return frozenset(labels), frozenset(pair_map.keys()), label_row, pair_map


def _diagnose_refs_file(ref_path: Path, src_iri_to_label: Dict[str, str], tgt_iri_to_label: Dict[str, str],
                        seen_labels: frozenset, seen_pairs: frozenset,
                        label_row: Dict[str, Tuple[str, str]],
                        pair_map: Dict[Tuple[str, str], Tuple[str, str]]
                        ) -> Tuple[List[Dict], List[Dict]]:
    removed: List[Dict] = []
    kept: List[Dict] = []
    with open(ref_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            src_iri = row.get("SrcEntity", "").strip()
            tgt_iri = row.get("TgtEntity", "").strip()
            if not (src_iri and tgt_iri):
                continue
            src_label = src_iri_to_label.get(src_iri, "")
            tgt_label = tgt_iri_to_label.get(tgt_iri, "")
            s, t = _clean_label(src_label), _clean_label(tgt_label)
            if not (s in seen_labels or t in seen_labels):
                kept.append({"src_iri": src_iri, "tgt_iri": tgt_iri,
                             "src_label": src_label, "tgt_label": tgt_label})
                continue
            if (s, t) in seen_pairs:
                reason = "exact_pair"
                csv_subj, csv_obj = pair_map[(s, t)]
            else:
                reason = "label_disjoint"
                csv_subj, csv_obj = label_row.get(s) or label_row[t]
            removed.append({
                "src_iri": src_iri, "tgt_iri": tgt_iri,
                "src_label": src_label, "tgt_label": tgt_label,
                "reason": reason,
                "matched_csv_subject": csv_subj, "matched_csv_object": csv_obj,
            })
    return removed, kept


def _count_data_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


_REMOVED_COLS = ["src_iri", "tgt_iri", "src_label", "tgt_label",
                 "reason", "matched_csv_subject", "matched_csv_object"]
_KEPT_COLS = ["src_iri", "tgt_iri", "src_label", "tgt_label"]

_SUMMARY_COLS = ["task", "ref_file", "n_total", "n_kept", "n_removed",
                 "n_exact_pair", "n_label_disjoint_only"]


def _filtered_exists(task_dir: Path) -> bool:
    """Sanity check: at least one materialized filtered variant should exist."""
    refs = task_dir / "refs_equiv"
    if not refs.is_dir():
        return False
    return any(p.name.endswith(("_disjoint.tsv", "_nopair.tsv",
                                "_disjoint.cands.tsv", "_nopair.cands.tsv"))
               for p in refs.iterdir())


def main() -> None:
    seen_labels, seen_pairs, label_row, pair_map = _load_csv_index()
    print(f"[DIAG] {len(seen_labels)} labels, {len(seen_pairs)} pairs from {SAPBERT_CSV_PATH}")

    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []

    for task in TASKS:
        task_dir = LEONMAP_ROOT / BIOML_DATA_DIR / task
        if not task_dir.is_dir():
            print(f"[SKIP] {task}: data dir missing")
            continue
        if not _filtered_exists(task_dir):
            print(f"[SKIP] {task}: no filtered variants found (run leonmap_oaei_patch.py first)")
            continue

        src_owl_name, tgt_owl_name = _owl_files(task)
        src_owl = task_dir / src_owl_name
        tgt_owl = task_dir / tgt_owl_name
        if not (src_owl.exists() and tgt_owl.exists()):
            print(f"[SKIP] {task}: OWL missing")
            continue

        print(f"\n[DIAG] {task}")
        src_iri_to_label, tgt_iri_to_label = _db_iri_to_label(task, src_owl, tgt_owl)
        print(f"  loaded {len(src_iri_to_label)} src labels, {len(tgt_iri_to_label)} tgt labels (from FAISS DB)")

        out_dir = DIAG_ROOT / task
        out_dir.mkdir(parents=True, exist_ok=True)

        for ref_name in REF_FILES:
            ref_path = task_dir / "refs_equiv" / ref_name
            if not ref_path.exists():
                continue
            n_total = _count_data_rows(ref_path)
            removed, kept = _diagnose_refs_file(ref_path, src_iri_to_label, tgt_iri_to_label,
                                                seen_labels, seen_pairs, label_row, pair_map)
            stem = ref_name[:-len(".tsv")]
            with open(out_dir / f"{stem}_removed.tsv", "w", encoding="utf-8", newline="") as g:
                w = csv.DictWriter(g, fieldnames=_REMOVED_COLS, delimiter="\t")
                w.writeheader()
                for r in removed:
                    w.writerow(r)
            with open(out_dir / f"{stem}_kept.tsv", "w", encoding="utf-8", newline="") as g:
                w = csv.DictWriter(g, fieldnames=_KEPT_COLS, delimiter="\t")
                w.writeheader()
                for r in kept:
                    w.writerow(r)
            n_pair = sum(1 for r in removed if r["reason"] == "exact_pair")
            n_disjoint_only = len(removed) - n_pair
            summary_rows.append({
                "task": task, "ref_file": ref_name, "n_total": n_total,
                "n_kept": len(kept), "n_removed": len(removed),
                "n_exact_pair": n_pair, "n_label_disjoint_only": n_disjoint_only,
            })
            print(f"  {ref_name}: total={n_total} kept={len(kept)} removed={len(removed)} "
                  f"(exact_pair={n_pair}, disjoint_only={n_disjoint_only})")

    summary_path = DIAG_ROOT / "summary.tsv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_COLS, delimiter="\t")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\n[DIAG] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
