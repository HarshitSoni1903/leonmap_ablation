"""
Collect metrics from each method's output dir into a single table.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


_SAPBERT_LABELS = {
    ("base", False): "Base SapBERT",
    ("base", True):  "Base SapBERT + boost",
    ("ft", False):   "Fine-tuned SapBERT (no boost)",
    ("ft", True):    "LeonMap (fine-tuned + boost)",
}


def _display(spec: Dict) -> str:
    name = spec["name"]
    if name == "bm25":
        return "BM25"
    if name == "tfidf":
        return "TF-IDF"
    if name == "sapbert":
        return _SAPBERT_LABELS.get(
            (spec.get("model"), spec.get("boost", False)),
            f"sapbert_{spec.get('model','')}_{spec.get('boost','')}",
        )
    return name


def aggregate(
    method_results: List[Tuple[Dict, Path]],
    ks: List[int],
    out_path: Path,
) -> List[Dict]:
    """
    method_results: list of (method_spec, metrics_json_path)
    Writes a TSV with one row per method, columns: Method, R@k for each k.
    Returns the rows for printing.
    """
    rows = []
    for spec, mpath in method_results:
        metrics = json.loads(Path(mpath).read_text())
        row = {"Method": _display(spec), "Evaluated": metrics.get("evaluated", 0)}
        for k in ks:
            row[f"R@{k}"] = round(metrics.get(f"recall@{k}", 0.0), 4)
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    return rows


def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("(empty)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def write_full_metrics(
    scenario_method_results: List[Tuple[str, Dict, Path]],
    out_path: Path,
) -> None:
    """
    scenario_method_results: list of (scenario_name, method_spec, metrics_json_path)
    Writes one row per (scenario, method) with all metrics fields.
    """
    rows = []
    for scenario_name, spec, mpath in scenario_method_results:
        metrics = json.loads(Path(mpath).read_text())
        row = {"scenario": scenario_name, "method": _display(spec)}
        row.update(metrics)
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        # union of all keys (in case some methods report fields others don't)
        all_keys = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    all_keys.append(k); seen.add(k)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)