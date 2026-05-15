"""
SapBERT method wrapper. Shells out to leonmap-ablation to do the actual
retrieval, then re-evaluates the produced predictions through
utils.eval_predictions to attach per-bucket recall fields.

Also exposes the build / mapper shell-outs because they share the same
sys.argv-swap helper.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from leonmap_ablation.utils import eval_predictions
from leonmap_ablation.v_config import SCENARIO_STUDY_KEY


def run_cli(entry_main, name: str, argv: List[str], logger) -> None:
    """Invoke a leonmap CLI entry point by swapping sys.argv around it."""
    logger.info(f"--- {name} {' '.join(argv)} ---")
    saved_argv = sys.argv
    sys.argv = [name] + argv
    try:
        entry_main()
    finally:
        sys.argv = saved_argv


def run_leonmap_build(v_config: Path, collections: List[str], logger) -> None:
    import leonmap.config as _cfg
    from leonmap.build_vdb import main as build_main
    from leonmap.config_loader import load_user_config

    saved_init = _cfg.BuildConfig.__init__
    load_user_config(str(v_config))
    try:
        run_cli(build_main, "leonmap-build",
                ["--collections", *collections, "--monitor", "0"], logger)
    finally:
        _cfg.BuildConfig.__init__ = saved_init


def run_leonmap_mapper(v_config: Path, logger) -> None:
    from leonmap.mapper import main as mapper_main
    run_cli(mapper_main, "leonmap-mapper",
            ["--study", SCENARIO_STUDY_KEY, "--config", str(v_config)], logger)


def _run_leonmap_ablation(
    v_config: Path,
    model: str,
    ks: List[int],
    scenario: Dict,
    logger,
) -> Path:
    from leonmap.ablation import main as ablation_main
    from leonmap.config import PROJECT_ROOT

    argv = [
        "--study", SCENARIO_STUDY_KEY,
        "--config", str(v_config),
        "--models", model,
        "--modes", "full_src",
        "--ks", *[str(k) for k in ks],
    ]
    run_cli(ablation_main, "leonmap-ablation", argv, logger)

    src = scenario["source_prefix"]
    tgt = scenario["target_prefix"]
    parent = Path(PROJECT_ROOT) / "ablation_results" / f"{src}_{tgt}"
    runs = sorted(parent.glob("run_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1]


def _find_ablation_dir(run_dir: Path, model: str) -> Path:
    """Find the dir holding metrics.json + predictions_at_<k>.jsonl for `model`."""
    matches = list(run_dir.rglob(f"{model}/full_src/metrics.json"))
    if not matches:
        raise RuntimeError(f"No metrics.json found under {run_dir} for model={model}")
    return matches[0].parent


def _read_predictions(jsonl_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(
    scenario: Dict,
    ks: List[int],
    out_dir: Path,
    logger,
    model: str,
    boost: bool,
    gold_pairs: List[Tuple[str, str]],
    bucket_map: Dict[Tuple[str, str], str],
    v_on: Path,
    v_off: Path,
) -> Path:
    """
    Shell out to leonmap-ablation, re-evaluate its predictions through
    eval_predictions to add per-bucket recall, overwrite metrics.json, and
    copy everything into out_dir.
    """
    v_cfg = v_on if boost else v_off
    ab_run = _run_leonmap_ablation(v_cfg, model, ks, scenario, logger)
    ab_dir = _find_ablation_dir(ab_run, model)

    # Re-evaluate using the largest-K predictions file (it has the longest
    # top_matches list, so we can compute recall@k for every requested k).
    max_k = max(ks)
    preds_path = ab_dir / f"predictions_at_{max_k}.jsonl"
    if not preds_path.exists():
        raise RuntimeError(f"Missing {preds_path} — cannot re-evaluate with buckets")
    predictions = _read_predictions(preds_path)

    # Preserve skipped counts from leonmap's own metrics.json
    orig_metrics = json.loads((ab_dir / "metrics.json").read_text())

    metrics = eval_predictions(
        predictions, gold_pairs, bucket_map, ks,
        model_name=orig_metrics.get("model", model),
        query_mode=orig_metrics.get("query_mode", "full_src"),
        direction=orig_metrics.get(
            "direction",
            f"{scenario['source_prefix']}->{scenario['target_prefix']}",
        ),
    )
    metrics["skipped_src_missing"] = orig_metrics.get("skipped_src_missing", 0)
    metrics["skipped_tgt_missing"] = orig_metrics.get("skipped_tgt_missing", 0)

    # Copy every file from the ablation output dir, then overwrite metrics.json
    # with the bucket-enriched version. leonmap's predictions_at_<k>.jsonl files
    # are already correct (and per-k truncated), so we don't rewrite them.
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ab_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "leonmap_ablation_run_dir.txt").write_text(str(ab_run))

    recall_str = ", ".join(f"r@{k}={metrics[f'recall@{k}']:.4f}" for k in ks)
    logger.info(
        f"SapBERT[{model},boost={boost}]: evaluated={metrics['evaluated']}, {recall_str}"
    )
    return out_dir / "metrics.json"
