"""
leonmap_ablation orchestrator.

usage:
    python -m leonmap_ablation.main --config leonmap_ablation/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from leonmap.config import BuildConfig
from leonmap.utils import load_collection, load_gold_pairs

from leonmap_ablation import aggregate, bm25, evidence as evidence_mod, sapbert, tfidf, utils
from leonmap_ablation.v_config import write_v_config


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("leonmap_ablation")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ----- data prep -------------------------------------------------------------

def ensure_ontology_files(data_prep_cfg: Dict, logger) -> None:
    mondo_cfg = data_prep_cfg.get("mondo", {})
    if mondo_cfg.get("out_path"):
        out = Path(mondo_cfg["out_path"])
        if not out.exists():
            url = mondo_cfg["url"]
            logger.info(f"Downloading MONDO from {url} -> {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(url, out)
        else:
            logger.info(f"MONDO OWL exists: {out}")

    mesh_cfg = data_prep_cfg.get("mesh", {})
    if mesh_cfg.get("out_path"):
        out = Path(mesh_cfg["out_path"])
        if not out.exists():
            logger.info(f"Building MeSH disease OWL via indra.mesh_client -> {out}")
            utils.build_mesh_disease_owl(out, logger)
        else:
            logger.info(f"MeSH OWL exists: {out}")


def _method_dir_name(m: Dict) -> str:
    if m["name"] == "sapbert":
        boost = "boost" if m.get("boost", False) else "noboost"
        return f"sapbert_{m['model']}_{boost}"
    return m["name"]


# ----- per-scenario driver ---------------------------------------------------

def run_scenario(
    scenario_cfg: Dict,
    methods: List[Dict],
    ks: List[int],
    run_dir: Path,
    logger,
    keep_temp_config: bool = True,
) -> List[Tuple[Dict, Path]]:
    """End-to-end run for one scenario. Returns (method_spec, metrics_json) pairs."""
    scenario = scenario_cfg["scenario"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. data prep
    ensure_ontology_files(scenario_cfg.get("data_prep", {}), logger)

    # 2. evidence union
    evidence_tsv = evidence_mod.build_evidence(scenario, scenario_cfg["evidence"], logger)

    # 3. write two v-configs: boost on, boost off
    v_on = write_v_config(scenario_cfg, evidence_tsv,
                          run_dir / "v_config_boost_on.yaml",
                          enable_boost=True, ks=ks)
    v_off = write_v_config(scenario_cfg, evidence_tsv,
                           run_dir / "v_config_boost_off.yaml",
                           enable_boost=False, ks=ks)

    # 4. decide which collections to build
    needed = set()
    for m in methods:
        if m["name"] == "sapbert":
            if m["model"] == "ft":
                needed.update([scenario["source_prefix"], scenario["target_prefix"]])
            elif m["model"] == "base":
                needed.update([f"{scenario['source_prefix']}_base",
                               f"{scenario['target_prefix']}_base"])
        else:
            needed.update([scenario["source_prefix"], scenario["target_prefix"]])
    sapbert.run_leonmap_build(v_on, sorted(needed), logger)

    # 5. load collections + gold pairs once, compute buckets (writes buckets.tsv inline)
    src_col = scenario_cfg["evidence"].get("src_col", "subject_id")
    tgt_col = scenario_cfg["evidence"].get("tgt_col", "object_id")

    cfg = BuildConfig()
    src_db = load_collection(cfg, scenario["source_prefix"])
    tgt_db = load_collection(cfg, scenario["target_prefix"])
    gold_pairs = load_gold_pairs(evidence_tsv, src_col, tgt_col)
    bucket_map = utils.compute_buckets(gold_pairs, src_db, tgt_db,
                                       out_path=run_dir / "buckets.tsv")

    for kind in utils.BUCKET_KINDS:
        counts = {b: 0 for b in utils.BUCKETS}
        for kinds in bucket_map.values():
            counts[kinds[kind]] += 1
        logger.info(
            f"Buckets[{kind}]: " + ", ".join(f"{b}={counts[b]}" for b in utils.BUCKETS)
        )

    # 6. run methods
    method_results: List[Tuple[Dict, Path]] = []
    for m in methods:
        name = m["name"]
        method_dir = run_dir / "methods" / _method_dir_name(m)
        method_dir.mkdir(parents=True, exist_ok=True)

        if name == "bm25":
            mpath = bm25.run(scenario, ks, method_dir, logger,
                             src_db=src_db, tgt_db=tgt_db,
                             gold_pairs=gold_pairs, bucket_map=bucket_map)
        elif name == "tfidf":
            mpath = tfidf.run(scenario, ks, method_dir, logger,
                              src_db=src_db, tgt_db=tgt_db,
                              gold_pairs=gold_pairs, bucket_map=bucket_map)
        elif name == "sapbert":
            mpath = sapbert.run(scenario, ks, method_dir, logger,
                                model=m["model"], boost=m.get("boost", False),
                                gold_pairs=gold_pairs, bucket_map=bucket_map,
                                v_on=v_on, v_off=v_off)
        else:
            raise ValueError(f"Unknown method: {name}")

        method_results.append((m, mpath))

    # 7. per-scenario aggregate
    table = aggregate.aggregate(method_results, ks, run_dir / "recall_at_k.tsv")
    print()
    aggregate.print_table(table)
    logger.info(f"Table written: {run_dir / 'recall_at_k.tsv'}")

    # 8. final mapper run
    mapper_cfg = scenario_cfg.get("mapper", {})
    if mapper_cfg.get("run", False):
        logger.info("Running final leonmap-mapper")
        sapbert.run_leonmap_mapper(v_on, logger)

    if not keep_temp_config:
        v_on.unlink(missing_ok=True)
        v_off.unlink(missing_ok=True)

    return method_results


# ----- main ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Scenario config YAML")
    args = ap.parse_args()

    logger = _get_logger()
    top_cfg = yaml.safe_load(Path(args.config).read_text())

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base_run_dir = Path("results") / f"run_{stamp}"
    base_run_dir.mkdir(parents=True, exist_ok=True)
    (base_run_dir / "config.yaml").write_text(Path(args.config).read_text())
    logger.info(f"Base run dir: {base_run_dir}")

    methods = top_cfg["methods"]
    ks = top_cfg["ks"]
    keep_temp_config = top_cfg.get("keep_temp_config", True)
    scenario_blocks = top_cfg.get("scenarios", [top_cfg])

    all_results: List[Tuple[str, Dict, Path]] = []
    for block in scenario_blocks:
        scenario_name = block["scenario"]["name"]
        run_dir = base_run_dir / scenario_name
        logger.info(f"=== Scenario: {scenario_name} ===")

        method_results = run_scenario(block, methods, ks, run_dir, logger,
                                      keep_temp_config=keep_temp_config)
        for spec, mpath in method_results:
            all_results.append((scenario_name, spec, mpath))

    aggregate.write_full_metrics(all_results, base_run_dir / "all_scenarios.tsv")
    logger.info(f"Combined table: {base_run_dir / 'all_scenarios.tsv'}")
    logger.info(f"DONE. {base_run_dir}")


if __name__ == "__main__":
    main()
