"""
Build the temporary leonmap v-config from the scenario config + evidence path.
The original leonmap config is never touched. Every collection, ablation, and
mapping used by this run is declared fresh here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


SCENARIO_STUDY_KEY = "_emnlp_scenario"   # used as both ABLATIONS and MAPPINGS key


def write_v_config(
    scenario_cfg: Dict,
    evidence_tsv: Path,
    out_path: Path,
    enable_boost: bool,
    ks: list,
) -> Path:
    """
    Construct a leonmap-compatible YAML that declares:
      - build defaults (with the enable_boost flag set)
      - all collections used by this scenario (ft + base variants)
      - one ABLATIONS entry pointing at the evidence TSV
      - one MAPPINGS entry for the final mapper run
    """
    scenario = scenario_cfg["scenario"]
    src = scenario["source_prefix"]   # e.g. "mondo"
    tgt = scenario["target_prefix"]   # e.g. "mesh"

    collections = dict(scenario_cfg["collections"])

    # one ABLATIONS entry. Models and modes are fixed; the wrapper drives which
    # model variant runs by overriding --models on the leonmap-ablation CLI.
    ablations = {
        SCENARIO_STUDY_KEY: {
            "src_collection": src,
            "tgt_collection": tgt,
            "gold_file": str(evidence_tsv.resolve()),
            "src_col": scenario_cfg["evidence"].get("src_col", "subject_id"),
            "tgt_col": scenario_cfg["evidence"].get("tgt_col", "object_id"),
            "ks": ks,
            "models": ["ft"],         # default; overridden per call
            "modes": ["full_src"],
            "reverse": False,
        }
    }

    # one MAPPINGS entry for the final mapper run
    mapper_cfg = scenario_cfg.get("mapper", {})
    mappings = {
        SCENARIO_STUDY_KEY: {
            "src_collection": src,
            "tgt_collection": tgt,
            "gold_file": ablations[SCENARIO_STUDY_KEY]["gold_file"],
            "src_col": scenario_cfg["evidence"].get("src_col", "subject_id"),
            "tgt_col": scenario_cfg["evidence"].get("tgt_col", "object_id"),
            "threshold": mapper_cfg.get("threshold", 0.9),
            "top_k": mapper_cfg.get("top_k", 1),
            "reverse": False,
        }
    }

    v_config = {
        "build": {
            "enable_boost": bool(enable_boost),
            # everything else inherits leonmap defaults
        },
        "collections": collections,
        "ablations": ablations,
        "mappings": mappings,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(v_config, f, sort_keys=False)
    return out_path
