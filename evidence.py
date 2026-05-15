"""
Build the evidence union (biomappings + OBO xrefs + SeMRA) and write it as a
flat TSV with subject_id/object_id columns. Used as the "gold" file for the
leonmap ablation study and the leonmap mapper run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import polars as pl

from biomappings.resources import POSITIVES_SSSOM_PATH, PREDICTIONS_SSSOM_PATH
from mapnet.utils.filtering import load_semera_landscape_df
from mapnet.utils.obo import load_known_mappings_df
from mapnet.utils.utils import make_undirected


def build_evidence(scenario: Dict, evidence_cfg: Dict, logger) -> Path:
    # if a prebuilt TSV is provided, use it directly (skip everything else)
    prebuilt = evidence_cfg.get("prebuilt_tsv")
    if prebuilt:
        prebuilt = Path(prebuilt)
        if not prebuilt.exists():
            raise FileNotFoundError(f"prebuilt_tsv not found: {prebuilt}")
        n = sum(1 for _ in open(prebuilt)) - 1
        logger.info(f"Using prebuilt evidence TSV: {prebuilt} ({n} rows)")
        return prebuilt

    out_path = Path(evidence_cfg["cache_tsv"])
    if out_path.exists() and not evidence_cfg.get("force_rebuild", False):
        n = sum(1 for _ in open(out_path)) - 1
        logger.info(f"Using cached evidence TSV: {out_path} ({n} rows)")
        return out_path

    src = scenario["source_prefix"]
    tgt = scenario["target_prefix"]
    landscape = scenario["semra_landscape"]

    evidence = None
    if evidence_cfg.get("use_biomappings", True):
        bio = _load_biomappings_sssom(src, tgt)
        logger.info(f"Biomappings: {len(bio)} rows")
        evidence = bio

    if evidence_cfg.get("use_known_mappings", True):
        known = make_undirected(load_known_mappings_df(
            resources={src: {}, tgt: {}},
            meta={"landscape": landscape},
            additional_namespaces={src: src, tgt: tgt},
            sssom=False,
        ))
        logger.info(f"Known mappings (OBO xrefs): {len(known)} rows")
        evidence = known if evidence is None else evidence.vstack(known)

    if evidence_cfg.get("use_semra", True):
        semra = load_semera_landscape_df(
            landscape_name=landscape,
            resources={src: {}, tgt: {}},
            additional_namespaces={src: src, tgt: tgt},
            sssom=False,
        )
        logger.info(f"SeMRA: {len(semra)} rows")
        evidence = semra if evidence is None else evidence.vstack(semra)

    if evidence is None:
        raise RuntimeError("All evidence sources disabled, nothing to build.")

    evidence = make_undirected(evidence.unique())

    # restrict to canonical src -> tgt direction
    flat = evidence.filter(
        (pl.col("source prefix") == src) & (pl.col("target prefix") == tgt)
    ).select([
        pl.col("source identifier").alias("subject_id"),
        pl.col("target identifier").alias("object_id"),
    ]).unique()

    # multi-target diagnostic
    per_src = flat.group_by("subject_id").agg(pl.len().alias("n"))
    n_sources = len(per_src)
    n_multi = int((per_src["n"] >= 2).sum())
    pct = (100 * n_multi / n_sources) if n_sources else 0
    logger.info(
        f"Evidence: {len(flat)} pairs, {n_sources} unique sources, "
        f"{n_multi} ({pct:.1f}%) with >=2 valid targets"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat.write_csv(str(out_path), separator="\t")
    logger.info(f"Evidence TSV written: {out_path}")
    return out_path

def _canonical(cid: str) -> str:
    """Lowercase the namespace, keep local part. e.g. MONDO:0001 -> mondo:0001"""
    if ":" not in cid:
        return cid.lower()
    ns, local = cid.split(":", 1)
    return f"{ns.lower()}:{local}"


def _load_biomappings_sssom(src: str, tgt: str) -> pl.DataFrame:
    """
    Read biomappings POSITIVES + PREDICTIONS SSSOM files directly and return
    rows in mapnet's biomappings format (source identifier / source prefix /
    etc.), normalized to canonical src -> tgt direction.
    """
    records = []
    for path in [POSITIVES_SSSOM_PATH, PREDICTIONS_SSSOM_PATH]:
        try:
            df = pd.read_csv(path, comment="#", sep="\t")
        except Exception:
            continue
        for _, row in df.iterrows():
            s = _canonical(str(row.get("subject_id", "")))
            o = _canonical(str(row.get("object_id", "")))
            sl = str(row.get("subject_label", ""))
            ol = str(row.get("object_label", ""))
            if s.startswith(f"{src}:") and o.startswith(f"{tgt}:"):
                records.append({
                    "source identifier": s, "source name": sl, "source prefix": src,
                    "target identifier": o, "target name": ol, "target prefix": tgt,
                })
            elif s.startswith(f"{tgt}:") and o.startswith(f"{src}:"):
                # normalize reverse-direction rows to canonical src -> tgt
                records.append({
                    "source identifier": o, "source name": ol, "source prefix": src,
                    "target identifier": s, "target name": sl, "target prefix": tgt,
                })
    return pl.DataFrame(records)