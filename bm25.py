"""
BM25 baseline. Output shape matches leonmap/ablation.py so downstream
aggregation is uniform. Per-bucket lexical-overlap metrics are added by
utils.eval_predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from leonmap_ablation.utils import (
    build_prediction_row,
    doc_tokens,
    finalize_method,
)


def run(
    scenario: Dict,
    ks: List[int],
    out_dir: Path,
    logger,
    *,
    src_db,
    tgt_db,
    gold_pairs: List[Tuple[str, str]],
    bucket_map: Dict[Tuple[str, str], str],
) -> Path:
    corpus_ids: List[str] = []
    corpus_tokens: List[List[str]] = []
    for pos in range(tgt_db.count()):
        pid = tgt_db.id_at_pos(pos)
        if not pid:
            continue
        corpus_ids.append(pid)
        corpus_tokens.append(doc_tokens(tgt_db.get_payload_by_id(pid) or {}))

    logger.info(f"BM25 corpus: {len(corpus_ids)} target docs")
    bm25 = BM25Okapi(corpus_tokens)

    max_k = max(ks)
    predictions: List[Dict] = []
    skipped_src = 0
    skipped_tgt = 0

    for src_id, tgt_id in tqdm(gold_pairs, desc="BM25", unit="pair"):
        if tgt_id not in tgt_db.id2pos:
            skipped_tgt += 1
            continue
        src_payload = src_db.get_payload_by_id(src_id)
        if not src_payload:
            skipped_src += 1
            continue
        qtok = doc_tokens(src_payload)
        if not qtok:
            skipped_src += 1
            continue

        scores = bm25.get_scores(qtok)
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:max_k]
        predictions.append(build_prediction_row(
            src_id, src_payload, tgt_id, top_idx, scores, corpus_ids, tgt_db,
        ))

    return finalize_method(
        out_dir, predictions, gold_pairs, bucket_map, ks,
        model_name="bm25",
        direction=f"{scenario['source_prefix']}->{scenario['target_prefix']}",
        skipped_src=skipped_src,
        skipped_tgt=skipped_tgt,
        logger=logger,
        log_label="BM25",
    )
