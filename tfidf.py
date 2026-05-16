"""
TF-IDF baseline. Same output shape as bm25.py and leonmap/ablation.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from leonmap_ablation.utils import (
    build_prediction_row,
    doc_text,
    finalize_method,
)


_TOKEN_RE = r"[a-z0-9]+"


def run(
    scenario: Dict,
    ks: List[int],
    out_dir: Path,
    logger,
    *,
    src_db,
    tgt_db,
    gold_pairs: List[Tuple[str, str]],
    bucket_map: Dict[Tuple[str, str], Dict[str, str]],
) -> Path:
    corpus_ids: List[str] = []
    corpus_texts: List[str] = []
    for pos in range(tgt_db.count()):
        pid = tgt_db.id_at_pos(pos)
        if not pid:
            continue
        corpus_ids.append(pid)
        corpus_texts.append(doc_text(tgt_db.get_payload_by_id(pid) or {}))

    logger.info(f"TF-IDF corpus: {len(corpus_ids)} target docs")
    vec = TfidfVectorizer(lowercase=True, token_pattern=_TOKEN_RE)
    tgt_mat = vec.fit_transform(corpus_texts)
    norms = np.sqrt(tgt_mat.multiply(tgt_mat).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    tgt_mat_norm = tgt_mat.multiply(1.0 / norms[:, None]).tocsr()

    max_k = max(ks)
    predictions: List[Dict] = []
    skipped_src = 0
    skipped_tgt = 0

    for src_id, tgt_id in tqdm(gold_pairs, desc="TF-IDF", unit="pair"):
        if tgt_id not in tgt_db.id2pos:
            skipped_tgt += 1
            continue
        src_payload = src_db.get_payload_by_id(src_id)
        if not src_payload:
            skipped_src += 1
            continue
        qtext = doc_text(src_payload)
        if not qtext.strip():
            skipped_src += 1
            continue

        qvec = vec.transform([qtext])
        qnorm = float(np.sqrt(qvec.multiply(qvec).sum()))
        if qnorm == 0:
            skipped_src += 1
            continue
        qvec = qvec.multiply(1.0 / qnorm)

        scores = (qvec @ tgt_mat_norm.T).toarray().ravel()
        top_idx = np.argpartition(-scores, min(max_k, len(scores) - 1))[:max_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        predictions.append(build_prediction_row(
            src_id, src_payload, tgt_id, top_idx, scores, corpus_ids, tgt_db,
        ))

    return finalize_method(
        out_dir, predictions, gold_pairs, bucket_map, ks,
        model_name="tfidf",
        direction=f"{scenario['source_prefix']}->{scenario['target_prefix']}",
        skipped_src=skipped_src,
        skipped_tgt=skipped_tgt,
        logger=logger,
        log_label="TF-IDF",
    )
