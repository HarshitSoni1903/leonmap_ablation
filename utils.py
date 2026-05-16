"""
Shared helpers for the leonmap_ablation wrapper:
- tokenization
- payload -> text / token-list adapters used by BM25 and TF-IDF
- gold-pair lexical-overlap bucketing (three Jaccard variants per pair)
- per-method finalize: eval predictions + write metrics + write jsonl
- MeSH disease OWL builder (formerly in main.py)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Jaccard cut-offs for bucket assignment.
# bucket = zero if J == 0
#        | low if 0 < J <= 0.33
#        | medium if 0.33 < J <= 0.66
#        | high if J > 0.66
_BUCKET_THRESHOLDS: Tuple[float, float] = (0.33, 0.66)
BUCKETS: List[str] = ["zero", "low", "medium", "high"]

# Three Jaccard "views" of a pair:
#   label   : tokens(src.label)               vs tokens(tgt.label)               — what the old code did
#   pooled  : tokens(label + all synonyms)    vs same on tgt                     — what BM25/TF-IDF actually see
#   max_syn : max over (s in {label, syns}, t in {label, syns}) of jaccard(s, t) — best alignment over surface forms
BUCKET_KINDS: List[str] = ["label", "pooled", "max_syn"]


# ----- tokenization / payload adapters --------------------------------------


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanum, drop empties."""
    return _TOKEN_RE.findall((text or "").lower())


def doc_text(payload: Dict) -> str:
    """'label syn1 syn2 ...' — used by the TF-IDF vectorizer."""
    parts = [payload.get("label", "") or ""]
    parts.extend(payload.get("synonyms", []) or [])
    return " ".join(parts)


def doc_tokens(payload: Dict) -> List[str]:
    """Flat token list for BM25: tokens(label) + tokens(syn1) + ..."""
    tokens = tokenize(payload.get("label", ""))
    for s in payload.get("synonyms", []) or []:
        tokens.extend(tokenize(s))
    return tokens


# ----- bucketing -------------------------------------------------------------


def _bucket_for_jaccard(j: float) -> str:
    if j == 0.0:
        return "zero"
    if j <= _BUCKET_THRESHOLDS[0]:
        return "low"
    if j <= _BUCKET_THRESHOLDS[1]:
        return "medium"
    return "high"


def _surface_forms(payload: Dict) -> List[str]:
    forms = [payload.get("label", "") or ""]
    forms.extend(payload.get("synonyms", []) or [])
    return [f for f in forms if f and f.strip()]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _max_synonym_jaccard(src_forms: List[set], tgt_forms: List[set]) -> float:
    best = 0.0
    for s in src_forms:
        if not s:
            continue
        for t in tgt_forms:
            if not t:
                continue
            j = _jaccard(s, t)
            if j > best:
                best = j
                if best == 1.0:
                    return best
    return best


def compute_buckets(
    gold_pairs: Iterable[Tuple[str, str]],
    src_db,
    tgt_db,
    out_path: Optional[Path] = None,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    For each gold pair, compute three Jaccard variants and bucket each one.

    Returns {(src_id, tgt_id): {"label": bucket, "pooled": bucket, "max_syn": bucket}}.

    If `out_path` is given, writes buckets.tsv with full per-pair diagnostics:
    src_id, tgt_id, src_label, tgt_label, j_label, b_label, j_pooled, b_pooled,
    j_max_syn, b_max_syn, n_inter_pooled, n_union_pooled.
    """
    bucket_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    tsv = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tsv = open(out_path, "w", encoding="utf-8")
        tsv.write(
            "src_id\ttgt_id\tsrc_label\ttgt_label"
            "\tj_label\tb_label"
            "\tj_pooled\tb_pooled"
            "\tj_max_syn\tb_max_syn"
            "\tn_inter_pooled\tn_union_pooled\n"
        )
    try:
        for src_id, tgt_id in gold_pairs:
            src_payload = src_db.get_payload_by_id(src_id) or {}
            tgt_payload = tgt_db.get_payload_by_id(tgt_id) or {}

            src_form_sets = [set(tokenize(f)) for f in _surface_forms(src_payload)]
            tgt_form_sets = [set(tokenize(f)) for f in _surface_forms(tgt_payload)]

            s_lab = src_form_sets[0] if src_form_sets else set()
            t_lab = tgt_form_sets[0] if tgt_form_sets else set()
            j_label = _jaccard(s_lab, t_lab)

            s_pool = set().union(*src_form_sets) if src_form_sets else set()
            t_pool = set().union(*tgt_form_sets) if tgt_form_sets else set()
            j_pooled = _jaccard(s_pool, t_pool)
            n_inter = len(s_pool & t_pool)
            n_union = len(s_pool | t_pool)

            j_max_syn = _max_synonym_jaccard(src_form_sets, tgt_form_sets)

            buckets = {
                "label": _bucket_for_jaccard(j_label),
                "pooled": _bucket_for_jaccard(j_pooled),
                "max_syn": _bucket_for_jaccard(j_max_syn),
            }
            bucket_map[(src_id, tgt_id)] = buckets

            if tsv is not None:
                src_label = (src_payload.get("label") or "").replace("\t", " ")
                tgt_label = (tgt_payload.get("label") or "").replace("\t", " ")
                tsv.write(
                    f"{src_id}\t{tgt_id}\t{src_label}\t{tgt_label}"
                    f"\t{j_label:.6f}\t{buckets['label']}"
                    f"\t{j_pooled:.6f}\t{buckets['pooled']}"
                    f"\t{j_max_syn:.6f}\t{buckets['max_syn']}"
                    f"\t{n_inter}\t{n_union}\n"
                )
    finally:
        if tsv is not None:
            tsv.close()
    return bucket_map


# ----- prediction rows / evaluation -----------------------------------------


def build_prediction_row(
    src_id: str,
    src_payload: Dict,
    tgt_id: str,
    top_idx: Iterable[int],
    scores,
    corpus_ids: List[str],
    tgt_db,
) -> Dict:
    """Common (src_id, src_label, tgt_id_gold, top_matches) row used by BM25/TF-IDF."""
    top_matches = []
    for i in top_idx:
        pid = corpus_ids[i]
        tgt_meta = tgt_db.get_payload_by_id(pid) or {}
        top_matches.append({
            "id": pid,
            "label": tgt_meta.get("label", ""),
            "score": round(float(scores[i]), 6),
        })
    return {
        "src_id": src_id,
        "src_label": str(src_payload.get("label", "") or ""),
        "tgt_id_gold": tgt_id,
        "top_matches": top_matches,
    }


def eval_predictions(
    predictions: List[Dict],
    gold_pairs: Iterable[Tuple[str, str]],
    bucket_map: Dict[Tuple[str, str], Dict[str, str]],
    ks: List[int],
    model_name: str,
    query_mode: str,
    direction: str,
) -> Dict:
    """
    Build the full metrics dict from per-pair prediction rows.

    Overall recall@k denominator is len(predictions).

    Per-bucket recall is reported for THREE bucket families: label, pooled,
    and max_syn (see BUCKET_KINDS / compute_buckets). Each per-bucket recall
    uses the EVALUATED-IN-BUCKET denominator: only pairs that produced a
    prediction count. Skipped pairs are reported separately as
    `skipped_<kind>_<bucket>` so they don't pollute recall.

    Emitted keys per kind/bucket:
      gold_<kind>_<bucket>       : gold-pair count in bucket (informational)
      evaluated_<kind>_<bucket>  : pairs in bucket that produced a prediction
      skipped_<kind>_<bucket>    : gold - evaluated for this bucket
      recall@<k>_<kind>_<bucket> : evaluated_hits / evaluated_in_bucket
    """
    total = len(predictions)
    hits_at = {k: 0 for k in ks}

    gold_per_bucket = {kind: {b: 0 for b in BUCKETS} for kind in BUCKET_KINDS}
    for pair in gold_pairs:
        kinds = bucket_map.get(pair)
        if not kinds:
            continue
        for kind in BUCKET_KINDS:
            b = kinds.get(kind)
            if b in gold_per_bucket[kind]:
                gold_per_bucket[kind][b] += 1

    eval_per_bucket = {kind: {b: 0 for b in BUCKETS} for kind in BUCKET_KINDS}
    hits_per_bucket = {kind: {b: {k: 0 for k in ks} for b in BUCKETS} for kind in BUCKET_KINDS}

    for row in predictions:
        tgt_id = row["tgt_id_gold"]
        kinds = bucket_map.get((row["src_id"], tgt_id))
        top_ids = [m["id"] for m in row.get("top_matches", [])]
        hit_at_k = {k: tgt_id in top_ids[:k] for k in ks}
        for k in ks:
            if hit_at_k[k]:
                hits_at[k] += 1
        if not kinds:
            continue
        for kind in BUCKET_KINDS:
            b = kinds.get(kind)
            if b not in eval_per_bucket[kind]:
                continue
            eval_per_bucket[kind][b] += 1
            for k in ks:
                if hit_at_k[k]:
                    hits_per_bucket[kind][b][k] += 1

    metrics: Dict = {
        "direction": direction,
        "model": model_name,
        "query_mode": query_mode,
        "evaluated": total,
    }
    for k in ks:
        metrics[f"recall@{k}"] = (hits_at[k] / total) if total else 0.0

    for kind in BUCKET_KINDS:
        for b in BUCKETS:
            gold_n = gold_per_bucket[kind][b]
            eval_n = eval_per_bucket[kind][b]
            metrics[f"gold_{kind}_{b}"] = gold_n
            metrics[f"evaluated_{kind}_{b}"] = eval_n
            metrics[f"skipped_{kind}_{b}"] = gold_n - eval_n
        for k in ks:
            for b in BUCKETS:
                eval_n = eval_per_bucket[kind][b]
                metrics[f"recall@{k}_{kind}_{b}"] = (
                    hits_per_bucket[kind][b][k] / eval_n
                ) if eval_n else 0.0

    return metrics


def write_predictions(predictions: List[Dict], out_dir: Path, ks: List[int]) -> None:
    """One predictions_at_<k>.jsonl per k. top_matches in each row is truncated to k."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ks:
        with open(out_dir / f"predictions_at_{k}.jsonl", "w", encoding="utf-8") as f:
            for row in predictions:
                tgt_id = row["tgt_id_gold"]
                top_k = row.get("top_matches", [])[:k]
                trimmed = {
                    "src_id": row["src_id"],
                    "src_label": row.get("src_label", ""),
                    "tgt_id_gold": tgt_id,
                    "hit": tgt_id in [m["id"] for m in top_k],
                    "top_matches": top_k,
                }
                f.write(json.dumps(trimmed, ensure_ascii=False) + "\n")


def finalize_method(
    out_dir: Path,
    predictions: List[Dict],
    gold_pairs: Iterable[Tuple[str, str]],
    bucket_map: Dict[Tuple[str, str], Dict[str, str]],
    ks: List[int],
    *,
    model_name: str,
    direction: str,
    skipped_src: int,
    skipped_tgt: int,
    logger,
    query_mode: str = "label_plus_synonyms",
    log_label: Optional[str] = None,
) -> Path:
    """Eval predictions, attach skip counts, write metrics.json + jsonls, log, return metrics path."""
    metrics = eval_predictions(
        predictions, gold_pairs, bucket_map, ks,
        model_name=model_name, query_mode=query_mode, direction=direction,
    )
    metrics["skipped_src_missing"] = skipped_src
    metrics["skipped_tgt_missing"] = skipped_tgt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_predictions(predictions, out_dir, ks)

    recall_str = ", ".join(f"r@{k}={metrics[f'recall@{k}']:.4f}" for k in ks)
    logger.info(f"{log_label or model_name}: evaluated={metrics['evaluated']}, {recall_str}")
    return out_dir / "metrics.json"


# ----- MeSH disease OWL build (used by main.ensure_ontology_files) ----------


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def build_mesh_disease_owl(out_path: Path, logger) -> None:
    """Generate an OWL file with one owl:Class per MeSH disease descriptor."""
    from indra.databases import mesh_client

    mesh_ids: List[Tuple[str, str]] = []
    for mid, name in mesh_client.mesh_id_to_name.items():
        try:
            if mesh_client.is_disease(mid):
                mesh_ids.append((mid, name))
        except Exception:
            continue
    logger.info(f"Collected {len(mesh_ids)} disease MeSH descriptors")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n')
        f.write('         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"\n')
        f.write('         xmlns:owl="http://www.w3.org/2002/07/owl#">\n')
        f.write('<owl:Ontology rdf:about="http://example.org/mesh_disease"/>\n')
        for mid, name in mesh_ids:
            iri = f"http://example.org/mesh_{mid}"
            f.write(f'<owl:Class rdf:about="{iri}">\n')
            f.write(f'  <rdfs:label>{_xml_escape(name)}</rdfs:label>\n')
            f.write('</owl:Class>\n')
        f.write('</rdf:RDF>\n')
