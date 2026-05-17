"""Phase 1 benchmark harness.

CLI:

    python benchmarks/run.py --dataset scifact --backend mojo
    python benchmarks/run.py --dataset trec-covid --backend numba --k 100

Per (dataset, backend) measurement:

- **indexing time** (s) — wall clock for ``BM25.index(corpus_tokens)``
- **QPS** — queries per second over the timed retrieve loop
- **latency p50 / p95 / p99** (ms) — per-query wall clock
- **peak RSS** (MB) — high-water mark via ``resource.getrusage``

The numba backend JITs on first query; we warm up (and discard the
warmup wall time) before the timed loop. The mojo backend monkey-
patches after indexing — `mojo_bm25s.patch_bm25s` is itself a no-op
on the score data.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

# Ensure peers are importable when invoked as `python benchmarks/run.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from benchmarks.datasets import load_beir
from benchmarks.backends import (
    build_retriever, retrieve_one, retrieve_batch,
)


def _peak_rss_mb() -> float:
    # On Linux, ru_maxrss is in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_one(
    *,
    dataset: str,
    backend: str,
    k: int = 10,
    corpus_subsample: int | None = None,
    queries_subsample: int | None = None,
    warmup: int = 3,
    repeats: int = 1,
) -> dict:
    """Run one bench cell. Returns a metrics dict."""
    ds = load_beir(
        dataset,
        corpus_subsample=corpus_subsample,
        queries_subsample=queries_subsample,
    )
    corpus_tokens = ds.corpus_tokens()
    query_tokens = ds.query_tokens()

    t0 = time.perf_counter()
    retriever = build_retriever(backend, corpus_tokens)
    index_secs = time.perf_counter() - t0

    # Warmup — JIT compile for numba, page-warm everything for the others.
    for q in query_tokens[: max(1, warmup)]:
        retrieve_one(backend, retriever, q, k)

    # Throughput phase: one batched call over all queries, repeated.
    # This is the apples-to-apples comparison — numba batches via
    # `_retrieve_numba_functional`, mojo batches via `retrieve_batch`,
    # numpy still sequential-maps internally (no batched native path).
    t0 = time.perf_counter()
    for _ in range(repeats):
        retrieve_batch(backend, retriever, query_tokens, k)
    batch_total_secs = time.perf_counter() - t0
    batch_total_queries = len(query_tokens) * repeats
    qps = batch_total_queries / batch_total_secs

    # Latency phase: per-query loop, recording individual wall times.
    # Captures the small-batch (interactive) cost the throughput
    # number hides.
    latencies_us: list[float] = []
    for _ in range(repeats):
        for q in query_tokens:
            t_q = time.perf_counter()
            retrieve_one(backend, retriever, q, k)
            latencies_us.append((time.perf_counter() - t_q) * 1_000_000)

    return {
        "dataset": dataset,
        "backend": backend,
        "k": k,
        "corpus_size": len(corpus_tokens),
        "queries": len(query_tokens),
        "repeats": repeats,
        "index_secs": round(index_secs, 4),
        "qps": round(qps, 2),
        "latency_p50_ms": round(statistics.median(latencies_us) / 1000, 4),
        "latency_p95_ms": round(
            statistics.quantiles(latencies_us, n=20)[18] / 1000, 4
        ) if len(latencies_us) >= 20 else None,
        "latency_p99_ms": round(
            statistics.quantiles(latencies_us, n=100)[98] / 1000, 4
        ) if len(latencies_us) >= 100 else None,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--dataset", default="scifact",
        help="BEIR dataset name (scifact, trec-covid, ...)",
    )
    p.add_argument(
        "--backend", default="mojo", choices=["numpy", "numba", "mojo"],
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--corpus-subsample", type=int, default=None)
    p.add_argument("--queries-subsample", type=int, default=None)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--format", choices=["json", "table"], default="table",
    )
    p.add_argument(
        "--all-backends", action="store_true",
        help="Run all three backends sequentially; emit a markdown table.",
    )
    return p.parse_args(argv)


def _emit_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = [
        "backend", "qps", "index_secs",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "peak_rss_mb",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.all_backends:
        results = []
        for be in ("numpy", "numba", "mojo"):
            r = run_one(
                dataset=args.dataset, backend=be, k=args.k,
                corpus_subsample=args.corpus_subsample,
                queries_subsample=args.queries_subsample,
                warmup=args.warmup, repeats=args.repeats,
            )
            results.append(r)
        meta = results[0]
        print(
            f"## {meta['dataset']} — corpus={meta['corpus_size']:,} "
            f"queries={meta['queries']} k={meta['k']} repeats={meta['repeats']}"
        )
        print()
        print(_emit_table(results))
        return 0

    r = run_one(
        dataset=args.dataset, backend=args.backend, k=args.k,
        corpus_subsample=args.corpus_subsample,
        queries_subsample=args.queries_subsample,
        warmup=args.warmup, repeats=args.repeats,
    )
    if args.format == "json":
        print(json.dumps(r, indent=2))
    else:
        print(_emit_table([r]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
