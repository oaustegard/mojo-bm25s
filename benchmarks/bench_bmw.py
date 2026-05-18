"""Synthetic-workload bench for BMW vs scan-everything.

Per the orchestrator brief, trec-covid is infeasible to fetch on CCotw,
so we use the same synthetic 100K × 5K sparse workload that issue #34
benched against (and is representative of "many long postings, queries
hit a tiny fraction of the corpus").

Runs three sizes × three query lengths × k=10, reports BMW speedup over
the (parallel) scan-everything path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import bm25s

import mojo_bm25s


@dataclass
class BenchSample:
    n_docs: int
    n_vocab: int
    query_len: int
    k: int
    n_queries: int
    block_size: int
    scan_serial_s: float
    scan_parallel_s: float
    bmw_serial_s: float
    bmw_parallel_s: float

    def lines(self) -> list[str]:
        speedup_serial = self.scan_serial_s / max(self.bmw_serial_s, 1e-12)
        speedup_par = self.scan_parallel_s / max(self.bmw_parallel_s, 1e-12)
        return [
            f"n_docs={self.n_docs:>7d}  vocab={self.n_vocab:>5d}  "
            f"qlen={self.query_len:>3d}  k={self.k:>3d}  n_q={self.n_queries:>4d}  "
            f"B={self.block_size:>3d}",
            f"  scan-serial : {self.scan_serial_s*1000:9.2f} ms / batch",
            f"  scan-parallel: {self.scan_parallel_s*1000:9.2f} ms / batch",
            f"  bmw-serial  : {self.bmw_serial_s*1000:9.2f} ms / batch  "
            f"(vs scan-serial: {speedup_serial:5.2f}x)",
            f"  bmw-parallel: {self.bmw_parallel_s*1000:9.2f} ms / batch  "
            f"(vs scan-parallel: {speedup_par:5.2f}x)",
            "",
        ]


def make_corpus(n_docs: int, n_vocab: int, doc_len: int, seed: int = 0):
    """Synthetic sparse corpus: each doc samples `doc_len` token-ids
    uniformly from [0, n_vocab). Returns the bm25s-indexed retriever
    plus the vocab strings (so queries can be constructed)."""
    rng = np.random.default_rng(seed)
    vocab_strs = [f"tok{i}" for i in range(n_vocab)]
    corpus = []
    for _ in range(n_docs):
        # Use a Zipfian distribution so the corpus is realistic — a few
        # frequent terms, many rare ones. This is where BMW gets its win.
        token_ids = rng.zipf(1.2, size=doc_len) % n_vocab
        doc = [vocab_strs[i] for i in token_ids]
        corpus.append(doc)
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    return r, vocab_strs


def make_queries(vocab_strs: list[str], n_queries: int, query_len: int,
                 seed: int = 7) -> list[list[str]]:
    """Synthetic queries: sample query_len terms from the vocab. We use
    Zipfian sampling so most queries hit moderately common terms (the
    realistic case for BMW, where rare-term queries would scan-everything
    cheaply anyway)."""
    rng = np.random.default_rng(seed)
    queries = []
    for _ in range(n_queries):
        ids = rng.zipf(1.3, size=query_len) % len(vocab_strs)
        queries.append([vocab_strs[i] for i in ids])
    return queries


def time_fn(fn, *args, repeats: int = 3, **kwargs) -> float:
    """Best-of-N timer."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        if t1 - t0 < best:
            best = t1 - t0
    return best


def bench_one(
    n_docs: int, n_vocab: int, doc_len: int, query_len: int,
    k: int = 10, n_queries: int = 50, block_size: int = 128,
    repeats: int = 3, seed: int = 0,
) -> BenchSample:
    r, vocab_strs = make_corpus(n_docs, n_vocab, doc_len, seed=seed)
    queries = make_queries(vocab_strs, n_queries, query_len, seed=seed + 1)
    # warm: prime the BMW metadata cache so we don't include build cost
    mojo_bm25s.retrieve_batch_bmw(r, queries[:1], k=k, num_workers=1,
                                  block_size=block_size)
    mojo_bm25s.retrieve_batch(r, queries[:1], k=k, num_workers=1)

    n_workers = os.cpu_count() or 1

    scan_serial = time_fn(
        mojo_bm25s.retrieve_batch, r, queries, k=k, num_workers=1,
        repeats=repeats,
    )
    scan_parallel = time_fn(
        mojo_bm25s.retrieve_batch, r, queries, k=k, num_workers=n_workers,
        repeats=repeats,
    )
    bmw_serial = time_fn(
        mojo_bm25s.retrieve_batch_bmw, r, queries, k=k, num_workers=1,
        block_size=block_size, repeats=repeats,
    )
    bmw_parallel = time_fn(
        mojo_bm25s.retrieve_batch_bmw, r, queries, k=k, num_workers=n_workers,
        block_size=block_size, repeats=repeats,
    )
    return BenchSample(
        n_docs=n_docs, n_vocab=n_vocab, query_len=query_len, k=k,
        n_queries=n_queries, block_size=block_size,
        scan_serial_s=scan_serial, scan_parallel_s=scan_parallel,
        bmw_serial_s=bmw_serial, bmw_parallel_s=bmw_parallel,
    )


def report_index_overhead(n_docs: int, n_vocab: int, doc_len: int,
                          block_size: int = 128, seed: int = 0):
    r, _ = make_corpus(n_docs, n_vocab, doc_len, seed=seed)
    data = r.scores["data"]
    indices = r.scores["indices"]
    indptr = r.scores["indptr"]
    bmax, boff = mojo_bm25s.build_block_max_metadata(
        data, indices, indptr, block_size=block_size,
    )
    data_bytes = data.nbytes
    bmax_bytes = bmax.nbytes + boff.nbytes
    print(
        f"BMW metadata overhead at n_docs={n_docs}, n_vocab={n_vocab}, "
        f"B={block_size}: {bmax_bytes/1024:.1f} KiB / "
        f"{data_bytes/1024:.1f} KiB data = "
        f"{100*bmax_bytes/max(data_bytes,1):.2f}%"
    )


def main():
    print("BMW vs scan-everything synthetic bench (proxy for trec-covid)")
    print("=" * 70)
    print()

    samples = []
    configs = [
        # (n_docs, n_vocab, doc_len, query_len, k)
        ( 10_000, 1_000, 40, 2, 10),
        ( 10_000, 1_000, 40, 5, 10),
        ( 10_000, 1_000, 40, 20, 10),
        (100_000, 5_000, 50, 2, 10),
        (100_000, 5_000, 50, 5, 10),
        (100_000, 5_000, 50, 20, 10),
    ]
    for cfg in configs:
        n_docs, n_vocab, doc_len, qlen, k = cfg
        print(f"== n_docs={n_docs}, n_vocab={n_vocab}, qlen={qlen}, k={k} ==")
        s = bench_one(n_docs, n_vocab, doc_len, qlen, k=k, n_queries=50,
                      block_size=128, repeats=3)
        samples.append(s)
        for line in s.lines():
            print(line)

    print("\nIndex overhead at the largest config:")
    report_index_overhead(100_000, 5_000, 50, block_size=128)


if __name__ == "__main__":
    main()
