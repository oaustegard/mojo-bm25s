"""Command-line interface for mojo-bm25s (issue #28).

Subcommands:
    mojo-bm25s index --corpus FILE --out DIR [--method ...] [--k1] [--b] [--delta]
    mojo-bm25s query --index DIR --queries FILE --k N --out FILE

JSONL formats (BEIR-compatible):
    corpus  : {"_id": "doc1", "text": "..."}
    queries : {"_id": "q1", "text": "..."}
    results : {"_id": "q1", "doc_ids": ["doc7", ...], "scores": [4.21, ...]}

Thin wrapper around ``mojo_bm25s.Retriever`` plus a ``doc_ids.json``
sidecar inside the index dir — the Retriever stores the CSC index by
internal int id, this sidecar holds the original ``_id`` strings in
corpus order so ``query`` can map retrieved int ids back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List

from .retriever import Retriever


_DOC_IDS_SIDECAR = "doc_ids.json"


def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{lineno}: not valid JSON: {e.msg}"
                ) from e


def _load_corpus(path: Path) -> tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    texts: List[str] = []
    for lineno, row in enumerate(_read_jsonl(path), start=1):
        if "_id" not in row:
            raise ValueError(
                f"{path}:{lineno}: corpus entry missing required field '_id'"
            )
        if "text" not in row:
            raise ValueError(
                f"{path}:{lineno}: corpus entry missing required field 'text'"
            )
        doc_ids.append(str(row["_id"]))
        texts.append(str(row["text"]))
    return doc_ids, texts


def _load_queries(path: Path) -> tuple[List[str], List[str]]:
    q_ids: List[str] = []
    texts: List[str] = []
    for lineno, row in enumerate(_read_jsonl(path), start=1):
        if "_id" not in row:
            raise ValueError(
                f"{path}:{lineno}: query entry missing required field '_id'"
            )
        if "text" not in row:
            raise ValueError(
                f"{path}:{lineno}: query entry missing required field 'text'"
            )
        q_ids.append(str(row["_id"]))
        texts.append(str(row["text"]))
    return q_ids, texts


def _cmd_index(args: argparse.Namespace) -> int:
    corpus_path = Path(args.corpus)
    out_dir = Path(args.out)

    try:
        doc_ids, texts = _load_corpus(corpus_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    retriever = Retriever(
        k1=args.k1,
        b=args.b,
        delta=args.delta,
        method=args.method,
    ).index(texts)

    retriever.save(out_dir)
    (out_dir / _DOC_IDS_SIDECAR).write_text(
        json.dumps({"doc_ids": doc_ids}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"indexed {len(doc_ids)} docs (vocab={len(retriever._vocab)}) "
        f"→ {out_dir}",
        file=sys.stderr,
    )
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    index_dir = Path(args.index)
    queries_path = Path(args.queries)
    out_path = Path(args.out)

    if not index_dir.is_dir():
        print(f"error: index dir does not exist: {index_dir}", file=sys.stderr)
        return 2

    sidecar = index_dir / _DOC_IDS_SIDECAR
    if not sidecar.exists():
        print(
            f"error: index dir missing {_DOC_IDS_SIDECAR}; was it built "
            f"with `mojo-bm25s index`?",
            file=sys.stderr,
        )
        return 2

    try:
        retriever = Retriever.load(index_dir)
    except Exception as e:
        print(f"error: failed to load index: {e}", file=sys.stderr)
        return 2

    doc_ids: List[str] = json.loads(sidecar.read_text(encoding="utf-8"))["doc_ids"]

    try:
        q_ids, q_texts = _load_queries(queries_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    scores, ids = retriever.retrieve(q_texts, k=args.k)

    with open(out_path, "w", encoding="utf-8") as f:
        for qi, qid in enumerate(q_ids):
            row = {
                "_id": qid,
                "doc_ids": [doc_ids[int(idx)] for idx in ids[qi]],
                "scores": [float(s) for s in scores[qi]],
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    print(f"wrote {len(q_ids)} result rows → {out_path}", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mojo-bm25s",
        description="Mojo-native BM25 retrieval — index a corpus, run queries.",
    )
    sub = p.add_subparsers(dest="cmd", metavar="{index,query}")

    p_index = sub.add_parser("index", help="Build an index from a JSONL corpus.")
    p_index.add_argument(
        "--corpus", required=True,
        help="Path to corpus JSONL (one {\"_id\", \"text\"} per line).",
    )
    p_index.add_argument(
        "--out", required=True,
        help="Output index directory (created if absent).",
    )
    p_index.add_argument(
        "--method", default="lucene",
        choices=("lucene", "robertson", "atire", "bm25l", "bm25+"),
        help="BM25 TFC variant (default: lucene).",
    )
    p_index.add_argument(
        "--k1", type=float, default=1.5,
        help="BM25 k1 parameter (default: 1.5).",
    )
    p_index.add_argument(
        "--b", type=float, default=0.75,
        help="BM25 b parameter (default: 0.75).",
    )
    p_index.add_argument(
        "--delta", type=float, default=0.5,
        help="BM25 delta parameter for bm25l / bm25+ (default: 0.5).",
    )

    p_query = sub.add_parser("query", help="Run queries against a saved index.")
    p_query.add_argument(
        "--index", required=True,
        help="Saved index directory.",
    )
    p_query.add_argument(
        "--queries", required=True,
        help="Path to queries JSONL (one {\"_id\", \"text\"} per line).",
    )
    p_query.add_argument(
        "--out", required=True,
        help="Output results JSONL.",
    )
    p_query.add_argument(
        "--k", type=int, default=10,
        help="Top-k retrieved per query (default: 10).",
    )

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "index":
        return _cmd_index(args)
    if args.cmd == "query":
        return _cmd_query(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
