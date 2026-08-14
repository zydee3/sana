"""Cross-encoder re-ranking of a candidate set — local ONNX, no API calls.

The dense scan and BM25 each score a chunk without ever reading it next to the query:
one compares two independently-made vectors, the other counts shared terms. A cross
encoder reads the pair together, which is why it can rank material neither arm ranks
well — the measured motivation is that BM25's top-10 holds relevant chunks the dense
scan does not return at depth 50, while fusing the two arms by rank made retrieval
worse (iteration 17). Re-ranking is the other way to spend a wide candidate set.

Two models so a null result cannot be blamed on a weak one: the standard cheap
MS MARCO MiniLM, and the much larger bge-reranker-base. Both emit a single logit per
pair, so only the ordering matters and no score calibration is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from tokenizers import Tokenizer

from .embed import MODELS_DIR, download

Floats = npt.NDArray[np.float32]

BATCH_SIZE = 32


@dataclass(frozen=True)
class CrossSpec:
    """One re-ranker candidate: the exact ONNX file and how much of a pair it reads."""

    name: str
    repo: str
    onnx_path: str
    max_length: int = 512
    tokenizer_repo: str | None = None


RERANKERS: dict[str, CrossSpec] = {
    "ms-marco-minilm": CrossSpec(
        name="ms-marco-minilm",
        repo="cross-encoder/ms-marco-MiniLM-L-6-v2",
        onnx_path="onnx/model.onnx",
    ),
    "bge-reranker-int8": CrossSpec(
        name="bge-reranker-int8",
        repo="Xenova/bge-reranker-base",
        onnx_path="onnx/model_quantized.onnx",
        tokenizer_repo="BAAI/bge-reranker-base",
    ),
}


def ensure_reranker(spec: CrossSpec) -> tuple[str, str]:
    """Fetch the graph + tokenizer into MODELS_DIR. Returns (onnx, tokenizer) paths."""
    root = MODELS_DIR / spec.name
    onnx = root / "model.onnx"
    tok = root / "tokenizer.json"
    download(f"https://huggingface.co/{spec.repo}/resolve/main/{spec.onnx_path}", onnx, print)
    tok_repo = spec.tokenizer_repo or spec.repo
    download(f"https://huggingface.co/{tok_repo}/resolve/main/tokenizer.json", tok, print)
    return str(onnx), str(tok)


def build_feed(
    ids: npt.NDArray[np.int64],
    mask: npt.NDArray[np.int64],
    type_ids: npt.NDArray[np.int64],
    input_names: set[str],
) -> dict[str, Any]:
    """Inputs for one batch, matched to what the graph actually declares.

    The BERT-family rerankers separate query from passage with token_type_ids and rank
    differently without them; the XLM-R-family ones have no such input at all. Feeding
    the tokenizer's real type ids (never zeros) keeps both correct.
    """
    feed: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = type_ids
    return feed


class CrossEncoder:
    """A loaded ONNX cross encoder. One per process — sessions are not cheap to build."""

    def __init__(self, spec: CrossSpec, threads: int = 8) -> None:
        self.spec = spec
        onnx, tok_path = ensure_reranker(spec)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx, opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=spec.max_length)
        self.tokenizer.enable_padding()

    def score(self, query: str, texts: Sequence[str], *, batch_size: int = BATCH_SIZE) -> Floats:
        """Relevance logit per (query, text) pair, in the input order."""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        out = np.zeros(len(texts), dtype=np.float32)
        # Length-sorted batches: padding is per batch, and chunk lengths vary enough that
        # sorting removes most of the wasted compute. Order is restored before returning.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            encoded = self.tokenizer.encode_batch([(query, texts[i]) for i in idx])
            feed = build_feed(
                np.array([e.ids for e in encoded], dtype=np.int64),
                np.array([e.attention_mask for e in encoded], dtype=np.int64),
                np.array([e.type_ids for e in encoded], dtype=np.int64),
                self.input_names,
            )
            logits = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
            out[idx] = logits.reshape(len(idx), -1)[:, 0]
        return out


def union(*rankings: Sequence[str]) -> list[str]:
    """Every candidate any arm returned, first-seen order, no duplicates."""
    return list(dict.fromkeys(cid for ranking in rankings for cid in ranking))


def top_k(ids: Sequence[str], scores: Sequence[float] | Floats, k: int) -> list[str]:
    """The k highest-scoring ids, best first, ties broken by candidate order."""
    ranked = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i))
    return [ids[i] for i in ranked[:k]]
