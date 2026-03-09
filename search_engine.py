from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import storage

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_FILE = BASE_DIR / "data" / "vector_store.npz"
TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
SPACE_RE = re.compile(r"\s+")


@dataclass
class EmbeddingBackend:
    name: str
    model_name: str
    dim: Optional[int]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self._model = None
        super().__init__(name="sentence-transformers", model_name=model_name, dim=384)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


_BACKEND: Optional[EmbeddingBackend] = None
_BACKEND_ERROR: Optional[str] = None


def get_embedding_backend() -> Optional[EmbeddingBackend]:
    global _BACKEND, _BACKEND_ERROR
    if _BACKEND is not None:
        return _BACKEND
    if _BACKEND_ERROR is not None:
        return None
    try:
        importlib.import_module('sentence_transformers')
        _BACKEND = SentenceTransformerBackend()
        return _BACKEND
    except Exception as exc:  # pragma: no cover - optional dependency path
        _BACKEND_ERROR = str(exc)
        return None


def get_embedding_backend_status() -> Dict[str, Any]:
    backend = get_embedding_backend()
    return {
        "available": backend is not None,
        "backend": backend.name if backend else "disabled",
        "model_name": backend.model_name if backend else None,
        "error": _BACKEND_ERROR,
    }


def normalize_for_search(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


RUSSIAN_SUFFIXES = [
    "иями", "ями", "ами", "иях", "ях", "ов", "ев", "ом", "ем", "ам", "ям", "ах", "ях",
    "ия", "ья", "ие", "ье", "ий", "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие",
    "ов", "ев", "ы", "и", "а", "я", "у", "ю", "е", "о",
]


def normalize_token(token: str) -> str:
    token = token.lower().replace("ё", "е")
    for suffix in RUSSIAN_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    return [normalize_token(token) for token in TOKEN_RE.findall(text or "")]


STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "под",
    "над",
    "не",
    "что",
    "как",
    "к",
    "ко",
    "из",
    "за",
    "у",
    "о",
    "об",
    "про",
    "для",
    "или",
    "где",
    "когда",
    "кто",
    "это",
    "эта",
    "этот",
    "хочу",
    "хоч",
    "книгу",
    "книга",
    "книг",
    "почитать",
    "почит",
    "найди",
    "найд",
    "подбери",
    "подбер",
    "посоветуй",
    "посовет",
}


def informative_tokens(text: str) -> List[str]:
    return [token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS]



def load_vector_store() -> Tuple[Optional[np.ndarray], List[int], Optional[str]]:
    if not VECTOR_STORE_FILE.exists():
        return None, [], None
    try:
        payload = np.load(VECTOR_STORE_FILE, allow_pickle=True)
        vectors = payload["vectors"]
        chunk_ids = payload["chunk_ids"].astype(np.int64).tolist()
        backend_name = None
        if "backend_name" in payload:
            backend_name = str(payload["backend_name"].item())
        return np.asarray(vectors, dtype=np.float32), chunk_ids, backend_name
    except Exception:
        return None, [], None



def save_vector_store(vectors: np.ndarray, chunk_ids: Sequence[int], backend_name: str) -> None:
    storage.ensure_data_dirs()
    np.savez_compressed(
        VECTOR_STORE_FILE,
        vectors=np.asarray(vectors, dtype=np.float32),
        chunk_ids=np.asarray(list(chunk_ids), dtype=np.int64),
        backend_name=np.array([backend_name], dtype=object),
    )



def clear_vector_store() -> None:
    if VECTOR_STORE_FILE.exists():
        VECTOR_STORE_FILE.unlink()



def rebuild_vector_store() -> Dict[str, Any]:
    chunks = storage.get_all_chunks()
    if not chunks:
        clear_vector_store()
        return {
            "indexed_chunks": 0,
            "dense_available": False,
            "backend": None,
            "message": "Библиотека пуста.",
        }

    backend = get_embedding_backend()
    if backend is None:
        clear_vector_store()
        return {
            "indexed_chunks": len(chunks),
            "dense_available": False,
            "backend": None,
            "message": "Dense-эмбеддинги не созданы: sentence-transformers недоступен.",
        }

    texts = [chunk["text"] for chunk in chunks]
    try:
        vectors = backend.encode(texts)
    except Exception as exc:
        clear_vector_store()
        return {
            "indexed_chunks": len(chunks),
            "dense_available": False,
            "backend": None,
            "message": f"Dense-эмбеддинги не созданы: {exc}",
        }
    chunk_ids = [int(chunk["chunk_id"]) for chunk in chunks]
    save_vector_store(vectors, chunk_ids, backend.name)
    return {
        "indexed_chunks": len(chunks),
        "dense_available": True,
        "backend": backend.name,
        "message": f"Индекс обновлен: {len(chunks)} чанков.",
    }



def cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = np.clip(matrix_norm * query_norm, 1e-8, None)
    scores = (matrix @ query_vector) / denom
    return scores.astype(np.float32)



def build_df(chunks: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for chunk in chunks:
        unique_tokens = set(informative_tokens(chunk["text"]))
        for token in unique_tokens:
            df[token] = df.get(token, 0) + 1
    return df



def keyword_score(query_tokens: Sequence[str], text_tokens: Sequence[str], df: Dict[str, int], total_docs: int) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    token_set = set(text_tokens)
    score = 0.0
    max_score = 0.0
    for token in query_tokens:
        inv_df = math.log((1 + total_docs) / (1 + df.get(token, 0))) + 1.0
        max_score += inv_df
        if token in token_set:
            score += inv_df
    if max_score == 0:
        return 0.0
    return min(score / max_score, 1.0)



def exact_phrase_bonus(query_text: str, chunk_text: str) -> float:
    normalized_query = normalize_for_search(query_text)
    normalized_chunk = normalize_for_search(chunk_text)
    if not normalized_query or len(normalized_query) < 3:
        return 0.0
    if normalized_query in normalized_chunk:
        return 1.0
    return 0.0



def quoted_phrase(query: str) -> Optional[str]:
    match = re.search(r"[\"“”«](.+?)[\"“”»]", query)
    return match.group(1).strip() if match else None



def _dense_score_map(query_text: str) -> Tuple[Dict[int, float], bool]:
    backend = get_embedding_backend()
    if backend is None:
        return {}, False

    vectors, chunk_ids, backend_name = load_vector_store()
    if vectors is None or not chunk_ids:
        return {}, False
    if backend_name and backend_name != backend.name:
        return {}, False

    try:
        query_vector = backend.encode([query_text])[0]
    except Exception:
        return {}, False
    scores = cosine_scores(query_vector, vectors)
    return {int(chunk_id): float(score) for chunk_id, score in zip(chunk_ids, scores)}, True



def _combine_scores(
    *,
    dense: float,
    keyword: float,
    phrase: float,
    intent: str,
) -> float:
    if intent == "exact_quote":
        return min(1.0, 0.7 * phrase + 0.2 * keyword + 0.1 * max(dense, 0.0))
    if intent == "discovery":
        return 0.45 * max(dense, 0.0) + 0.45 * keyword + 0.10 * phrase
    if intent == "complex":
        return 0.50 * max(dense, 0.0) + 0.35 * keyword + 0.15 * phrase
    return 0.55 * max(dense, 0.0) + 0.35 * keyword + 0.10 * phrase



def _chunk_matches_constraint(chunk_text: str, constraints: Sequence[str]) -> float:
    if not constraints:
        return 0.0
    normalized = normalize_for_search(chunk_text)
    matched = 0
    for constraint in constraints:
        if normalize_for_search(constraint) in normalized:
            matched += 1
    return matched / max(len(constraints), 1)



def hybrid_search(plan: Dict[str, Any], top_k: int = 8) -> Dict[str, Any]:
    query_text = plan.get("search_query") or plan.get("original_query") or ""
    intent = plan.get("intent", "semantic_lookup")
    chunks = storage.get_all_chunks()
    if not chunks:
        return {
            "query": query_text,
            "intent": intent,
            "dense_available": False,
            "chunk_results": [],
            "book_results": [],
        }

    q_phrase = plan.get("quoted_phrase") or quoted_phrase(query_text)
    keyword_tokens = plan.get("keywords") or informative_tokens(query_text)
    constraints = plan.get("constraints") or []
    total_docs = max(len(chunks), 1)
    df = build_df(chunks)
    dense_map, dense_available = _dense_score_map(query_text)

    chunk_results: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = chunk["text"]
        phrase_score = exact_phrase_bonus(q_phrase or query_text, text) if (q_phrase or query_text) else 0.0
        key_score = keyword_score(keyword_tokens, informative_tokens(text), df, total_docs)
        dense_score = dense_map.get(int(chunk["chunk_id"]), 0.0)
        constraint_score = _chunk_matches_constraint(text, constraints)
        combined = _combine_scores(
            dense=dense_score,
            keyword=key_score,
            phrase=max(phrase_score, constraint_score),
            intent=intent,
        )
        if intent == "complex" and constraints:
            combined = 0.75 * combined + 0.25 * constraint_score
        chunk_results.append(
            {
                **chunk,
                "dense_score": float(dense_score),
                "keyword_score": float(key_score),
                "phrase_score": float(phrase_score),
                "constraint_score": float(constraint_score),
                "score": float(combined),
            }
        )

    min_threshold = 0.16 if intent in {"semantic_lookup", "complex"} else 0.12
    if intent == "exact_quote":
        min_threshold = 0.10

    chunk_results = [item for item in chunk_results if item["score"] >= min_threshold or item["phrase_score"] > 0]
    chunk_results.sort(
        key=lambda item: (
            item["score"],
            item["phrase_score"],
            item["keyword_score"],
            item["dense_score"],
        ),
        reverse=True,
    )

    deduped_chunks: List[Dict[str, Any]] = []
    seen_texts = set()
    for item in chunk_results:
        normalized_text = normalize_for_search(item["text"][:300])
        if normalized_text in seen_texts:
            continue
        deduped_chunks.append(item)
        seen_texts.add(normalized_text)
        if len(deduped_chunks) >= max(top_k * 3, top_k):
            break

    grouped: Dict[str, Dict[str, Any]] = {}
    for item in deduped_chunks:
        book_id = item["book_id"]
        current = grouped.setdefault(
            book_id,
            {
                "book_id": book_id,
                "title": item["title"],
                "author": item.get("author"),
                "format": item.get("format"),
                "score": item["score"],
                "best_chunks": [],
                "last_opened_at": item.get("last_opened_at"),
                "open_count": int(item.get("open_count") or 0),
                "added_at": item.get("added_at"),
            },
        )
        current["score"] = max(current["score"], item["score"])
        if len(current["best_chunks"]) < 3:
            current["best_chunks"].append(item)

    book_results = list(grouped.values())
    if intent == "discovery":
        book_results.sort(
            key=lambda item: (
                item["last_opened_at"] is not None,
                item["last_opened_at"] or "",
                -item["score"],
                item["title"].lower(),
            )
        )
    else:
        book_results.sort(key=lambda item: (item["score"], item["title"].lower()), reverse=True)

    return {
        "query": query_text,
        "intent": intent,
        "dense_available": dense_available,
        "chunk_results": deduped_chunks[: max(top_k, 5)],
        "book_results": book_results[:top_k],
    }



def vector_store_file_size_mb() -> float:
    if not VECTOR_STORE_FILE.exists():
        return 0.0
    return round(VECTOR_STORE_FILE.stat().st_size / (1024 * 1024), 3)



def estimate_dense_ram_mb() -> float:
    vectors, _, _ = load_vector_store()
    if vectors is None:
        return 0.0
    return round(vectors.nbytes / (1024 * 1024), 3)
