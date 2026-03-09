from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chardet
import PyPDF2
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT


WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def compute_file_hash(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_encoding(file_path: str) -> str:
    with open(file_path, "rb") as file_obj:
        raw = file_obj.read(200_000)
    detected = chardet.detect(raw)
    return detected.get("encoding") or "utf-8"


def read_txt(file_path: str) -> Dict[str, Any]:
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding, errors="ignore") as file_obj:
            text = file_obj.read()
        title = Path(file_path).stem
        return {
            "title": title,
            "author": None,
            "content": normalize_text(text),
            "format": "txt",
        }
    except Exception as exc:  # pragma: no cover - defensive branch
        return {
            "title": Path(file_path).name,
            "author": None,
            "content": "",
            "format": "txt",
            "error": str(exc),
        }


def read_fb2(file_path: str) -> Dict[str, Any]:
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding, errors="ignore") as file_obj:
            xml_content = file_obj.read()
        soup = BeautifulSoup(xml_content, "xml")

        title = Path(file_path).stem
        title_info = soup.find("title-info")
        author = None
        if title_info:
            title_tag = title_info.find("book-title")
            if title_tag and title_tag.get_text(strip=True):
                title = title_tag.get_text(strip=True)
            author_tag = title_info.find("author")
            if author_tag:
                first = author_tag.find("first-name")
                last = author_tag.find("last-name")
                author = " ".join(
                    part.get_text(strip=True)
                    for part in (first, last)
                    if part and part.get_text(strip=True)
                ).strip() or None

        body = soup.find("body")
        text_parts: List[str] = []
        if body:
            for tag in body.find_all(["title", "subtitle", "p", "poem", "stanza", "v"]):
                content = tag.get_text(" ", strip=True)
                if content:
                    text_parts.append(content)

        return {
            "title": title,
            "author": author,
            "content": normalize_text("\n\n".join(text_parts)),
            "format": "fb2",
        }
    except Exception as exc:  # pragma: no cover - defensive branch
        return {
            "title": Path(file_path).name,
            "author": None,
            "content": "",
            "format": "fb2",
            "error": str(exc),
        }


def read_pdf(file_path: str) -> Dict[str, Any]:
    try:
        title = Path(file_path).stem
        author = None
        pages: List[str] = []
        with open(file_path, "rb") as file_obj:
            reader = PyPDF2.PdfReader(file_obj)
            if reader.metadata:
                meta_title = reader.metadata.get("/Title")
                meta_author = reader.metadata.get("/Author")
                if isinstance(meta_title, str) and meta_title.strip():
                    title = meta_title.strip()
                if isinstance(meta_author, str) and meta_author.strip():
                    author = meta_author.strip()
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                if page_text.strip():
                    pages.append(page_text)
        return {
            "title": title,
            "author": author,
            "content": normalize_text("\n\n".join(pages)),
            "format": "pdf",
        }
    except Exception as exc:  # pragma: no cover - defensive branch
        return {
            "title": Path(file_path).name,
            "author": None,
            "content": "",
            "format": "pdf",
            "error": str(exc),
        }


def read_epub(file_path: str) -> Dict[str, Any]:
    try:
        book = epub.read_epub(file_path)
        title = Path(file_path).stem
        author = None

        title_meta = book.get_metadata("DC", "title")
        if title_meta and title_meta[0] and title_meta[0][0]:
            title = str(title_meta[0][0]).strip()

        author_meta = book.get_metadata("DC", "creator")
        if author_meta and author_meta[0] and author_meta[0][0]:
            author = str(author_meta[0][0]).strip()

        parts: List[str] = []
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(" ", strip=True)
                if text:
                    parts.append(text)

        return {
            "title": title,
            "author": author,
            "content": normalize_text("\n\n".join(parts)),
            "format": "epub",
        }
    except Exception as exc:  # pragma: no cover - defensive branch
        return {
            "title": Path(file_path).name,
            "author": None,
            "content": "",
            "format": "epub",
            "error": str(exc),
        }


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    sentences: List[str] = []
    for paragraph in paragraphs:
        candidates = SENTENCE_RE.split(paragraph)
        cleaned_candidates = [candidate.strip() for candidate in candidates if candidate.strip()]
        if cleaned_candidates:
            sentences.extend(cleaned_candidates)
        else:
            sentences.append(paragraph)
    return sentences


def build_chunks(
    text: str,
    *,
    target_chars: int = 1400,
    overlap_sentences: int = 1,
    min_chars: int = 250,
) -> List[Dict[str, Any]]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[Dict[str, Any]] = []
    current: List[str] = []
    chunk_start = 0
    running_char_pos = 0
    sentence_offsets: List[int] = []
    for sentence in sentences:
        sentence_offsets.append(running_char_pos)
        running_char_pos += len(sentence) + 1

    def flush(chunk_sentences: List[str], start_idx: int, end_idx: int) -> None:
        joined = " ".join(chunk_sentences).strip()
        if not joined:
            return
        if chunks and len(joined) < min_chars:
            chunks[-1]["text"] = f"{chunks[-1]['text']} {joined}".strip()
            chunks[-1]["char_end"] = sentence_offsets[end_idx] + len(sentences[end_idx])
            chunks[-1]["token_count"] = len(chunks[-1]["text"].split())
            return
        chunks.append(
            {
                "chunk_index": len(chunks),
                "text": joined,
                "char_start": sentence_offsets[start_idx],
                "char_end": sentence_offsets[end_idx] + len(sentences[end_idx]),
                "token_count": len(joined.split()),
            }
        )

    start_idx = 0
    for idx, sentence in enumerate(sentences):
        if not current:
            start_idx = idx
        current.append(sentence)
        current_text = " ".join(current)
        is_last = idx == len(sentences) - 1
        if len(current_text) >= target_chars or is_last:
            flush(current, start_idx, idx)
            if not is_last:
                overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
                current = overlap[:]
                start_idx = max(idx - len(overlap) + 1, 0) if overlap else idx + 1
            else:
                current = []

    return chunks


READERS = {
    ".txt": read_txt,
    ".fb2": read_fb2,
    ".pdf": read_pdf,
    ".epub": read_epub,
}


def process_file(file_path: str) -> Dict[str, Any]:
    ext = Path(file_path).suffix.lower()
    reader = READERS.get(ext)
    if not reader:
        return {
            "title": Path(file_path).name,
            "author": None,
            "content": "",
            "format": ext.lstrip("."),
            "error": f"Неподдерживаемый формат: {ext}",
        }
    book_data = reader(file_path)
    content = normalize_text(book_data.get("content", ""))
    book_data["content"] = content
    book_data["char_count"] = len(content)
    book_data["word_count"] = len(content.split()) if content else 0
    book_data["chunks"] = build_chunks(content)
    book_data["file_hash"] = compute_file_hash(file_path)
    return book_data
