from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import streamlit as st

import converter
import search_engine
import storage
from agents import MultiAgentBookAssistant
from local_llm import OllamaClient


storage.init_db()
st.set_page_config(
    page_title="Novellect Applied",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

assistant = MultiAgentBookAssistant()
ollama_client = OllamaClient()

EXAMPLE_QUERIES = [
    "хочу почитать про драконов",
    'найди точную фразу "неймарец"',
    "найди книгу в жанре фэнтези, где главный герой — дракон, и в сюжете есть предательство родителями",
    "в какой книге есть сильная тема одиночества и взросления",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")



def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Zа-яА-Я0-9._-]+", "_", name)
    return safe.strip("._") or f"book_{uuid.uuid4().hex[:8]}"



def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()



def read_markdown(relative_path: str) -> str:
    path = Path(__file__).resolve().parent / relative_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")



def rebuild_dense_index(show_message: bool = True) -> Dict[str, Any]:
    result = search_engine.rebuild_vector_store()
    if show_message:
        if result.get("dense_available"):
            st.success(result["message"])
        else:
            st.info(result["message"])
    return result



def ingest_uploaded_files(uploaded_files: List[Any]) -> Dict[str, Any]:
    added = 0
    skipped = 0
    failed = 0
    messages: List[str] = []
    progress = st.progress(0.0, text="Подготовка файлов...")
    total = max(len(uploaded_files), 1)

    for index, uploaded in enumerate(uploaded_files, start=1):
        payload = uploaded.getvalue()
        file_hash = hash_bytes(payload)
        duplicate = storage.get_book_by_hash(file_hash)
        if duplicate:
            skipped += 1
            messages.append(f"⚠️ {uploaded.name}: такой файл уже есть в библиотеке ({duplicate['title']}).")
            progress.progress(index / total, text=f"Пропуск дубликата: {uploaded.name}")
            continue

        book_id = uuid.uuid4().hex
        file_name = sanitize_filename(uploaded.name)
        file_path = storage.UPLOADS_DIR / f"{book_id}_{file_name}"
        with open(file_path, "wb") as file_obj:
            file_obj.write(payload)

        progress.progress(index / total, text=f"Обработка: {uploaded.name}")
        book_data = converter.process_file(str(file_path))
        if book_data.get("error"):
            failed += 1
            messages.append(f"❌ {uploaded.name}: {book_data['error']}")
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        if not book_data.get("content"):
            failed += 1
            messages.append(f"❌ {uploaded.name}: не удалось извлечь текст из файла.")
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                pass
            continue

        text_path = storage.CACHE_DIR / f"{book_id}.txt"
        text_path.write_text(book_data["content"], encoding="utf-8")

        storage.insert_book(
            {
                "id": book_id,
                "file_hash": file_hash,
                "title": book_data.get("title") or Path(uploaded.name).stem,
                "author": book_data.get("author"),
                "format": book_data.get("format"),
                "file_path": str(file_path),
                "text_path": str(text_path),
                "added_at": now_iso(),
                "last_opened_at": None,
                "open_count": 0,
                "char_count": int(book_data.get("char_count", 0)),
                "word_count": int(book_data.get("word_count", 0)),
            }
        )
        storage.replace_chunks(book_id, book_data.get("chunks", []))
        added += 1
        messages.append(
            f"✅ {book_data.get('title')}: добавлено {len(book_data.get('chunks', []))} чанков, {book_data.get('word_count', 0)} слов."
        )

    progress.progress(1.0, text="Загрузка завершена")
    rebuild_info = rebuild_dense_index(show_message=False)
    return {
        "added": added,
        "skipped": skipped,
        "failed": failed,
        "messages": messages,
        "rebuild_info": rebuild_info,
    }



def rerun_last_search() -> None:
    query = st.session_state.get("last_query")
    if not query:
        return
    result = assistant.run(
        query,
        top_k=st.session_state.get("top_k", 8),
        use_llm=st.session_state.get("use_llm", False) and not st.session_state.get("lite_mode", False),
        llm_model=st.session_state.get("llm_model", "qwen2.5:1.5b-instruct"),
    )
    st.session_state["last_result"] = result



def render_result_card(book: Dict[str, Any], discovery_mode: bool) -> None:
    title_line = book["title"]
    if book.get("author"):
        title_line += f" — {book['author']}"
    with st.container(border=True):
        cols = st.columns([4, 1, 1, 1, 1])
        cols[0].subheader(title_line)
        cols[1].metric("Score", f"{book['score']:.2f}")
        cols[2].metric("Формат", str(book.get("format") or "—").upper())
        cols[3].metric("Открытия", int(book.get("open_count") or 0))
        last_opened = book.get("last_opened_at")
        cols[4].caption("Последнее открытие")
        cols[4].write(last_opened[:19].replace("T", " ") if last_opened else "ещё не открывалась")

        if discovery_mode:
            st.caption("Режим рекомендаций: книги отсортированы по давности последнего открытия.")

        button_cols = st.columns([1, 1, 6])
        if button_cols[0].button("Отметить как открытую", key=f"open_{book['book_id']}"):
            storage.mark_book_opened(book["book_id"], now_iso())
            rerun_last_search()
            st.rerun()
        if button_cols[1].button("Удалить книгу", key=f"delete_{book['book_id']}"):
            storage.delete_book(book["book_id"])
            rebuild_dense_index(show_message=False)
            rerun_last_search()
            st.success(f"Удалена книга: {book['title']}")
            st.rerun()

        for idx, chunk in enumerate(book.get("best_chunks", []), start=1):
            st.markdown(f"**Фрагмент {idx}.** {chunk['text'][:700]}{'...' if len(chunk['text']) > 700 else ''}")
            score_line = (
                f"dense={chunk['dense_score']:.2f} | keyword={chunk['keyword_score']:.2f} | "
                f"phrase={chunk['phrase_score']:.2f} | total={chunk['score']:.2f}"
            )
            st.caption(score_line)


st.title("📚 Novellect Applied")
st.markdown(
    "Локальная интеллектуальная система рекомендаций книг: гибридный поиск, мультиагентная обработка запросов, "
    "поддержка .txt/.fb2/.epub/.pdf и опциональная локальная LLM через Ollama."
)

with st.sidebar:
    st.header("⚙️ Режим работы")
    st.checkbox("Lite-режим для слабого железа", key="lite_mode", value=True)
    st.checkbox(
        "Использовать локальную LLM через Ollama",
        key="use_llm",
        value=False,
        disabled=st.session_state.get("lite_mode", True),
        help="В Lite-режиме генерация отключена. Поиск и рекомендации остаются полностью локальными.",
    )
    st.text_input("Модель Ollama", key="llm_model", value="qwen2.5:1.5b-instruct")
    st.slider("Сколько книг показывать", min_value=3, max_value=12, value=8, key="top_k")

    st.header("📥 Загрузка книг")
    uploaded_files = st.file_uploader(
        "Выберите файлы библиотеки",
        type=["txt", "fb2", "pdf", "epub"],
        accept_multiple_files=True,
        help="Поиск и рекомендации выполняются только по вашей локальной библиотеке.",
    )
    if uploaded_files and st.button("Добавить в библиотеку", type="primary"):
        result = ingest_uploaded_files(uploaded_files)
        if result["added"]:
            st.success(
                f"Добавлено: {result['added']}. Пропущено: {result['skipped']}. Ошибок: {result['failed']}"
            )
        else:
            st.warning(
                f"Новых книг не добавлено. Пропущено: {result['skipped']}. Ошибок: {result['failed']}"
            )
        for message in result["messages"]:
            st.write(message)
        st.info(result["rebuild_info"]["message"])

    if st.button("Переиндексировать библиотеку"):
        rebuild_dense_index(show_message=True)

    if st.button("Очистить всю библиотеку"):
        storage.clear_library()
        search_engine.clear_vector_store()
        st.session_state.pop("last_result", None)
        st.success("Локальная библиотека и индексы очищены.")
        st.rerun()

    st.header("📊 Статистика")
    stats = storage.get_stats()
    backend = search_engine.get_embedding_backend_status()
    st.metric("Книг", stats["books_count"])
    st.metric("Чанков", stats["chunks_count"])
    st.metric("Объем векторного индекса", f"{search_engine.vector_store_file_size_mb():.3f} MB")
    st.metric("RAM на dense-матрицу", f"{search_engine.estimate_dense_ram_mb():.3f} MB")
    if backend["available"]:
        st.success(f"Dense-модель: {backend['model_name']}")
    else:
        st.info("Dense-модель недоступна. Работает lexical-only fallback.")

    if st.button("Проверить Ollama"):
        ok, message = ollama_client.ping()
        if ok:
            st.success(message)
        else:
            st.warning(message)

search_tab, library_tab  = st.tabs(["🔎 Поиск и рекомендации", "📚 Библиотека"])

with search_tab:
    st.markdown("**Примеры запросов:**")
    sample_cols = st.columns(len(EXAMPLE_QUERIES))
    for idx, example in enumerate(EXAMPLE_QUERIES):
        if sample_cols[idx].button(example, key=f"example_{idx}"):
            st.session_state["query_input"] = example

    query = st.text_area(
        "Запрос к библиотеке",
        key="query_input",
        height=120,
        placeholder="Например: хочу почитать про драконов или найди книгу, где главный герой — дракон, а в сюжете есть предательство.",
    )

    run_search = st.button("Найти", type="primary", disabled=not bool(query.strip()))
    if run_search and query.strip():
        start = time.perf_counter()
        use_llm = st.session_state.get("use_llm", False) and not st.session_state.get("lite_mode", False)
        result = assistant.run(
            query.strip(),
            top_k=st.session_state.get("top_k", 8),
            use_llm=use_llm,
            llm_model=st.session_state.get("llm_model", "qwen2.5:1.5b-instruct"),
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        st.session_state["last_query"] = query.strip()
        st.session_state["last_result"] = result
        storage.log_query(
            query_text=query.strip(),
            intent=result["plan"]["intent"],
            executed_at=now_iso(),
            result_count=len(result["retrieval"].get("book_results", [])),
            latency_ms=elapsed_ms,
            used_llm=bool(result["response"].get("used_llm")),
            notes=result["plan"].get("reasoning", ""),
        )

    result = st.session_state.get("last_result")
    if result:
        response = result["response"]
        retrieval = result["retrieval"]
        plan = result["plan"]

        answer_box = st.container(border=True)
        with answer_box:
            st.markdown("### Итоговый ответ")
            st.write(response.get("answer", ""))
            if response.get("used_llm"):
                st.caption("Ответ сформирован локальной LLM через Ollama.")
            elif response.get("llm_error"):
                st.caption(f"LLM недоступна, поэтому использован детерминированный шаблонный ответ. Причина: {response['llm_error']}")
            else:
                st.caption("Ответ собран без генерации: только по найденным локальным фрагментам.")

        info_cols = st.columns(4)
        info_cols[0].metric("Intent", plan.get("intent", "—"))
        info_cols[1].metric("Найдено книг", len(retrieval.get("book_results", [])))
        info_cols[2].metric("Dense-поиск", "Да" if retrieval.get("dense_available") else "Нет")
        info_cols[3].metric("Lite-режим", "Да" if st.session_state.get("lite_mode", False) else "Нет")

        with st.expander("Показать работу агентов"):
            st.markdown("**Агент 1 — разбор запроса**")
            st.json(plan)
            st.markdown("**Агент 2 — гибридный retrieval**")
            preview = retrieval.get("book_results", [])[:3]
            st.json(
                {
                    "intent": retrieval.get("intent"),
                    "query": retrieval.get("query"),
                    "top_books_preview": [
                        {
                            "title": book["title"],
                            "score": round(book["score"], 3),
                            "last_opened_at": book.get("last_opened_at"),
                        }
                        for book in preview
                    ],
                }
            )
            st.markdown("**Агент 3 — формирование ответа**")
            st.json(response)

        st.markdown("### Подходящие книги")
        if not retrieval.get("book_results"):
            st.info("В библиотеке пока нет уверенных совпадений по этому запросу.")
        else:
            discovery_mode = plan.get("intent") == "discovery"
            for book in retrieval.get("book_results", []):
                render_result_card(book, discovery_mode=discovery_mode)
    else:
        st.info("Загрузите книги и задайте запрос. Система работает только по вашей локальной библиотеке.")

with library_tab:
    books = storage.list_books()
    if not books:
        st.info("Библиотека пока пуста. Добавьте .txt, .fb2, .epub или .pdf файлы через боковую панель.")
    else:
        st.markdown("### Все книги")
        table_rows = []
        for book in books:
            table_rows.append(
                {
                    "Название": book["title"],
                    "Автор": book.get("author") or "—",
                    "Формат": str(book.get("format") or "—").upper(),
                    "Чанков": book.get("chunks_count", 0),
                    "Слов": book.get("word_count", 0),
                    "Открытий": book.get("open_count", 0),
                    "Последнее открытие": (book.get("last_opened_at") or "—")[:19].replace("T", " "),
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.markdown("### Управление библиотекой")
        for book in books:
            with st.expander(f"{book['title']} ({str(book.get('format') or '—').upper()})"):
                st.write(f"**Автор:** {book.get('author') or 'не указан'}")
                st.write(f"**Чанков:** {book.get('chunks_count', 0)}")
                st.write(f"**Слов:** {book.get('word_count', 0)}")
                st.write(f"**Открытий:** {book.get('open_count', 0)}")
                st.write(
                    f"**Последнее открытие:** {(book.get('last_opened_at') or 'ещё не открывалась')[:19].replace('T', ' ')}"
                )
                cols = st.columns([1, 1, 5])
                if cols[0].button("Открыта", key=f"library_open_{book['id']}"):
                    storage.mark_book_opened(book["id"], now_iso())
                    st.rerun()
                if cols[1].button("Удалить", key=f"library_delete_{book['id']}"):
                    storage.delete_book(book["id"])
                    rebuild_dense_index(show_message=False)
                    st.rerun()

        st.markdown("### История запросов")
        query_history = storage.list_recent_queries(limit=30)
        if query_history:
            history_rows = []
            for item in query_history:
                history_rows.append(
                    {
                        "Запрос": item["query_text"],
                        "Intent": item.get("intent") or "—",
                        "Время": (item.get("executed_at") or "—")[:19].replace("T", " "),
                        "Результатов": item.get("result_count", 0),
                        "Latency, ms": item.get("latency_ms", 0),
                        "LLM": "Да" if item.get("used_llm") else "Нет",
                    }
                )
            st.dataframe(history_rows, use_container_width=True, hide_index=True)
