import streamlit as st
import os
import uuid
import time
from converter import process_file
from search_engine import add_to_index, load_index, clear_cache
from agents import AgentOrchestrator
from fine_tune import fine_tune_model
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Novellect Agent System", page_icon="🤖", layout="wide")
st.title("🤖 Novellect: Мультиагентная система поиска книг")
st.markdown("Локальная библиотека книг с интеллектуальными агентами")

UPLOAD_DIR = "uploads"
TXT_CACHE_DIR = "txt_cache"

for dir_path in [UPLOAD_DIR, TXT_CACHE_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Инициализация оркестратора в сессии
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()

# Боковая панель
with st.sidebar:
    st.header("📊 Состояние системы")

    # Статистика
    index = load_index()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Всего книг", len(index))
    with col2:
        total_chunks = sum(book.get('chunks_count', 0) for book in index)
        st.metric("Всего чанков", total_chunks)

    # Информация об агентах
    with st.expander("🤖 Активные агенты", expanded=True):
        st.markdown("""
        - **QueryAnalyzerAgent**: Анализ запросов
        - **RetrievalAgent**: Поиск по RAG
        - **RankingAgent**: Ранжирование результатов
        - **ResponseAgent**: Форматирование ответов
        """)

    # Информация о кэше txt
    with st.expander("💾 Кэш TXT-версий"):
        if os.path.exists(TXT_CACHE_DIR):
            cache_files = os.listdir(TXT_CACHE_DIR)
            txt_files = [f for f in cache_files if f.endswith('.txt')]

            st.metric("Кэшированные книги", len(txt_files))

            # Размер кэша
            total_size = 0
            for f in cache_files:
                f_path = os.path.join(TXT_CACHE_DIR, f)
                if os.path.isfile(f_path):
                    total_size += os.path.getsize(f_path)

            st.metric("Размер кэша", f"{total_size / 1024 / 1024:.1f} MB")

            if st.button("🧹 Очистить кэш", use_container_width=True):
                for f in os.listdir(TXT_CACHE_DIR):
                    try:
                        os.remove(os.path.join(TXT_CACHE_DIR, f))
                    except:
                        pass
                st.success("Кэш очищен")
                time.sleep(1)
                st.rerun()
        else:
            st.write("Кэш пуст")

    # Управление
    st.header("🛠 Управление")

    # Загрузка книг
    with st.expander("📤 Загрузить книги", expanded=False):
        uploaded_files = st.file_uploader(
            "Выберите файлы (.txt, .fb2, .pdf, .epub)",
            type=['txt', 'fb2', 'pdf', 'epub'],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🚀 Начать индексацию", use_container_width=True):
            index = load_index()
            existing_titles = [book.get('title', '') for book in index]
            files_to_process = []

            for f in uploaded_files:
                title = os.path.splitext(f.name)[0]
                if title not in existing_titles:
                    files_to_process.append(f)
                else:
                    st.warning(f"Пропуск: {f.name} уже в базе")

            if files_to_process:
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()

                start_total = time.time()

                for i, uploaded_file in enumerate(files_to_process):
                    status_text.text(f"Обработка: {uploaded_file.name}")

                    file_id = str(uuid.uuid4())
                    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{uploaded_file.name}")

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Засекаем время на книгу
                    start_book = time.time()
                    book_data = process_file(file_path)

                    if not book_data.get('error'):
                        record = add_to_index(book_data, file_id)
                        elapsed_book = time.time() - start_book
                        time_text.text(f"⏱ Книга {i + 1}/{len(files_to_process)}: {elapsed_book:.1f} сек")
                    else:
                        st.error(f"Ошибка в {uploaded_file.name}: {book_data['error']}")

                    progress_bar.progress((i + 1) / len(files_to_process))

                total_elapsed = time.time() - start_total
                status_text.text(f"✅ Готово! Всего: {total_elapsed:.1f} сек")
                time.sleep(1.5)
                st.rerun()

    # Дообучение модели
    with st.expander("🧠 Дообучение модели", expanded=False):
        st.markdown("""
        Дообучите модель на ваших книгах для лучшего понимания литературных контекстов.
        """)

        if st.button("🚀 Запустить дообучение", use_container_width=True):
            with st.spinner("Дообучение модели... Это может занять несколько минут"):
                try:
                    success = fine_tune_model(epochs=1, batch_size=2)
                    if success:
                        st.success("✅ Модель успешно дообучена!")
                    else:
                        st.error("❌ Ошибка дообучения. Нужно больше книг в библиотеке.")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")

    # Очистка
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Очистить кэш поиска", use_container_width=True):
            clear_cache()
            st.success("Кэш поиска очищен")

    with col2:
        if st.button("🗑 Очистить всё", use_container_width=True):
            for f in ['storage.json', 'vector_db.npz', 'search_cache.pkl']:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
            st.success("Все данные очищены")
            time.sleep(1)
            st.rerun()

    # Список книг
    with st.expander("📚 Библиотека"):
        if index:
            # Безопасная сортировка по last_opened
            def get_sort_key(book):
                last = book.get('last_opened')
                return last if last is not None else 0


            sorted_books = sorted(index, key=get_sort_key, reverse=True)

            for book in sorted_books:
                title = book.get('title', 'Без названия')
                last = book.get('last_opened')
                last_str = datetime.fromtimestamp(last).strftime('%d.%m.%Y') if last else 'никогда'
                opens = book.get('open_count', 0)

                # ========== НОВОЕ: Показываем настроение книги ==========
                mood_info = ""
                if 'mood' in book:
                    top_moods = sorted(book['mood'].items(), key=lambda x: x[1], reverse=True)[:2]
                    top_moods = [f"{m}: {s:.0%}" for m, s in top_moods if s > 0.1]
                    if top_moods:
                        mood_info = f" | {', '.join(top_moods)}"

                st.write(f"• {title} ({opens} откр, посл: {last_str}){mood_info}")
        else:
            st.write("Библиотека пуста")

# Основная область - поиск
st.header("🔍 Поиск по библиотеке")

# История запросов в сессии
if 'history' not in st.session_state:
    st.session_state.history = []

# Форма поиска
with st.form("search_form"):
    query = st.text_input(
        "Введите запрос:",
        placeholder="Например: хочу прочитать про драконов или в какой книге главный герой - неймарек?",
        key="query_input"
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        use_agents = st.checkbox("🤖 Использовать агентов", value=True,
                                 help="Агенты лучше обрабатывают сложные и неопределенные запросы")
    with col2:
        update_stats = st.checkbox("📊 Учитывать открытия", value=True)
    with col3:
        submitted = st.form_submit_button("🔍 Найти", use_container_width=True)

if submitted and query:
    start_time = time.time()

    with st.spinner("Агенты анализируют запрос..."):
        try:
            if use_agents:
                # Используем мультиагентную систему
                response = st.session_state.orchestrator.process_query(query, update_opened=update_stats)
            else:
                # Используем обычный поиск
                from search_engine import search_hybrid

                results = search_hybrid(query, top_k=5)
                response = {
                    'type': 'simple',
                    'query': query,
                    'results': [{'title': r.get('title', ''),
                                 'snippet': r.get('snippet', ''),
                                 'relevance': r.get('similarity', 0)}
                                for r in results]
                }

            elapsed = time.time() - start_time

            # Сохраняем в историю
            st.session_state.history.append({
                'query': query,
                'time': elapsed,
                'type': response.get('type', 'simple')
            })

            # Отображаем результаты
            st.markdown(f"⏱️ Время выполнения: **{elapsed:.2f} сек**")

            if response['type'] == 'empty':
                st.info(response.get('message', 'Ничего не найдено'))

            elif response['type'] == 'vague':
                st.success(f"📚 Найдены рекомендации по запросу: _{response.get('query', '')}_")
                if response.get('genre'):
                    st.caption(f"Определенный жанр: {response['genre']}")
                if response.get('mood'):
                    st.caption(f"🎭 Настроение: {response['mood']}")

                for rec in response.get('recommendations', []):
                    with st.expander(f"📖 {rec.get('title', '')} (релевантность: {rec.get('relevance_score', 0):.2f})"):
                        last_opened = rec.get('last_opened')
                        if last_opened:
                            last_date = datetime.fromtimestamp(last_opened).strftime('%d.%m.%Y %H:%M')
                            st.caption(f"📅 Последнее открытие: {last_date} | Открытий: {rec.get('open_count', 0)}")
                        else:
                            st.caption("📅 Книга еще не открывалась")

                        # ========== НОВОЕ: Показываем соответствие настроению ==========
                        if rec.get('mood_match'):
                            st.caption(rec['mood_match'])

                        for i, snippet in enumerate(rec.get('snippets', [])[:2], 1):
                            st.markdown(f"**Отрывок {i}:**")
                            st.markdown(f"> {snippet}")
                            st.markdown("---")

            elif response['type'] == 'specific':
                st.success(f"🔎 Точные результаты по запросу: _{response.get('query', '')}_")

                for answer in response.get('answers', []):
                    with st.expander(
                            f"📖 {answer.get('book_title', '')} — релевантность: {answer.get('relevance', 0):.2f}"):
                        if answer.get('exact_match'):
                            st.markdown("✅ **Высокая точность совпадения!**")
                        st.markdown(f"> {answer.get('snippet', '')}")

            elif response['type'] == 'general':
                st.info(f"📋 Результаты по запросу: _{response.get('query', '')}_")
                st.caption(f"Ключевые слова: {', '.join(response.get('keywords', []))}")

                for result in response.get('results', []):
                    with st.expander(f"📖 {result.get('title', '')} — {result.get('relevance', 0):.2f}"):
                        st.markdown(f"> {result.get('snippet', '')}")

            else:  # simple search
                st.info(f"📋 Результаты обычного поиска")
                for result in response.get('results', []):
                    with st.expander(f"📖 {result.get('title', '')}"):
                        st.markdown(f"> {result.get('snippet', '')}")
                        st.caption(f"Релевантность: {result.get('relevance', 0):.2f}")

        except Exception as e:
            st.error(f"❌ Ошибка при обработке запроса: {e}")

# История запросов
if st.session_state.history:
    with st.expander("📋 История запросов"):
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            st.text(f"{i + 1}. '{h['query']}' - {h['type']} ({h['time']:.2f} сек)")

# Информация о системе
with st.expander("ℹ️ О системе"):
    st.markdown("""
    ### Novellect: Мультиагентная система поиска книг

    **Агенты:**
    - **QueryAnalyzerAgent**: Анализирует запрос, определяет тип (vague/specific/general) и настроение
    - **RetrievalAgent**: Выполняет гибридный поиск (семантика + BM25)
    - **RankingAgent**: Специализированное ранжирование для разных типов запросов и настроений
    - **ResponseAgent**: Форматирует ответ в зависимости от типа запроса

    **Новые возможности:**
    - 🎭 **Анализ настроения** книг при индексации
    - 🎯 **Ранжирование по настроению** для запросов типа "хочу юморную историю"
    - 📊 Отображение "профиля настроения" каждой книги

    **Оптимизации:**
    - Кэширование TXT-версий для быстрой повторной индексации
    - Кэширование частых запросов
    - Batch processing для ускорения
    - Безопасная обработка всех типов данных
    """)

st.markdown("---")
st.caption("Novellect Agent System | 2026 | Мультиагентная система с анализом настроения")
