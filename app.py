import os
import time
import pandas as pd
import uuid
import streamlit as st
import storage
from agents import AgentOrchestrator
from converter import process_file
from search_engine import add_to_index, clear_cache
import hashlib
from datetime import datetime


st.set_page_config(page_title="Novellect Agent System", page_icon="🤖", layout="wide")
st.title("🤖 Novellect: Мультиагентная система поиска книг")
st.markdown("Локальная библиотека книг с интеллектуальными агентами")

UPLOAD_DIR = "uploads"
TXT_CACHE_DIR = "txt_cache"

# Создаем папки
for dir_path in [UPLOAD_DIR, TXT_CACHE_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("📊 Состояние системы")

    # 1. КОНТРОЛЬ ЛИМИТА 1 ГБ
    size_mb = storage.get_library_size() / (1024 * 1024)
    st.progress(min(size_mb / 1024, 1.0))
    st.caption(f"Занято: {size_mb:.2f} МБ из 1024 МБ (1 ГБ)")
    if size_mb > 900:
        st.warning("⚠️ Память почти заполнена!")

    # 2. СТАТИСТИКА
    index = storage.load_index()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Всего книг", len(index))
    with col2:
        total_chunks = sum(book.get('chunks_count', 0) for book in index)
        st.metric("Всего чанков", total_chunks)

    st.header("🛠 Управление")

    # 3. ЗАГРУЗКА КНИГ (Deduplication + 1GB Limit)
    with st.expander("📤 Загрузить книги", expanded=True):
        uploaded_files = st.file_uploader(
            "Выберите файлы (.txt, .fb2, .pdf, .epub)",
            type=['txt', 'fb2', 'pdf', 'epub'],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🚀 Начать индексацию", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                # 1. Читаем байты и считаем уникальный ХЕШ (SHA-256)
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.sha256(file_bytes).hexdigest()

                # 2. ПРОВЕРКА НА ДУБЛИКАТ (используем твою новую функцию из storage.py)
                duplicate = storage.get_book_by_hash(file_hash)
                if duplicate:
                    st.warning(
                        f"Пропуск: книга '{uploaded_file.name}' уже есть в библиотеке (как '{duplicate['title']}')")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    continue

                # 3. ПРОВЕРКА ЛИМИТА 1 ГБ
                if storage.is_limit_exceeded(uploaded_file.size):
                    st.error(f"Лимит 1 ГБ превышен! Файл {uploaded_file.name} пропущен.")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    continue

                # 4. Сохранение файла на диск
                status_text.text(f"Обработка: {uploaded_file.name}")
                file_id = str(uuid.uuid4())
                file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{uploaded_file.name}")

                with open(file_path, "wb") as f:
                    f.write(file_bytes)

                # 5. Конвертация (извлечение текста)
                book_data = process_file(file_path)

                # Проверка на пустой или битый файл (уже внутри process_file)
                if book_data.get('error'):
                    st.error(f"Ошибка в {uploaded_file.name}: {book_data['error']}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                else:
                    # ВАЖНО: Добавляем хеш в данные книги перед индексацией
                    book_data['file_hash'] = file_hash

                    # 6. Индексация (генерация векторов и запись в JSON)
                    add_to_index(book_data, file_id)

                progress_bar.progress((i + 1) / len(uploaded_files))

            st.success("Индексация успешно завершена!")
            time.sleep(1)
            st.rerun()

    # 4. ОЧИСТКА
    st.markdown("---")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑 Кэш поиска", use_container_width=True):
            clear_cache()  # Тот самый вызов
            st.success("Очищено")
    with col_c2:
        if st.button("🗑 Очистить всё", use_container_width=True):
            # 1. Физическое удаление файлов
            for f in ['storage.json', 'vector_db.npz', 'search_cache.pkl']:
                if os.path.exists(f): os.remove(f)

            # 2. Очистка папок
            for folder in [UPLOAD_DIR, TXT_CACHE_DIR]:
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        os.remove(os.path.join(folder, file))

            # 3. КРИТИЧЕСКИ ВАЖНО: Очистка памяти Streamlit
            st.session_state.clear()

            st.success("Система полностью обнулена")
            st.rerun()

    # 5. БИБЛИОТЕКА
    with st.expander("📚 Библиотека"):
        if index:
            # Сортировка: сначала те, что открывались давно (по ТЗ)
            sorted_books = sorted(index, key=lambda x: x.get('last_opened') or 0)

            for book in sorted_books:
                title = book.get('title', 'Без названия')
                last = book.get('last_opened')
                last_str = datetime.fromtimestamp(last).strftime('%d.%m %H:%M') if last else 'никогда'

                # Отображаем информацию о книге
                st.write(f"**{title}**")
                st.caption(f"📅 Открыто: {last_str}")

                # Кнопка удаления для конкретной книги
                # Используем уникальный ключ del_ + id книги, чтобы кнопки не конфликтовали
                if st.button(f"🗑 Удалить", key=f"del_{book['id']}", use_container_width=True):
                    storage.delete_book_physically(book['id'])  # Удаляем физически и из индекса
                    st.success(f"Удалено: {title}")
                    time.sleep(0.5)  # Даем пользователю увидеть сообщение
                    st.rerun()  # Перезагружаем интерфейс, чтобы обновить список и индикатор памяти

                st.markdown("---")  # Разделительная черта между книгами
        else:
            st.write("Библиотека пуста")

# --- ПОИСК ---
st.header("🔍 Поиск по библиотеке")

if 'history' not in st.session_state:
    st.session_state.history = []

with st.form("search_form"):
    query = st.text_input("Введите запрос:", placeholder="Например: хочу прочитать про драконов...")
    submitted = st.form_submit_button("🔍 Найти", use_container_width=True)

if submitted and query:
    start_time = time.time()
    with st.spinner("🤖 Группа агентов анализирует ваш запрос..."):
        try:
            # 1. ЗАПУСК ОРКЕСТРАТОРА
            response = st.session_state.orchestrator.process_query(query)
            elapsed = time.time() - start_time

            # --- ВИЗУАЛИЗАЦИЯ РАБОТЫ АГЕНТОВ ---
            st.markdown("### 🧠 Ход мыслей системы")

            # Показываем работу первого агента (Analyzer)
            with st.status("🕵️ QueryAnalyzerAgent закончил анализ", expanded=False):
                st.write(f"**Тип запроса:** {response.get('type', 'определяется')}")
                if 'genre' in response and response['genre']:
                    st.write(f"**Выделенный жанр:** {response['genre']}")
                st.write("Агент определил намерения пользователя и выбрал стратегию поиска.")

            # Показываем работу второго и третьего (Retrieval & Ranking)
            with st.status("📊 Retrieval & Ranking агенты отобрали лучшее", expanded=False):
                st.write("Сравнение семантических векторов и ключевых слов завершено.")
                st.write("Результаты отранжированы с учетом ваших предпочтений и истории открытий.")

            # ВЫВОД ФИНАЛЬНОГО ОТВЕТА (Response Agent)
            st.markdown("---")
            st.subheader("📝 Ответ Response-агента")

            if response['type'] == 'empty':
                st.info("Агенты не нашли подходящих материалов в вашей локальной библиотеке.")

            elif response['type'] == 'vague':
                st.success(f"🤖 Найдено по вашему запросу:")
                for rec in response.get('recommendations', []):
                    # Делаем заголовок красивым (без ID файла)
                    clean_title = rec['title']
                    with st.expander(f"📖 {clean_title}"):
                        st.write(rec['snippets'][0] if rec['snippets'] else "Нет доступного фрагмента")
                        st.caption(f"Релевантность: {rec.get('relevance_score', 0):.2f}")

            else:
                results_list = response.get('results', response.get('answers', []))
                for res in results_list:
                    title = res.get('title') or res.get('book_title') or "Книга"
                    with st.expander(f"📖 {title}"):
                        st.write(res.get('snippet', ''))
                        st.caption(f"Сходство: {res.get('relevance', 0):.2f}")

            st.write(f"⏱️ Суммарное время работы агентов: {elapsed:.2f} сек")

            st.session_state.history.append({
                'Запрос': query,
                'Время (сек)': round(elapsed, 2),
                'Тип': response.get('type', 'simple')
            })

            st.markdown("---")
            if st.session_state.history:
                with st.expander("📋 Журнал последних запросов", expanded=False):
                    # Превращаем историю в таблицу для презентабельности
                    df = pd.DataFrame(st.session_state.history).iloc[::-1]  # Последние запросы сверху
                    st.table(df.head(5))  # Показываем 5 последних запросов

                    if st.button("🗑 Очистить журнал", use_container_width=True):
                        st.session_state.history = []
                        st.rerun()

        except Exception as e:
            st.error(f"❌ Критическая ошибка агента: {e}")

# Информация о системе
with st.expander("ℹ️ О системе"):
    st.markdown("""
    **Novellect Ultimate (Proto 5)**
    - **SHA-256 Deduplication**: Умный контроль повторов.
    - **Local Inference**: Модели работают на CPU без API.
    - **1GB Quota**: Защита от переполнения памяти.
    - **Multi-Agent RAG**: Анализ намерений и настроений.
    """)