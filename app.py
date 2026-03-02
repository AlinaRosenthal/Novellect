import streamlit as st
import os
import uuid
from converter import process_file
from search_engine import add_to_index, search_keywords

# Настройки страницы
st.set_page_config(page_title="Novellect Proto 1", page_icon="📚")

st.title("📚 Novellect: Прототип №1")
st.markdown("Локальная библиотека книг с поиском по ключевым словам.")

# Создание папки для загрузок
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Боковая панель для загрузки
st.sidebar.header("Загрузка книг")
uploaded_files = st.sidebar.file_uploader("Выберите файлы (.txt, .fb2)", type=['txt', 'fb2'],
                                          accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Сохранение файла
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{uploaded_file.name}")

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Обработка и индексация
        with st.spinner(f"Обработка {uploaded_file.name}..."):
            book_data = process_file(file_path)

            if book_data.get('error'):
                st.sidebar.error(f"Ошибка в {uploaded_file.name}: {book_data['error']}")
            else:
                add_to_index(book_data, file_id)
                st.sidebar.success(f"Добавлено: {book_data['title']}")

# Основная часть: Поиск
st.header("🔍 Поиск по библиотеке")
query = st.text_input("Введите ключевое слово или фразу:")

if query:
    results = search_keywords(query)
    st.subheader(f"Результаты ({len(results)})")

    if not results:
        st.info("Ничего не найдено.")
    else:
        for res in results:
            with st.expander(f"📖 {res['title']} ({res['format']})"):
                st.write(res['snippet'])
                st.caption("Найдено совпадение по запросу.")

# Информация о системе
st.markdown("---")
st.caption("Novellect Prototype 1 | Февраль 2024 | Поиск по ключевым словам + Конвертация TXT/FB2")