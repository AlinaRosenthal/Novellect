import streamlit as st
import os
import uuid
from converter import process_file
from search_engine import add_to_index, search_semantic, load_index

st.set_page_config(page_title="Novellect Proto 2", page_icon="📚")
st.title("📚 Novellect: Прототип №2")
st.markdown("Локальная библиотека книг с семантическим поиском.")

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

with st.sidebar:
    st.header("📤 Загрузка книг")
    uploaded_files = st.file_uploader(
        "Выберите файлы (.txt, .fb2, .pdf, .epub)", 
        type=['txt', 'fb2', 'pdf', 'epub'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        index = load_index()
        existing_titles = [book['title'] for book in index]
        new_files = []
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                title = os.path.splitext(uploaded_file.name)[0]
                if title not in existing_titles:
                    new_files.append(uploaded_file)
                    st.session_state.processed_files.add(uploaded_file.name)
                else:
                    st.warning(f"{uploaded_file.name} уже существует в библиотеке")
        
        if new_files:
            progress_bar = st.progress(0, text="Подготовка...")
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(new_files):
                status_text.text(f"Обработка: {uploaded_file.name}")
                progress_bar.progress((i + 1) / len(new_files))
                
                file_id = str(uuid.uuid4())
                file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{uploaded_file.name}")
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                book_data = process_file(file_path)
                
                if book_data.get('error'):
                    st.error(f"{uploaded_file.name}: {book_data['error']}")
                else:
                    result = add_to_index(book_data, file_id)
                    if not result:
                        st.warning(f"{uploaded_file.name}: не удалось добавить в индекс")
            
            progress_bar.progress(1.0, text="Завершено!")
            status_text.text(f"Обработано новых файлов: {len(new_files)}")
            st.rerun()
    
    st.header("📊 Статистика")
    index = load_index()
    st.metric("Всего книг", len(index))
    
    with st.expander("📚 Список книг"):
        if index:
            for book in index:
                st.write(f"• {book['title']}")
        else:
            st.write("Нет добавленных книг")
    
    if st.button("🗑️ Очистить индекс"):
        if os.path.exists('storage.json'):
            os.remove('storage.json')
        if os.path.exists('vector_db.npz'):
            os.remove('vector_db.npz')
        st.session_state.processed_files = set()
        st.success("Индекс очищен")
        st.rerun()

st.header("🔍 Поиск по библиотеке")

query = st.text_input("Введите поисковый запрос:", placeholder="Например: искусственный интеллект...")

if query:
    with st.spinner("Поиск..."):
        results = search_semantic(query, top_k=3)
    
    if not results:
        st.info("😕 Ничего не найдено. Попробуйте изменить запрос.")
    else:
        for i, res in enumerate(results):
            with st.expander(f"📖 {res['title']} ({res['format']}) - совпадение: {res['similarity']:.2f}"):
                st.write(res['snippet'])
                st.caption("Фрагмент текста с наибольшим совпадением")

st.markdown("---")
st.caption("Novellect Prototype 2 | Февраль 2024 | Семантический поиск по книгам")