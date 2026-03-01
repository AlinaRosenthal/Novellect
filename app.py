import streamlit as st
import os
import uuid
import time
from converter import process_file
from search_engine import NovellectEngine

st.set_page_config(page_title="Novellect Proto 2", page_icon="📚")

@st.cache_resource(show_spinner = False)
def get_engine():
    return NovellectEngine()

engine = get_engine()

st.title("📚 Novellect: Прототип №2")
st.markdown("Локальная библиотека книг с семантическим поиском.")

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

with st.sidebar:
    st.header("📤 Загрузка книг")
    uploaded_files = st.file_uploader("Выберите файлы", type=['txt', 'fb2', 'pdf', 'epub'], accept_multiple_files=True)
    
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.processed_files:
                file_id = str(uuid.uuid4())
                path = os.path.join("uploads", f"{file_id}_{f.name}")
                os.makedirs("uploads", exist_ok=True)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                
                data = process_file(path)
                if "error" not in data:
                    if engine.add_to_index(data, file_id):
                        st.session_state.processed_files.add(f.name)
        st.rerun()

    st.header("📊 Статистика")
    _, meta = engine.load_db()
    
    if meta:
        unique_books = {}
        for m in meta:
            book_id = m.get('book_id')
            if book_id not in unique_books:
                unique_books[book_id] = {
                    "title": m.get('book_title', 'Без названия'),
                    "format": m.get('format', 'unknown')
                }
        
        st.metric("Всего книг", len(unique_books))
        
        with st.expander("📚 Список книг"):
            for b_id, info in unique_books.items():
                st.write(f"• {info['title']} ({info['format'].upper()})")
    else:
        st.metric("Всего книг", 0)
        with st.expander("📚 Список книг"):
            st.write("Библиотека пуста")
    
    if st.button("🗑️ Очистить индекс"):
        if os.path.exists('vector_db.npz'): os.remove('vector_db.npz')
        st.session_state.processed_files = set()
        st.rerun()

st.header("🔍 Поиск по библиотеке")
query = st.text_input("Введите поисковый запрос:", placeholder="Например: искусственный интеллект...")

if query:
    start_time = time.time()
    
    with st.spinner("Поиск..."):
        
        results = engine.search(query, top_k=3)
    
    elapsed_time = time.time() - start_time
    
    if not results:
        st.info(f"😕 Ничего не найдено. (Поиск занял {elapsed_time:.2f} сек.)")
    else:
        st.caption(f"⏱️ Время поиска: {elapsed_time:.2f} сек.")
        for res in results:
            with st.expander(f"📖 {res['title']} ({res['format']}) - совпадение: {res['similarity']:.2f}"):
                st.write(res['snippet'])