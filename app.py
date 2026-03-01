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
    books_count = len(set([m['book_id'] for m in meta])) if meta else 0
    st.metric("Всего книг", books_count)
    
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