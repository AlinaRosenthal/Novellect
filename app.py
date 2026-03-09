import streamlit as st
import os
import uuid
import time
from converter import process_file
from search_engine import add_to_index, load_index, search_hybrid

st.set_page_config(page_title="Novellect Proto 2", page_icon="📚")
st.title("📚 Novellect: Прототип №2")
st.markdown("Локальная библиотека книг с гибридным поиском.")

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

with st.sidebar:
    st.header("📤 Загрузка книг")
    uploaded_files = st.file_uploader(
        "Выберите файлы (.txt, .fb2, .pdf, .epub)", 
        type=['txt', 'fb2', 'pdf', 'epub'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Начать индексацию"):
        index = load_index()
        existing_titles = [book['title'] for book in index]
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
            
            for i, uploaded_file in enumerate(files_to_process):
                status_text.text(f"Обработка: {uploaded_file.name}")
                
                file_id = str(uuid.uuid4())
                file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{uploaded_file.name}")
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                book_data = process_file(file_path)
                
                if not book_data.get('error'):
                    add_to_index(book_data, file_id)
                else:
                    st.error(f"Ошибка в {uploaded_file.name}: {book_data['error']}")
                
                progress_bar.progress((i + 1) / len(files_to_process))
            
            status_text.text("✅ Готово!")
            time.sleep(1.5)
            st.rerun()

    st.header("📊 Статистика")
    index = load_index()
    st.metric("Всего книг", len(index))
    
    with st.expander("📚 Список книг"):
        if index:
            for book in index:
                st.write(f"• {book['title']} ({book.get('chunks_count', 0)} чанков)")
        else:
            st.write("Библиотека пуста")
    
    if st.button("🗑️ Очистить индекс"):
        if os.path.exists('storage.json'): os.remove('storage.json')
        if os.path.exists('vector_db.npz'): os.remove('vector_db.npz')
        st.success("Индекс очищен")
        time.sleep(1)
        st.rerun()

st.header("🔍 Поиск по библиотеке")

with st.form("search_form"):
    query = st.text_input("Введите поисковый запрос:", placeholder = "Например: искусственный интеллект...")
    alpha = st.slider("Баланс: Ключевые слова (0.0) --- Семантика (1.0)", 0.0, 1.0, 0.7)
    submit_button = st.form_submit_button("Найти")

if submit_button and query:
    start_time = time.time()
    with st.spinner("Поиск..."):
        results = search_hybrid(query, top_k=3, alpha=alpha)
    end_time = time.time()
    
    st.write(f"⏱️ Время выполнения: {(end_time - start_time):.3f} сек")
    
    if not results:
        st.info("Ничего не найдено.")
    else:
        for i, res in enumerate(results):
            label = f"📖 {res['title']} ({res['format']}) — {res['similarity']:.2f}"
            with st.expander(label):
                st.write(res['snippet'])
                st.caption(f"Sem: {res['sem_score']:.2f} | Key: {res['bm25_score']:.2f}")

st.markdown("---")
st.caption("Novellect Prototype 2 | 2026")