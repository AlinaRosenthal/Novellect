import json
import os
import re
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

STORAGE_FILE = 'storage.json'
VECTOR_DB_FILE = 'vector_db.npz'

STOP_WORDS = {
    'про', 'книга', 'книгу', 'книги', 'автор', 'сюжет', 'прочитать', 'найти',
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 
    'все', 'она', 'так', 'из', 'за', 'вы', 'же', 'бы', 'по', 'только', 'ее', 
    'мне', 'было', 'вот', 'от', 'меня', 'еще', 'о', 'из', 'ему', 'теперь', 'когда',
    'быть', 'был', 'была', 'это', 'для', 'кто', 'дом', 'год', 'купить', 'читать'
}

_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        print("[INIT] Загрузка модели SentenceTransformer...")
        _model_cache = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model_cache

def load_index():
    if not os.path.exists(STORAGE_FILE): 
        return []
    with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_index(index):
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def tokenize_smart(text):
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]

def split_text_semantic(text, threshold=0.3):
    print("[PROCESS] Семантическое разбиение текста...")
    model = get_model()
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2: return sentences
    
    embeddings = model.encode(sentences, show_progress_bar=False)
    chunks, current_chunk = [], [sentences[0]]
    for i in range(len(sentences) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        if sim > threshold:
            current_chunk.append(sentences[i+1])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
    if current_chunk: chunks.append(" ".join(current_chunk))
    print(f"[INFO] Текст разбит на {len(chunks)} чанков.")
    return chunks

def load_vector_db():
    if not os.path.exists(VECTOR_DB_FILE): return None, None
    try:
        data = np.load(VECTOR_DB_FILE, allow_pickle=True)
        embeddings = data['embeddings']
        metadata = json.loads(data['metadata'].item()) if data['metadata'].size > 0 else []
        return embeddings, metadata
    except: return None, None

def add_to_index(book_data, file_id):
    title = book_data.get('title', 'Без названия')
    print(f"[START] Добавление книги в индекс: {title}")
    
    index = load_index()
    index = [b for b in index if b['id'] != file_id]
    model = get_model()
    
    content = book_data.get('content', "")
    chunks = book_data.get('chunks') or split_text_semantic(content)
    if not chunks: 
        print("[ERROR] Не удалось создать чанки.")
        return None
    
    print(f"[PROCESS] Генерация эмбеддингов для {len(chunks)} чанков...")
    new_embeddings = model.encode(chunks, show_progress_bar=True)
    
    print("[PROCESS] Обновление векторной базы данных...")
    old_embeddings, old_metadata = load_vector_db()
    
    new_meta = [{
        "book_id": file_id,
        "book_title": title,
        "format": book_data.get('format', 'unknown'),
        "chunk": c
    } for c in chunks]
    
    if old_embeddings is not None and len(old_embeddings) > 0:
        mask = [m.get('book_id') != file_id for m in (old_metadata or [])]
        old_embeddings_filtered = old_embeddings[mask]
        old_metadata_filtered = [m for i, m in enumerate(old_metadata or []) if mask[i]]
        combined_embeddings = np.vstack([old_embeddings_filtered, new_embeddings]) if len(old_embeddings_filtered) > 0 else new_embeddings
        final_metadata = old_metadata_filtered + new_meta
    else:
        combined_embeddings, final_metadata = new_embeddings, new_meta
    
    np.savez(VECTOR_DB_FILE, embeddings=combined_embeddings, metadata=json.dumps(final_metadata))
    
    record = {
        "id": file_id, 
        "title": title, 
        "format": book_data.get('format', 'unknown'),
        "chunks_count": len(chunks)
    }
    index.append(record)
    save_index(index)
    print(f"[SUCCESS] Книга '{title}' успешно проиндексирована.")
    return record

class HybridSearch:
    def __init__(self):
        self.bm25 = None
        self.corpus_id = None

    def search(self, query, model, embeddings, metadata, top_k=5, alpha=0.7):
        print(f"[SEARCH] Запрос: '{query}'")
        
        start_time = time.time()
        q_emb = model.encode([query], show_progress_bar=False)
        sem_scores = cosine_similarity(q_emb, embeddings)[0]
        
        tokenized_query = tokenize_smart(query)
        current_corpus_id = hash(str(metadata))
        
        if self.bm25 is None or self.corpus_id != current_corpus_id:
            print("[SEARCH] Обновление индекса BM25...")
            corpus = [tokenize_smart(m.get('chunk', '')) for m in metadata]
            self.bm25 = BM25Okapi(corpus)
            self.corpus_id = current_corpus_id
        
        bm25_raw = self.bm25.get_scores(tokenized_query)
        max_b = np.max(bm25_raw)
        bm25_norm = bm25_raw / max_b if max_b > 0 else bm25_raw

        combined = (alpha * sem_scores) + ((1 - alpha) * bm25_norm)
        top_indices = np.argsort(combined)[-top_k:][::-1]
        
        duration = time.time() - start_time
        print(f"[SEARCH] Поиск завершен за {duration:.3f} сек.")
        
        results = []
        for idx in top_indices:
            results.append({
                "title": metadata[idx].get('book_title', 'Без названия'),
                "format": metadata[idx].get('format', 'unknown'),
                "snippet": metadata[idx].get('chunk', '')[:400] + "...",
                "similarity": float(combined[idx]),
                "sem_score": float(sem_scores[idx]),
                "bm25_score": float(bm25_norm[idx])
            })
        return results

_engine = HybridSearch()

def search_hybrid(query, top_k=5, alpha=0.7):
    emb, meta = load_vector_db()
    if emb is None or not meta: 
        print("[WARNING] База данных пуста.")
        return []
    return _engine.search(query, get_model(), emb, meta, top_k, alpha)