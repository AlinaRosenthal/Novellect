import json
import os
import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

STORAGE_FILE = 'storage.json'
VECTOR_DB_FILE = 'vector_db.npz'

@st.cache_resource(show_spinner=False)
def get_model():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

def load_index():
    if not os.path.exists(STORAGE_FILE): 
        return []
    with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_index(index):
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def split_text_semantic(text, threshold=0.6):
    model = get_model()
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 2: 
        return sentences if sentences else []
    
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return []
    
    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(len(sentences) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        if sim > threshold:
            current_chunk.append(sentences[i+1])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
    
    if current_chunk: 
        chunks.append(" ".join(current_chunk))
    return chunks

def load_vector_db():
    if not os.path.exists(VECTOR_DB_FILE): 
        return None, None
    try:
        data = np.load(VECTOR_DB_FILE, allow_pickle=True)
        embeddings = data['embeddings']
        metadata = json.loads(data['metadata'].item()) if data['metadata'].size > 0 else []
        return embeddings, metadata
    except Exception:
        return None, None

def add_to_index(book_data, file_id):
    index = load_index()
    index = [b for b in index if b['id'] != file_id]
    model = get_model()
    
    content = book_data.get('content', "")
    
    if 'chunks' in book_data and book_data['chunks']:
        chunks = book_data['chunks']
    else:
        chunks = split_text_semantic(content)
    
    if not chunks:
        return None
    
    new_embeddings = model.encode(chunks)
    old_embeddings, old_metadata = load_vector_db()
    
    new_metadata_entries = []
    for i, chunk in enumerate(chunks):
        new_metadata_entries.append({
            "book_id": file_id,
            "book_title": book_data.get('title', 'Без названия'),
            "format": book_data.get('format', 'unknown'),
            "chunk_index": i,
            "chunk": chunk
        })
    
    if old_embeddings is not None and len(old_embeddings) > 0:
        mask = [m.get('book_id') != file_id for m in (old_metadata or [])]
        old_embeddings_filtered = old_embeddings[mask]
        old_metadata_filtered = [m for i, m in enumerate(old_metadata or []) if mask[i]]
        
        combined_embeddings = np.vstack([old_embeddings_filtered, new_embeddings]) if len(old_embeddings_filtered) > 0 else new_embeddings
        final_metadata = old_metadata_filtered + new_metadata_entries
    else:
        combined_embeddings = new_embeddings
        final_metadata = new_metadata_entries
    
    try:
        np.savez(VECTOR_DB_FILE, embeddings=combined_embeddings, metadata=json.dumps(final_metadata))
    except Exception:
        return None
    
    book_record = {
        "id": file_id, 
        "title": book_data.get('title', 'Без названия'), 
        "format": book_data.get('format', 'unknown'),
        "content": content[:500] + "..." if len(content) > 500 else content,
        "chunks_count": len(chunks)
    }
    index.append(book_record)
    save_index(index)
    
    return book_record

def search_semantic(query, top_k=5, min_similarity=0.2):
    embeddings, metadata = load_vector_db()
    
    if embeddings is None or len(embeddings) == 0 or not metadata:
        return []
    
    model = get_model()
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    seen_chunks = set()
    
    for idx in top_indices:
        if idx < len(metadata) and similarities[idx] > min_similarity:
            chunk_text = metadata[idx].get('chunk', '')
            if len(chunk_text) < 10 or chunk_text in seen_chunks:
                continue
                
            seen_chunks.add(chunk_text)
            results.append({
                "title": metadata[idx].get('book_title', 'Без названия'),
                "format": metadata[idx].get('format', 'unknown'),
                "snippet": chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                "similarity": float(similarities[idx])
            })
    
    return results