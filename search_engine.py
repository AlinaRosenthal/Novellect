import json
import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

VECTORS_PATH = 'vector_db.npz'
STORAGE_PATH = 'storage.json'

class NovellectEngine:
    def __init__(self):
        self.model = self._load_model()
        
    @st.cache_resource(show_spinner=False)
    def _load_model(_self):
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def split_text(self, text, chunk_size=700):
        sentences = text.replace('\n', ' ').split('. ')
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) < chunk_size:
                current += s + ". "
            else:
                chunks.append(current.strip())
                current = s + ". "
        if current: chunks.append(current.strip())
        return chunks

    def add_to_index(self, book_data, file_id):
        content = book_data.get('content', "")
        chunks = self.split_text(content)
        if not chunks: return False
        
        embeddings = self.model.encode(chunks)
        old_vecs, old_meta = self.load_db()
        
        new_meta = [{
            "book_id": file_id,
            "book_title": book_data.get('title'),
            "format": book_data.get('format'),
            "chunk": c
        } for c in chunks]
        
        if old_vecs is not None:
            combined_vecs = np.vstack([old_vecs, embeddings])
            final_meta = old_meta + new_meta
        else:
            combined_vecs, final_meta = embeddings, new_meta
            
        np.savez(VECTORS_PATH, embeddings=combined_vecs, metadata=json.dumps(final_meta))
        return True

    def load_db(self):
        if not os.path.exists(VECTORS_PATH): return None, None
        data = np.load(VECTORS_PATH, allow_pickle=True)
        return data['embeddings'], json.loads(str(data['metadata']))

    def search(self, query, top_k=3):
        vecs, meta = self.load_db()
        if vecs is None: return []
        
        query_vec = self.model.encode([query])
        sims = cosine_similarity(query_vec, vecs)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            if sims[i] > 0.2:
                results.append({
                    "title": meta[i]['book_title'],
                    "format": meta[i]['format'],
                    "snippet": meta[i]['chunk'],
                    "similarity": float(sims[i])
                })
        return results