import os
import json
from pathlib import Path

# Константы путей
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
STORAGE_FILE = BASE_DIR / "storage.json"
MAX_SIZE_GB = 1

if not UPLOADS_DIR.exists():
    UPLOADS_DIR.mkdir(parents=True)

def get_library_size():
    """Считает общий размер всех загруженных файлов в байтах"""
    return sum(f.stat().st_size for f in UPLOADS_DIR.glob('**/*') if f.is_file())

def is_limit_exceeded(new_file_size_bytes=0):
    """Проверяет, не будет ли превышен лимит в 1 ГБ"""
    limit_bytes = MAX_SIZE_GB * 1024 * 1024 * 1024
    return (get_library_size() + new_file_size_bytes) > limit_bytes

def load_index():
    if not STORAGE_FILE.exists(): return []
    with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_index(index):
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def get_book_by_hash(file_hash):
    """Ищет книгу в индексе по её хешу"""
    index = load_index()
    for book in index:
        if book.get('file_hash') == file_hash:
            return book
    return None


def delete_book_physically(book_id):
    index = load_index()
    new_index = []
    for book in index:
        if book.get('id') == book_id:
            # Находим путь к файлу и удаляем его
            file_path = book.get('file_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

        else:
            new_index.append(book)
    save_index(new_index)