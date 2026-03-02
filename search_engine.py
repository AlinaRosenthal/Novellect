import json
import os

STORAGE_FILE = 'storage.json'


def load_index():
    """Загрузка индекса книг из JSON"""
    if not os.path.exists(STORAGE_FILE):
        return []
    with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_index(index):
    """Сохранение индекса книг в JSON"""
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def add_to_index(book_data, file_id):
    """Добавление книги в индекс"""
    index = load_index()
    # Проверка на дубликаты по ID
    index = [book for book in index if book['id'] != file_id]

    book_record = {
        "id": file_id,
        "title": book_data['title'],
        "format": book_data['format'],
        "content": book_data['content']  # В прототипе храним полный текст, в MVP нужно будет чанковать
    }
    index.append(book_record)
    save_index(index)


def search_keywords(query):
    """Поиск по ключевым словам"""
    index = load_index()
    results = []
    query_lower = query.lower()

    for book in index:
        if query_lower in book['content'].lower():
            # Находим контекст (snippet)
            content_lower = book['content'].lower()
            start = content_lower.find(query_lower)
            snippet_start = max(0, start - 50)
            snippet_end = min(len(book['content']), start + len(query) + 50)
            snippet = book['content'][snippet_start:snippet_end].replace('\n', ' ')

            results.append({
                "title": book['title'],
                "format": book['format'],
                "snippet": f"...{snippet}..."
            })

    return results