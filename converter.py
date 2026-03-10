import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import chardet
import PyPDF2
from ebooklib import epub
import re
import hashlib
import pickle
from pathlib import Path

# Папка для кэша txt-версий
TXT_CACHE_DIR = "txt_cache"
if not os.path.exists(TXT_CACHE_DIR):
    os.makedirs(TXT_CACHE_DIR)


def get_cache_path(original_file_path):
    """Создает путь для кэшированной txt-версии"""
    # Создаем хеш от оригинального файла для уникальности
    with open(original_file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    cache_filename = f"{file_hash}_{os.path.basename(original_file_path)}.txt"
    return os.path.join(TXT_CACHE_DIR, cache_filename)


def save_txt_cache(original_path, content, metadata):
    """Сохраняет txt-версию в кэш"""
    cache_path = get_cache_path(original_path)

    # Сохраняем метаданные отдельно
    meta_path = cache_path + '.meta'
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    # Сохраняем текст
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return cache_path


def load_txt_cache(original_path):
    """Загружает txt-версию из кэша если есть"""
    cache_path = get_cache_path(original_path)
    meta_path = cache_path + '.meta'

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()

            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)

            print(f"[CACHE] Загружена txt-версия: {cache_path}")
            return content, metadata
        except Exception as e:
            print(f"[CACHE] Ошибка загрузки кэша: {e}")
            return None, None

    return None, None


def read_txt(file_path):
    """Читает txt файл с определением кодировки"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'

        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            text = f.read()

        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r' +', ' ', text)

        title = os.path.splitext(os.path.basename(file_path))[0]

        metadata = {"title": title, "format": "txt"}

        return {
            "title": title,
            "content": text,
            "format": "txt",
            "metadata": metadata
        }
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "txt", "error": str(e)}


def read_fb2(file_path):
    """Читает fb2 и конвертирует в txt для индексации"""
    try:
        # Проверяем кэш
        cached_content, cached_meta = load_txt_cache(file_path)
        if cached_content and cached_meta:
            return {
                "title": cached_meta.get('title', 'Неизвестно'),
                "content": cached_content,
                "format": "fb2",
                "metadata": cached_meta,
                "from_cache": True
            }

        # Если нет в кэше - парсим
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'xml')

        title_info = soup.find('title-info')
        book_title = "Неизвестно"
        if title_info:
            book_name_tag = title_info.find('book-title')
            if book_name_tag:
                book_title = book_name_tag.get_text().strip()

        body = soup.find('body')
        text_content = ""
        if body:
            for tag in body.find_all(['p', 'title', 'subtitle', 'poem', 'stanza']):
                text_content += tag.get_text().strip() + "\n\n"

        # Закрываем soup (освобождаем память)
        soup.decompose()

        # Сохраняем в кэш
        metadata = {"title": book_title, "format": "fb2", "original": file_path}
        save_txt_cache(file_path, text_content, metadata)

        return {
            "title": book_title,
            "content": text_content,
            "format": "fb2",
            "metadata": metadata
        }
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "fb2", "error": str(e)}


def read_pdf(file_path):
    """Читает pdf и конвертирует в txt для индексации"""
    try:
        # Проверяем кэш
        cached_content, cached_meta = load_txt_cache(file_path)
        if cached_content and cached_meta:
            return {
                "title": cached_meta.get('title', 'Неизвестно'),
                "content": cached_content,
                "format": "pdf",
                "metadata": cached_meta,
                "from_cache": True
            }

        # Если нет в кэше - парсим
        text = ""
        title = "Неизвестно"

        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)

            for page_num in range(len(reader.pages)):
                try:
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue

            if reader.metadata and reader.metadata.get('/Title'):
                title = reader.metadata.get('/Title')
            else:
                title = os.path.splitext(os.path.basename(file_path))[0]

        # Сохраняем в кэш
        metadata = {"title": title, "format": "pdf", "original": file_path}
        save_txt_cache(file_path, text, metadata)

        return {
            "title": title,
            "content": text,
            "format": "pdf",
            "metadata": metadata
        }
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "pdf", "error": str(e)}


def read_epub(file_path):
    """Читает epub и конвертирует в txt для индексации"""
    try:
        # Проверяем кэш
        cached_content, cached_meta = load_txt_cache(file_path)
        if cached_content and cached_meta:
            return {
                "title": cached_meta.get('title', 'Неизвестно'),
                "content": cached_content,
                "format": "epub",
                "metadata": cached_meta,
                "from_cache": True
            }

        # Если нет в кэше - парсим
        text = ""
        title = "Неизвестно"

        book = epub.read_epub(file_path)

        if book.get_metadata('DC', 'title'):
            title = book.get_metadata('DC', 'title')[0][0]

        for item in book.get_items():
            if item.get_type() == 9:  # Документ
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text += soup.get_text() + "\n\n"
                soup.decompose()  # Закрываем soup

        # Закрываем книгу (освобождаем память)
        book = None

        # Сохраняем в кэш
        metadata = {"title": title, "format": "epub", "original": file_path}
        save_txt_cache(file_path, text, metadata)

        return {
            "title": title,
            "content": text,
            "format": "epub",
            "metadata": metadata
        }
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "epub", "error": str(e)}


def process_file(file_path):
    """Основная функция обработки файлов"""
    if not os.path.exists(file_path):
        return {"error": "Файл не найден"}

    ext = os.path.splitext(file_path)[1].lower()

    readers = {
        '.txt': read_txt,
        '.fb2': read_fb2,
        '.pdf': read_pdf,
        '.epub': read_epub
    }

    reader = readers.get(ext)
    if not reader:
        return {
            "title": os.path.basename(file_path),
            "content": "",
            "format": ext,
            "error": f"Неподдерживаемый формат: {ext}"
        }

    book_data = reader(file_path)

    if 'from_cache' in book_data:
        print(f"[INFO] Загружено из кэша: {file_path}")

    return book_data


def clear_txt_cache(max_age_days=30):
    """Очищает старые файлы из кэша"""
    import time
    now = time.time()

    if not os.path.exists(TXT_CACHE_DIR):
        return

    for file in os.listdir(TXT_CACHE_DIR):
        file_path = os.path.join(TXT_CACHE_DIR, file)
        if os.path.isfile(file_path):
            # Удаляем файлы старше max_age_days
            if os.stat(file_path).st_mtime < now - (max_age_days * 86400):
                try:
                    os.remove(file_path)
                    print(f"[CACHE] Удален старый кэш: {file}")
                except:
                    pass
