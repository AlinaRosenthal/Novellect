import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import chardet


def read_txt(file_path):
    """Чтение .txt файлов с определением кодировки"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'

        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            text = f.read()

        # Название файла как заголовок
        title = os.path.basename(file_path)
        return {"title": title, "content": text, "format": "txt"}
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "txt", "error": str(e)}


def read_fb2(file_path):
    """Чтение .fb2 файлов (XML структура)"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'xml')  # Используем xml parser для FB2

        # Извлечение названия книги
        title_info = soup.find('title-info')
        book_title = "Неизвестно"
        if title_info:
            book_name_tag = title_info.find('book-title')
            if book_name_tag:
                book_title = book_name_tag.get_text()

        # Извлечение основного текста (тело книги)
        body = soup.find('body')
        text_content = ""
        if body:
            # Удаляем теги, оставляем текст
            text_content = body.get_text(separator='\n')

        return {"title": book_title, "content": text_content, "format": "fb2"}
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "fb2", "error": str(e)}


def process_file(file_path):
    """Универсальная функция обработки"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        return read_txt(file_path)
    elif ext == '.fb2':
        return read_fb2(file_path)
    else:
        return {"title": os.path.basename(file_path), "content": "", "format": ext, "error": "Unsupported format"}