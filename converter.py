import os
import chardet
import PyPDF2
from bs4 import BeautifulSoup
from ebooklib import epub

def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.txt':
            with open(file_path, 'rb') as f:
                enc = chardet.detect(f.read())['encoding'] or 'utf-8'
            with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                return {"title": os.path.splitext(os.path.basename(file_path))[0], "content": f.read(), "format": "txt"}
        elif ext == '.fb2':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'xml')
            return {"title": soup.find('book-title').text if soup.find('book-title') else "FB2", "content": soup.text, "format": "fb2"}
        elif ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return {"title": reader.metadata.get('/Title') or "PDF", "content": "".join([p.extract_text() for p in reader.pages]), "format": "pdf"}
        elif ext == '.epub':
            book = epub.read_epub(file_path)
            text = "".join([BeautifulSoup(i.get_content(), 'html.parser').text for i in book.get_items_of_type(9)])
            return {"title": "EPUB", "content": text, "format": "epub"}
    except:
        return {"error": "Ошибка обработки"}
    return {"error": "Формат не поддерживается"}