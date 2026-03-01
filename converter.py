import os
import chardet
import PyPDF2
from bs4 import BeautifulSoup
from ebooklib import epub

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.txt':
            with open(file_path, 'rb') as f:
                raw = f.read()
                enc = chardet.detect(raw)['encoding'] or 'utf-8'
            with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                return f.read(), os.path.splitext(os.path.basename(file_path))[0]
        
        elif ext == '.fb2':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'xml')
            title = soup.find('book-title').get_text() if soup.find('book-title') else "FB2 Book"
            text = "\n\n".join([p.get_text() for p in soup.find_all('p')])
            return text, title
            
        elif ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join([p.extract_text() or "" for p in reader.pages])
                title = reader.metadata.get('/Title') or os.path.basename(file_path)
            return text, title
            
        elif ext == '.epub':
            book = epub.read_epub(file_path)
            title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "EPUB"
            text = ""
            for item in book.get_items_of_type(9):
                text += BeautifulSoup(item.get_content(), 'html.parser').get_text() + "\n"
            return text, title
    except:
        return None, None
    return None, None