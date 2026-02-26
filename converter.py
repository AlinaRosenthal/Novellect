import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import chardet
import PyPDF2
from ebooklib import epub
import re

def read_txt(file_path):
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
        return {"title": title, "content": text, "format": "txt"}
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "txt", "error": str(e)}

def read_fb2(file_path):
    try:
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
        
        return {"title": book_title, "content": text_content, "format": "fb2"}
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "fb2", "error": str(e)}

def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            
            for page_num in range(len(reader.pages)):
                try:
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue
            
            title = "Неизвестно"
            if reader.metadata and reader.metadata.get('/Title'):
                title = reader.metadata.get('/Title')
            else:
                title = os.path.splitext(os.path.basename(file_path))[0]
            
            return {"title": title, "content": text, "format": "pdf"}
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "pdf", "error": str(e)}

def read_epub(file_path):
    try:
        book = epub.read_epub(file_path)
        text = ""
        
        title = "Неизвестно"
        if book.get_metadata('DC', 'title'):
            title = book.get_metadata('DC', 'title')[0][0]
        
        for item in book.get_items():
            if item.get_type() == 9:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text += soup.get_text() + "\n\n"
        
        return {"title": title, "content": text, "format": "epub"}
    except Exception as e:
        return {"title": os.path.basename(file_path), "content": "", "format": "epub", "error": str(e)}

def chunk_text(text, max_chunk_size=1000, overlap=100):
    if not text or len(text.strip()) == 0:
        return []
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        else:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    chunks = [c for c in chunks if len(c) > 50]
    
    return chunks

def process_file(file_path):
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
    
    if not book_data.get('error') and book_data.get('content', '').strip():
        book_data['chunks'] = chunk_text(book_data['content'])
        book_data['chunks_count'] = len(book_data['chunks'])
    
    return book_data