import json
import os
import re
import numpy as np
import time
import hashlib
import pickle
from collections import OrderedDict
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

STORAGE_FILE = 'storage.json'
VECTOR_DB_FILE = 'vector_db.npz'
FINE_TUNED_MODEL_PATH = 'fine_tuned_literary_model'
CACHE_FILE = 'search_cache.pkl'

STOP_WORDS = {
    'про', 'книга', 'книгу', 'книги', 'автор', 'сюжет', 'прочитать', 'найти',
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то',
    'все', 'она', 'так', 'из', 'за', 'вы', 'же', 'бы', 'по', 'только', 'ее',
    'мне', 'было', 'вот', 'от', 'меня', 'еще', 'о', 'из', 'ему', 'теперь', 'когда',
    'быть', 'был', 'была', 'это', 'для', 'кто', 'дом', 'год', 'купить', 'читать'
}

# ========== УНИВЕРСАЛЬНЫЙ АНАЛИЗ ТЕКСТА ==========
TEXT_FEATURES = {
    # Настроения (moods)
    'mood': {
        'позитивное': ['радост', 'счаст', 'весел', 'светл', 'добр', 'хорош', 'прекрасн', 'замечательн'],
        'негативное': ['груст', 'печал', 'мрачн', 'темн', 'зл', 'плох', 'ужасн', 'кошмарн'],
        'нейтральное': ['спокойн', 'ровн', 'обычн', 'просто', 'стандартн'],
        'тревожное': ['тревож', 'страшн', 'жутк', 'пугающ', 'опаса', 'беспоко'],
        'романтичное': ['любов', 'нежн', 'романтик', 'чувств', 'сердц', 'страст']
    },

    # Стиль повествования (style)
    'style': {
        'диалоговый': ['—', 'сказал', 'спросил', 'ответил', 'промолвил', 'воскликнул', 'произнес'],
        'описательный': ['был', 'стоял', 'находил', 'виднел', 'представлял', 'казался'],
        'динамичный': ['вдруг', 'внезапно', 'быстро', 'резко', 'мгновенно', 'стремительно'],
        'медленный': ['медленно', 'долго', 'постепенно', 'неторопливо', 'плавно']
    },

    # Элементы сюжета (plot_elements)
    'plot': {
        'преодоление': ['преодол', 'справил', 'победил', 'выдержал', 'переборол', 'одолел'],
        'борьба': ['борьб', 'сражен', 'битв', 'противосто', 'сопротивл', 'воевал'],
        'путешествие': ['путешеств', 'поход', 'дорог', 'путь', 'странств', 'поездк'],
        'отношения': ['отношен', 'дружб', 'любов', 'ссор', 'конфликт', 'примир'],
        'тайна': ['тайн', 'загад', 'мистик', 'секрет', 'неизвестн'],
        'приключение': ['приключ', 'авантюр', 'риск', 'опасн']
    },

    # Персонажи (characters)
    'characters': {
        'герой': ['герой', 'спаситель', 'защитник', 'рыцарь', 'спасатель'],
        'антигерой': ['антигерой', 'падший', 'циничн', 'сломленн'],
        'злодей': ['злодей', 'враг', 'противник', 'антагонист', 'негодяй'],
        'жертва': ['жертв', 'пострадав', 'бедняг', 'несчастн']
    },

    # Атмосфера (atmosphere)
    'atmosphere': {
        'таинственная': ['тайн', 'загад', 'мистик', 'потусторон', 'необъясним'],
        'напряженная': ['напряж', 'нервн', 'тревож', 'взволнованн', 'наэлектризован'],
        'уютная': ['уют', 'тепл', 'домашн', 'комфортн', 'приютн'],
        'мрачная': ['мрач', 'темн', 'гнетущ', 'сумрачн', 'угрюм'],
        'легкая': ['легк', 'воздушн', 'непринужд', 'свободн']
    },

    # Тон повествования (tone)
    'tone': {
        'ироничный': ['ирони', 'насмешк', 'сарказм', 'язвительн'],
        'серьезный': ['серьезн', 'важн', 'существенн', 'значим'],
        'философский': ['философ', 'мудр', 'размышл', 'смысл'],
        'эмоциональный': ['эмоциональн', 'чувствен', 'душевн']
    }
}

_reranker_model = None


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        print("[INIT] Загрузка Cross-Encoder реранкера...")
        try:
            _reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device='cpu')
        except:
            print("[INIT] Не удалось загрузить BGE, пробую альтернативу...")
            _reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLM-L12-v2', max_length=512, device='cpu')
    return _reranker_model


class UniversalTextAnalyzer:
    """
    Универсальный анализатор текста по множеству категорий
    """

    def __init__(self):
        self.features = TEXT_FEATURES

    def analyze_text(self, text):
        """
        Анализирует текст по всем категориям
        Возвращает словарь с весами каждой характеристики
        """
        if not text:
            return {}

        text_lower = text.lower()
        results = {}

        for category, subcategories in self.features.items():
            results[category] = {}
            total_matches = 0

            for subcat, keywords in subcategories.items():
                # Считаем вхождения ключевых слов
                matches = sum(1 for kw in keywords if kw in text_lower)
                results[category][subcat] = matches
                total_matches += matches

            # Нормализуем (превращаем в доли)
            if total_matches > 0:
                for subcat in results[category]:
                    results[category][subcat] = results[category][subcat] / max(total_matches, 1)

        return results

    def analyze_book(self, chunks):
        """
        Анализирует книгу по всем чанкам
        Возвращает усредненные характеристики всей книги
        """
        if not chunks:
            return {}

        # Инициализируем структуру для агрегации
        aggregated = {}
        for category in self.features:
            aggregated[category] = {}
            for subcat in self.features[category]:
                aggregated[category][subcat] = 0.0

        # Суммируем по чанкам
        for chunk in chunks:
            # Извлекаем текст из чанка (может быть строкой или словарем)
            if isinstance(chunk, str):
                text = chunk
            else:
                text = chunk.get('chunk', chunk.get('text', ''))

            chunk_analysis = self.analyze_text(text)

            for category in aggregated:
                for subcat in aggregated[category]:
                    if category in chunk_analysis and subcat in chunk_analysis[category]:
                        aggregated[category][subcat] += chunk_analysis[category][subcat]

        # Усредняем
        num_chunks = len(chunks)
        for category in aggregated:
            for subcat in aggregated[category]:
                aggregated[category][subcat] /= num_chunks

        return aggregated

    def get_dominant_features(self, book_features, threshold=0.1):
        """
        Возвращает доминирующие характеристики книги (со значением выше порога)
        """
        dominant = {}
        for category, subcats in book_features.items():
            for subcat, value in subcats.items():
                if value > threshold:
                    if category not in dominant:
                        dominant[category] = []
                    dominant[category].append((subcat, value))

        # Сортируем по убыванию
        for category in dominant:
            dominant[category].sort(key=lambda x: x[1], reverse=True)

        return dominant


class QueryAnalyzer:
    """
    Анализирует запрос пользователя для определения приоритетов поиска
    """

    def __init__(self):
        self.analyzer = UniversalTextAnalyzer()

    def analyze_query(self, query):
        """
        Определяет, что ищет пользователь
        """
        query_lower = query.lower()

        # Анализируем сам запрос
        query_features = self.analyzer.analyze_text(query_lower)

        # Определяем приоритетные категории (где есть ненулевые значения)
        priorities = {}

        for category, subcats in query_features.items():
            for subcat, score in subcats.items():
                if score > 0:
                    if category not in priorities:
                        priorities[category] = []
                    priorities[category].append((subcat, score))

        # Сортируем по значимости (убывание score)
        for category in priorities:
            priorities[category].sort(key=lambda x: x[1], reverse=True)

        return {
            'original': query,
            'features': query_features,
            'priorities': priorities,
            'has_mood': any(v > 0 for v in query_features.get('mood', {}).values()),
            'has_style': any(v > 0 for v in query_features.get('style', {}).values()),
            'has_plot': any(v > 0 for v in query_features.get('plot', {}).values()),
            'has_atmosphere': any(v > 0 for v in query_features.get('atmosphere', {}).values()),
            'has_tone': any(v > 0 for v in query_features.get('tone', {}).values())
        }

    def expand_query_with_features(self, query, analysis):
        """
        Расширяет запрос синонимами из найденных категорий
        """
        expanded_terms = []

        for category, priorities in analysis['priorities'].items():
            for subcat, score in priorities[:2]:  # Берем топ-2
                # Добавляем само подкатегорию как термин
                expanded_terms.append(subcat)

        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"

        return query


# Глобальные экземпляры анализаторов
_text_analyzer = UniversalTextAnalyzer()
_query_analyzer = QueryAnalyzer()

# ========== МОДЕЛЬ ==========
_model_cache = None


class ModelManager:
    """Управление моделями с поддержкой дообученной версии"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            print("[INIT] Загрузка модели SentenceTransformer...")
            if os.path.exists(FINE_TUNED_MODEL_PATH):
                print("[INIT] ✅ Используется дообученная литературная модель")
                self._model = SentenceTransformer(FINE_TUNED_MODEL_PATH)
            else:
                print("[INIT] ⚠️ Используется базовая модель (дообученная не найдена)")
                self._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            # Оптимизация для CPU
            self._model = self._model.to('cpu')
            self._model.eval()

        return self._model


def get_model():
    return ModelManager().get_model()


# ========== КЭШ ==========
class SearchCache:
    """Кэш для частых запросов с сохранением на диск"""

    def __init__(self, maxsize=100, cache_file=CACHE_FILE):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.cache_file = cache_file
        self._load_cache()

    def _make_key(self, query, alpha, top_k):
        """Создает уникальный ключ для запроса"""
        content = f"{query}_{alpha}_{top_k}".lower()
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self):
        """Загружает кэш с диска"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"[CACHE] Загружено {len(self.cache)} записей из кэша")
            except Exception as e:
                print(f"[CACHE] Ошибка загрузки кэша: {e}")
                self.cache = OrderedDict()

    def _save_cache(self):
        """Сохраняет кэш на диск"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"[CACHE] Ошибка сохранения кэша: {e}")

    def get(self, query, alpha, top_k):
        key = self._make_key(query, alpha, top_k)
        if key in self.cache:
            self.cache.move_to_end(key)
            print(f"[CACHE] Найдено в кэше: '{query[:50]}...'")
            return self.cache[key]
        return None

    def set(self, query, alpha, top_k, results):
        key = self._make_key(query, alpha, top_k)
        self.cache[key] = results
        self.cache.move_to_end(key)

        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

        # Периодически сохраняем
        if len(self.cache) % 10 == 0:
            self._save_cache()

    def clear(self):
        self.cache.clear()
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except:
                pass


# Глобальный экземпляр кэша
_search_cache = SearchCache()


# ========== РАБОТА С ИНДЕКСОМ ==========
def load_index():
    if not os.path.exists(STORAGE_FILE):
        return []
    try:
        with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def save_index(index):
    try:
        with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Ошибка сохранения индекса: {e}")


def update_last_opened(book_id):
    """Обновляет время последнего открытия книги"""
    index = load_index()
    updated = False
    for book in index:
        if book.get('id') == book_id:
            book['last_opened'] = datetime.now().timestamp()
            book['open_count'] = book.get('open_count', 0) + 1
            updated = True
            break
    if updated:
        save_index(index)


# ========== ТОКЕНИЗАЦИЯ ==========
def tokenize_smart(text):
    if not text:
        return []
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


# ========== СЕМАНТИЧЕСКОЕ РАЗБИЕНИЕ ==========
def split_text_semantic(text, model, threshold=0.3, batch_size=32):
    """
    Оптимизированное семантическое разбиение с batch processing
    Возвращает чанки и их эмбеддинги
    """
    if not text:
        return [], np.array([])

    print("[PROCESS] Семантическое разбиение текста...")
    start_time = time.time()

    # Разбиваем на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        if sentences:
            embeddings = model.encode(sentences, show_progress_bar=False, batch_size=batch_size)
            return sentences, embeddings
        return [], np.array([])

    # Генерируем эмбеддинги для всех предложений за один проход
    print(f"[PROCESS] Генерация эмбеддингов для {len(sentences)} предложений...")
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=batch_size)

    # Формируем чанки на основе семантической близости
    chunks = []
    chunk_embeddings_list = []
    current_chunk = [sentences[0]]
    current_emb = [embeddings[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]

        if sim > threshold:
            current_chunk.append(sentences[i])
            current_emb.append(embeddings[i])
        else:
            chunk_text = " ".join(current_chunk)
            chunk_emb = np.mean(current_emb, axis=0)
            chunks.append(chunk_text)
            chunk_embeddings_list.append(chunk_emb)

            current_chunk = [sentences[i]]
            current_emb = [embeddings[i]]

    # Добавляем последний чанк
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_emb = np.mean(current_emb, axis=0)
        chunks.append(chunk_text)
        chunk_embeddings_list.append(chunk_emb)

    chunk_embeddings = np.array(chunk_embeddings_list) if chunk_embeddings_list else np.array([])

    elapsed = time.time() - start_time
    print(f"[INFO] Текст разбит на {len(chunks)} чанков за {elapsed:.1f} сек")

    return chunks, chunk_embeddings


# ========== ВЕКТОРНАЯ БД ==========
def load_vector_db():
    """Безопасная загрузка векторной базы данных"""
    if not os.path.exists(VECTOR_DB_FILE):
        return None, None

    try:
        data = np.load(VECTOR_DB_FILE, allow_pickle=True)

        if 'embeddings' not in data or 'metadata' not in data:
            print("[ERROR] Неверный формат файла векторной БД")
            return None, None

        embeddings = data['embeddings']
        metadata_str = data['metadata'].item() if data['metadata'].size > 0 else "[]"
        metadata = json.loads(metadata_str) if metadata_str else []

        if len(embeddings) == 0 or len(metadata) == 0:
            return None, None

        if len(embeddings) != len(metadata):
            print(f"[WARNING] Несоответствие размеров")
            min_len = min(len(embeddings), len(metadata))
            embeddings = embeddings[:min_len]
            metadata = metadata[:min_len]

        return embeddings, metadata

    except Exception as e:
        print(f"[ERROR] Ошибка загрузки векторной БД: {e}")
        return None, None


# ========== BM25 ДЛЯ ГИБРИДНОГО ПОИСКА ==========
class BM25:
    """Простая реализация BM25 для гибридного поиска"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = {}
        self.doc_len = []
        self.avg_doc_len = 0
        self.idf = {}

    def fit(self, corpus):
        """Обучает BM25 на корпусе документов"""
        self.corpus = [tokenize_smart(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len)

        # Вычисляем частоту терминов в документах
        for doc in self.corpus:
            terms = set(doc)
            for term in terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Вычисляем IDF
        N = len(self.corpus)
        for term, df in self.doc_freqs.items():
            self.idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens, doc_idx):
        """Вычисляет BM25 score для документа"""
        doc = self.corpus[doc_idx]
        score = 0
        doc_len = self.doc_len[doc_idx]

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            # Частота термина в документе
            tf = doc.count(term)
            if tf == 0:
                continue

            # BM25 формула
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (numerator / denominator)

        return score

    def search(self, query, top_k=100):
        """Поиск с BM25"""
        query_tokens = tokenize_smart(query)
        scores = []

        for i in range(len(self.corpus)):
            scores.append(self.score(query_tokens, i))

        scores = np.array(scores)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return scores, top_indices


# ========== ГИБРИДНЫЙ ПОИСК ==========
class HybridSearch:
    """Оптимизированный гибридный поиск с кэшированием BM25"""

    def __init__(self):
        self.bm25 = None
        self.bm25_corpus = None

    def _build_bm25(self, metadata):
        """Строит BM25 индекс из метаданных"""
        self.bm25_corpus = [m.get('chunk', '') for m in metadata]
        self.bm25 = BM25()
        self.bm25.fit(self.bm25_corpus)

    def search(self, query, model, embeddings, metadata, top_k=5, alpha=0.7):
        # Строим BM25 если нужно
        if self.bm25 is None or self.bm25_corpus != [m.get('chunk', '') for m in metadata]:
            self._build_bm25(metadata)

        # Количество кандидатов для реранжирования
        candidate_count = top_k * 4

        # 1. Семантический поиск (Bi-Encoder)
        q_emb = model.encode([query], show_progress_bar=False)
        sem_scores = cosine_similarity(q_emb, embeddings)[0]

        # 2. BM25 поиск
        query_tokens = tokenize_smart(query)
        bm25_scores = []
        for i in range(len(self.bm25_corpus)):
            bm25_scores.append(self.bm25.score(query_tokens, i))
        bm25_scores = np.array(bm25_scores)

        # Нормализуем BM25 scores
        bm25_max = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1
        bm25_norm = bm25_scores / bm25_max

        # 3. Комбинируем оценки
        combined = (alpha * sem_scores) + ((1 - alpha) * bm25_norm)
        top_indices = np.argsort(combined)[-candidate_count:][::-1]

        initial_candidates = []
        for idx in top_indices:
            initial_candidates.append({
                "text": metadata[idx].get('chunk', ''),
                "metadata_idx": idx
            })

        # 4. РЕРАНЖИРОВАНИЕ (Cross-Encoder)
        reranker = get_reranker()
        # Подготавливаем пары [запрос, текст]
        pairs = [[query, c['text']] for c in initial_candidates]
        rerank_scores = reranker.predict(pairs)

        # 5. Сборка финальных результатов
        final_results = []
        for i, score in enumerate(rerank_scores):
            idx = initial_candidates[i]['metadata_idx']
            final_results.append({
                "book_id": metadata[idx].get('book_id'),
                "title": metadata[idx].get('book_title'),
                "snippet": metadata[idx].get('chunk', '')[:400] + "...",
                "similarity": float(score),
                "chunk_id": metadata[idx].get('chunk_id')
            })

        # Сортируем по новому скору
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        return final_results[:top_k]


_searcher = HybridSearch()


def search_hybrid(query, top_k=5, alpha=0.7, use_cache=True):
    """Поиск с поддержкой кэширования"""
    if not query:
        return []

    if use_cache:
        cached = _search_cache.get(query, alpha, top_k)
        if cached is not None:
            return cached

    embeddings, metadata = load_vector_db()
    if embeddings is None or not metadata:
        return []

    results = _searcher.search(query, get_model(), embeddings, metadata, top_k, alpha)

    if use_cache and results:
        _search_cache.set(query, alpha, top_k, results)

    return results


def clear_cache():
    """Очищает кэш поиска"""
    _search_cache.clear()
    print("[CACHE] Кэш очищен")


# ========== ДОБАВЛЕНИЕ В ИНДЕКС ==========
def add_to_index(book_data, file_id):
    """Оптимизированная индексация с универсальным анализом текста"""
    title = book_data.get('title', 'Без названия')
    print(f"[START] Добавление книги в индекс: {title}")
    start_total = time.time()

    # Загружаем существующие данные
    old_embeddings, old_metadata = load_vector_db()
    index = load_index()

    # Удаляем старую версию книги если есть
    index = [b for b in index if b.get('id') != file_id]

    if old_metadata is not None and len(old_metadata) > 0:
        mask = [m.get('book_id') != file_id for m in old_metadata]
        if any(mask):
            old_embeddings = old_embeddings[mask] if len(old_embeddings) > 0 else old_embeddings
            old_metadata = [m for i, m in enumerate(old_metadata) if mask[i]]

    # Получаем модель
    model = get_model()
    content = book_data.get('content', "")

    if not content:
        print(f"[ERROR] Пустое содержание книги: {title}")
        return None

    # Разбиваем на чанки и получаем эмбеддинги
    chunks, chunk_embeddings = split_text_semantic(content, model)

    if not chunks or len(chunk_embeddings) == 0:
        print("[ERROR] Не удалось создать чанки.")
        return None

    # ========== УНИВЕРСАЛЬНЫЙ АНАЛИЗ КНИГИ ==========
    print(f"[ANALYZE] Универсальный анализ книги...")
    book_features = _text_analyzer.analyze_book(chunks)
    dominant = _text_analyzer.get_dominant_features(book_features)

    print(f"[ANALYZE] Доминирующие характеристики:")
    for category, features in dominant.items():
        print(f"  {category}: {', '.join([f'{f[0]}({f[1]:.2f})' for f in features[:3]])}")

    # Создаем метаданные для новых чанков
    new_meta = [{
        "book_id": file_id,
        "book_title": title,
        "format": book_data.get('format', 'unknown'),
        "chunk": c,
        "chunk_id": f"{file_id}_{i}"
    } for i, c in enumerate(chunks)]

    # Объединяем с существующими данными
    if old_embeddings is not None and len(old_embeddings) > 0:
        combined_embeddings = np.vstack([old_embeddings, chunk_embeddings])
        final_metadata = old_metadata + new_meta
    else:
        combined_embeddings, final_metadata = chunk_embeddings, new_meta

    # Сохраняем векторную БД
    try:
        np.savez(VECTOR_DB_FILE,
                 embeddings=combined_embeddings,
                 metadata=json.dumps(final_metadata, ensure_ascii=False))
    except Exception as e:
        print(f"[ERROR] Ошибка сохранения векторной БД: {e}")
        return None

    # Обновляем индекс книг с универсальными характеристиками
    record = {
        "id": file_id,
        "title": book_data.get('title', 'Без названия'),
        "file_hash": book_data.get('file_hash'),
        "format": book_data.get('format', 'unknown'),
        "chunks_count": len(chunks),
        "added_date": time.time(),
        "last_opened": None,
        "features": book_features
    }
    index.append(record)
    save_index(index)

    elapsed = time.time() - start_total
    print(f"[SUCCESS] Книга '{title}' проиндексирована за {elapsed:.1f} сек")
    return record


# ========== ЭКСПОРТ ФУНКЦИЙ ==========
def get_text_analyzer():
    return _text_analyzer


def get_query_analyzer():
    return _query_analyzer


# Список того, что можно импортировать из этого модуля
__all__ = [
    'add_to_index',
    'search_hybrid',
    'load_index',
    'save_index',
    'update_last_opened',
    'clear_cache',
    'get_text_analyzer',
    'get_query_analyzer',
    'get_model',
    'load_vector_db'
]
