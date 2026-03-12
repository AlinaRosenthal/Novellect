import re
from datetime import datetime
from search_engine import get_query_analyzer, get_text_analyzer
from search_engine import search_hybrid, load_index, update_last_opened

# Глобальные анализаторы
_query_analyzer = get_query_analyzer()
_text_analyzer = get_text_analyzer()


class QueryAnalyzerAgent:
    """
    Агент для анализа запросов с универсальным определением характеристик
    """

    # Паттерны для разных типов запросов
    VAGUE_PATTERNS = [
        r'хочу прочитать (?:про|о)',
        r'посоветуй (?:книгу|что почитать)',
        r'что почитать (?:о|про)',
        r'интересуют книги (?:о|про)',
        r'подбери (?:книгу|книги)',
        r'книги (?:про|о)'
    ]

    SPECIFIC_PATTERNS = [
        r'в какой книге',
        r'найди книгу (?:где|в которой)',
        r'кто главный герой в',
        r'что такое',
        r'кто такой',
        r'как зовут'
    ]

    def __init__(self):
        self.genre_keywords = {
            'фэнтези': ['фэнтези', 'магия', 'волшебник', 'эльф', 'дракон'],
            'фантастика': ['фантастика', 'космос', 'будущее', 'робот', 'инопланетянин'],
            'детектив': ['детектив', 'убийство', 'расследование', 'сыщик', 'преступление'],
            'роман': ['роман', 'любовь', 'отношения', 'чувства'],
            'исторический': ['исторический', 'история', 'война', 'эпоха', 'древний']
        }

    def analyze(self, query):
        """
        Анализирует запрос и возвращает структурированное представление
        """
        query_lower = query.lower().strip()

        # Определяем тип запроса
        query_type = self._determine_type(query_lower)

        # Извлекаем ключевые слова
        keywords = self._extract_keywords(query_lower)

        # Определяем жанр если есть
        genre = self._detect_genre(query_lower)

        # Извлекаем сущности
        entities = self._extract_entities(query_lower)

        # ========== УНИВЕРСАЛЬНЫЙ АНАЛИЗ ЗАПРОСА ==========
        query_analysis = _query_analyzer.analyze_query(query)

        # Расширяем запрос для лучшего поиска
        expanded_query = _query_analyzer.expand_query_with_features(query, query_analysis)

        # Оцениваем сложность
        complexity = self._assess_complexity(query)

        return {
            'original_query': query,
            'expanded_query': expanded_query,
            'type': query_type,
            'keywords': keywords,
            'genre': genre,
            'entities': entities,
            'complexity': complexity,
            'query_analysis': query_analysis,  # полный анализ запроса
            'priorities': query_analysis['priorities'],  # приоритеты для поиска
            'has_features': {
                'mood': query_analysis['has_mood'],
                'style': query_analysis['has_style'],
                'plot': query_analysis['has_plot'],
                'atmosphere': query_analysis['has_atmosphere'],
                'tone': query_analysis['has_tone']
            }
        }

    def _determine_type(self, query):
        """Определяет тип запроса: vague, specific или general"""
        for pattern in self.VAGUE_PATTERNS:
            if re.search(pattern, query):
                return 'vague'

        for pattern in self.SPECIFIC_PATTERNS:
            if re.search(pattern, query):
                return 'specific'

        return 'general'

    def _extract_keywords(self, query):
        """Извлекает ключевые слова из запроса"""
        stop_words = {'про', 'книгу', 'книге', 'найди', 'хочу', 'посоветуй',
                      'что', 'где', 'как', 'кто', 'в', 'на', 'с', 'о'}

        words = re.findall(r'\w+', query)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _detect_genre(self, query):
        """Определяет жанр из запроса"""
        for genre, keywords in self.genre_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    return genre
        return None

    def _extract_entities(self, query):
        """Извлекает именованные сущности"""
        entities = {
            'characters': [],
            'places': [],
            'concepts': []
        }

        words = query.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 1:
                if i > 0 or word.lower() != word:
                    entities['characters'].append(word)

        return entities

    def _assess_complexity(self, query):
        """Оценивает сложность запроса"""
        words = len(query.split())
        conditions = len(re.findall(r'и|или|где|котор', query.lower()))

        if words > 10 or conditions > 1:
            return 'complex'
        elif words > 5 or conditions > 0:
            return 'medium'
        else:
            return 'simple'


class RetrievalAgent:
    """
    Агент для поиска по базе
    Использует расширенный запрос для лучших результатов
    """

    def __init__(self):
        self.last_results = None

    def search(self, analysis, top_k=10):
        """
        Выполняет поиск на основе анализа запроса
        Использует расширенный запрос если доступен
        """
        # Используем расширенный запрос если есть
        query = analysis.get('expanded_query', analysis['original_query'])

        # Для vague запросов берем больше результатов
        actual_top_k = top_k * 2 if analysis['type'] == 'vague' else top_k

        # Баланс между семантикой и ключевыми словами
        if analysis['type'] == 'specific':
            alpha = 0.5
        elif analysis['type'] == 'vague':
            alpha = 0.8
        else:
            alpha = 0.7

        # Выполняем поиск
        results = search_hybrid(query, top_k=actual_top_k, alpha=alpha)

        self.last_results = results
        return results


class RankingAgent:
    """
    Агент для ранжирования результатов
    Универсальная сортировка по множеству критериев
    """

    def __init__(self):
        self.index = load_index()

    def rank(self, results, analysis):
        """
        Ранжирует результаты в зависимости от анализа запроса
        """
        if not results:
            return []

        self.index = load_index()

        # Если есть приоритеты из анализа запроса - используем универсальное ранжирование
        if analysis.get('priorities'):
            return self._rank_by_features(results, analysis)

        # Иначе используем старые методы
        if analysis['type'] == 'vague':
            return self._rank_by_last_opened(results)
        elif analysis['type'] == 'specific':
            return results
        elif analysis['complexity'] == 'complex':
            return self._rank_complex(results, analysis)
        else:
            return results

    def _rank_by_features(self, results, analysis):
        """
        Универсальное ранжирование по множеству характеристик
        """
        # Группируем результаты по книгам
        books = {}
        for r in results:
            book_id = r['book_id']
            if book_id not in books:
                book_info = next((b for b in self.index if b['id'] == book_id), None)
                books[book_id] = {
                    'book_info': book_info,
                    'results': [],
                    'feature_score': 0
                }
            books[book_id]['results'].append(r)

        # Для каждой книги вычисляем соответствие запросу
        priorities = analysis['priorities']

        for book_id, book_data in books.items():
            book_info = book_data['book_info']
            if book_info and 'features' in book_info:
                book_features = book_info['features']

                # Вычисляем общий score соответствия
                total_score = 0
                total_weight = 0

                for category, cat_priorities in priorities.items():
                    if category in book_features:
                        for subcat, query_score in cat_priorities:
                            book_score = book_features[category].get(subcat, 0)
                            # Чем выше приоритет в запросе, тем больше вес
                            weight = query_score
                            total_score += book_score * weight
                            total_weight += weight

                if total_weight > 0:
                    book_data['feature_score'] = total_score / total_weight
            else:
                book_data['feature_score'] = 0

        # Сортируем книги по feature_score
        sorted_books = sorted(
            books.values(),
            key=lambda x: x['feature_score'],
            reverse=True
        )

        # Собираем результаты
        ranked_results = []
        for book in sorted_books:
            # Внутри книги сортируем по релевантности
            book_results = sorted(book['results'], key=lambda x: x['similarity'], reverse=True)
            ranked_results.extend(book_results)

        # Логируем топ-книги
        if sorted_books:
            print(f"[RANKING] Топ по характеристикам:")
            for i, book in enumerate(sorted_books[:3]):
                if book['book_info']:
                    print(f"  {i + 1}. {book['book_info']['title']}: {book['feature_score']:.2f}")

        return ranked_results

    def _rank_by_last_opened(self, results):
        """
        Сортирует книги по last_opened (для vague запросов)
        """
        books = {}
        for r in results:
            book_id = r['book_id']
            if book_id not in books:
                book_info = next((b for b in self.index if b['id'] == book_id), None)
                books[book_id] = {
                    'book_info': book_info,
                    'results': []
                }
            books[book_id]['results'].append(r)

        sorted_books = sorted(
            books.values(),
            key=lambda x: (
                x['book_info'].get('last_opened', float('inf')) if x['book_info'] else float('inf'),
                -len(x['results'])
            )
        )

        ranked_results = []
        for book in sorted_books:
            book_results = sorted(book['results'], key=lambda x: x['similarity'], reverse=True)
            ranked_results.extend(book_results)

        return ranked_results

    def _rank_complex(self, results, analysis):
        """
        Ранжирование для сложных запросов
        """
        keywords = set(analysis['keywords'])

        for r in results:
            snippet_lower = r['snippet'].lower()
            keyword_matches = sum(1 for k in keywords if k in snippet_lower)
            r['adjusted_score'] = r['similarity'] * (1 + 0.1 * keyword_matches)

        return sorted(results, key=lambda x: x.get('adjusted_score', x['similarity']), reverse=True)


class ResponseAgent:
    """
    Агент для формирования финального ответа
    Учитывает все характеристики в ответе
    """

    def __init__(self):
        self.index = load_index()

    def format(self, results, analysis):
        """
        Форматирует ответ с учетом всех характеристик
        """
        if not results:
            return self._format_empty_response(analysis)

        # Определяем тип ответа по анализу
        if analysis['type'] == 'vague':
            return self._format_vague_response(results, analysis)
        elif analysis['type'] == 'specific':
            return self._format_specific_response(results, analysis)
        else:
            return self._format_general_response(results, analysis)

    def _format_vague_response(self, results, analysis):
        """Форматирует ответ для неопределенных запросов с учетом характеристик"""
        # Группируем по книгам
        books = {}
        for r in results:
            if r['title'] not in books:
                books[r['title']] = {
                    'results': [],
                    'book_id': r['book_id']
                }
            books[r['title']]['results'].append(r)

        response = {
            'type': 'vague',
            'query': analysis['original_query'],
            'genre': analysis.get('genre'),
            'recommendations': []
        }

        # Добавляем информацию о том, что искали
        if analysis.get('has_features'):
            features_found = []
            if analysis['has_features'].get('mood'):
                features_found.append("по настроению")
            if analysis['has_features'].get('style'):
                features_found.append("по стилю")
            if analysis['has_features'].get('plot'):
                features_found.append("по сюжету")
            if features_found:
                response['search_criteria'] = f"Подбор {', '.join(features_found)}"

        for title, data in books.items():
            book_info = next((b for b in self.index if b['id'] == data['book_id']), None)

            # Формируем информацию о соответствии характеристикам
            feature_matches = []
            if book_info and 'features' in book_info and analysis.get('priorities'):
                book_features = book_info['features']
                priorities = analysis['priorities']

                for category, cat_priorities in priorities.items():
                    if category in book_features:
                        for subcat, query_score in cat_priorities[:2]:  # Топ-2
                            book_score = book_features[category].get(subcat, 0)
                            if book_score > 0.1:  # Если есть заметное соответствие
                                feature_matches.append(f"{subcat}: {book_score:.0%}")

            recommendation = {
                'title': title,
                'format': data['results'][0]['format'],
                'relevance_score': max(r['similarity'] for r in data['results']),
                'last_opened': book_info.get('last_opened') if book_info else None,
                'open_count': book_info.get('open_count', 0) if book_info else 0,
                'snippets': [r['snippet'] for r in data['results'][:2]],
                'feature_matches': feature_matches[:3]  # Топ-3 совпадения
            }
            response['recommendations'].append(recommendation)

        return response

    def _format_specific_response(self, results, analysis):
        """Форматирует ответ для точных запросов"""
        response = {
            'type': 'specific',
            'query': analysis['original_query'],
            'answers': []
        }

        for r in results[:3]:
            answer = {
                'book_title': r['title'],
                'format': r['format'],
                'exact_match': r['similarity'] > 0.8,
                'snippet': r['snippet'],
                'relevance': r['similarity']
            }
            response['answers'].append(answer)

        return response

    def _format_general_response(self, results, analysis):
        """Форматирует ответ для общих запросов"""
        response = {
            'type': 'general',
            'query': analysis['original_query'],
            'keywords': analysis['keywords'],
            'results': []
        }

        for r in results[:5]:
            result = {
                'title': r['title'],
                'snippet': r['snippet'],
                'relevance': r['similarity']
            }
            response['results'].append(result)

        return response

    def _format_empty_response(self, analysis):
        """Форматирует ответ, когда ничего не найдено"""
        return {
            'type': 'empty',
            'query': analysis['original_query'],
            'message': 'Ничего не найдено. Попробуйте изменить запрос или добавить больше книг.'
        }


class AgentOrchestrator:
    """
    Оркестратор агентов - главный класс мультиагентной системы
    """

    def __init__(self):
        self.analyzer = QueryAnalyzerAgent()
        self.retriever = RetrievalAgent()
        self.ranker = RankingAgent()
        self.formatter = ResponseAgent()
        self.conversation_history = []

    def process_query(self, query, update_opened=True):
        """
        Обрабатывает запрос через всех агентов
        """
        print(f"[ORCHESTRATOR] Обработка запроса: '{query}'")
        start_time = datetime.now()

        # Шаг 1: Анализ запроса
        analysis = self.analyzer.analyze(query)
        print(f"[ANALYZER] Тип запроса: {analysis['type']}")

        # Выводим информацию о найденных характеристиках
        if analysis.get('has_features'):
            features = [k for k, v in analysis['has_features'].items() if v]
            if features:
                print(f"[ANALYZER] Найдены характеристики: {', '.join(features)}")

        if analysis.get('priorities'):
            for category, priorities in analysis['priorities'].items():
                print(f"[ANALYZER] Приоритет {category}: {', '.join([p[0] for p in priorities[:2]])}")

        # Шаг 2: Поиск
        results = self.retriever.search(analysis)

        if results:
            # Шаг 3: Ранжирование
            ranked_results = self.ranker.rank(results, analysis)

            # Шаг 4: Обновляем статистику открытий
            if update_opened:
                for r in ranked_results[:3]:
                    update_last_opened(r['book_id'])

            # Шаг 5: Форматирование ответа
            response = self.formatter.format(ranked_results, analysis)
        else:
            response = self.formatter._format_empty_response(analysis)

        # Сохраняем в историю
        self.conversation_history.append({
            'query': query,
            'analysis': analysis,
            'response_type': response['type'],
            'timestamp': datetime.now().isoformat()
        })

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[ORCHESTRATOR] Запрос обработан за {elapsed:.2f} сек")

        return response


# Глобальный экземпляр оркестратора
_orchestrator = AgentOrchestrator()


def process_with_agents(query):
    """Удобная функция для вызова агентной системы"""
    return _orchestrator.process_query(query)