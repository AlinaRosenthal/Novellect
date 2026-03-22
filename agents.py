import re
from datetime import datetime
from typing import Dict, List

from search_engine import (
    QueryProfile,
    _sanitize_search_text,
    find_title_matches,
    get_query_analyzer,
    load_index,
    search_hybrid,
    tokenize_smart,
)

_query_analyzer = get_query_analyzer()


class QueryAnalyzerAgent:
    """Определяет тип запроса и готовит безопасный search text без мусорного expansion."""

    RECOMMENDATION_MARKERS = (
        'хочу', 'посоветуй', 'подбери', 'что почитать', 'интересуют', 'что-то', 'что то',
        'мрач', 'тревож', 'атмосфер', 'настроен', 'сюжет', 'похож', 'напомина',
    )
    SPECIFIC_PATTERNS = [
        r'\bв какой книге\b',
        r'\bв каком произведении\b',
        r'\bгде есть\b',
        r'\bкто такой\b',
        r'\bкто такая\b',
        r'\bкак зовут\b',
        r'\bнайди книгу где\b',
    ]
    TITLE_HINTS = {'роман', 'повесть', 'рассказ', 'книга', 'произведение'}

    def __init__(self):
        self.stop_words = {
            'про', 'книгу', 'книге', 'найди', 'хочу', 'посоветуй', 'что', 'где', 'как', 'кто', 'в', 'на', 'с', 'о',
            'или', 'и', 'мне', 'что-то', 'что', 'то', 'книга', 'роман', 'повесть', 'рассказ',
        }

    def _extract_keywords(self, query: str) -> List[str]:
        words = re.findall(r'\w+', query.lower())
        return [word for word in words if word not in self.stop_words and len(word) > 2]

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {'characters': [], 'places': [], 'concepts': []}
        matches = re.findall(r'\b[А-ЯA-Z][а-яa-zёЁ-]+(?:\s+[А-ЯA-Z][а-яa-zёЁ-]+)*\b', query)
        seen = set()
        for match in matches:
            value = match.strip()
            if value and value not in seen:
                seen.add(value)
                entities['characters'].append(value)
        return entities

    def _determine_type(self, original_query: str, cleaned_query: str, keywords: List[str], title_matches: List[dict], query_analysis: dict):
        lowered = original_query.lower()
        strong_title = title_matches and title_matches[0]['score'] >= 0.9
        soft_title = title_matches and title_matches[0]['score'] >= 0.8

        if strong_title and len(keywords) <= 5 and not any(marker in lowered for marker in self.RECOMMENDATION_MARKERS):
            return 'title'

        if any(re.search(pattern, lowered) for pattern in self.SPECIFIC_PATTERNS):
            return 'specific'

        if soft_title and len(keywords) <= 4 and not any(marker in lowered for marker in self.RECOMMENDATION_MARKERS):
            return 'title'

        if any(marker in lowered for marker in self.RECOMMENDATION_MARKERS):
            return 'recommendation'

        if any(query_analysis.get(flag) for flag in ('has_mood', 'has_style', 'has_plot', 'has_atmosphere', 'has_tone')) and len(keywords) >= 3:
            return 'recommendation'

        if len(keywords) <= 2:
            return 'entity'

        if len(cleaned_query.split()) >= 5:
            return 'recommendation'

        return 'keyword'

    def analyze(self, query: str):
        original_query = (query or '').strip()
        cleaned_query = _sanitize_search_text(original_query)
        keywords = self._extract_keywords(cleaned_query or original_query)
        title_matches = find_title_matches(original_query, limit=5)
        entities = self._extract_entities(original_query)
        query_analysis = _query_analyzer.analyze_query(cleaned_query or original_query)
        query_type = self._determine_type(original_query, cleaned_query, keywords, title_matches, query_analysis)

        if query_type in {'title', 'entity', 'specific'}:
            priorities = {}
            has_features = {'mood': False, 'style': False, 'plot': False, 'atmosphere': False, 'tone': False}
        else:
            priorities = query_analysis['priorities']
            has_features = {
                'mood': query_analysis['has_mood'],
                'style': query_analysis['has_style'],
                'plot': query_analysis['has_plot'],
                'atmosphere': query_analysis['has_atmosphere'],
                'tone': query_analysis['has_tone'],
            }

        return {
            'original_query': original_query,
            'search_text': cleaned_query or original_query,
            'type': query_type,
            'keywords': keywords,
            'entities': entities,
            'query_analysis': query_analysis,
            'priorities': priorities,
            'title_matches': title_matches,
            'title_like': query_type == 'title',
            'require_exact': query_type in {'title', 'entity', 'specific'},
            'has_features': has_features,
        }


class RetrievalAgent:
    def search(self, analysis: dict, top_k: int = 10):
        profile = QueryProfile(
            original_query=analysis['original_query'],
            search_text=analysis['search_text'],
            query_type=analysis['type'],
            keywords=analysis.get('keywords', []),
            priorities=analysis.get('priorities', {}),
            has_features=analysis.get('has_features', {}),
            title_matches=analysis.get('title_matches', []),
            title_like=analysis.get('title_like', False),
            require_exact=analysis.get('require_exact', False),
        )
        title_boosts = {item['book_id']: float(item['score']) for item in analysis.get('title_matches', [])}
        return search_hybrid(
            analysis['search_text'],
            top_k=top_k,
            title_boosts=title_boosts or None,
            query_profile={
                'type': profile.query_type,
                'search_text': profile.search_text,
                'original_query': profile.original_query,
                'keywords': profile.keywords,
                'priorities': profile.priorities,
                'has_features': profile.has_features,
                'title_like': profile.title_like,
                'require_exact': profile.require_exact,
            },
        )


class RankingAgent:
    def rank(self, results: List[dict], analysis: dict):
        # Основное ранжирование уже выполняется в search_engine на уровне книг.
        return results


class ResponseAgent:
    def __init__(self):
        self.index = load_index()

    def _get_book_info(self, book_id: str):
        if not self.index:
            self.index = load_index()
        return next((book for book in self.index if book.get('id') == book_id), None)

    def _recommendation_response(self, results: List[dict], analysis: dict):
        payload = {
            'type': 'recommendation',
            'query': analysis['original_query'],
            'query_type': analysis['type'],
            'recommendations': [],
        }
        for result in results[:5]:
            book_info = self._get_book_info(result['book_id']) or {}
            feature_matches = []
            for category, priorities in (analysis.get('priorities') or {}).items():
                category_scores = (book_info.get('features') or {}).get(category, {})
                for feature_name, _weight in priorities[:2]:
                    score = float(category_scores.get(feature_name, 0.0))
                    if score >= 0.12:
                        feature_matches.append(f'{feature_name}: {score:.0%}')
            payload['recommendations'].append(
                {
                    'title': result['title'],
                    'format': result.get('format', 'unknown'),
                    'relevance_score': float(result['similarity']),
                    'snippets': result.get('snippets', [])[:2],
                    'feature_matches': feature_matches[:3],
                }
            )
        return payload

    def _result_response(self, results: List[dict], analysis: dict):
        response_type = analysis['type'] if analysis['type'] in {'title', 'entity', 'specific', 'keyword'} else 'general'
        payload = {
            'type': response_type,
            'query': analysis['original_query'],
            'query_type': analysis['type'],
            'results': [],
        }
        for result in results[:5]:
            payload['results'].append(
                {
                    'title': result['title'],
                    'format': result.get('format', 'unknown'),
                    'snippet': result.get('snippet', ''),
                    'relevance': float(result['similarity']),
                    'exact_match': bool(result.get('title_match_score', 0) >= 0.95 or result.get('coverage_score', 0) >= 0.95),
                }
            )
        return payload

    def _empty_message(self, analysis: dict):
        qtype = analysis.get('type', 'keyword')
        if qtype == 'title':
            return 'Точное название не найдено. Попробуйте другое написание или более короткую форму названия.'
        if qtype in {'entity', 'specific'}:
            return 'Точных совпадений по персонажу или фразе не найдено. Попробуйте указать больше контекста.'
        if qtype == 'recommendation':
            return 'Не удалось подобрать релевантные книги. Попробуйте уточнить тему, настроение или сюжет.'
        return 'Ничего не найдено. Попробуйте изменить формулировку запроса.'

    def format(self, results: List[dict], analysis: dict):
        if not results:
            return {
                'type': 'empty',
                'query': analysis['original_query'],
                'query_type': analysis['type'],
                'message': self._empty_message(analysis),
            }
        if analysis['type'] == 'recommendation':
            return self._recommendation_response(results, analysis)
        return self._result_response(results, analysis)


class AgentOrchestrator:
    def __init__(self):
        self.analyzer = QueryAnalyzerAgent()
        self.retriever = RetrievalAgent()
        self.ranker = RankingAgent()
        self.formatter = ResponseAgent()
        self.history = []

    def process_query(self, query: str, update_opened: bool = False):
        started = datetime.now()
        analysis = self.analyzer.analyze(query)
        results = self.retriever.search(analysis)
        ranked = self.ranker.rank(results, analysis)
        response = self.formatter.format(ranked, analysis)
        self.history.append(
            {
                'query': query,
                'type': analysis['type'],
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'latency_sec': round((datetime.now() - started).total_seconds(), 3),
            }
        )
        return response


_orchestrator = AgentOrchestrator()


def process_with_agents(query: str):
    return _orchestrator.process_query(query)
