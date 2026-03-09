from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import search_engine
from local_llm import OllamaClient


@dataclass
class QueryPlan:
    original_query: str
    intent: str
    quoted_phrase: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    search_query: str = ""
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not payload["search_query"]:
            payload["search_query"] = self.original_query
        return payload


class QueryAgent:
    BROAD_MARKERS = [
        "хочу",
        "что почитать",
        "посоветуй",
        "подбери",
        "подберите",
        "ищу книгу",
        "ищу что",
        "порекомендуй",
        "рекоменд",
    ]
    COMPLEX_MARKERS = [
        "главный герой",
        "в сюжете",
        "жанр",
        "где",
        "при этом",
        "но не",
        "сначала",
        "затем",
        "одновременно",
        "предательство",
        "отношения",
    ]

    def analyze(self, query: str) -> QueryPlan:
        cleaned = (query or "").strip()
        normalized = search_engine.normalize_for_search(cleaned)
        phrase = search_engine.quoted_phrase(cleaned)
        keywords = search_engine.informative_tokens(cleaned)
        constraints = self._extract_constraints(cleaned)

        intent = "semantic_lookup"
        reasoning = "Запрос распознан как обычный семантический поиск по библиотеке."

        if phrase or "цитат" in normalized or "точная фраза" in normalized:
            intent = "exact_quote"
            reasoning = "В запросе есть точная фраза или цитата, поэтому приоритет отдан exact match и поиску по фрагментам."
        elif any(marker in normalized for marker in self.BROAD_MARKERS):
            intent = "discovery"
            reasoning = "Запрос похож на рекомендацию по теме, поэтому система ищет релевантные книги и сортирует их по давности последнего открытия."
        else:
            complex_signals = sum(marker in normalized for marker in self.COMPLEX_MARKERS)
            if complex_signals >= 2 or ("," in cleaned and len(keywords) >= 5) or len(constraints) >= 2:
                intent = "complex"
                reasoning = "Запрос содержит несколько условий, поэтому используется мультиагентный сценарий: разбор условий -> гибридный поиск -> итоговый ответ."

        if intent == "discovery":
            if constraints:
                constraints = constraints[:1]
            else:
                constraints = keywords[:3]
        if phrase and phrase not in constraints:
            constraints = [phrase] + constraints

        if phrase:
            search_query = phrase
        elif constraints:
            search_query = "; ".join(constraints)
        else:
            search_query = cleaned

        return QueryPlan(
            original_query=cleaned,
            intent=intent,
            quoted_phrase=phrase,
            keywords=keywords,
            constraints=constraints,
            search_query=search_query,
            reasoning=reasoning,
        )

    def _extract_constraints(self, query: str) -> List[str]:
        lowered = search_engine.normalize_for_search(query)
        cleaned = re.sub(
            r"\b(найди|подбери|подберите|посоветуй|порекомендуй|хочу|хочу\s+прочитать|ищу|покажи|мне)\b",
            " ",
            lowered,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:")

        parts = re.split(r",|;|\bи\b|\bно\b", cleaned)
        constraints: List[str] = []
        for part in parts:
            part = part.strip(" .,:;-")
            if len(part.split()) < 2:
                continue
            if part not in constraints:
                constraints.append(part)

        # Дополнительная эвристика для конструкций "про X"
        pro_match = re.search(r"\bпро\s+([a-zа-яё0-9\- ]{3,})", lowered)
        if pro_match:
            candidate = pro_match.group(1).strip(" .,:;-")
            candidate = re.split(r"\b(где|котор|и|но)\b", candidate)[0].strip(" .,:;-")
            if candidate and candidate not in constraints:
                constraints.insert(0, candidate)

        return constraints[:5]


class RetrievalAgent:
    def retrieve(self, plan: QueryPlan, top_k: int = 8) -> Dict[str, Any]:
        return search_engine.hybrid_search(plan.to_dict(), top_k=top_k)


class ResponseAgent:
    def __init__(self, ollama_client: Optional[OllamaClient] = None) -> None:
        self.ollama_client = ollama_client or OllamaClient()

    def compose(
        self,
        *,
        plan: QueryPlan,
        retrieval: Dict[str, Any],
        use_llm: bool,
        llm_model: str,
    ) -> Dict[str, Any]:
        fallback_text = self._deterministic_answer(plan, retrieval)
        if not use_llm:
            return {
                "answer": fallback_text,
                "used_llm": False,
                "llm_error": None,
            }

        try:
            llm_text = self._generate_with_ollama(plan, retrieval, llm_model)
            if llm_text:
                return {
                    "answer": llm_text,
                    "used_llm": True,
                    "llm_error": None,
                }
        except Exception as exc:
            return {
                "answer": fallback_text,
                "used_llm": False,
                "llm_error": str(exc),
            }

        return {
            "answer": fallback_text,
            "used_llm": False,
            "llm_error": "Локальная LLM не вернула ответ.",
        }

    def _generate_with_ollama(self, plan: QueryPlan, retrieval: Dict[str, Any], model: str) -> str:
        contexts = self._contexts_for_prompt(retrieval)
        if not contexts:
            return ""
        system = (
            "Ты — локальный помощник по личной библиотеке. "
            "Отвечай только на основе переданных фрагментов. "
            "Не придумывай книги, персонажей и детали. Если данных недостаточно, прямо скажи об этом. "
            "Отвечай на русском языке кратко и по делу."
        )
        prompt = (
            f"Тип запроса: {plan.intent}\n"
            f"Запрос пользователя: {plan.original_query}\n"
            f"Условия: {', '.join(plan.constraints) if plan.constraints else 'нет'}\n\n"
            "Контексты из локальной библиотеки:\n"
            f"{contexts}\n\n"
            "Сформируй ответ в формате:\n"
            "1) краткий вывод;\n"
            "2) 2-5 подходящих книг;\n"
            "3) по каждой книге — чем она подходит;\n"
            "4) если уверенность низкая, укажи это."
        )
        response = self.ollama_client.generate(model=model, prompt=prompt, system=system)
        return response.get("text", "").strip()

    def _contexts_for_prompt(self, retrieval: Dict[str, Any], max_books: int = 5) -> str:
        contexts: List[str] = []
        for index, book in enumerate(retrieval.get("book_results", [])[:max_books], start=1):
            header = f"[{index}] {book['title']}"
            if book.get("author"):
                header += f" — {book['author']}"
            header += f" (score={book['score']:.2f})"
            snippets = []
            for chunk in book.get("best_chunks", [])[:2]:
                snippets.append(f"- {self._shorten(chunk['text'])}")
            contexts.append(f"{header}\n" + "\n".join(snippets))
        return "\n\n".join(contexts)

    def _deterministic_answer(self, plan: QueryPlan, retrieval: Dict[str, Any]) -> str:
        books = retrieval.get("book_results", [])
        if not books:
            return (
                "По вашей локальной библиотеке ничего уверенного не найдено. "
                "Попробуйте уточнить тему, персонажа, троп или привести более точную фразу."
            )

        if plan.intent == "discovery":
            lines = [
                f"Нашёл {len(books)} книг(и) по теме запроса. Список отсортирован по давности последнего открытия: сначала давно не открывавшиеся книги.",
            ]
            for book in books[:5]:
                meta = self._book_meta(book)
                snippet = self._shorten(book["best_chunks"][0]["text"]) if book.get("best_chunks") else ""
                lines.append(f"• {book['title']}{meta}: {snippet}")
            return "\n".join(lines)

        if plan.intent == "exact_quote":
            lines = ["Нашёл наиболее вероятные совпадения по точной фразе или очень близкому фрагменту:"]
            for book in books[:5]:
                snippet = self._shorten(book["best_chunks"][0]["text"]) if book.get("best_chunks") else ""
                lines.append(f"• {book['title']}: {snippet}")
            return "\n".join(lines)

        if plan.intent == "complex":
            lines = [
                "Обработал запрос как многосоставной: сначала выделил условия, затем выполнил гибридный поиск по библиотеке.",
                f"Выделенные условия: {', '.join(plan.constraints) if plan.constraints else 'без явных условий'}.",
                "Наиболее подходящие книги:",
            ]
            for book in books[:5]:
                snippet = self._shorten(book["best_chunks"][0]["text"]) if book.get("best_chunks") else ""
                lines.append(f"• {book['title']}: {snippet}")
            return "\n".join(lines)

        lines = ["Вот самые релевантные книги и фрагменты из вашей локальной библиотеки:"]
        for book in books[:5]:
            snippet = self._shorten(book["best_chunks"][0]["text"]) if book.get("best_chunks") else ""
            lines.append(f"• {book['title']}: {snippet}")
        return "\n".join(lines)

    def _book_meta(self, book: Dict[str, Any]) -> str:
        last_opened = book.get("last_opened_at")
        open_count = int(book.get("open_count") or 0)
        extras: List[str] = []
        if open_count:
            extras.append(f"открывалась {open_count} раз")
        if last_opened:
            extras.append(f"последнее открытие: {last_opened[:19].replace('T', ' ')}")
        if extras:
            return f" ({'; '.join(extras)})"
        return " (ещё не открывалась)"

    def _shorten(self, text: str, limit: int = 260) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


class MultiAgentBookAssistant:
    def __init__(self, ollama_client: Optional[OllamaClient] = None) -> None:
        self.query_agent = QueryAgent()
        self.retrieval_agent = RetrievalAgent()
        self.response_agent = ResponseAgent(ollama_client=ollama_client)

    def run(
        self,
        query: str,
        *,
        top_k: int = 8,
        use_llm: bool = False,
        llm_model: str = "qwen2.5:1.5b-instruct",
    ) -> Dict[str, Any]:
        plan = self.query_agent.analyze(query)
        retrieval = self.retrieval_agent.retrieve(plan, top_k=top_k)
        response = self.response_agent.compose(
            plan=plan,
            retrieval=retrieval,
            use_llm=use_llm,
            llm_model=llm_model,
        )
        return {
            "plan": plan.to_dict(),
            "retrieval": retrieval,
            "response": response,
        }
