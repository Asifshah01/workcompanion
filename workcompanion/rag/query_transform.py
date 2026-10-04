"""Query transformation: rewriting, de-contextualisation and expansion.

Runs *before* retrieval so that queries like "explain this better" or "give me
more like that" become standalone, searchable questions.  An LLM does the heavy
lifting when one is configured; a solid rule-based fallback keeps the pipeline
functional (and free) otherwise.
"""

from __future__ import annotations

import re

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.llm.base import LLMProvider, LLMRequest, Message, SystemMessage, UserMessage
from workcompanion.schemas.retrieval import RetrievalQuery
from workcompanion.utils.text_utils import extract_key_terms, normalize_query

logger = get_logger(__name__)

SYSTEM_PROMPT = """You rewrite a learner's question into retrieval queries for a \
personal study knowledge base.

Return STRICT JSON only, with these keys:
{
  "standalone": "the question rewritten so it makes sense with no conversation history",
  "queries": ["2 to 4 alternative search queries, including synonyms and technical terms"],
  "keywords": ["4 to 8 key content words"],
  "resolved": true if you had to use conversation context to resolve a pronoun or vague reference
}

Rules:
- Preserve the learner's intent exactly; never answer the question.
- Expand jargon into equivalent technical terms (e.g. "F=ma" -> "Newton second law force mass acceleration").
- If the question is already standalone, keep it as-is.
- Never invent information that is not implied by the question."""

_PRONOUNS = (
    "this", "that", "it", "these", "those", "they", "them", "he", "she", "his", "her",
    "their", "the above", "the same", "more", "again", "explain better", "elaborate",
)

_FOLLOW_UP_PATTERNS = (
    re.compile(r"^\s*(explain|elaborate|expand)\s+(this|that|it)\b", re.I),
    re.compile(r"^\s*(more|anything else|what else)\b", re.I),
    re.compile(r"^\s*and\s+(what|why|how|when|where)\b", re.I),
    re.compile(r"^\s*(why|how)\s+(is|are|does|do|can)\s+(that|this|it)\b", re.I),
    re.compile(r"^\s*(tell me )?(more )?about (it|that|this)\b", re.I),
)

_EXPANSION_HINTS: dict[str, list[str]] = {
    "newton": ["newton second law", "F = ma", "force mass acceleration", "equation of motion"],
    "entropy": ["entropy definition", "disorder measure", "second law entropy generation", "dS"],
    "photosynthesis": ["calvin benson cycle", "carbon fixation", "chloroplast light reaction"],
    "big o": ["asymptotic complexity", "time complexity analysis", "growth rate"],
}


class QueryTransformer:
    """Produce a :class:`RetrievalQuery` from raw user input + chat history."""

    def __init__(self, llm: LLMProvider | None = None, settings: Settings | None = None):
        self._llm = llm
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    def transform(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        document_filter: str | None = None,
    ) -> RetrievalQuery:
        """Rewrite, expand and enrich a user question."""
        cleaned = normalize_query(question)
        result = RetrievalQuery(original=cleaned, standalone=cleaned)
        result.filters = {"document_name": document_filter} if document_filter else {}
        result.keywords = extract_key_terms(cleaned, limit=8)
        result.conversation_resolved = False

        needs_work = self._is_vague(cleaned) and bool(history)
        llm_rewrite: dict[str, object] = {}

        if self._settings.enable_query_rewrite and (needs_work or self._is_vague(cleaned)):
            llm_rewrite = self._llm_rewrite(cleaned, history or [])
            if llm_rewrite.get("standalone"):
                result.standalone = str(llm_rewrite["standalone"])
                result.conversation_resolved = bool(llm_rewrite.get("resolved"))
            keywords = llm_rewrite.get("keywords")
            if isinstance(keywords, list) and keywords:
                result.keywords = [str(k) for k in keywords][:8]
        elif needs_work:
            resolved = self._rule_resolve(cleaned, history or [])
            if resolved:
                result.standalone = resolved
                result.conversation_resolved = True

        expansions: list[str] = []
        if self._settings.enable_query_expansion:
            expansions = llm_rewrite.get("queries") or self._rule_expand(result.standalone or cleaned)

        # De-duplicate queries case-insensitively, keeping the rewritten
        # standalone question first so it is always searched.
        ordered: list[str] = [result.standalone or cleaned]
        seen = {ordered[0].lower()}
        for candidate in expansions:
            value = (candidate or "").strip()
            if not value or value.lower() in seen:
                continue
            seen.add(value.lower())
            ordered.append(value)

        # `queries` holds the *alternatives*; the primary is `standalone`.
        alternatives = ordered[1:] if ordered[0] == (result.standalone or cleaned) else ordered
        result.queries = alternatives[: self._settings.multi_query_count]
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _is_vague(question: str) -> bool:
        lowered = question.lower().strip()
        if not lowered:
            return True
        words = lowered.split()
        if len(words) <= 3 and any(pronoun in words for pronoun in _PRONOUNS):
            return True
        return any(pattern.search(lowered) for pattern in _FOLLOW_UP_PATTERNS)

    @staticmethod
    def _rule_resolve(question: str, history: list[tuple[str, str]]) -> str | None:
        """Resolve pronouns by borrowing the topic from the previous turn."""
        if not history:
            return None
        for role, content in reversed(history):
            if role != "user" or not content.strip():
                continue
            topic = extract_key_terms(content, limit=5)
            if topic:
                return f"{question.rstrip('?.')} (about: {', '.join(topic)})"
        return None

    def _rule_expand(self, question: str) -> list[str]:
        """Cheap synonym/alternate-phrase expansion for the obvious cases."""
        lowered = question.lower()
        expansions: list[str] = []
        for needle, alternatives in _EXPANSION_HINTS.items():
            if needle in lowered:
                expansions.extend(alt for alt in alternatives if alt.lower() not in lowered)
        terms = extract_key_terms(question, limit=5)
        if len(terms) >= 2:
            # A definition-style probe usually retrieves better than the raw question.
            expansions.append(f"{terms[0]} definition")
            expansions.append(f"{' '.join(terms[:3])}")
        return expansions[: self._settings.multi_query_count]

    def _llm_rewrite(
        self, question: str, history: list[tuple[str, str]]
    ) -> dict[str, object]:
        """Ask the LLM to rewrite the query; never fatal on failure."""
        if self._llm is None:
            return {}
        recent = history[-4:]
        conversation = (
            "\n".join(f"{'Learner' if role == 'user' else 'Tutor'}: {content}" for role, content in recent)
            or "(no prior conversation)"
        )
        request = LLMRequest(
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                UserMessage(
                    content=f"CONVERSATION:\n{conversation}\n\nCURRENT QUESTION:\n{question}"
                ),
            ],
            temperature=0.0,
            max_tokens=420,
            metadata={"task": "query_transform"},
        )
        try:
            payload = self._llm.generate_json(request, retries=2)
        except Exception as exc:  # never block retrieval on rewrite failures
            logger.warning("Query rewriting failed (%s); using the original question.", exc)
            return {}
        if not isinstance(payload, dict):
            return {}
        queries = [str(q).strip() for q in (payload.get("queries") or []) if str(q).strip()]
        keywords = [str(k).strip() for k in (payload.get("keywords") or []) if str(k).strip()]
        return {
            "standalone": str(payload.get("standalone") or question).strip(),
            "queries": queries,
            "keywords": keywords,
            "resolved": bool(payload.get("resolved")),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def conversation_prompt(
        question: str, history: list[tuple[str, str]] | None, limit: int = 6
    ) -> str:
        """Render recent chat turns for inclusion in an agent prompt."""
        if not history:
            return "(no prior conversation)"
        recent = history[-limit:]
        return "\n".join(
            f"{'Learner' if role == 'user' else 'Tutor'}: {content.strip()[:600]}"
            for role, content in recent
        )

    @staticmethod
    def build_messages(system: str, user: str, history: list[tuple[str, str]] | None) -> list[Message]:
        """Convenience message builder used by the agents."""
        return [
            SystemMessage(content=system),
            UserMessage(content=f"{QueryTransformer.conversation_prompt(user, history)}\n\nCURRENT QUESTION:\n{user}"),
        ]