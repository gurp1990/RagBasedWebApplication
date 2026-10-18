"""
Step 3 — Question generation (the "G" in RAG).

For a stored document:
  1. Retrieve its chunks from ChromaDB and pick passages spread across the document.
  2. Ask the LLM for one question per passage (structured JSON output), including
     the exact sentence the question is based on.
  3. Check the question is grounded and correct:
       a. code check  — the quoted sentence really appears in the passage;
       b. blind check — the LLM answers the question again without seeing the
          intended answer; both answers must agree.
  4. Store the accepted questions in ChromaDB.

Each question keeps its source passage and page number, so the answer can
always be traced back to the document.
"""

import logging
import random
import re
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app import vector_store
from app.config import settings
from app.llm import get_llm
from app.prompts import QUESTION_PROMPT, VERIFY_PROMPT

log = logging.getLogger(__name__)

MAX_ATTEMPTS_PER_PASSAGE = 3
MIN_QUOTE_OVERLAP = 0.8  # share of quote words that must appear in the passage
LETTERS = "ABCD"
LABEL_PREFIX = re.compile(r"^\(?[A-Da-d1-4][\).:]\s+")  # "A) ", "b. ", "(3) "


# ---------- structured output schemas (field order guides the LLM) ----------

class QuestionDraft(BaseModel):
    source_quote: str = Field(description="Sentence copied word for word from the context")
    question: str
    correct_answer: str
    wrong_answers: list[str] = Field(description="Exactly 3 wrong answers")
    explanation: str


class BlindAnswer(BaseModel):
    reasoning: str
    answer: Literal["A", "B", "C", "D"]


# ---------- public API ----------

def generate_questions(source: str, count: int | None = None) -> list[dict]:
    """Generate, verify and store `count` questions for a document."""
    count = count or settings.num_questions
    chunks = vector_store.get_chunks(source)
    if not chunks:
        raise ValueError(f"Document '{source}' has not been ingested yet.")

    questions: list[dict] = []
    for chunk in candidate_passages(chunks, count):
        if len(questions) == count:
            break
        question = generate_one(chunk, existing=questions)
        if question:
            questions.append(question)

    vector_store.save_questions(source, questions)
    return questions


def candidate_passages(chunks: list[Document], count: int) -> list[Document]:
    """
    Passages to build questions from: first `count` chunks spread evenly over
    the document, then every chunk twice more as back-up if a passage fails.
    """
    step = len(chunks) / count
    spread = [chunks[int(i * step) % len(chunks)] for i in range(count)]
    return spread + chunks * 2


def generate_one(chunk: Document, existing: list[dict]) -> dict | None:
    """Try a few times to produce a valid, verified question for one passage."""
    for attempt in range(1, MAX_ATTEMPTS_PER_PASSAGE + 1):
        try:
            draft = _clean(_draft_question(chunk, existing))
            problem = _find_problem(draft, chunk, existing)
            if problem is None:
                options = _shuffled_options(draft)
                if _blind_answer_matches(draft, options, chunk):
                    return _to_record(draft, options, chunk)
                problem = "blind answer disagrees"
            log.info("Rejected draft (chunk %s, attempt %d): %s",
                     chunk.metadata["chunk_index"], attempt, problem)
        except Exception as error:  # LLM or JSON errors: just try again
            log.warning("Attempt %d failed: %s", attempt, error)
    return None


# ---------- LLM calls ----------

def _draft_question(chunk: Document, existing: list[dict]) -> QuestionDraft:
    chain = QUESTION_PROMPT | get_llm().with_structured_output(QuestionDraft)
    return chain.invoke(
        {
            "context": chunk.page_content,
            "page": chunk.metadata["page"],
            "source": chunk.metadata["source"],
            "existing_questions": "\n".join(f"- {q['question']}" for q in existing) or "- (none)",
        }
    )


def _blind_answer_matches(draft: QuestionDraft, options: list[str], chunk: Document) -> bool:
    chain = VERIFY_PROMPT | get_llm().with_structured_output(BlindAnswer)
    result = chain.invoke(
        {
            "context": chunk.page_content,
            "question": draft.question,
            "options": "\n".join(f"{letter}) {o}" for letter, o in zip(LETTERS, options)),
        }
    )
    return options[LETTERS.index(result.answer)] == draft.correct_answer


# ---------- checks and helpers ----------

def _clean(draft: QuestionDraft) -> QuestionDraft:
    """Trim whitespace and remove 'A)'-style labels (the web page adds its own)."""
    def strip(text: str) -> str:
        return LABEL_PREFIX.sub("", text.strip()).strip()

    return QuestionDraft(
        source_quote=draft.source_quote.strip(),
        question=draft.question.strip(),
        correct_answer=strip(draft.correct_answer),
        wrong_answers=[strip(w) for w in draft.wrong_answers],
        explanation=draft.explanation.strip(),
    )


def _find_problem(draft: QuestionDraft, chunk: Document, existing: list[dict]) -> str | None:
    """Return a description of what is wrong with the draft, or None if it is fine."""
    if len(draft.wrong_answers) != 3:
        return f"expected 3 wrong answers, got {len(draft.wrong_answers)}"
    all_options = [draft.correct_answer, *draft.wrong_answers]
    if not all(all_options) or len({o.lower() for o in all_options}) != 4:
        return "empty or duplicate options"
    if any(q["question"].lower() == draft.question.lower() for q in existing):
        return "duplicate question"
    if _quote_overlap(draft.source_quote, chunk.page_content) < MIN_QUOTE_OVERLAP:
        return "quote not found in passage"
    return None


def _quote_overlap(quote: str, passage: str) -> float:
    """Share of the quote's words that appear in the passage (0..1)."""
    def words(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    quote_words = words(quote)
    passage_words = set(words(passage))
    if not quote_words:
        return 0.0
    return sum(w in passage_words for w in quote_words) / len(quote_words)


def _shuffled_options(draft: QuestionDraft) -> list[str]:
    options = [draft.correct_answer, *draft.wrong_answers]
    random.shuffle(options)
    return options


def _to_record(draft: QuestionDraft, options: list[str], chunk: Document) -> dict:
    return {
        "question": draft.question,
        "options": options,
        "correct_answer": draft.correct_answer,
        "explanation": draft.explanation,
        "source_quote": draft.source_quote,
        "page": chunk.metadata["page"],
        "source_text": chunk.page_content,
    }
