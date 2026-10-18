"""
All LLM prompts in one place, so they are easy to read and tune.
"""

from langchain_core.prompts import ChatPromptTemplate

# Step 1: write a question. The model must first quote the sentence it uses,
# which keeps the question grounded in the passage (and lets code check it).
QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher who writes multiple-choice exam questions. "
            "Use ONLY facts stated in the given context. Never use outside knowledge.",
        ),
        (
            "human",
            """Context (from page {page} of "{source}"):
\"\"\"
{context}
\"\"\"

Write ONE multiple-choice question about a single fact in the context.

Fill the fields in this order:
- "source_quote": copy, word for word, the one sentence from the context that contains the fact.
- "question": a clear question about that fact.
- "correct_answer": the short, correct answer, taken from the quote.
- "wrong_answers": exactly 3 short answers that are clearly WRONG according to the
  context (for example, facts about a different term in the context).
- "explanation": one or two sentences explaining why the answer is correct.

Do not ask about the same fact as these existing questions:
{existing_questions}""",
        ),
    ]
)

# Step 2: answer the question "blind" (without seeing the intended answer).
# If this independent answer matches, the question is unambiguous and correct.
VERIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful student. Answer using ONLY the given context.",
        ),
        (
            "human",
            """Context:
\"\"\"
{context}
\"\"\"

Question: {question}
{options}

First write your short "reasoning" based on the context, then give the letter of
the single correct option in "answer".""",
        ),
    ]
)
