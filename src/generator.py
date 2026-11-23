"""
Generator v1 (baseline): prompt B, the generic RAG prompt.

Deliberately has NO rules about formulas, rounding or reading tables.
Those get added only if error analysis shows they are needed.
"""

import os
import time

from openai import OpenAI

PROMPT_VERSION = "v1_generic"

PROMPT = """You are a financial analyst assistant. Answer the question using ONLY
the information in the context below. If the context does not contain
the answer, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""


def format_context(chunks: list[dict]) -> str:
    """Each chunk needs "text"; "doc_name" is shown as the source if present."""
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("doc_name", "unknown")
        parts.append(f"[Chunk {i} | source: {source}]\n{c['text']}")
    return "\n\n".join(parts)


_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set. Put it in .env.")
        _client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _client


def generate(question: str, chunks: list[dict], model: str = "deepseek-flash") -> dict:
    prompt = PROMPT.format(context=format_context(chunks), question=question)
    t0 = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return {
        "answer": resp.choices[0].message.content.strip(),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
