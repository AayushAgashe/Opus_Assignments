import re
from collections import Counter

from modules.embeddings import embed
from modules.vectorstore import build_index
from modules.llm_phi2 import generate
from modules.query_utils import extract_error_code


# ==================================================
# Safety & Validation
# ==================================================
def error_code_exists_in_db(error_code: str, metadata) -> bool:
    """Checks whether an error code exists as a full token in metadata."""
    if not error_code:
        return True

    pattern = re.compile(rf"\b{re.escape(error_code)}\b", re.IGNORECASE)
    return any(pattern.search(str(doc)) for doc in metadata["document"].values)


# ==================================================
# Core Retrieval
# ==================================================
def retrieve(query, index, metadata, k=10):
    qv = embed([query])
    _, idx = index.search(qv, k)
    return metadata.iloc[idx[0]]["document"].tolist()


def ask(question, index, metadata):
    """Safe RAG answer generation for DB-backed queries."""
    error_code = extract_error_code(question)

    if error_code and not error_code_exists_in_db(error_code, metadata):
        return "This document does not contain sufficient information for this error."

    docs = retrieve(question, index, metadata)
    context = "\n\n---\n\n".join(docs[:5])

    prompt = f"""
You are a Payment Failure Analysis Assistant.

Question:
{question}

Context:
{context}

Explain the failure clearly with root cause and recommended actions.

Answer:
"""
    return generate(prompt)


# ==================================================
# Analytics Helpers (DB)
# ==================================================
def get_query_specific_docs(question, index, metadata, k=50):
    qv = embed([question])
    _, idxs = index.search(qv, k)
    return [metadata.iloc[i]["document"] for i in idxs[0]]


def extract_failure_reasons(docs):
    reasons = []
    for doc in docs:
        for line in doc.splitlines():
            if line.lower().startswith("failure reason:"):
                reasons.append(line.split(":", 1)[1].strip())
    return reasons


def get_effective_top_n(docs, requested_top_n):
    unique = list(dict.fromkeys(extract_failure_reasons(docs)))
    return min(requested_top_n, len(unique))


def extract_ranked_list_only(text: str):
    return "\n".join(
        line for line in text.splitlines()
        if line.strip().startswith(tuple(str(i) for i in range(1, 10)))
    )


def explain_top_n_reasons(question, index, metadata, top_n=3):
    docs = retrieve(question, index, metadata, k=30)

    counts = Counter(extract_failure_reasons(docs)).most_common(top_n)
    summary = "\n".join(
        f"{i+1}. {reason} — {count} occurrences"
        for i, (reason, count) in enumerate(counts)
    )

    prompt = f"""
You are analyzing payment failure trends.

Question:
{question}

Top {top_n} Observed Failure Reasons:
{summary}

Explain why these failures occur and what actions can reduce them.

Answer:
"""
    return generate(prompt)


# ==================================================
# Uploaded / Hybrid Retrieval
# ==================================================
def retrieve_with_error_focus(query, raw_chunks, k=5):
    """
    Retrieves chunks that explicitly match an error code.
    Handles both symbolic (U16) and numeric issuer codes (14).
    """
    import re

    error_code = extract_error_code(query)

    if error_code:
        # Define pattern HERE (this was missing)
        pattern = re.compile(
            rf"\b{re.escape(error_code)}\b",
            re.IGNORECASE
        )

        literal_matches = [
            c for c in raw_chunks
            if (
                pattern.search(c)
                and (
                    "failure reason" in c.lower()
                    or "response code" in c.lower()
                    or "issuer" in c.lower()
                    or "error code" in c.lower()
                )
            )
        ]

        # Strict unknown‑error rule
        if not literal_matches:
            return []

        chunks = literal_matches
    else:
        chunks = raw_chunks

    # Semantic ranking within matched chunks
    emb = embed(chunks)
    index = build_index(emb)

    qv = embed([query])
    _, idx = index.search(qv, k)

    return [chunks[i] for i in idx[0]]


def generate_focused_answer(question, context_chunks):
    """
    Generates a focused explanation strictly for one error code,
    supporting both Failure Codes and Issuer Response Codes.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
You are a Payment Failure Analysis Assistant.

Rules:
- Use ONLY the provided context
- Do NOT speculate or guess
- If context is insufficient, say so

Question:
{question}

Context:
{context}

If the error is an Issuer Response Code:
- Extract the numeric code
- State its meaning clearly

If the error is a Failure Code:
- Extract the failure code

Answer in this format:

Error Code:
Meaning:
Description:

Answer:
"""
    return generate(prompt)




def hybrid_retrieve(query, main_index, main_metadata, temp_index, temp_chunks, k=10):
    qv = embed([query])

    _, idx_main = main_index.search(qv, k)
    main_docs = [main_metadata.iloc[i]["document"] for i in idx_main[0]]

    _, idx_temp = temp_index.search(qv, k)
    temp_docs = [temp_chunks[i] for i in idx_temp[0]]

    return main_docs + temp_docs


# ==================================================
# SINGLE Authoritative Handler (Default DB)
# ==================================================
def handle_default_db_query(question, top_n, index, metadata):
    error_code = extract_error_code(question)

    if error_code and not error_code_exists_in_db(error_code, metadata):
        return {"type": "error"}

    # Root cause / Top‑1
    if error_code or top_n == 1:
        return {
            "type": "explanation",
            "text": ask(question, index, metadata),
        }

    # Analytics
    docs = get_query_specific_docs(question, index, metadata)
    effective_top_n = get_effective_top_n(docs, top_n)

    summary = explain_top_n_reasons(
        question, index, metadata, top_n=effective_top_n
    )

    return {
        "type": "analytics",
        "summary": extract_ranked_list_only(summary),
        "docs": docs,
        "top_n": effective_top_n,
    }
