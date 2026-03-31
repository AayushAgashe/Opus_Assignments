from collections import Counter
from modules.embeddings import embed
from modules.vectorstore import build_index
from modules.llm_phi2 import generate
from modules.query_utils import extract_error_code


# Helper: check if an error code exists in DB
import re

def error_code_exists_in_db(error_code: str, metadata) -> bool:
    """
    Checks whether the error code exists as an explicit error code,
    not as a substring of another value.
    """

    if not error_code:
        return True

    # Match exact error code as a whole token (word boundary)
    pattern = re.compile(rf"\b{re.escape(error_code)}\b", re.IGNORECASE)

    for doc in metadata["document"].values:
        if pattern.search(str(doc)):
            return True

    return False


# Default retrieval from historical DB
def retrieve(query, index, metadata, k=10):
    qv = embed([query])
    _, idx = index.search(qv, k)
    return metadata.iloc[idx[0]]["document"].tolist()


def ask(question, index, metadata):
    """
    SAFE RAG entrypoint.
    Guarantees zero hallucinations for unknown error codes.
    """

    # --- Inline error code extraction ---
    match = re.search(r"\b[A-Z]{1,5}-?\d{1,4}\b", question)
    error_code = match.group(0) if match else None

    # --- HARD UNKNOWN-ERROR GATE ---
    if error_code:
        pattern = re.compile(rf"\b{re.escape(error_code)}\b", re.IGNORECASE)
        found = False

        for doc in metadata["document"].values:
            if pattern.search(str(doc)):
                found = True
                break

        if not found:
            return "This document does not contain sufficient information for this error."

    # --- Normal retrieval ---
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

# Analytical (Top-N) explanation
def extract_failure_reason(doc: str):
    """
    Extracts 'Failure Reason: XYZ' from a document.
    """
    for line in doc.splitlines():
        if line.lower().startswith("failure reason:"):
            return line.split(":", 1)[1].strip()
    return None


def explain_top_n_reasons(question, index, metadata, top_n=3):
    """
    Used for analytical questions like:
    - Top failure reasons
    - Common causes
    """
    docs = retrieve(question, index, metadata, k=30)

    reasons = [
        extract_failure_reason(doc)
        for doc in docs
        if extract_failure_reason(doc)
    ]

    counts = Counter(reasons).most_common(top_n)

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


# Error-focused retrieval (Uploaded files)
def retrieve_with_error_focus(query, raw_chunks, k=5):
    """
    Retrieves only chunks that explicitly match the error code.
    If no such chunks exist, returns an empty list.
    """
    error_code = extract_error_code(query)

    if error_code:
        literal_matches = [
            c for c in raw_chunks
            if error_code in c.upper()
        ]

        # Strict unknown-error rule
        if not literal_matches:
            return []

        chunks = literal_matches
    else:
        chunks = raw_chunks

    emb = embed(chunks)
    index = build_index(emb)

    qv = embed([query])
    _, idx = index.search(qv, k)

    return [chunks[i] for i in idx[0]]


def generate_focused_answer(question, context_chunks):
    """
    Generates a focused explanation strictly for one error code.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
You are a Payment Failure Analysis Assistant.

Rules:
- Answer ONLY for the error code mentioned in the question.
- Do NOT speculate.
- Do NOT generalize.
- If context is insufficient, say so.

Question:
{question}

Context:
{context}

Answer in the following format:

Failure Code:
Explanation:
Likely Causes:
Recommended Actions:

If information is insufficient, reply with:
"This document does not contain sufficient information for this error."

Answer:
"""
    return generate(prompt)


# Hybrid retrieval (DB + Uploaded files)
def hybrid_retrieve(
    query,
    main_index,
    main_metadata,
    temp_index,
    temp_chunks,
    k=10
):
    """
    Retrieves results from:
    - Permanent FAISS (historical DB)
    - Temporary FAISS (uploaded files)
    """
    qv = embed([query])

    # Search historical DB
    _, idx_main = main_index.search(qv, k)
    main_docs = [
        main_metadata.iloc[i]["document"]
        for i in idx_main[0]
    ]

    # Search uploaded docs
    _, idx_temp = temp_index.search(qv, k)
    temp_docs = [
        temp_chunks[i]
        for i in idx_temp[0]
    ]

    return main_docs + temp_docs