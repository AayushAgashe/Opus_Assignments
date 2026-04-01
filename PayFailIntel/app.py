import streamlit as st

# Load FAISS index and metadata (historical DB)
from modules.vectorstore import load

# Utility to extract error codes (U16, issuer code 14, etc.)
from modules.query_utils import extract_error_code

# Core RAG logic functions
from modules.rag_engine import (
    handle_default_db_query,      # main decision-maker for DB mode
    retrieve_with_error_focus,    # strict retrieval for uploaded docs
    generate_focused_answer,      # structured answer generation
    hybrid_retrieve,              # combine DB + uploaded docs
)

# Chart rendering for analytics
from modules.charts import show_top_failure_reasons_chart

# File ingestion and preprocessing
from modules.ingestion import extract_text_from_file
from modules.preprocessing import preprocess_text
from modules.chunking import chunk_text

# Embeddings and FAISS utilities (used for uploaded docs fallback)
from modules.embeddings import embed
from modules.vectorstore import build_index


# --------------------------------------------------
# Load historical database (FAISS index + metadata)
# --------------------------------------------------
MAIN_INDEX, MAIN_METADATA = load()


# --------------------------------------------------
# Streamlit UI setup
# --------------------------------------------------
st.set_page_config(page_title="Payment Failure Intelligence Copilot", layout="wide")
st.title("Payment Failure Intelligence Copilot")


# --------------------------------------------------
# UI controls
# --------------------------------------------------

# Select retrieval mode
mode = st.radio(
    "Retrieval Mode:",
    [
        "Default Database Only",
        "Uploaded Files Only",
        "Hybrid (Database + Uploaded Files)",
    ],
)

# Select Top-N depth for analytics queries
top_n_value = int(
    st.selectbox("Failure reason depth:", ["Top 1", "Top 3", "Top 5"]).split()[-1]
)

# Optional document upload
uploads = st.file_uploader(
    "Upload CSV / PDF / DOCX / TXT (optional):",
    accept_multiple_files=True,
)

# User question input
question = st.text_input("Ask your question:")

# Trigger execution
run = st.button("Analyze")


# --------------------------------------------------
# Main execution block
# --------------------------------------------------
if run:
    # Guard against empty input
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    # ==============================
    # DEFAULT DATABASE ONLY MODE
    # ==============================
    if mode == "Default Database Only":

        # Delegate ALL logic to rag_engine
        result = handle_default_db_query(
            question, top_n_value, MAIN_INDEX, MAIN_METADATA
        )

        # Render result based on returned type
        if result["type"] == "error":
            st.warning("This document does not contain sufficient information.")

        elif result["type"] == "explanation":
            # Root-cause explanation (Top-1 or error-specific)
            st.write(result["text"])

        else:
            # Analytics mode (Top-N)
            st.subheader("Top Failure Insights")
            st.write(result["summary"])
            show_top_failure_reasons_chart(result["docs"], result["top_n"])


    # ==============================
    # UPLOADED FILES ONLY MODE
    # ==============================
    elif mode == "Uploaded Files Only":

        # Read and combine uploaded files into raw text
        raw = ""
        for f in uploads or []:
            raw += extract_text_from_file(f) + "\n"

        # Clean and chunk the text
        chunks = chunk_text(preprocess_text(raw))

        # Try strict error-focused retrieval first
        selected = retrieve_with_error_focus(question, chunks)

        # Fallback: semantic retrieval if strict match fails
        if not selected:
            if not chunks:
                st.warning("No valid content found in uploaded documents.")
                st.stop()

            emb = embed(chunks)

            # Guard against empty embeddings
            if emb is None or len(emb) == 0:
                st.warning("Uploaded documents do not contain usable information.")
                st.stop()

            # Build temporary FAISS index
            idx = build_index(emb)

            # Retrieve most relevant chunks semantically
            qv = embed([question])
            _, ids = idx.search(qv, min(5, len(chunks)))
            selected = [chunks[i] for i in ids[0]]

        # Generate structured explanation from selected chunks
        st.write(generate_focused_answer(question, selected))


    # ==============================
    # HYBRID MODE (DB + UPLOADS)
    # ==============================
    else:
        # Read uploaded files
        raw = ""
        for f in uploads or []:
            raw += extract_text_from_file(f) + "\n"

        # Preprocess and chunk uploaded docs
        chunks = chunk_text(preprocess_text(raw))

        # Build temporary FAISS index for uploaded docs
        t_index = build_index(embed(chunks))

        # Retrieve context from both DB and uploaded docs
        context = hybrid_retrieve(
            question,
            MAIN_INDEX,
            MAIN_METADATA,
            t_index,
            chunks,
        )

        # Root-cause queries → focused explanation
        if extract_error_code(question) or top_n_value == 1:
            st.write(generate_focused_answer(question, context))

        # Analytics queries → reuse DB analytics logic + hybrid context
        else:
            st.subheader("Top Failure Insights")
            st.write(
                handle_default_db_query(
                    question, top_n_value, MAIN_INDEX, MAIN_METADATA
                )["summary"]
            )
            show_top_failure_reasons_chart(context, top_n_value)