from collections import Counter
import streamlit as st


def extract_failure_reasons(docs):
    """
    Extracts failure reasons from retrieved documents.
    """
    reasons = []
    for doc in docs:
        for line in doc.splitlines():
            if line.lower().startswith("failure reason:"):
                reasons.append(line.split(":", 1)[1].strip())
    return reasons


def show_top_failure_reasons_chart(docs, top_n: int):
    """
    Displays a bar chart of top-N failure reasons.
    This function assumes it is called ONLY for analytical queries.
    """

    reasons = extract_failure_reasons(docs)

    if not reasons:
        st.info("No sufficient data available for chart visualization.")
        return

    counts = Counter(reasons).most_common(top_n)

    chart_data = {
        "Failure Reason": [reason for reason, _ in counts],
        "Count": [count for _, count in counts],
    }

    st.markdown("### Top Failure Reasons")
    st.bar_chart(chart_data, x="Failure Reason", y="Count")