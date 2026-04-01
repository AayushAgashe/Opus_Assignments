

from collections import Counter
import streamlit as st
import pandas as pd
import altair as alt


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
    Displays a clean horizontal bar chart for top-N failure reasons.
    Optimized for readability and compact spacing.
    """

    reasons = extract_failure_reasons(docs)

    if not reasons:
        st.info("No sufficient data available for chart visualization.")
        return

    counts = Counter(reasons).most_common(top_n)
    df = pd.DataFrame(counts, columns=["Failure Reason", "Count"])

    color_palette = [
        "#6f1d1b",
        "#bb9457",
        "#432818",
        "#99582a",
        "#9467bd",
    ]

    chart = (
        alt.Chart(df)
        .mark_bar(size=22)  # controls bar thickness
        .encode(
            y=alt.Y(
                "Failure Reason:N",
                sort="-x",
                title="Failure Reason",
                scale=alt.Scale(
                    paddingInner=0.2,   # tight spacing between bars
                    paddingOuter=0.05
                ),
            ),
            x=alt.X("Count:Q", title="Count"),
            color=alt.Color(
                "Failure Reason:N",
                scale=alt.Scale(range=color_palette),
                legend=None,
            ),
            tooltip=["Failure Reason", "Count"],
        )
        .properties(
            width=600,
            height=300,
            title="Top Failure Reasons",
        )
    )

    st.altair_chart(chart, use_container_width=True)
