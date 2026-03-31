import re


def extract_error_code(query: str):
    """
    Extract payment error codes from user query.
    """
    patterns = [
        r"\bU\d{2}\b",
        r"\b[A-Z]{3,5}-\d{2,4}\b",
        r"\bDO_NOT_HONOR\b",
        r"\bINVALID_PIN\b",
        r"\bINSUFFICIENT_FUNDS\b",
        r"\bTIMEOUT\b"
    ]

    q = query.upper()
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(0)

    return None