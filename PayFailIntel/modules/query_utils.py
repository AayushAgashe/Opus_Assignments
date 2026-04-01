import re


def extract_error_code(question: str):
    """
    Extracts error codes from a query.
    Supports:
    - Symbolic codes (U16, RZP_021)
    - Numeric issuer response codes (e.g. Issuer Response Code: 14)
    """

    # Symbolic error codes (U16, RZP-021 etc.)
    match = re.search(r"\b[A-Z]{1,5}-?\d{1,4}\b", question)
    if match:
        return match.group(0)

    # Numeric issuer response codes
    issuer_match = re.search(
        r"(issuer\s+response\s+code|response\s+code)\s*[:\-]?\s*(\d{2})",
        question,
        re.IGNORECASE,
    )
    if issuer_match:
        return issuer_match.group(2)

    return None



def is_definition_query(question: str) -> bool:
    """
    Identifies definition / lookup style queries such as:
    - Which error code indicates insufficient funds?
    - What error code means incorrect PIN?
    """

    q = question.lower()
    return any(
        phrase in q
        for phrase in [
            "which error code",
            "what error code",
            "error code indicates",
            "error code for",
            "issuer response code",
        ]
    )
