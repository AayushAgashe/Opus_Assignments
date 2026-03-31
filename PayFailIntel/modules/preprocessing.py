import pandas as pd

def preprocess_base():
    banks = pd.read_csv("data/banks.csv")
    merchants = pd.read_csv("data/merchants.csv")
    users = pd.read_csv("data/users.csv")
    tx = pd.read_csv("data/transactions.csv")

    tx['txn_time'] = pd.to_datetime(tx['txn_time'], format="%d-%m-%Y %H:%M")
    tx = tx[tx['status'] != "SUCCESS"].copy()

    merged = (
        tx.merge(banks, on="bank_id", how="left")
          .merge(merchants, on="merchant_id", how="left")
          .merge(users, on="user_id", how="left")
    )

    merged = merged.rename(columns={
        "name_x": "merchant_name",
        "name_y": "user_name",
        "city": "user_city"
    })

    def doc(r):
        return f"""
Transaction ID: {r['txn_id']}
Timestamp: {r['txn_time']}
Amount: {r['amount']}
Status: {r['status']}
Failure Reason: {r['failure_reason']}
Bank: {r['bank_name']} (Server: {r['server_status']})
Merchant: {r['merchant_name']} ({r['category']})
User City: {r['user_city']}
"""

    merged["document"] = merged.apply(doc, axis=1)
    merged = merged.dropna(subset=["failure_reason"])

    merged.to_csv("outputs/final_failed_transactions.csv", index=False)
    return merged

def remove_instructional_lines(text: str):
    """
    Removes checklist-style and instructional lines that
    should NOT be treated as content.
    """
    bad_prefixes = [
        "provide a sample",
        "instructions:",
        "task:",
        "you should",
        "steps:",
        "example:",
    ]

    cleaned_lines = []
    for line in text.splitlines():
        if not any(line.lower().strip().startswith(p) for p in bad_prefixes):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def preprocess_text(text: str):
    """
    Cleans raw extracted text before chunking:
    - Removes checklist / instructional lines
    - Keeps only factual content
    """
    bad_prefixes = [
        "provide ",
        "instructions",
        "task:",
        "you should",
        "steps:",
        "example:",
        "sample:",
    ]

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if not any(stripped.startswith(p) for p in bad_prefixes):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)