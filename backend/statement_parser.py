"""
NeuroCard — Bank Statement Parser
==================================
Turns a real bank/credit-card statement export (CSV or Excel) into clean
(date, merchant, amount) transactions ready for the NLP classifier.

Handles the big-bank formats without asking the user anything:
  - HDFC  : "Date | Narration | ... | Withdrawal Amt. | Deposit Amt."
  - ICICI : "Transaction Date | Transaction Remarks | Withdrawal Amount ..."
  - SBI   : "Txn Date | Description | Debit | Credit"
  - Axis  : "Tran Date | PARTICULARS | DR | CR"
  - Generic: any sheet with a detectable date / description / amount column.

Banks bury the header row under logos and disclaimers, so we scan the first
40 rows for the most convincing header before parsing.

Narrations like "UPI-SWIGGY INSTAMART-9012345@ybl-YESB0YBLUPI-1234-Order"
are stripped down to "SWIGGY INSTAMART" before classification — exactly the
messy-real-world input the char-level TF-IDF model was built for.
"""

import io
import re
import pandas as pd

# Narration prefixes that are money movement, not card-routable spending:
# fund transfers, account-to-account, ATM/cash withdrawals, self transfers.
TRANSFER_PREFIXES = re.compile(
    r"^\s*(FT-|A2A|TPT-|IMPS[-/ ]|RTGS[-/ ]|NEFT[-/ ]|ATW[-/ ]|EAW[-/ ]|NWD[-/ ]|"
    r"IB\s+FUNDS|FD\s+THROUGH|SELF\b|CASH\b|CHQ\b)", re.I)

# A masked account number anywhere in the narration ("XXXXXXX7525") means a
# transfer to another bank account — never card-routable spending.
MASKED_ACCOUNT = re.compile(r"\bX{4,}\d{2,}\b", re.I)

# Words that mark a UPI counterparty as a business, not a person
BUSINESS_HINTS = re.compile(
    r"\b(ENTERPRISES?|TRADERS?|STORES?|MART|BAZAA?R|SHOPP?E?|JUICE|CAFE|COFFEE|HOTEL|RESTAURANT|"
    r"DHABA|FOODS?|SWEETS?|DAIRY|KIRANA|GENERAL|PROVISION|FRUITS?|VEGETABLES?|CHICKEN|MEDICAL|"
    r"PHARMA\w*|CLINIC|HOSPITAL|SALON|GYM|FITNESS|ACADEMY|CLASSES|TOURS?|TRAVELS?|PETROL|FUEL|"
    r"SERVICES?|SOLUTIONS?|AGENCY|STATIONER\w*|XEROX|BOOKS?|HARDWARE|MOBILE|TELECOM|ELECTRONICS?|"
    r"FASHION|COLLECTIONS?|GARMENTS?|TAILOR|LAUNDRY|BAKERY|CATERERS?|PVT|LTD|LIMITED|LLP|INDIA|"
    r"TECHNOLOG\w*|PAY\w*|RECHARGE|BROADBAND|CORP\w*|COMPANY|CO\b)", re.I)

# A cleaned name that looks like "FIRSTNAME [MIDDLE] LASTNAME" with no business hint
PERSON_NAME = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+){1,3}$")

DATE_HINTS = ("date",)
DESC_HINTS = ("narration", "description", "particulars", "remarks", "details",
              "transaction remarks", "merchant", "transaction details")
DEBIT_HINTS = ("withdrawal", "debit", "dr", "amount (dr)", "debit amount", "withdrawal amt")
AMOUNT_HINTS = ("amount", "txn amount", "transaction amount")
CREDIT_HINTS = ("deposit", "credit", "cr")


def _read_any(filename: str, blob: bytes) -> pd.DataFrame:
    """Read csv/xls/xlsx bytes with no assumption about header position."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(blob), header=None, dtype=str)
    # try a few encodings for CSV. names=range(60) fixes the classic bank-CSV
    # trap: the first row is a 1-column title, and pandas would otherwise
    # infer 1 column and discard every real transaction row as "bad".
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(blob), header=None, dtype=str, names=range(60),
                             encoding=enc, on_bad_lines="skip", engine="python")
            return df.dropna(axis=1, how="all")
        except Exception:
            continue
    raise ValueError("Could not read the file as CSV or Excel")


def _find_header(df: pd.DataFrame):
    """Scan the first 40 rows for the row that looks most like a header."""
    best, best_score = None, 0
    for i in range(min(40, len(df))):
        cells = [str(c).strip().lower() for c in df.iloc[i].tolist()]
        score = 0
        if any(any(h in c for h in DATE_HINTS) for c in cells):
            score += 2
        if any(any(h in c for h in DESC_HINTS) for c in cells):
            score += 2
        if any(any(h in c for h in DEBIT_HINTS + AMOUNT_HINTS) for c in cells):
            score += 1
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score < 4:
        raise ValueError("No header row found — is this a transaction statement?")
    return best


def _pick(cols, hints, exclude=()):
    for h in hints:  # hint priority order
        for c in cols:
            cl = str(c).strip().lower()
            if h in cl and not any(x in cl for x in exclude):
                return c
    return None


def clean_narration(raw: str) -> str:
    """'UPI-SWIGGY INSTAMART-9012@ybl-YESB0-1234-food' -> 'SWIGGY INSTAMART'"""
    s = str(raw).strip()
    s = re.sub(r"^(UPI|POS|IMPS|NEFT|RTGS|ACH D?|ECOM|VPS|ME DC|EAW|ATW)[-/: ]+", "", s, flags=re.I)
    s = re.sub(r"\b[\w.]+@\w+\b", " ", s)               # VPAs (no hyphens — don't eat brand names)
    s = re.sub(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", " ", s)     # IFSC codes
    s = re.sub(r"\b\d{4,6}X{3,10}\d{2,6}\b", " ", s, flags=re.I)  # masked card numbers
    s = re.sub(r"\b\d{6,}\b", " ", s)                   # long reference numbers
    parts = [p for p in re.split(r"[-/|]", s) if p.strip()]
    if parts:
        # first part that contains letters and isn't a generic word
        for part in parts:
            if re.search(r"[A-Za-z]{3}", part) and not re.match(
                    r"\s*(payment|paytm qr|bharatqr|refund|transfer|to|from)\s*$", part.strip(), re.I):
                s = part
                break
    s = re.sub(r"\s{2,}", " ", s).strip(" .-_")
    return s[:60] if s else str(raw)[:60]


def parse_statement(filename: str, blob: bytes) -> list[dict]:
    """Returns [{date, merchant, merchant_raw, amount}] — spends only."""
    raw = _read_any(filename, blob)
    hdr = _find_header(raw)
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    df = df.dropna(how="all")

    cols = list(df.columns)
    date_col = _pick(cols, DATE_HINTS, exclude=("value",))  # prefer txn date over value date
    desc_col = _pick(cols, DESC_HINTS)
    debit_col = _pick(cols, DEBIT_HINTS)
    amount_col = debit_col or _pick(cols, AMOUNT_HINTS)
    credit_col = _pick(cols, CREDIT_HINTS)

    if not (date_col and desc_col and amount_col):
        raise ValueError(f"Could not identify columns. Found: date={date_col}, "
                         f"description={desc_col}, amount={amount_col}. Columns: {cols[:10]}")

    out = []
    for _, r in df.iterrows():
        desc = str(r.get(desc_col, "")).strip()
        if not desc or desc.lower() == "nan":
            continue
        amt_raw = str(r.get(amount_col, "")).replace(",", "").replace("₹", "").strip()
        try:
            amount = abs(float(amt_raw))
        except ValueError:
            continue
        if not (amount > 0) or amount != amount:  # rejects 0, negatives AND NaN
            continue
        # skip credit/deposit rows when statement mixes both in one amount column
        if not debit_col and credit_col:
            cr_raw = str(r.get(credit_col, "")).replace(",", "").strip()
            try:
                if float(cr_raw) > 0:
                    continue
            except ValueError:
                pass
        if re.search(r"payment received|payment credit|autopay si|thank ?you", desc, re.I):
            continue  # bill payments to the card are not spends
        date = pd.to_datetime(str(r.get(date_col, "")), dayfirst=True, errors="coerce")
        if pd.isna(date):
            continue
        merchant = clean_narration(desc)
        is_transfer = bool(TRANSFER_PREFIXES.match(desc)) or bool(MASKED_ACCOUNT.search(desc))
        # UPI to an individual: name-shaped counterparty with no business words.
        # Final call is made by the caller, who can consult the classifier.
        maybe_person = (not is_transfer
                        and bool(PERSON_NAME.match(merchant))
                        and not BUSINESS_HINTS.search(merchant))
        out.append({
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "merchant": merchant,
            "merchant_raw": desc[:120],
            "amount": round(amount, 2),
            "is_transfer": is_transfer,
            "maybe_person": maybe_person,
        })
    if not out:
        raise ValueError("Parsed the file but found no debit transactions")
    return out
