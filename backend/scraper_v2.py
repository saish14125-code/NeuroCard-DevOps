"""
NeuroCard — Live Card Scraper v2
=================================
Fetches real credit card data from TWO independent sources and merges them:

  Source A: 1finance.co.in  — server-rendered pages with per-category reward
            percentages in a predictable format. Parsed DETERMINISTICALLY
            (regex/text), no LLM needed. This is our numeric ground truth.
  Source B: PaisaBazaar card pages — rich prose with caps, fee waivers,
            milestone details. Used as context for Gemini.

  Merge:    Gemini gets (parsed 1finance numbers + both page texts) and
            produces one row in our exact schema. The parsed numbers are
            passed as ground truth it must not contradict.

  Guardrails (same as extract_card.py):
    - Pydantic validation on every row; failures go to review_queue.jsonl,
      never into the CSV.
    - Upsert only — existing rows are updated, never mass-deleted, so the
      catalog can only grow or improve. Hand-curated cards are skipped
      unless --overwrite is passed.

Usage:
  python scraper_v2.py             # scrape all targets not already in CSV
  python scraper_v2.py --limit 5   # first 5 missing targets (test run)
  python scraper_v2.py --overwrite # re-scrape even existing cards
  python scraper_v2.py --enrich    # Gemini-verify caps on auto-parsed rows (~20/day quota)
"""

import re
import sys
import time
import json
import random
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

from extract_card import CardRow, upsert, CSV_PATH, MODELS
from dotenv import load_dotenv
from google import genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
REVIEW_PATH = os.path.join(BASE_DIR, "review_queue.jsonl")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

# ── TARGET CARDS ─────────────────────────────────────────────────────────────
# (display name we want, 1finance slug, bank). Popular, currently-issued cards.
TARGETS = [
    ("Axis Magnus",                 "axis-magnus-card",                        "Axis"),
    ("Axis Select",                 "axis-select-card",                        "Axis"),
    ("Axis Neo",                    "axis-neo-card",                           "Axis"),
    ("Airtel Axis",                 "axis-airtel-card",                        "Axis"),
    ("Axis Horizon",                "axis-horizon-card",                       "Axis"),
    ("HDFC Regalia",                "hdfc-regalia-card",                       "HDFC"),
    ("HDFC Diners Club Black",      "hdfc-diners-club-black-card",             "HDFC"),
    ("HDFC Diners Club Black Metal","hdfc-diners-club-black-metal-edition",    "HDFC"),
    ("HDFC Diners Club Privilege",  "hdfc-diners-club-privilege-card",         "HDFC"),
    ("HDFC MoneyBack Plus",         "hdfc-moneyback-plus-card",                "HDFC"),
    ("Tata Neu Plus HDFC",          "hdfc-tata-neu-plus-card",                 "HDFC"),
    ("Marriott Bonvoy HDFC",        "hdfc-marriott-bonvoy-card",               "HDFC"),
    ("HDFC IRCTC",                  "hdfc-irctc-card",                         "HDFC"),
    ("HDFC Pixel Play",             "hdfc-pixels-play-card",                   "HDFC"),
    ("SBI Elite",                   "sbi-elite-card",                          "SBI"),
    ("SBI Prime",                   "sbi-prime-card",                          "SBI"),
    ("SBI SimplyCLICK",             "sbi-simplyclick-card",                    "SBI"),
    ("SBI SimplySAVE",              "sbi-simplysave-card",                     "SBI"),
    ("SBI BPCL Octane",             "sbi-bpcl-octane-card",                    "SBI"),
    ("SBI IRCTC Premier",           "sbi-irctc-premier-card",                  "SBI"),
    ("ICICI Coral",                 "icici-coral-card",                        "ICICI"),
    ("ICICI Sapphiro",              "icici-sapphiro-card",                     "ICICI"),
    ("ICICI Emeralde Private Metal","icici-emeralde-private-metal-card",       "ICICI"),
    ("ICICI HPCL Super Saver",      "icici-hpcl-super-saver-card",             "ICICI"),
    ("ICICI MakeMyTrip Signature",  "icici-makemytrip-signature-card",         "ICICI"),
    ("Amex Platinum Travel",        "american-express-platinum-travel-card",   "American Express"),
    ("Amex SmartEarn",              "american-express-smart-earn-card",        "American Express"),
    ("AU LIT",                      "au-lit-live-it-today-card",               "AU"),
    ("AU Zenith",                   "au-zenith-card",                          "AU"),
    ("IDFC First Select",           "idfc-first-select-card",                  "IDFC"),
    ("IDFC First Wealth",           "idfc-first-wealth-card",                  "IDFC"),
    ("IDFC First Millennia",        "idfc-first-millennia-card",               "IDFC"),
    ("IDFC First WOW",              "idfc-first-wow-card",                     "IDFC"),
    ("IndusInd Legend",             "indusind-legend-card",                    "IndusInd"),
    ("IndusInd Tiger",              "indusind-tiger-card",                     "IndusInd"),
    ("RBL ShopRite",                "rbl-shop-rite-card",                      "RBL"),
    ("HSBC Cashback",               "hsbc-cashback-card",                      "HSBC"),
    ("Standard Chartered Smart",    "standard-chartered-smart-card",           "Standard Chartered"),
    ("Standard Chartered Ultimate", "standard-chartered-ultimate-card",        "Standard Chartered"),
    ("Federal Scapia",              "federal-scapia-card",                     "Federal"),
    ("Kotak Myntra",                "kotak-mahindra-myntra-card",              "Kotak"),
    ("Kotak 811 Dream Different",   "kotak-mahindra-811-dreamdifferent-card",  "Kotak"),
    ("BOB Eterna",                  "bank-of-baroda-eterna-card",              "BOB"),
    ("YES First Exclusive",         "yes-first-exclusive-card",                "YES"),
    ("DBS Vantage",                 "dbs-vantage-card",                        "DBS"),
    ("SBM OneCard",                 "sbm-onecard",                             "SBM"),
]

PB_BANK_PATH = {
    "Axis": "axis-bank", "HDFC": "hdfc-bank", "SBI": "sbi-card",
    "ICICI": "icici-bank", "IndusInd": "indusind-bank", "RBL": "rbl-bank",
    "HSBC": "hsbc-bank", "Standard Chartered": "standard-chartered-bank",
    "Federal": "federal-bank", "Kotak": "kotak-mahindra-bank",
    "AU": "au-small-finance-bank", "IDFC": "idfc-first-bank",
    "YES": "yes-bank", "American Express": "american-express",
}

# 1finance expense buckets → NeuroCard categories
ONEF_SECTIONS = [
    "Groceries", "Transportation", "Utilities", "Clothing and Fashion",
    "Technology and Electronics", "Food and Dining", "Entertainment",
    "Travel and Leisure",
]


def fetch(url, timeout=25):
    time.sleep(random.uniform(1.5, 3.0))  # polite
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200 and len(r.text) > 5000:
            return r.text
    except requests.RequestException as e:
        print(f"    fetch failed: {e}")
    return None


def page_text(html, limit=16000):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    return text[:limit]


def parse_1finance(text):
    """Deterministic parse of 1finance's structured sections."""
    out = {"sections": {}, "notes": {}}

    m = re.search(r"Annual Fee\s*\n\s*(.+)", text)
    if m:
        fee_raw = m.group(1).strip()
        out["annual_fee_raw"] = fee_raw
        if re.search(r"life\s*time\s*free|lifetime free", fee_raw, re.I):
            out["annual_fee"] = 0
        else:
            digits = re.sub(r"[^\d]", "", fee_raw.split("+")[0])
            out["annual_fee"] = int(digits) if digits else None

    for sec in ONEF_SECTIONS:
        # e.g. "Food and Dining\n\nCashback of 1.0% (5% cashback on online spends)"
        pat = re.escape(sec) + r"\s*\n+\s*(Zero[^\n]*|Cashback of [\d.]+%[^\n]*|Earn Reward Points at a rate of [\d.]+%[^\n]*|[^\n]*%[^\n]*)"
        m = re.search(pat, text)
        if not m:
            continue
        line = m.group(1).strip()
        if re.match(r"zero", line, re.I):
            rate = 0.0
        else:
            rm = re.search(r"(\d+(?:\.\d+)?)\s*%", line)  # strict: '1..5' on some pages must not crash
            rate = float(rm.group(1)) if rm else 0.0
        out["sections"][sec] = rate
        note = re.search(r"\((.*?)\)", line)
        if note:
            out["notes"][sec] = note.group(1)

    m = re.search(r"conversion rate of 1 RP = ₹?\s*([\d.]+)", text)
    if m:
        out["point_value_rs"] = float(m.group(1))
    elif re.search(r"Cashback of", text):
        out["point_value_rs"] = 1.0

    for label, key in [("Welcome Benefits", "welcome"), ("Milestone Benefits", "milestone"),
                       ("Domestic Lounge Access", "lounge_dom"), ("International Lounge Access", "lounge_intl"),
                       ("Eligibility Criteria", "eligibility")]:
        m = re.search(re.escape(label) + r"\s*\n+\s*([^\n]+)", text)
        if m:
            out[key] = m.group(1).strip()
    return out


def map_to_neurocard(parsed):
    """1finance buckets → our 6 categories (best-effort ground truth)."""
    s = parsed.get("sections", {})
    def g(k): return s.get(k, None)
    cats = {}
    if g("Food and Dining") is not None: cats["dining"] = g("Food and Dining")
    if g("Travel and Leisure") is not None: cats["travel"] = g("Travel and Leisure")
    shopping_vals = [v for v in [g("Clothing and Fashion"), g("Technology and Electronics")] if v is not None]
    if shopping_vals: cats["shopping"] = max(shopping_vals)
    if g("Transportation") is not None: cats["fuel"] = g("Transportation")
    if g("Utilities") is not None: cats["utility"] = g("Utilities")
    base_vals = [v for v in s.values() if v and v > 0]
    cats["others"] = min(base_vals) if base_vals else 0.0
    return cats


MERGE_PROMPT = """You are a credit card data analyst building a card database.

Produce ONE JSON object in EXACTLY this schema (fill every key):
{schema}

GROUND TRUTH (deterministically parsed from 1finance — your reward_points_*
values must match these where present; they are effective value-back percents):
{ground_truth}

Additional parsed facts: {facts}

RULES:
- reward_points_<cat> = effective value-back percent (float). Use ground truth
  values where given. Where a category is missing, derive it from the page text
  using (points per Rs.100) x (rupee value of 1 point). 0 = category excluded.
- cap_<cat> = max reward VALUE in RUPEES per MONTH for that category, from the
  text (convert points caps to rupees via point value; quarterly caps / 3).
  0 if uncapped or unknown. NEVER invent a cap not supported by the text.
- min_income: annual, in rupees (monthly figures x 12). 0 if not stated.
- card_tier: Basic (fee<1000) / Mid (1000-2999) / Premium (3000-9999) /
  Super_Premium (10000+) — adjust if the text clearly positions it otherwise.
- eligibility_level: Low/Medium/High/Elite matching the tier.
- reward_type: cashback | points | miles | neucoins | coins.
- highlights: 3-5 short selling points separated by "|", include cap warnings.
- exclusions: comma-separated spend types that earn nothing.
- welcome_benefit / milestone_benefit: one concise sentence each, "" if none.
- lounge_domestic_yr / lounge_intl_yr: visits per year; 99 means unlimited.
- Card name must be exactly: {name}  | bank_name exactly: {bank}
- Return ONLY the JSON object. No markdown fences, no commentary.

PAGE TEXT — 1finance:
---
{text_1f}
---

PAGE TEXT — PaisaBazaar (may be empty):
---
{text_pb}
---"""


class DailyQuotaExhausted(Exception):
    pass


def build_deterministic(name, bank, parsed, cats, source_url):
    """No-LLM row built purely from the deterministic 1finance parse.
    Used when Gemini's daily quota is gone — caps stay 0 (unknown) and the
    row is tagged for later enrichment. Better a real-rates row today than
    no row at all; Gemini can polish it on a future run."""
    fee = parsed.get("annual_fee") or 0
    tier = ("Basic" if fee < 1000 else "Mid" if fee < 3000 else
            "Premium" if fee < 10000 else "Super_Premium")
    elig = {"Basic": "Low", "Mid": "Medium", "Premium": "High", "Super_Premium": "Elite"}[tier]
    is_cashback = parsed.get("point_value_rs", 0) == 1.0 and "Cashback" in json.dumps(parsed.get("notes", {})) or \
                  any("cashback" in str(v).lower() for v in parsed.get("notes", {}).values())
    top_cat = max(cats, key=cats.get) if cats else "others"
    lounge_dom = parsed.get("lounge_dom", "")
    has_lounge = bool(lounge_dom) and not re.search(r"zero|not available", str(lounge_dom), re.I)
    notes = [f"{k}: {v}" for k, v in list(parsed.get("notes", {}).items())[:3]]
    highlights = "|".join(notes + ["Auto-parsed from 1finance — caps pending verification"])
    return {
        "card_name": name, "bank_name": bank, "card_category": top_cat.capitalize(),
        "network_type": "", "annual_fee": int(fee), "joining_fee": int(fee),
        "fee_waiver_spend": 0, "min_income": 0,
        "reward_type": "cashback" if is_cashback else "points",
        "point_value_rs": float(parsed.get("point_value_rs", 1.0) or 1.0),
        "reward_points_dining": cats.get("dining", 0), "reward_points_travel": cats.get("travel", 0),
        "reward_points_shopping": cats.get("shopping", 0), "reward_points_fuel": cats.get("fuel", 0),
        "reward_points_utilities": cats.get("utility", 0), "reward_points_others": cats.get("others", 0),
        "cap_dining": 0, "cap_travel": 0, "cap_shopping": 0, "cap_fuel": 0, "cap_utility": 0, "cap_others": 0,
        "fuel_surcharge_waiver": 0, "forex_markup_pct": 3.5,
        "lounge_access": 1 if has_lounge else 0,
        "lounge_domestic_yr": 99 if re.search(r"unlimited", str(lounge_dom), re.I) else (4 if has_lounge else 0),
        "lounge_intl_yr": 0, "lounge_type": "Domestic" if has_lounge else "",
        "card_tier": tier, "eligibility_level": elig,
        "max_benefit_category": top_cat.capitalize(),
        "welcome_benefit": str(parsed.get("welcome", ""))[:200],
        "milestone_benefit": str(parsed.get("milestone", ""))[:200],
        "exclusions": "", "highlights": highlights[:500],
        "source_url": source_url, "data_updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "caps_verified": 0,
    }


def gemini_merge(client, name, bank, parsed, cats, text_1f, text_pb, schema_example):
    prompt = MERGE_PROMPT.format(
        schema=schema_example,
        ground_truth=json.dumps(cats),
        facts=json.dumps({k: v for k, v in parsed.items() if k != "sections"}, ensure_ascii=False)[:1500],
        name=name, bank=bank,
        text_1f=text_1f[:12000], text_pb=(text_pb or "")[:12000],
    )
    for model in MODELS:
        for attempt in range(3):
            try:
                r = client.models.generate_content(model=model, contents=prompt)
                raw = r.text.strip()
                if raw.startswith("```"):
                    raw = raw[raw.index("{"):raw.rindex("}") + 1]
                return json.loads(raw)
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg and "PerDay" in msg:
                    raise DailyQuotaExhausted(model)  # no point retrying today
                if "RESOURCE_EXHAUSTED" in msg:      # per-minute limit — wait it out
                    wait = 30
                    m = re.search(r"retry in ([\d.]+)s", msg, re.I)
                    if m:
                        wait = float(m.group(1)) + 2
                    print(f"    rate-limited, waiting {int(wait)}s...")
                    time.sleep(wait)
                    continue
                print(f"    {model}: {msg[:160]}")
                break
    return None


def guess_paisabazaar(name, bank):
    bank_path = PB_BANK_PATH.get(bank)
    if not bank_path:
        return None
    nm = re.sub(r"[^a-z0-9 ]", "", name.lower().replace(bank.lower(), "").strip())
    nm = re.sub(r"\s+", "-", nm.strip())
    candidates = [
        f"https://www.paisabazaar.com/{bank_path}/{nm}-credit-card/",
        f"https://www.paisabazaar.com/{bank_path}/{bank.lower().split()[0]}-{nm}-credit-card/",
    ]
    for url in candidates:
        html = fetch(url)
        if html and "Page not found" not in html[:4000]:
            print(f"    + PaisaBazaar: {url}")
            return page_text(html)
    return None


def cross_check(row, cats):
    """Flag when Gemini's numbers drift from the deterministic parse."""
    issues = []
    keymap = {"dining": "reward_points_dining", "travel": "reward_points_travel",
              "shopping": "reward_points_shopping", "fuel": "reward_points_fuel",
              "utility": "reward_points_utilities", "others": "reward_points_others"}
    for cat, col in keymap.items():
        if cat in cats and abs(float(row.get(col, 0)) - cats[cat]) > 1.5:
            issues.append(f"{cat}: gemini={row.get(col)} vs 1finance={cats[cat]}")
    return issues


def main():
    overwrite = "--overwrite" in sys.argv
    enrich = "--enrich" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY missing"); return
    client = genai.Client(api_key=api_key)

    from extract_card import SCHEMA_EXAMPLE
    existing, unverified = set(), set()
    if os.path.exists(CSV_PATH):
        _df = pd.read_csv(CSV_PATH)
        existing = set(_df["card_name"].str.lower())
        if "caps_verified" in _df.columns:
            unverified = set(_df[pd.to_numeric(_df["caps_verified"], errors="coerce").fillna(0) < 1]["card_name"].str.lower())

    if enrich:
        # target only the rows still waiting for Gemini-verified caps
        todo = [(n, s, b) for n, s, b in TARGETS if n.lower() in unverified]
    else:
        todo = [(n, s, b) for n, s, b in TARGETS if overwrite or n.lower() not in existing]
    if limit:
        todo = todo[:limit]
    print(f"Scraping {len(todo)} cards ({len(existing)} already in catalog)...\n")

    added, failed = 0, []
    for name, slug, bank in todo:
        print(f"→ {name}")
        url_1f = f"https://1finance.co.in/product-scoring/credit-card/{slug}"
        html = fetch(url_1f)
        if not html or "Annual Fee" not in html:
            print("    1finance page missing — skipped"); failed.append((name, "no 1finance page")); continue

        text_1f = page_text(html)
        parsed = parse_1finance(text_1f)
        cats = map_to_neurocard(parsed)
        if not cats or all(v == 0 for v in cats.values()):
            print("    could not parse any rates — skipped"); failed.append((name, "no rates parsed")); continue

        data = None
        if not getattr(main, "_no_gemini", False):
            text_pb = guess_paisabazaar(name, bank)
            try:
                data = gemini_merge(client, name, bank, parsed, cats, text_1f, text_pb, SCHEMA_EXAMPLE)
                time.sleep(4)  # stay far under free-tier RPM
            except DailyQuotaExhausted:
                if enrich:
                    print("
⚠ Gemini DAILY quota exhausted — enrichment stops here. "
                          "Run 'python scraper_v2.py --enrich' again tomorrow; progress is saved.")
                    break
                print("    ⚠ Gemini DAILY quota exhausted — switching to deterministic-only mode")
                main._no_gemini = True

        if not data:
            if enrich:
                failed.append((name, "gemini unavailable")); continue
            data = build_deterministic(name, bank, parsed, cats, url_1f)
            print("    (deterministic row — enrich with Gemini on a later run)")

        data["card_name"] = name
        data["bank_name"] = bank
        data["source_url"] = url_1f
        data["data_updated"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        data.setdefault("caps_verified", 1)

        try:
            row = CardRow(**data).model_dump()
        except Exception as e:
            print(f"    validation failed → review_queue")
            with open(REVIEW_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"card": name, "error": str(e)[:500], "data": data}, ensure_ascii=False) + "\n")
            failed.append((name, "validation")); continue

        issues = cross_check(row, cats)
        if issues:
            print(f"    ⚠ cross-check drift: {issues} — 1finance values enforced")
            keymap = {"dining": "reward_points_dining", "travel": "reward_points_travel",
                      "shopping": "reward_points_shopping", "fuel": "reward_points_fuel",
                      "utility": "reward_points_utilities", "others": "reward_points_others"}
            for cat, col in keymap.items():
                if cat in cats:
                    row[col] = cats[cat]

        upsert(row)
        added += 1

    print(f"\nDone. {added} cards written, {len(failed)} failed:")
    for n, why in failed:
        print(f"  ✗ {n}: {why}")


if __name__ == "__main__":
    main()
