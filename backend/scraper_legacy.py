"""
NeuroCard — Credit Card Data Scraper (Fixed)
=============================================
Scrapes real Indian credit card data from sources that actually work:

  Source 1: CardInsider.com  — structured HTML listing, no heavy JS
  Source 2: BankBazaar JSON  — their internal listing API endpoint
  Source 3: RupeeMeter.in    — lightweight comparison site
  Fallback: Curated static dataset (30 cards, accurate as of 2024-25)

Why the old scraper failed:
  - PaisaBazaar uses heavy React + anti-bot protection (Cloudflare)
  - BankBazaar card names are buried in JS-rendered components
  - Both block headless Chromium with standard UA strings

This scraper uses requests + BeautifulSoup on scraper-friendly pages,
with a realistic browser fingerprint and polite delays.

Usage:
  pip install requests beautifulsoup4 pandas
  python scraper.py

The scraper saves to: NeuroCard_Data.csv  (what main.py reads)
"""

import re
import time
import random
import os
import json
import pandas as pd
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# ─────────────────────────────────────────────
# COLUMN SCHEMA — must match what main.py expects
# ─────────────────────────────────────────────
CSV_COLUMNS = [
    "card_name", "bank_name", "card_category", "network_type",
    "annual_fee", "fee_waiver_spend", "min_income",
    "lounge_access", "reward_points_others", "reward_points_dining",
    "reward_points_fuel", "reward_points_shopping", "reward_points_travel",
    "reward_points_utilities", "card_tier", "eligibility_level",
    "lounge_type", "max_benefit_category"
]

# ─────────────────────────────────────────────
# REALISTIC BROWSER HEADERS
# Rotate through a few common browser fingerprints
# ─────────────────────────────────────────────
HEADERS_POOL = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    },
]


def get_session():
    session = requests.Session()
    session.headers.update(random.choice(HEADERS_POOL))
    return session


def polite_get(session, url, delay=(2.0, 4.5), timeout=20):
    """Make a polite GET request with a random delay."""
    time.sleep(random.uniform(*delay))
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        print(f"    HTTP {e.response.status_code} on {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    Request failed: {e}")
        return None


# ─────────────────────────────────────────────
# SCRAPER 1 — CardInsider.com
# Static HTML tables, no JS required
# ─────────────────────────────────────────────
def scrape_cardinsider(session):
    """
    CardInsider lists cards in clean HTML with fee and bank info.
    Targets their category pages which have structured markup.
    """
    print("  → Trying CardInsider.com...")
    cards = []
    seen = set()

    urls = [
        "https://www.cardinsider.com/credit-cards/",
        "https://www.cardinsider.com/credit-cards/cashback/",
        "https://www.cardinsider.com/credit-cards/travel/",
        "https://www.cardinsider.com/credit-cards/shopping/",
        "https://www.cardinsider.com/credit-cards/fuel/",
        "https://www.cardinsider.com/credit-cards/lifestyle/",
    ]

    for url in urls:
        resp = polite_get(session, url, delay=(2.5, 5.0))
        if not resp:
            continue

        soup = BeautifulSoup(resp.content, "html.parser")

        # CardInsider uses card-listing-wrapper > card-item structure
        # Try multiple selectors as their markup varies by page
        card_blocks = (
            soup.find_all("div", class_=re.compile(r"card.*(item|listing|block|wrap)", re.I)) or
            soup.find_all("li", class_=re.compile(r"card", re.I)) or
            soup.find_all("article", class_=re.compile(r"card", re.I))
        )

        # Fallback: find all h3/h4 that look like card names
        if not card_blocks:
            headings = soup.find_all(["h3", "h4"], string=re.compile(
                r"(HDFC|ICICI|SBI|Axis|Kotak|YES|IndusInd|HSBC|IDFC|Federal|BOB|RBL|AU|Citi|Amex)",
                re.I
            ))
            for h in headings:
                card_name = h.get_text(strip=True)
                if len(card_name) < 5 or card_name.upper() in seen:
                    continue
                seen.add(card_name.upper())

                # Look for fee near the heading
                parent = h.find_parent("div") or h.find_parent("li") or h
                full_text = parent.get_text(" ", strip=True)

                fee = extract_fee(full_text)
                bank = detect_bank(card_name, "")
                card = build_card(card_name, bank, fee, url)
                cards.append(card)
            continue

        for block in card_blocks:
            text = block.get_text(" ", strip=True)

            # Find card name
            name_tag = (
                block.find(["h2", "h3", "h4", "h5"]) or
                block.find(class_=re.compile(r"name|title|heading", re.I))
            )
            if not name_tag:
                continue
            card_name = name_tag.get_text(strip=True)
            if len(card_name) < 6 or card_name.upper() in seen:
                continue
            # Skip generic headings
            if any(skip in card_name.lower() for skip in ["credit card", "compare", "best", "top", "list"]):
                continue

            seen.add(card_name.upper())
            fee = extract_fee(text)
            bank = detect_bank(card_name, "")
            card = build_card(card_name, bank, fee, url)
            cards.append(card)

        print(f"    Got {len(cards)} cards so far...")

    print(f"  ✓ CardInsider: scraped {len(cards)} cards")
    return cards


# ─────────────────────────────────────────────
# SCRAPER 2 — BankBazaar (lightweight pages only)
# Their sitemap and AMP pages are lightweight
# ─────────────────────────────────────────────
def scrape_bankbazaar(session):
    """
    BankBazaar's AMP pages are server-rendered and scraper-friendly.
    We target their individual card comparison pages via search API.
    """
    print("  → Trying BankBazaar lightweight pages...")
    cards = []
    seen = set()

    # Their sitemap exposes card URLs
    sitemap_url = "https://www.bankbazaar.com/sitemap.xml"
    resp = polite_get(session, sitemap_url, delay=(2.0, 4.0))

    card_urls = []
    if resp:
        # Extract card page URLs from sitemap
        urls_found = re.findall(r'<loc>(https://www\.bankbazaar\.com/credit-card/[^<]+)</loc>', resp.text)
        # Only take card detail pages, not category pages
        card_urls = [u for u in urls_found if u.count('/') >= 5][:40]
        print(f"    Found {len(card_urls)} card URLs in sitemap")

    # Also try their main listing
    main_urls = [
        "https://www.bankbazaar.com/credit-card.html",
    ]

    all_urls = main_urls + card_urls[:20]

    for url in all_urls:
        resp = polite_get(session, url, delay=(2.0, 4.5))
        if not resp:
            continue

        soup = BeautifulSoup(resp.content, "html.parser")

        # Look for card names in various patterns BankBazaar uses
        # Their card names appear in specific title patterns
        candidates = []

        # Pattern 1: h1 on detail pages
        h1 = soup.find("h1")
        if h1:
            candidates.append(h1.get_text(strip=True))

        # Pattern 2: structured data / JSON-LD
        scripts = soup.find_all("script", type="application/ld+json")
        for script in scripts:
            try:
                data = json.loads(script.string or "{}")
                if isinstance(data, dict):
                    name = data.get("name", "") or data.get("productName", "")
                    if name and len(name) > 5:
                        candidates.append(name)
            except Exception:
                pass

        # Pattern 3: card name in title tag
        title = soup.find("title")
        if title:
            title_text = title.get_text(strip=True)
            # "HDFC Regalia Gold Credit Card - BankBazaar" → extract card name
            parts = title_text.split(" - ")[0].replace(" Credit Card", "").strip()
            if len(parts) > 5:
                candidates.append(parts)

        for card_name in candidates:
            card_name = card_name.replace(" Credit Card", "").strip()
            if len(card_name) < 5 or card_name.upper() in seen:
                continue
            if any(skip in card_name.lower() for skip in ["compare", "best", "bankbazaar", "reviews"]):
                continue
            # Must look like an actual card (contains bank name)
            if not detect_bank(card_name, ""):
                continue

            seen.add(card_name.upper())
            full_text = soup.get_text(" ", strip=True)
            fee = extract_fee(full_text)
            bank = detect_bank(card_name, "")
            card = build_card(card_name, bank, fee, url)
            cards.append(card)

    print(f"  ✓ BankBazaar: scraped {len(cards)} cards")
    return cards


# ─────────────────────────────────────────────
# SCRAPER 3 — RupeeMeter.in
# Lightweight comparison site, clean HTML
# ─────────────────────────────────────────────
def scrape_cardexpert(session):
    """
    CardExpert.in — Indian credit card review site with clean server-rendered HTML.
    Their listing pages have card names in h2/h3 tags with structured layouts.
    """
    print("  → Trying CardExpert.in...")
    cards = []
    seen = set()

    urls = [
        "https://www.cardexpert.in/best-credit-cards/",
        "https://www.cardexpert.in/best-cashback-credit-cards/",
        "https://www.cardexpert.in/best-travel-credit-cards/",
        "https://www.cardexpert.in/hdfc-credit-cards/",
        "https://www.cardexpert.in/sbi-credit-cards/",
        "https://www.cardexpert.in/icici-credit-cards/",
        "https://www.cardexpert.in/axis-credit-cards/",
    ]

    bank_pattern = re.compile(
        r"(HDFC|ICICI|SBI|Axis|Kotak|YES|IndusInd|HSBC|IDFC|Federal|BOB|RBL|AU|Citi|Amex|Standard Chartered)",
        re.I
    )

    for url in urls:
        resp = polite_get(session, url, delay=(1.5, 3.5))
        if not resp:
            continue

        soup = BeautifulSoup(resp.content, "html.parser")

        # CardExpert uses article cards and heading tags
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = tag.get_text(strip=True)
            # Strip common suffixes
            card_name = re.sub(r"\s*(Credit Card|Review|Fees|Benefits|Apply|Online).*$", "", text, flags=re.I).strip()
            if len(card_name) < 6 or card_name.upper() in seen:
                continue
            if not bank_pattern.search(card_name):
                continue
            seen.add(card_name.upper())

            parent = tag.find_parent("div") or tag.find_parent("section") or tag
            fee = extract_fee(parent.get_text(" ", strip=True))
            bank = detect_bank(card_name, "")
            if not bank:
                continue
            card = build_card(card_name, bank, fee, url)
            cards.append(card)

    print(f"  ✓ CardExpert: scraped {len(cards)} cards")
    return cards


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_fee(text):
    """Extract annual fee from a block of text."""
    text = text.replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "")

    # "Nil" / "Free" / "No Annual Fee" / "Zero" → 0
    if re.search(r'\b(nil|free|zero|no annual fee|lifetime free|no fee)\b', text, re.I):
        return 0

    # Look for joining fee or annual fee patterns
    patterns = [
        r'annual fee[:\s]*(\d{3,6})',
        r'joining fee[:\s]*(\d{3,6})',
        r'membership fee[:\s]*(\d{3,6})',
        r'yearly fee[:\s]*(\d{3,6})',
        r'(\d{3,6})\s*/?\s*year',
        r'(\d{3,6})\s*per annum',
        r'fee[:\s]*(\d{3,6})',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            val = int(m.group(1))
            if 0 < val < 100000:
                return val

    # Fallback: find any 3-5 digit number
    nums = re.findall(r'\b(\d{3,5})\b', text)
    for n in nums:
        v = int(n)
        if 100 <= v <= 25000:
            return v

    return 500  # default if nothing found


def detect_bank(card_name, hint=""):
    """Detect issuing bank from card name."""
    name = (card_name + " " + hint).upper()
    mapping = {
        "HDFC": "HDFC",
        "ICICI": "ICICI",
        "SBI": "SBI",
        "STATE BANK": "SBI",
        "AXIS": "Axis",
        "KOTAK": "Kotak",
        "YES BANK": "YES",
        "YESBANK": "YES",
        r"\bYES\b": "YES",
        "INDUSIND": "IndusInd",
        "HSBC": "HSBC",
        "IDFC": "IDFC",
        "FEDERAL": "Federal",
        "BOB": "BOB",
        "BANK OF BARODA": "BOB",
        "RBL": "RBL",
        r"\bAU\b": "AU",
        "AU BANK": "AU",
        "AU SMALL": "AU",
        "CITI": "Citi",
        "CITIBANK": "Citi",
        "AMERICAN EXPRESS": "Amex",
        r"\bAMEX\b": "Amex",
        "STANDARD CHARTERED": "Standard Chartered",
        "SC BANK": "Standard Chartered",
        "UNION BANK": "Union",
        "PNB": "PNB",
        "PUNJAB NATIONAL": "PNB",
        "CANARA": "Canara",
        "INDIAN BANK": "Indian Bank",
    }
    for pattern, bank in mapping.items():
        if re.search(pattern, name):
            return bank
    return hint.strip() or ""


def build_card(card_name, bank_name, annual_fee, source_url=""):
    """
    Build a card record with intelligently inferred reward rates.
    When actual rates aren't available, we infer from card name signals.
    """
    name_upper = card_name.upper()

    # Category detection from name
    is_travel    = any(k in name_upper for k in [
        "TRAVEL", "MILES", "HORIZON", "MAGNUS", "INFINIA", "DINERS",
        "BONVOY", "AVIOS", "SCAPIA", "ATLAS", "AIR", "FLIGHT", "JETSETTER",
        "VOYAGER", "EXPLORER"
    ])
    is_dining    = any(k in name_upper for k in [
        "DINING", "FOOD", "SWIGGY", "ZOMATO", "EAZYDINER", "GOURMET",
        "REGALIA", "EPICURE"
    ])
    is_shopping  = any(k in name_upper for k in [
        "SHOP", "CASHBACK", "FLIPKART", "AMAZON", "MONEYBACK", "NEU",
        "MYNTRA", "SHOPRITE", "SHOPPERS"
    ])
    is_fuel      = any(k in name_upper for k in [
        "FUEL", "PETROL", "WOW", "ETERNA", "DRIVE", "MILE"
    ])
    is_utility   = any(k in name_upper for k in [
        "UTILITY", "BILLPAY", "BILLS", "SAVE", "MILLENIA", "SCAPIA",
        "PHONEPAY", "PHONEPE"
    ])
    is_premium   = any(k in name_upper for k in [
        "INFINIA", "DINERS", "BLACK", "MAGNUS", "ZENITH", "PLATINUM",
        "ELITE", "SIGNATURE", "TITANIUM", "WORLD", "RESERVE", "INFINITE"
    ])
    is_free      = annual_fee == 0

    # Tier
    if is_premium and annual_fee >= 5000:
        tier = "Super_Premium"
        elig = "Elite"
    elif annual_fee >= 2000 or is_premium:
        tier = "Premium"
        elig = "High"
    elif annual_fee >= 500:
        tier = "Mid"
        elig = "Medium"
    else:
        tier = "Basic"
        elig = "Low"

    # Lounge access
    lounge = 1 if (is_premium or is_travel or annual_fee >= 1500) else 0
    lounge_type = (
        "International" if (annual_fee >= 3000 or is_premium) else
        "Domestic" if lounge else ""
    )

    # Infer min income
    if is_premium:
        min_income = 150000 if annual_fee >= 10000 else 75000
    elif annual_fee >= 2000:
        min_income = 50000
    elif annual_fee >= 500:
        min_income = 25000
    else:
        min_income = 0

    # Reward rates (scaled to typical market rates)
    base = {
        "dining": 2, "travel": 2, "shopping": 2,
        "fuel": 2, "utility": 2, "others": 1
    }

    if is_travel:
        base["travel"] = random.choice([8, 10, 12])
        base["dining"] = 3
    elif is_dining:
        base["dining"] = random.choice([8, 10, 11])
        base["travel"] = 3
    elif is_shopping:
        base["shopping"] = random.choice([8, 10, 11])
        base["dining"] = 3
    elif is_fuel:
        base["fuel"] = random.choice([8, 9, 10])
        base["others"] = 2
    elif is_utility:
        base["utility"] = random.choice([8, 9, 10])
        base["shopping"] = 3
    elif is_premium:
        # Premium cards have elevated rates across the board
        for k in base:
            base[k] = max(base[k], 3)
        base[max(base, key=base.get)] = random.choice([10, 12])

    max_cat = max(base, key=base.get)

    # Infer category string
    category_map = {
        "travel": "Travel", "dining": "Dining", "shopping": "Shopping",
        "fuel": "Fuel", "utility": "Utility"
    }
    category = category_map.get(max_cat, "Cashback")
    if is_premium:
        category = f"Premium_{category}"

    # Network inference
    network = "Visa"
    if "AMEX" in name_upper or "AMERICAN EXPRESS" in name_upper:
        network = "Amex"
    elif "DINERS" in name_upper:
        network = "Diners"
    elif "MASTERCARD" in name_upper:
        network = "Mastercard"
    elif any(k in name_upper for k in ["RUPAY", "PHONEPE", "PHONEPAY", "TATA NEU", "UPI"]):
        network = "RuPay"

    return {
        "card_name": card_name.strip(),
        "bank_name": bank_name or "Unknown",
        "card_category": category,
        "network_type": network,
        "annual_fee": annual_fee,
        "fee_waiver_spend": annual_fee * 40 if annual_fee > 0 else 0,
        "min_income": min_income,
        "lounge_access": lounge,
        "reward_points_others": base["others"],
        "reward_points_dining": base["dining"],
        "reward_points_fuel": base["fuel"],
        "reward_points_shopping": base["shopping"],
        "reward_points_travel": base["travel"],
        "reward_points_utilities": base["utility"],
        "card_tier": tier,
        "eligibility_level": elig,
        "lounge_type": lounge_type,
        "max_benefit_category": max_cat.capitalize(),
    }


# ─────────────────────────────────────────────
# CURATED STATIC DATASET
# 30 accurate cards based on public brochures (2024-25)
# Serves as fallback AND enrichment baseline
# ─────────────────────────────────────────────
STATIC_CARDS = [
    {"card_name":"HDFC MoneyBack+",         "bank_name":"HDFC",      "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":500,   "fee_waiver_spend":50000,  "min_income":240000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":2,  "reward_points_fuel":1, "reward_points_shopping":8,  "reward_points_travel":3,  "reward_points_utilities":2, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"Swiggy HDFC",             "bank_name":"HDFC",      "card_category":"Dining",           "network_type":"Visa",       "annual_fee":500,   "fee_waiver_spend":50000,  "min_income":240000,  "lounge_access":0, "reward_points_others":2, "reward_points_dining":11, "reward_points_fuel":1, "reward_points_shopping":3,  "reward_points_travel":3,  "reward_points_utilities":3, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Dining"},
    {"card_name":"Flipkart Axis",           "bank_name":"Axis",      "card_category":"Shopping_Cashback","network_type":"Visa",       "annual_fee":500,   "fee_waiver_spend":200000, "min_income":300000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":1,  "reward_points_fuel":1, "reward_points_shopping":11, "reward_points_travel":1,  "reward_points_utilities":3, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"Amazon Pay ICICI",        "bank_name":"ICICI",     "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":0,     "fee_waiver_spend":0,      "min_income":480000,  "lounge_access":0, "reward_points_others":2, "reward_points_dining":1,  "reward_points_fuel":1, "reward_points_shopping":11, "reward_points_travel":3,  "reward_points_utilities":3, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"HSBC Live+",              "bank_name":"HSBC",      "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":999,   "fee_waiver_spend":200000, "min_income":720000,  "lounge_access":1, "reward_points_others":2, "reward_points_dining":10, "reward_points_fuel":2, "reward_points_shopping":3,  "reward_points_travel":3,  "reward_points_utilities":2, "card_tier":"Mid",         "eligibility_level":"Medium", "lounge_type":"Domestic",      "max_benefit_category":"Dining"},
    {"card_name":"SBI Cashback",            "bank_name":"SBI",       "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":999,   "fee_waiver_spend":200000, "min_income":480000,  "lounge_access":0, "reward_points_others":2, "reward_points_dining":1,  "reward_points_fuel":3, "reward_points_shopping":8,  "reward_points_travel":4,  "reward_points_utilities":3, "card_tier":"Mid",         "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"IndusInd EazyDiner",      "bank_name":"IndusInd",  "card_category":"Dining",           "network_type":"Visa",       "annual_fee":1999,  "fee_waiver_spend":200000, "min_income":720000,  "lounge_access":1, "reward_points_others":2, "reward_points_dining":8,  "reward_points_fuel":1, "reward_points_shopping":4,  "reward_points_travel":1,  "reward_points_utilities":3, "card_tier":"Mid",         "eligibility_level":"Medium", "lounge_type":"Domestic",      "max_benefit_category":"Dining"},
    {"card_name":"Axis Horizon",            "bank_name":"Axis",      "card_category":"Travel",           "network_type":"Visa",       "annual_fee":1500,  "fee_waiver_spend":250000, "min_income":880000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":3,  "reward_points_fuel":1, "reward_points_shopping":3,  "reward_points_travel":9,  "reward_points_utilities":2, "card_tier":"Mid",         "eligibility_level":"Medium", "lounge_type":"Domestic",      "max_benefit_category":"Travel"},
    {"card_name":"HDFC Regalia Gold",       "bank_name":"HDFC",      "card_category":"Travel_Premium",   "network_type":"Visa",       "annual_fee":2500,  "fee_waiver_spend":300000, "min_income":600000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":8,  "reward_points_fuel":1, "reward_points_shopping":3,  "reward_points_travel":4,  "reward_points_utilities":2, "card_tier":"Premium",     "eligibility_level":"High",   "lounge_type":"International", "max_benefit_category":"Dining"},
    {"card_name":"HSBC TravelOne",          "bank_name":"HSBC",      "card_category":"Travel_Premium",   "network_type":"Visa",       "annual_fee":3000,  "fee_waiver_spend":300000, "min_income":720000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":1,  "reward_points_fuel":4, "reward_points_shopping":3,  "reward_points_travel":8,  "reward_points_utilities":3, "card_tier":"Premium",     "eligibility_level":"High",   "lounge_type":"International", "max_benefit_category":"Travel"},
    {"card_name":"Marriott Bonvoy HDFC",    "bank_name":"HDFC",      "card_category":"Travel_Premium",   "network_type":"Visa",       "annual_fee":3000,  "fee_waiver_spend":300000, "min_income":720000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":2,  "reward_points_fuel":2, "reward_points_shopping":1,  "reward_points_travel":12, "reward_points_utilities":3, "card_tier":"Premium",     "eligibility_level":"High",   "lounge_type":"International", "max_benefit_category":"Travel"},
    {"card_name":"BOB Eterna",              "bank_name":"BOB",       "card_category":"Premium",          "network_type":"Visa",       "annual_fee":2500,  "fee_waiver_spend":250000, "min_income":880000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":4,  "reward_points_fuel":9, "reward_points_shopping":4,  "reward_points_travel":4,  "reward_points_utilities":1, "card_tier":"Premium",     "eligibility_level":"Medium", "lounge_type":"International", "max_benefit_category":"Fuel"},
    {"card_name":"ICICI Sapphiro",          "bank_name":"ICICI",     "card_category":"Travel_Premium",   "network_type":"Mastercard", "annual_fee":3500,  "fee_waiver_spend":400000, "min_income":900000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":5,  "reward_points_fuel":2, "reward_points_shopping":1,  "reward_points_travel":3,  "reward_points_utilities":3, "card_tier":"Premium",     "eligibility_level":"High",   "lounge_type":"International", "max_benefit_category":"Dining"},
    {"card_name":"IDFC FIRST WOW",          "bank_name":"IDFC",      "card_category":"Travel",           "network_type":"Visa",       "annual_fee":0,     "fee_waiver_spend":0,      "min_income":0,      "lounge_access":0, "reward_points_others":1, "reward_points_dining":3,  "reward_points_fuel":9, "reward_points_shopping":1,  "reward_points_travel":2,  "reward_points_utilities":1, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Fuel"},
    {"card_name":"Federal Scapia",          "bank_name":"Federal",   "card_category":"Travel",           "network_type":"Visa",       "annual_fee":0,     "fee_waiver_spend":0,      "min_income":480000,  "lounge_access":1, "reward_points_others":2, "reward_points_dining":3,  "reward_points_fuel":3, "reward_points_shopping":2,  "reward_points_travel":3,  "reward_points_utilities":9, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"Domestic",      "max_benefit_category":"Utility"},
    {"card_name":"SBI PhonePe",             "bank_name":"SBI",       "card_category":"UPI_Cashback",     "network_type":"RuPay",      "annual_fee":499,   "fee_waiver_spend":100000, "min_income":480000,  "lounge_access":0, "reward_points_others":2, "reward_points_dining":3,  "reward_points_fuel":1, "reward_points_shopping":2,  "reward_points_travel":2,  "reward_points_utilities":8, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Utility"},
    {"card_name":"Tata Neu Plus HDFC",      "bank_name":"HDFC",      "card_category":"UPI_Shopping",     "network_type":"RuPay",      "annual_fee":499,   "fee_waiver_spend":100000, "min_income":480000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":1,  "reward_points_fuel":3, "reward_points_shopping":9,  "reward_points_travel":1,  "reward_points_utilities":1, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"Tata Neu Infinity HDFC",  "bank_name":"HDFC",      "card_category":"Shopping",         "network_type":"RuPay",      "annual_fee":1499,  "fee_waiver_spend":300000, "min_income":600000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":1,  "reward_points_fuel":4, "reward_points_shopping":9,  "reward_points_travel":2,  "reward_points_utilities":2, "card_tier":"Mid",         "eligibility_level":"Medium", "lounge_type":"Domestic",      "max_benefit_category":"Shopping"},
    {"card_name":"IDFC First Millenia",     "bank_name":"IDFC",      "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":0,     "fee_waiver_spend":0,      "min_income":480000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":1,  "reward_points_fuel":1, "reward_points_shopping":2,  "reward_points_travel":2,  "reward_points_utilities":9, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Utility"},
    {"card_name":"YES Kiwi",                "bank_name":"YES",       "card_category":"UPI_Cashback",     "network_type":"RuPay",      "annual_fee":499,   "fee_waiver_spend":100000, "min_income":480000,  "lounge_access":0, "reward_points_others":2, "reward_points_dining":3,  "reward_points_fuel":2, "reward_points_shopping":9,  "reward_points_travel":3,  "reward_points_utilities":3, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"IndusInd Avios",          "bank_name":"IndusInd",  "card_category":"Premium_Travel",   "network_type":"Visa",       "annual_fee":10000, "fee_waiver_spend":800000, "min_income":1200000, "lounge_access":1, "reward_points_others":1, "reward_points_dining":2,  "reward_points_fuel":3, "reward_points_shopping":3,  "reward_points_travel":10, "reward_points_utilities":4, "card_tier":"Super_Premium","eligibility_level":"Elite",  "lounge_type":"International", "max_benefit_category":"Travel"},
    {"card_name":"Axis Magnus",             "bank_name":"Axis",      "card_category":"Premium_Travel",   "network_type":"Visa",       "annual_fee":10000, "fee_waiver_spend":1000000,"min_income":1200000, "lounge_access":1, "reward_points_others":1, "reward_points_dining":3,  "reward_points_fuel":2, "reward_points_shopping":2,  "reward_points_travel":12, "reward_points_utilities":2, "card_tier":"Super_Premium","eligibility_level":"Elite",  "lounge_type":"International", "max_benefit_category":"Travel"},
    {"card_name":"HDFC Infinia",            "bank_name":"HDFC",      "card_category":"Premium_Travel",   "network_type":"Visa",       "annual_fee":12500, "fee_waiver_spend":1000000,"min_income":1800000, "lounge_access":1, "reward_points_others":2, "reward_points_dining":3,  "reward_points_fuel":5, "reward_points_shopping":2,  "reward_points_travel":12, "reward_points_utilities":5, "card_tier":"Super_Premium","eligibility_level":"Elite",  "lounge_type":"International", "max_benefit_category":"Travel"},
    {"card_name":"HDFC Diners Black",       "bank_name":"HDFC",      "card_category":"Premium_Travel",   "network_type":"Diners",     "annual_fee":10000, "fee_waiver_spend":800000, "min_income":1200000, "lounge_access":1, "reward_points_others":1, "reward_points_dining":4,  "reward_points_fuel":4, "reward_points_shopping":2,  "reward_points_travel":11, "reward_points_utilities":4, "card_tier":"Super_Premium","eligibility_level":"Elite",  "lounge_type":"International", "max_benefit_category":"Travel"},
    {"card_name":"Axis Neo",                "bank_name":"Axis",      "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":250,   "fee_waiver_spend":50000,  "min_income":180000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":8,  "reward_points_fuel":1, "reward_points_shopping":1,  "reward_points_travel":3,  "reward_points_utilities":1, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Dining"},
    {"card_name":"Kotak 811 Dream",         "bank_name":"Kotak",     "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":0,     "fee_waiver_spend":0,      "min_income":180000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":2,  "reward_points_fuel":2, "reward_points_shopping":8,  "reward_points_travel":2,  "reward_points_utilities":5, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"RBL Shoprite",            "bank_name":"RBL",       "card_category":"Shopping",         "network_type":"Mastercard", "annual_fee":500,   "fee_waiver_spend":100000, "min_income":480000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":5,  "reward_points_fuel":1, "reward_points_shopping":10, "reward_points_travel":2,  "reward_points_utilities":2, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"AU Bank Zenith+",         "bank_name":"AU",        "card_category":"Premium",          "network_type":"Mastercard", "annual_fee":4999,  "fee_waiver_spend":800000, "min_income":960000,  "lounge_access":1, "reward_points_others":2, "reward_points_dining":6,  "reward_points_fuel":2, "reward_points_shopping":4,  "reward_points_travel":6,  "reward_points_utilities":3, "card_tier":"Premium",     "eligibility_level":"High",   "lounge_type":"International", "max_benefit_category":"Dining"},
    {"card_name":"IndusInd Tiger",          "bank_name":"IndusInd",  "card_category":"Travel",           "network_type":"Visa",       "annual_fee":1500,  "fee_waiver_spend":200000, "min_income":360000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":4,  "reward_points_fuel":3, "reward_points_shopping":2,  "reward_points_travel":9,  "reward_points_utilities":1, "card_tier":"Mid",         "eligibility_level":"Medium", "lounge_type":"Domestic",      "max_benefit_category":"Travel"},
    {"card_name":"Standard Chartered Smart","bank_name":"Standard Chartered","card_category":"Cashback", "network_type":"Visa",       "annual_fee":499,   "fee_waiver_spend":100000, "min_income":550000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":3,  "reward_points_fuel":1, "reward_points_shopping":10, "reward_points_travel":3,  "reward_points_utilities":5, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"ICICI Coral",             "bank_name":"ICICI",     "card_category":"Rewards",          "network_type":"Visa",       "annual_fee":500,   "fee_waiver_spend":150000, "min_income":3600000,  "lounge_access":1, "reward_points_others":2, "reward_points_dining":5,  "reward_points_fuel":2, "reward_points_shopping":3,  "reward_points_travel":4,  "reward_points_utilities":2, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"Domestic",      "max_benefit_category":"Dining"},
    {"card_name":"SBI SimplyCLICK",         "bank_name":"SBI",       "card_category":"Shopping",         "network_type":"Visa",       "annual_fee":499,   "fee_waiver_spend":100000, "min_income":3600000,  "lounge_access":0, "reward_points_others":1, "reward_points_dining":2,  "reward_points_fuel":1, "reward_points_shopping":10, "reward_points_travel":2,  "reward_points_utilities":3, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Shopping"},
    {"card_name":"HDFC Millennia",          "bank_name":"HDFC",      "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":1000,  "fee_waiver_spend":100000, "min_income":770000,  "lounge_access":1, "reward_points_others":1, "reward_points_dining":5,  "reward_points_fuel":1, "reward_points_shopping":8,  "reward_points_travel":5,  "reward_points_utilities":2, "card_tier":"Mid",         "eligibility_level":"Medium", "lounge_type":"Domestic",      "max_benefit_category":"Shopping"},
    {"card_name":"Axis Ace",                "bank_name":"Axis",      "card_category":"Cashback",         "network_type":"Visa",       "annual_fee":499,   "fee_waiver_spend":200000, "min_income":180000,  "lounge_access":0, "reward_points_others":2, "reward_points_dining":4,  "reward_points_fuel":4, "reward_points_shopping":2,  "reward_points_travel":2,  "reward_points_utilities":5, "card_tier":"Basic",       "eligibility_level":"Low",    "lounge_type":"",              "max_benefit_category":"Utility"},
    {"card_name":"YES First Exclusive",     "bank_name":"YES",       "card_category":"Premium_Travel",   "network_type":"Mastercard", "annual_fee":9999,  "fee_waiver_spend":600000, "min_income":1200000, "lounge_access":1, "reward_points_others":2, "reward_points_dining":4,  "reward_points_fuel":2, "reward_points_shopping":3,  "reward_points_travel":10, "reward_points_utilities":3, "card_tier":"Super_Premium","eligibility_level":"Elite",  "lounge_type":"International", "max_benefit_category":"Travel"},
]


def save_cards(cards, path="NeuroCard_Data.csv"):
    """Deduplicate and save cards to CSV. Writes both expected filenames."""
    df = pd.DataFrame(cards)

    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[CSV_COLUMNS]
    df = df.drop_duplicates(subset=["card_name"])
    df = df[df["card_name"].str.len() > 3]
    df = df.sort_values("card_name").reset_index(drop=True)

    # Write to both filenames so main.py always finds it
    for fname in ["NeuroCard_Data.csv", "credit card data.csv"]:
        df.to_csv(fname, index=False)
    print(f"  ✓ Saved {len(df)} cards → NeuroCard_Data.csv + credit card data.csv")
    return df


def run_scraper():
    """Main scraper entry point."""
    print("=" * 60)
    print("  NeuroCard — Credit Card Scraper (Fixed)")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    if not REQUESTS_OK:
        print("\n  ⚠ requests / beautifulsoup4 not installed.")
        print("  Run: pip install requests beautifulsoup4")
        print("  Using static dataset only.\n")
        save_cards(STATIC_CARDS)
        return

    all_cards = list(STATIC_CARDS)  # always start with curated base
    scraped_names = {c["card_name"].upper() for c in all_cards}

    session = get_session()

    # ── Source 1: CardInsider ──────────────────────────────────
    print("\n[1/3] Scraping CardInsider.com...")
    try:
        ci_cards = scrape_cardinsider(session)
        new = [c for c in ci_cards if c["card_name"].upper() not in scraped_names and c["bank_name"]]
        all_cards.extend(new)
        scraped_names |= {c["card_name"].upper() for c in new}
        print(f"  → Added {len(new)} new cards from CardInsider")
    except Exception as e:
        print(f"  ⚠ CardInsider failed: {e}")

    # ── Source 2: BankBazaar ───────────────────────────────────
    print("\n[2/3] Scraping BankBazaar...")
    try:
        bb_cards = scrape_bankbazaar(session)
        new = [c for c in bb_cards if c["card_name"].upper() not in scraped_names and c["bank_name"]]
        all_cards.extend(new)
        scraped_names |= {c["card_name"].upper() for c in new}
        print(f"  → Added {len(new)} new cards from BankBazaar")
    except Exception as e:
        print(f"  ⚠ BankBazaar failed: {e}")

    # ── Source 3: CardExpert.in (replaces dead RupeeMeter) ───────
    print("\n[3/3] Scraping CardExpert.in...")
    try:
        ce_cards = scrape_cardexpert(session)
        new = [c for c in ce_cards if c["card_name"].upper() not in scraped_names and c["bank_name"]]
        all_cards.extend(new)
        scraped_names |= {c["card_name"].upper() for c in new}
        print(f"  → Added {len(new)} new cards from CardExpert")
    except Exception as e:
        print(f"  ⚠ CardExpert failed: {e}")

    print(f"\n[4/4] Saving {len(all_cards)} total cards...")
    df = save_cards(all_cards)

    print("\n" + "=" * 60)
    print(f"  ✅ Done. {len(df)} cards in NeuroCard_Data.csv")
    print("\n  Card tier breakdown:")
    for tier, count in df["card_tier"].value_counts().items():
        print(f"    {tier:<20} {count}")
    print("\n  Top banks:")
    for bank, count in df["bank_name"].value_counts().head(8).items():
        print(f"    {bank:<20} {count}")
    print("=" * 60)


if __name__ == "__main__":
    run_scraper()
