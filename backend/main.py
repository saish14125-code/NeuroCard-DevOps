"""
NeuroCard — Backend API
========================
Run train_model.py first to generate the model files.
Then: uvicorn main:app --reload  or  python main.py
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json
import random
import csv
import hashlib
import secrets
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

from google import genai

load_dotenv()

# ── PATH RESOLUTION ──────────────────────────────────────────────────────────
# All data files live in the same folder as main.py (backend/).
# Absolute paths mean the server works from any launch directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def p(filename):
    """Return absolute path to a file in the backend/ folder."""
    return os.path.join(BASE_DIR, filename)
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("⚠ GEMINI_API_KEY not set — LLM features in simulated mode.")

MODELS = [
    "gemini-2.5-flash",    # primary (2.0-flash has no free quota anymore)
    "gemini-2.5-flash-lite",  # fallback
]
# In-memory caches (live for the server process lifetime)
GEMINI_CACHE = {}      # prompt string → response text
_INSIGHT_CACHE = {}    # user_id → insight string (prevents re-calling on tab switch)

def generate(prompt):
    if not gemini_client:
        return None

    if prompt in GEMINI_CACHE:
        return GEMINI_CACHE[prompt]

    for model in MODELS:
        try:
            r = gemini_client.models.generate_content(
                model=model,
                contents=prompt
            )
            text = r.text.strip()
            GEMINI_CACHE[prompt] = text
            return text
        except Exception as e:
            print(f"{model} failed:", e)

    return None

app = FastAPI(title="NeuroCard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 🔑 KEYWORD OVERRIDE TABLE
# Rules-first: guaranteed-correct categories for known brands.
# The ML model is a fallback, not the primary classifier.
# ==========================================
KEYWORD_OVERRIDES = {
    "petrol": "fuel", "diesel": "fuel", "cng": "fuel", "fuel": "fuel",
    "pump": "fuel", "filling station": "fuel", "bpcl": "fuel", "hpcl": "fuel",
    "iocl": "fuel", "indian oil": "fuel", "bharat petroleum": "fuel",
    "hindustan petroleum": "fuel", "shell": "fuel", "essar oil": "fuel",
    "nayara": "fuel", "hp petrol": "fuel",

    "indigo": "travel", "spicejet": "travel", "air india": "travel",
    "vistara": "travel", "akasa": "travel", "goair": "travel",
    "irctc": "travel", "redbus": "travel", "rapido": "travel",
    "ola": "travel", "uber": "travel", "meru": "travel",
    "makemytrip": "travel", "goibibo": "travel", "cleartrip": "travel",
    "yatra": "travel", "ixigo": "travel", "oyo": "travel",
    "airbnb": "travel", "trivago": "travel",
    "airlines": "travel", "airport": "travel", "railway": "travel",
    "metro card": "travel",

    "zomato": "dining", "swiggy": "dining", "kfc": "dining",
    "mcdonalds": "dining", "mcdonald": "dining", "burger king": "dining",
    "dominos": "dining", "domino": "dining", "pizza hut": "dining",
    "starbucks": "dining", "cafe coffee day": "dining", "ccd": "dining",
    "barbeque nation": "dining", "fassos": "dining", "haldirams": "dining",
    "chaayos": "dining", "eazydiner": "dining", "dineout": "dining",
    "restaurant": "dining", "bistro": "dining", "dhaba": "dining",

    "amazon": "shopping", "flipkart": "shopping", "myntra": "shopping",
    "ajio": "shopping", "nykaa": "shopping", "meesho": "shopping",
    "zara": "shopping", "h&m": "shopping", "uniqlo": "shopping",
    "croma": "shopping", "reliance digital": "shopping",
    "vijay sales": "shopping", "tata cliq": "shopping",
    "snapdeal": "shopping", "decathlon": "shopping",

    "netflix": "utility", "hotstar": "utility", "disney": "utility",
    "prime video": "utility", "amazon prime": "utility",
    "spotify": "utility", "youtube premium": "utility",
    "jio": "utility", "airtel": "utility", "vodafone": "utility",
    "bsnl": "utility", "tata power": "utility", "bescom": "utility",
    "msedcl": "utility", "adani electricity": "utility",
    "mahanagar gas": "utility", "indane gas": "utility",
    "electricity": "utility", "broadband": "utility",

    "apollo pharmacy": "others", "medplus": "others", "practo": "others",
    "1mg": "others", "pharmeasy": "others", "netmeds": "others",
    "lic": "others", "hdfc life": "others", "icici prudential": "others",
    "policybazaar": "others", "coverfox": "others",
    "bookmyshow": "others", "cult fit": "others", "cult.fit": "others",
    "zerodha": "others", "groww": "others", "upstox": "others",
    "gym": "others", "hospital": "others", "clinic": "others",
}

def keyword_predict(merchant: str):
    m = merchant.lower().strip()
    for keyword, category in KEYWORD_OVERRIDES.items():
        if keyword in m:
            return category
    return None

# ==========================================
# 🤖 LOAD TRAINED MODELS
# If joblib files exist (run train_model.py first), load them.
# Otherwise fall back to inline training.
# ==========================================
CATEGORIES = ["dining", "travel", "shopping", "fuel", "utility", "others"]
PERSONAS = [
    "The Epicurean Hunter", "The Global Jetsetter", "The Lifestyle Curator",
    "The Highway Commuter", "The Pragmatic Saver", "The Balanced Strategist"
]

nlp_classifier = None
kmeans_model = None


# ==========================================
# 🃏 CARD CATALOG (v2 — real reward economics)
# Curated in cards_data.py: 10 popular Indian cards with real
# value-back %, monthly caps, and detail fields, verified against
# PaisaBazaar / 1finance pages. reward_points_<cat> now holds the
# TRUE effective value-back percent (float), not an invented score.
# ==========================================
from cards_data import REAL_CARDS, CARD_CSV_COLS

_CARD_CSV_COLS = CARD_CSV_COLS

def _csv_is_v2(path):
    try:
        return 'reward_type' in pd.read_csv(path, nrows=1).columns
    except Exception:
        return False

def bootstrap_card_data():
    """Self-healing catalog startup:
      - valid v2 CSV  → refresh the .backup copy (cheap insurance)
      - corrupt/old CSV → restore from backup, else rewrite curated seed
    Nothing outside the validated scraper pipeline should ever write this
    file, but if something does, one restart heals it."""
    path = p('NeuroCard_Data.csv')
    backup = p('NeuroCard_Data.backup.csv')

    if os.path.exists(path) and _csv_is_v2(path):
        df = pd.read_csv(path)
        if 'caps_verified' not in df.columns:
            # migrate: hand-curated / Gemini rows are verified, auto-parsed are not
            df['caps_verified'] = (~df['highlights'].astype(str)
                                   .str.contains('pending verification', case=False)).astype(int)
            df.to_csv(path, index=False)
            print(f"✓ Migrated catalog: {int(df['caps_verified'].sum())}/{len(df)} cards cap-verified.")
        df.to_csv(backup, index=False)
        print(f"✓ Card catalog OK ({len(df)} cards, {int(df.get('caps_verified', pd.Series([1]*len(df))).sum())} verified). Backup refreshed.")
        return

    if os.path.exists(path):  # exists but old-schema/corrupt
        os.replace(path, p('NeuroCard_Data_corrupt.csv'))
        print("⚠ Card CSV was old-schema or corrupt — quarantined as NeuroCard_Data_corrupt.csv")

    if os.path.exists(backup) and _csv_is_v2(backup):
        pd.read_csv(backup).to_csv(path, index=False)
        print(f"✓ Catalog RESTORED from backup ({len(pd.read_csv(path))} cards).")
    else:
        pd.DataFrame(REAL_CARDS)[_CARD_CSV_COLS].to_csv(path, index=False)
        print(f"✓ Wrote curated seed catalog ({len(REAL_CARDS)} cards). Run scraper_v2.py to expand.")

bootstrap_card_data()

def load_or_train_models():
    global nlp_classifier, kmeans_model

    nlp_path = p("nlp_model.joblib")
    kmeans_path = p("kmeans_model.joblib")

    if os.path.exists(nlp_path) and os.path.exists(kmeans_path):
        print("✓ Loading pre-trained models from disk...")
        nlp_classifier = joblib.load(nlp_path)
        kmeans_model = joblib.load(kmeans_path)
        print("✓ Models loaded. (TF-IDF + Logistic Regression, K-Means with RFM clustering)")
    else:
        print("⚠ Trained models not found. Training inline fallback...")
        print("  Run train_model.py for better accuracy and eval metrics.")
        _train_inline()

def _train_inline():
    """Fast inline training if joblib files are missing."""
    global nlp_classifier, kmeans_model

    INLINE_DATA = [
        ("ZOMATO ORDER", "dining"), ("SWIGGY FOOD", "dining"), ("KFC", "dining"),
        ("DOMINOS PIZZA", "dining"), ("MCDONALDS", "dining"), ("STARBUCKS", "dining"),
        ("PIZZA HUT", "dining"), ("BARBEQUE NATION", "dining"), ("CHAAYOS", "dining"),
        ("UBER TRIP", "travel"), ("OLA RIDE", "travel"), ("RAPIDO BIKE", "travel"),
        ("INDIGO AIRLINES", "travel"), ("IRCTC TICKET", "travel"), ("MAKEMYTRIP", "travel"),
        ("REDBUS BOOKING", "travel"), ("OYO ROOMS", "travel"), ("SPICEJET", "travel"),
        ("AMAZON PURCHASE", "shopping"), ("FLIPKART PAYMENT", "shopping"), ("MYNTRA FASHION", "shopping"),
        ("AJIO CLOTHING", "shopping"), ("NYKAA BEAUTY", "shopping"), ("CROMA STORE", "shopping"),
        ("RELIANCE DIGITAL", "shopping"), ("DECATHLON SPORTS", "shopping"),
        ("IOCL PETROL PUMP", "fuel"), ("BPCL FILLING", "fuel"), ("HPCL PUMP", "fuel"),
        ("SHELL PETROL", "fuel"), ("INDIAN OIL", "fuel"), ("BHARAT PETROLEUM", "fuel"),
        ("AIRTEL RECHARGE", "utility"), ("JIO BILL", "utility"), ("NETFLIX SUBSCRIPTION", "utility"),
        ("SPOTIFY PREMIUM", "utility"), ("BESCOM ELECTRICITY", "utility"), ("TATA POWER", "utility"),
        ("AMAZON PRIME", "utility"), ("HOTSTAR", "utility"),
        ("APOLLO PHARMACY", "others"), ("LIC PREMIUM", "others"), ("BOOKMYSHOW", "others"),
        ("CULT FIT GYM", "others"), ("ZERODHA", "others"), ("PRACTO", "others"),
    ]
    X = [d[0] for d in INLINE_DATA]
    y = [d[1] for d in INLINE_DATA]
    nlp_classifier = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True),
        LogisticRegression(max_iter=500, C=5.0, class_weight='balanced', solver='lbfgs')
    )
    nlp_classifier.fit(X, y)

    np.random.seed(42)
    persona_centers = np.array([
        [0.45, 0.10, 0.20, 0.08, 0.10, 0.07],
        [0.12, 0.50, 0.15, 0.07, 0.09, 0.07],
        [0.15, 0.08, 0.50, 0.05, 0.12, 0.10],
        [0.10, 0.08, 0.12, 0.45, 0.15, 0.10],
        [0.12, 0.07, 0.15, 0.08, 0.45, 0.13],
        [0.18, 0.17, 0.20, 0.15, 0.16, 0.14],
    ])
    cluster_data = np.vstack([
        np.random.dirichlet(c * 12, size=150) for c in persona_centers
    ])
    kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=500)
    kmeans_model.fit(cluster_data)
    print("✓ Inline models ready.")

load_or_train_models()

def smart_predict(merchant: str):
    """Rules first, ML fallback. Returns (category, confidence)."""
    override = keyword_predict(merchant)
    if override:
        return override, 1.0
    try:
        probs = nlp_classifier.predict_proba([merchant])[0]
        cat = nlp_classifier.classes_[np.argmax(probs)]
        return cat, round(float(max(probs)), 2)
    except Exception:
        return "others", 0.5

def predict_persona(spend_vector):
    vector = np.array(spend_vector)
    vector = vector / (vector.sum() or 1)
    pred_cluster = kmeans_model.predict([vector])[0]
    return PERSONAS[pred_cluster]

# ==========================================
# 🌐 GEMINI CATEGORIZATION
# ==========================================
def gemini_categorize(merchant: str):
    if not gemini_client:
        return smart_predict(merchant)

    prompt = (
        f"Classify the merchant '{merchant}' into EXACTLY ONE of: "
        f"dining, travel, shopping, fuel, utility, others.\n"
        f"dining=restaurants/food delivery, travel=cabs/flights/hotels, "
        f"shopping=ecommerce/retail, fuel=petrol/CNG/EV charging, "
        f"utility=telecom/electricity/streaming, others=pharmacy/insurance/gym.\n"
        f"Reply with ONLY the one-word category, nothing else."
    )

    valid = {"dining", "travel", "shopping", "fuel", "utility", "others"}

    result = generate(prompt)

    if result:
        result = result.lower().replace('.', '').replace('"', '').strip()
        if result in valid:
            return result, 0.90

    return smart_predict(merchant)

# ==========================================
# 📊 REDEMPTION VALUE ENGINE
# Converts raw points into real rupee estimates
# based on user's redemption preference
# ==========================================
REDEMPTION_VALUES = {
    "Cashback":        {"rate": 0.25, "label": "Statement Credit",       "tip": "Instant value against your bill. No expiry, no platform risk. Best default."},
    "Travel":          {"rate": 0.50, "label": "Flights & Hotels",       "tip": "Points worth 40–60% more for travel than cashback — best value if you fly even once a year."},
    "Shopping Vouchers":{"rate": 0.35,"label": "Amazon / Flipkart / Myntra","tip": "Great if you shop online regularly. Vouchers usually have no expiry."},
    "Dining Vouchers": {"rate": 0.38, "label": "Zomato / Swiggy Credits","tip": "Perfect for foodies. Swiggy Money and Zomato credits often come with extra bonuses."},
    "Fuel Surcharge":  {"rate": 0.20, "label": "Fuel Surcharge Waiver",  "tip": "Not points-based, but saves 1% on every fuel transaction. Best if you drive daily."},
    "Investments":     {"rate": 0.30, "label": "Gold / Mutual Funds",    "tip": "Stable long-term value. Specific cards like HDFC Infinia support this."},
    "Movie & Events":  {"rate": 0.32, "label": "BookMyShow / PVR Credits","tip": "Great for occasional use — tickets can be fully covered by accumulated points."},
    "Lounge Access":   {"rate": 0.45, "label": "Airport Lounge Visits",  "tip": "Each lounge visit is worth ₹400–800 if bought standalone. High value if you travel."},
}

def calculate_redemption_value(total_points: int, pref: str, summary: dict):
    """
    Calculate how much the user's points are worth in rupees,
    and suggest the best redemption strategy based on their spending.
    """
    rv = REDEMPTION_VALUES.get(pref, REDEMPTION_VALUES["Cashback"])
    base_value = int(total_points * rv["rate"])

    # Compare across all options
    all_options = []
    for pref_name, pref_data in REDEMPTION_VALUES.items():
        all_options.append({
            "mode": pref_name,
            "label": pref_data["label"],
            "estimated_value": int(total_points * pref_data["rate"]),
            "rate_per_point": pref_data["rate"],
            "tip": pref_data["tip"],
            "is_selected": pref_name == pref,
        })
    all_options.sort(key=lambda x: x["estimated_value"], reverse=True)

    # Personalized recommendation
    top_cat = max(summary, key=summary.get) if summary else "dining"
    if top_cat == "travel":
        recommendation = "You spend a lot on travel — redeeming for flights gives you up to 60% more value than cashback."
    elif top_cat == "dining":
        recommendation = "For a foodie, Swiggy/Zomato vouchers let you essentially get free meals. Worth trying."
    elif top_cat == "shopping":
        recommendation = "Amazon and Flipkart vouchers match your shopping habits perfectly — straightforward win."
    else:
        recommendation = f"Your top category is {top_cat} — vouchers or cashback are both solid options here."

    return {
        "total_points": total_points,
        "selected_mode": pref,
        "estimated_rupee_value": base_value,
        "rate_per_point": rv["rate"],
        "label": rv["label"],
        "tip": rv["tip"],
        "all_options": all_options,
        "recommendation": recommendation,
    }

# ==========================================
# 🛠️ DATA MODELS
# ==========================================
class SliderData(BaseModel):
    user_id: str; dining: float; travel: float; shopping: float
    fuel: float; utility: float; others: float

class WalletAction(BaseModel):
    user_id: str; card_name: str

class NewUser(BaseModel):
    name: str; password: str; persona: str; income: float
    redemption_pref: str; owned_cards: list[str]

class LoginData(BaseModel):
    user_id: str; password: str

class ChatRequest(BaseModel):
    user_id: str; message: str

class CategorizeMerchant(BaseModel):
    merchant: str

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK MERCHANT POOLS (used when Gemini is unavailable or for known brands)
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_MERCHANTS = {
    "dining":   ["KFC", "McDonalds", "Starbucks", "Zomato", "Swiggy", "Domino's", "Pizza Hut",
                 "Burger King", "Chaayos", "Haldirams", "Cafe Coffee Day", "Barbeque Nation",
                 "Fassos", "EazyDiner", "Natural Ice Cream", "Theobroma", "Social Offline"],
    "travel":   ["IndiGo", "MakeMyTrip", "Uber", "Ola", "Airbnb", "IRCTC", "RedBus", "OYO",
                 "Rapido", "Vistara", "SpiceJet", "Goibibo", "Cleartrip", "BluSmart", "Meru Cabs",
                 "Ixigo", "EaseMyTrip", "Treebo Hotels", "Yatra.com"],
    "shopping": ["Amazon", "Flipkart", "Myntra", "Zara", "Croma", "Ajio", "Nykaa", "Meesho",
                 "Decathlon", "Reliance Digital", "Vijay Sales", "H&M", "Uniqlo", "Pepperfry",
                 "Urban Ladder", "D-Mart", "Big Bazaar", "Lifestyle Stores", "Shoppers Stop"],
    "fuel":     ["Indian Oil", "Bharat Petroleum", "HP Petrol Pump", "Shell", "BPCL", "HPCL",
                 "IOCL XtraPremium", "Nayara Energy", "Essar Oil", "IGL CNG Station",
                 "MGL CNG Station", "Tata EV Charging"],
    "utility":  ["Airtel", "Jio", "BSNL", "Netflix", "Spotify", "Amazon Prime", "Hotstar",
                 "TATA Power", "BESCOM", "MSEDCL", "Vi Vodafone", "YouTube Premium",
                 "SonyLIV", "ZEE5", "ACT Fibernet", "Mahanagar Gas"],
    "others":   ["Apollo Pharmacy", "LIC Premium", "BookMyShow", "Cult Fit", "Practo",
                 "Zerodha", "Groww", "MedPlus", "1mg Medicines", "Netmeds", "PolicyBazaar",
                 "Upstox", "PharmEasy", "Tata 1mg", "Manipal Hospital", "Thyrocare"]
}

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI MERCHANT GENERATOR
# Asks Gemini to produce fresh, persona-aware brand names for each category.
# Called once at user-creation time. Falls back to FALLBACK_MERCHANTS if unavailable.
# ─────────────────────────────────────────────────────────────────────────────
def get_merchants_for_user(persona: str) -> dict:
    """
    Returns merchant lists for transaction generation.
    Uses static FALLBACK_MERCHANTS — Gemini is NOT used here to avoid
    burning free-tier quota on user creation (which is called infrequently
    and doesn't need AI-generated merchant names).
    """
    if True:  # always use static — saves Gemini quota for insights & chat
        return FALLBACK_MERCHANTS
    if not gemini_client:
        return FALLBACK_MERCHANTS

    persona_context = {
        "foodie":   "someone who loves eating out, ordering food online, trying new restaurants",
        "traveler": "a frequent traveller who books flights, hotels, and cabs regularly",
        "shopper":  "an avid online shopper who buys fashion, electronics, and home goods",
        "commuter": "a daily commuter who spends heavily on fuel, tolls, and local transport",
        "saver":    "a budget-conscious person focused on utility bills and essential services",
        "balanced": "a balanced spender across all categories",
    }.get(persona.lower(), "a typical Indian credit card user")

    prompt = f"""You are generating synthetic Indian transaction data for a credit card app.
The user persona is: {persona_context}.

Generate a JSON object with exactly these 6 keys: dining, travel, shopping, fuel, utility, others.
For each key, provide a list of exactly 20 realistic Indian merchant/brand names that this persona would transact with.

Rules:
- Include a MIX: ~8 well-known national brands AND ~12 realistic but less-common regional/niche brands
- Include some brands that are NOT in any standard list (e.g. regional restaurant chains, local services)
- All brands must be real or very realistic Indian businesses
- Do NOT use generic names like "Restaurant 1" or "Shop A"
- Return ONLY valid JSON, no markdown, no explanation

Example format:
{{"dining": ["Zomato", "Paradise Biryani", "Rameshwaram Cafe", ...], "travel": [...], ...}}"""

    # (Gemini merchant generation removed — saves quota for insights & chat)
    return FALLBACK_MERCHANTS

def get_all_cards_df(include_unverified: bool = False):
    """The app only serves cards whose caps are verified (hand-curated or
    Gemini-checked against source pages). Auto-parsed rows stay in the CSV,
    invisible, until `scraper_v2.py --enrich` verifies them."""
    if not os.path.exists(p('NeuroCard_Data.csv')):
        return pd.DataFrame()
    df = pd.read_csv(p('NeuroCard_Data.csv')).fillna(0)
    if not include_unverified and 'caps_verified' in df.columns:
        df = df[pd.to_numeric(df['caps_verified'], errors='coerce').fillna(0) >= 1]
    return df

def get_all_cards():
    df = get_all_cards_df()
    return [] if df.empty else df.to_dict('records')

def get_users_meta():
    if os.path.exists(p('users.json')):
        with open(p('users.json'), 'r') as f:
            return json.load(f)
    return {}

def save_users_meta(meta):
    with open(p('users.json'), 'w') as f:
        json.dump(meta, f, indent=4)

def get_rate(card, category):
    cat_map = {
        "dining": "reward_points_dining", "travel": "reward_points_travel",
        "shopping": "reward_points_shopping", "fuel": "reward_points_fuel",
        "utility": "reward_points_utilities", "others": "reward_points_others"
    }
    try:
        return float(card.get(cat_map.get(category), 0)) / 100.0
    except:
        return 0.0

def get_rate_from_df(df, card_name, category):
    try:
        card_row = df[df['card_name'] == card_name].iloc[0]
        cat_map = {
            "dining": "reward_points_dining", "travel": "reward_points_travel",
            "shopping": "reward_points_shopping", "fuel": "reward_points_fuel",
            "utility": "reward_points_utilities", "others": "reward_points_others"
        }
        return float(card_row.get(cat_map.get(category), 1)) / 100.0
    except:
        return 0.01

def get_cap(card, category):
    """Monthly ₹ cap on reward value for a category. 0 = uncapped."""
    try:
        return float(card.get(f"cap_{category}", 0) or 0)
    except (TypeError, ValueError):
        return 0.0

def capped_category_value(card, category, annual_spend):
    """Annual reward value in ₹ for one category, respecting the card's
    monthly earning cap — this is what separates a real engine from a
    naive rate × spend multiplication."""
    monthly_value = (float(annual_spend) / 12.0) * get_rate(card, category)
    cap = get_cap(card, category)
    if cap > 0:
        monthly_value = min(monthly_value, cap)
    return monthly_value * 12.0

def calculate_recs(summary, user_income=None, specific_cards=None):
    all_cards = get_all_cards()
    if specific_cards:
        cards_to_eval = [c for c in all_cards if c.get('card_name') in specific_cards]
    elif user_income:
        cards_to_eval = [c for c in all_cards if float(c.get('min_income', 0)) <= float(user_income)]
    else:
        cards_to_eval = all_cards
    if not cards_to_eval:
        return []

    cat_keys = ["dining", "travel", "shopping", "fuel", "utility", "others"]
    total_spend = sum(float(summary.get(k, 0)) for k in cat_keys) or 1
    recs = []
    for card in cards_to_eval:
        savings = sum(capped_category_value(card, k, float(summary.get(k, 0))) for k in cat_keys)
        match_score = (savings / total_spend) * 100 * 20
        card_min_income = float(card.get("min_income", 0))
        user_income_val = float(user_income) if user_income else 9999999
        income_eligible = user_income_val >= card_min_income

        recs.append({
            "card_name": card.get("card_name", "Unknown"),
            "bank_name": card.get("bank_name", "Unknown"),
            "annual_fee": card.get("annual_fee", 0),
            "fee_waiver_spend": card.get("fee_waiver_spend", 0),
            "min_income": card.get("min_income", 0),
            "card_tier": card.get("card_tier", "Standard"),
            "network": (lambda v: v if v and v not in ("0", "0.0", "nan") else "")(str(card.get("network_type", "")).strip()),
            "max_benefit": card.get("max_benefit_category", "General"),
            "lounge_access": card.get("lounge_access", 0),
            "lounge_type": str(card.get("lounge_type", "")) or "None",
            "card_category": card.get("card_category", "General"),
            "eligibility_level": card.get("eligibility_level", "Low"),
            "match": min(99.9, round(match_score, 1)),
            "annual_savings": int(savings),
            "income_eligible": income_eligible,       # NEW: True if user qualifies
            "rates": {k: card.get(f"reward_points_{k}" if k != "utility" else "reward_points_utilities", 0) for k in cat_keys},
            # v2 real-data fields (absent → sensible defaults for legacy rows)
            "caps": {k: get_cap(card, k) for k in cat_keys},
            "joining_fee": card.get("joining_fee", card.get("annual_fee", 0)),
            "reward_type": card.get("reward_type", "points"),
            "point_value_rs": card.get("point_value_rs", 1.0),
            "fuel_surcharge_waiver": int(card.get("fuel_surcharge_waiver", 0) or 0),
            "forex_markup_pct": card.get("forex_markup_pct", 3.5),
            "lounge_domestic_yr": int(card.get("lounge_domestic_yr", 0) or 0),
            "lounge_intl_yr": int(card.get("lounge_intl_yr", 0) or 0),
            "welcome_benefit": str(card.get("welcome_benefit", "") or ""),
            "milestone_benefit": str(card.get("milestone_benefit", "") or ""),
            "exclusions": str(card.get("exclusions", "") or ""),
            "highlights": str(card.get("highlights", "") or ""),
            "source_url": str(card.get("source_url", "") or ""),
            "data_updated": str(card.get("data_updated", "") or ""),
        })
    # Show ALL cards but sort: eligible first (by savings), then ineligible (still useful for aspiration)
    eligible   = sorted([r for r in recs if r["income_eligible"]],  key=lambda x: x["annual_savings"], reverse=True)
    ineligible = sorted([r for r in recs if not r["income_eligible"]], key=lambda x: x["annual_savings"], reverse=True)
    return (eligible + ineligible)[:5]

def generate_llm_insight(user_id, persona, top_category, top_perc, summary):
    # Return cached insight if we already generated one for this user this session.
    # This is the main quota saver — /profile is called on every tab switch.
    if user_id and user_id in _INSIGHT_CACHE:
        return _INSIGHT_CACHE[user_id]
    if not gemini_client:
        tips = {
            "dining": f"you're spending {top_perc}% on food — a good dining card could easily get you ₹3,000–5,000 back a year just from ordering in.",
            "travel": f"with {top_perc}% going to travel, you're leaving free flights on the table. A solid travel card pays for itself on your next booking.",
            "shopping": f"{top_perc}% on shopping means there's real cashback sitting unclaimed every month.",
            "fuel": f"spending {top_perc}% on fuel? A surcharge waiver card alone can save ₹1,500+ a year.",
            "utility": f"bills and subscriptions at {top_perc}% — boring category, but a utility card earns you back quietly every month.",
            "others": f"a solid {top_perc}% in misc spending. A flat-rate cashback card is your best friend here.",
        }
        tip = tips.get(top_category, f"your top spend is {top_category} at {top_perc}%.")
        return f"Looks like {tip} The right card won't change how you live — it'll just make your existing spending work harder for you."

    prompt = (
        f"You're a friendly personal finance buddy — not a bank bot. "
        f"A friend's spending shows they're a '{persona}', with {top_perc}% in {top_category}. "
        f"Write 2 short conversational sentences of practical insight. "
        f"Be casual but useful. No buzzwords. Slightly witty is good. Don't start with 'I'."
    )
    # Use the shared generate() helper which has caching + correct model list
    result = generate(prompt)
    if result:
        text = result.replace('"', '').replace('**', '').strip()
        if user_id:
            _INSIGHT_CACHE[user_id] = text
        return text
    fallback = f"Honestly, {top_perc}% in {top_category} is a strong pattern — the right card here could earn you ₹2,000–4,000 a year without changing anything."
    if user_id:
        _INSIGHT_CACHE[user_id] = fallback
    return fallback

# ==========================================
# 🌐 ENDPOINTS
# ==========================================

@app.get("/users")
def get_users_list():
    return list(get_users_meta().keys())

def hash_password(pw: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(8)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${digest}"

def verify_password(pw: str, stored: str) -> bool:
    if "$" not in str(stored):
        return str(stored) == pw  # legacy plaintext record
    salt = str(stored).split("$", 1)[0]
    return secrets.compare_digest(hash_password(pw, salt), str(stored))

@app.post("/login")
def login_user(data: LoginData):
    meta = get_users_meta()
    if data.user_id not in meta:
        return {"success": False, "error": "User not found"}
    stored = meta[data.user_id].get("password", "")
    if verify_password(data.password, stored):
        if "$" not in str(stored):  # upgrade legacy plaintext to a hash
            meta[data.user_id]["password"] = hash_password(data.password)
            save_users_meta(meta)
        return {"success": True}
    return {"success": False, "error": "Incorrect password"}

@app.delete("/delete-user/{user_id}")
def delete_user(user_id: str):
    meta = get_users_meta()
    if user_id in meta:
        del meta[user_id]
        save_users_meta(meta)
    if os.path.exists(p('transactions.csv')):
        df = pd.read_csv(p('transactions.csv'))
        df = df[df['user'].str.lower() != user_id.lower()]
        df.to_csv(p('transactions.csv'), index=False)
    return {"success": True}

@app.post("/create-user")
def create_user_api(user: NewUser):
    df = get_all_cards_df()
    if df.empty:
        return {"error": "NeuroCard_Data.csv not found"}
    meta = get_users_meta()
    meta[user.name] = {
        "name": user.name, "password": hash_password(user.password),
        "persona": user.persona.capitalize(), "income": user.income,
        "redemption_pref": user.redemption_pref, "owned_cards": user.owned_cards
    }
    save_users_meta(meta)

    file_path = p('transactions.csv')
    file_exists = os.path.isfile(file_path)

    # ── Income-scaled budgets ──────────────────────────────────────────────────
    annual_income = float(user.income)
    disposable = annual_income * 0.60          # 60% of income is spendable
    scale = min(disposable / 600000.0, 3.0)    # 6L income = scale 1.0 baseline; cap at 2×

    BASE_BUDGETS = {
        "foodie":   {"dining":250000,"travel":80000, "shopping":120000,"fuel":50000, "utility":60000, "others":40000},
        "traveler": {"dining":100000,"travel":350000,"shopping":80000, "fuel":40000, "utility":50000, "others":40000},
        "saver":    {"dining":60000, "travel":50000, "shopping":100000,"fuel":50000, "utility":180000,"others":150000},
        "shopper":  {"dining":120000,"travel":70000, "shopping":300000,"fuel":40000, "utility":60000, "others":50000},
        "commuter": {"dining":70000, "travel":40000, "shopping":80000, "fuel":200000,"utility":100000,"others":50000},
        "balanced": {"dining":150000,"travel":150000,"shopping":150000,"fuel":100000,"utility":100000,"others":100000},
    }
    base = BASE_BUDGETS.get(user.persona.lower(), BASE_BUDGETS["balanced"])
    targets = {cat: int(amt * scale) for cat, amt in base.items()}

    # Fetch persona-specific merchant list (Gemini-powered, falls back gracefully)
    # We do this BEFORE opening the file to avoid keeping the handle open during network calls
    user_merchants = get_merchants_for_user(user.persona)
    
    start_date = datetime.now() - timedelta(days=365)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["user","date","merchant","amount","paid_using","points_earned"])

        # ALL writing logic is now indented INSIDE the 'with' block
        for cat, target_amount in targets.items():
            num_txns = random.randint(40, 60)
            avg_txn = target_amount / num_txns
            merchant_pool = user_merchants.get(cat, FALLBACK_MERCHANTS[cat])
            for _ in range(num_txns):
                date = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
                amount = int(avg_txn * random.uniform(0.5, 1.5))
                merchant = random.choice(merchant_pool)
                card_used = random.choice(user.owned_cards) if user.owned_cards else "Debit/Cash"
                pred_cat, _ = smart_predict(merchant)
                rate = get_rate_from_df(df, card_used, pred_cat) if card_used != "Debit/Cash" else 0
                writer.writerow([user.name, date.strftime("%Y-%m-%d %H:%M"), merchant, amount, card_used, int(amount * rate)])

    return {"status": "success"}

@app.get("/all-cards-list")
def get_cards_list():
    unique = {}
    for c in get_all_cards():
        if c.get("card_name") not in unique:
            unique[c.get("card_name")] = c.get("bank_name", "Unknown")
    return sorted([{"card_name": k, "bank_name": v} for k, v in unique.items()], key=lambda x: x["card_name"])

@app.post("/wallet/add")
def add_to_wallet(action: WalletAction):
    meta = get_users_meta()
    if action.user_id in meta and action.card_name not in meta[action.user_id].get("owned_cards", []):
        meta[action.user_id]["owned_cards"].append(action.card_name)
        save_users_meta(meta)
    return {"owned_cards": meta[action.user_id].get("owned_cards", [])}

@app.post("/wallet/remove")
def remove_from_wallet(action: WalletAction):
    meta = get_users_meta()
    if action.user_id in meta and action.card_name in meta[action.user_id].get("owned_cards", []):
        meta[action.user_id]["owned_cards"].remove(action.card_name)
        save_users_meta(meta)
    return {"owned_cards": meta.get(action.user_id, {}).get("owned_cards", [])}

@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    if not os.path.exists(p('transactions.csv')):
        return {"error": "No transaction data. Please create a user first."}
    df = pd.read_csv(p('transactions.csv'))
    user_df = df[df['user'].str.lower() == user_id.lower()].copy()
    if user_df.empty:
        return {"error": "User not found"}

    user_df['category'] = user_df['merchant'].apply(lambda m: smart_predict(m)[0])

    summary = {k: float(v) for k, v in user_df.groupby('category')['amount'].sum().items()}
    for cat in CATEGORIES:
        if cat not in summary:
            summary[cat] = 0.0

    total_spend = sum(summary.values()) or 1
    spend_vector = [summary[c] / total_spend for c in CATEGORIES]
    ai_persona = predict_persona(spend_vector)

    sorted_cats = sorted(summary.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_perc = sorted_cats[0][0], int((sorted_cats[0][1] / total_spend) * 100)

    meta = get_users_meta()
    user_meta = meta.get(user_id, {"persona":"Member","income":500000,"redemption_pref":"Cashback","owned_cards":[]})

    insight = generate_llm_insight(user_id, ai_persona, top_cat, top_perc, summary)
    points_summary = user_df.groupby('paid_using')['points_earned'].sum().to_dict()
    global_recs = calculate_recs(summary, user_income=user_meta.get("income", 9999999))

    all_cards = get_all_cards()
    best_possible = sum(
        max((capped_category_value(card, cat, summary[cat]) for card in all_cards), default=0)
        for cat in CATEGORIES
    ) if all_cards else 0
    actual = float(user_df['points_earned'].sum())
    missed_value = max(0, int(best_possible - actual))

    # Redemption value calculation
    total_points = int(actual)
    redemption_info = calculate_redemption_value(total_points, user_meta.get("redemption_pref","Cashback"), summary)

    return {
        "info": {
            "name": user_id, "persona": ai_persona,
            "income": user_meta["income"],
            "redemption_pref": user_meta["redemption_pref"],
            "description": insight
        },
        "owned_cards": user_meta.get("owned_cards", []),
        "summary": summary,
        "points_summary": points_summary,
        "redemption_info": redemption_info,
        "retroactive_stats": {
            "total_spend": int(total_spend),
            "actual_savings": int(actual),
            "potential_savings": int(best_possible),
            "missed_value": missed_value
        },
        "transactions": user_df.sort_values(by='date', ascending=False).head(50).to_dict('records'),
        "recommendations": global_recs
    }

@app.post("/manual-analyze")
def manual_analyze(data: SliderData):
    meta = get_users_meta().get(data.user_id, {})
    return {"recommendations": calculate_recs(data.dict(), user_income=meta.get("income", 9999999))}

@app.get("/smart-swipe")
def smart_swipe(user: str, merchant: str):
    if not merchant.strip():
        return {"error": "No merchant provided"}

    override = keyword_predict(merchant)
    if override:
        pred_cat, confidence, source = override, 1.0, "rules"
    else:
        ml_cat, ml_conf = smart_predict(merchant)
        if ml_conf >= 0.55:
            pred_cat, confidence, source = ml_cat, ml_conf, "ml"
        else:
            g_cat, g_conf = gemini_categorize(merchant)
            pred_cat, confidence, source = g_cat, g_conf, "gemini"

    owned_names = get_users_meta().get(user, {}).get("owned_cards", [])
    if not owned_names:
        return {"error": "Wallet is empty. Add cards first."}

    simulated = {c: 0 for c in CATEGORIES}
    simulated[pred_cat] = 10000
    recs = calculate_recs(simulated, specific_cards=owned_names)
    if not recs:
        return {"error": "No cards available for evaluation."}

    all_cards = get_all_cards()
    owned_objs = [c for c in all_cards if c.get('card_name') in owned_names]
    if owned_objs:
        best = max(owned_objs, key=lambda c: get_rate(c, pred_cat) * (1 if c.get("max_benefit_category", "").lower() == pred_cat else 0.8))
        swipe_value = get_rate(best, pred_cat) * 10000
        best_cap = get_cap(best, pred_cat)
        if best_cap > 0:
            swipe_value = min(swipe_value, best_cap)
        return {
            "best_card": best.get("card_name"), "bank": best.get("bank_name"),
            "savings_on_10k": int(swipe_value),
            "category": pred_cat, "confidence": round(confidence, 2),
            "source": source, "reason": f"Best rewards in {pred_cat} among your wallet cards"
        }
    return {
        "best_card": recs[0]["card_name"], "bank": recs[0]["bank_name"],
        "savings_on_10k": recs[0]["annual_savings"],
        "category": pred_cat, "confidence": round(confidence, 2),
        "source": source, "reason": f"Highest {pred_cat} reward rate in your wallet"
    }

@app.post("/categorize-merchant")
def categorize_merchant_endpoint(req: CategorizeMerchant):
    override = keyword_predict(req.merchant)
    if override:
        return {"category": override, "confidence": 1.0, "source": "rules"}
    cat, conf = smart_predict(req.merchant)
    if conf >= 0.55:
        return {"category": cat, "confidence": conf, "source": "ml"}
    cat, conf = gemini_categorize(req.merchant)
    return {"category": cat, "confidence": conf, "source": "gemini"}

@app.post("/refresh-cards")
def refresh_cards():
    """Trigger the v2 scraper (validated pipeline — cannot corrupt the CSV)."""
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, p("scraper_v2.py"), "--limit", "10"],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode == 0:
            card_count = len(get_all_cards())
            return {"status": "success", "message": f"Card data refreshed. {card_count} cards loaded.", "output": result.stdout[-500:]}
        else:
            return {"status": "error", "message": "Scraper failed. Static data in use.", "error": result.stderr[-300:]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/upload-statement")
async def upload_statement(user_id: str = Form(...), file: UploadFile = File(...), replace: bool = Form(True)):
    """Ingest a real bank statement (CSV/Excel): parse → classify → analyze.
    By default this REPLACES the user's synthetic history so the dashboard
    reflects their real spending, not the demo data."""
    from statement_parser import parse_statement
    blob = await file.read()
    try:
        txns = parse_statement(file.filename, blob)
    except ValueError as e:
        return {"error": str(e)}

    file_path = p('transactions.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if replace:
            df = df[df['user'].str.lower() != user_id.lower()]
    else:
        df = pd.DataFrame(columns=["user", "date", "merchant", "amount", "paid_using", "points_earned"])

    rows = []
    for t in txns:
        rows.append({
            "user": user_id, "date": t["date"], "merchant": t["merchant"],
            "amount": t["amount"], "paid_using": "Statement Upload", "points_earned": 0,
        })
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(file_path, index=False)

    # quick classification preview so the user sees it worked
    preview = []
    for t in txns[:8]:
        cat, conf = smart_predict(t["merchant"])
        preview.append({"merchant": t["merchant"], "raw": t["merchant_raw"],
                        "amount": t["amount"], "category": cat, "confidence": conf})
    _INSIGHT_CACHE.pop(user_id, None)  # force fresh AI insight for real data
    return {"status": "success", "imported": len(txns), "replaced_synthetic": replace,
            "preview": preview}

@app.get("/rewards-analysis/{user_id}")
def rewards_analysis(user_id: str):
    """Per-card ROI: is each card in the wallet paying for itself?
    Uses the card's REAL rates/caps against the user's actual per-card spend."""
    if not os.path.exists(p('transactions.csv')):
        return {"error": "No transactions"}
    df = pd.read_csv(p('transactions.csv'))
    user_df = df[df['user'].str.lower() == user_id.lower()].copy()
    if user_df.empty:
        return {"error": "User not found"}

    user_df['category'] = user_df['merchant'].apply(lambda m: smart_predict(m)[0])
    meta = get_users_meta().get(user_id, {})
    owned = meta.get("owned_cards", [])
    all_cards = {c.get("card_name"): c for c in get_all_cards()}

    cards_out = []
    for name in owned:
        card = all_cards.get(name)
        if not card:
            continue
        card_df = user_df[user_df['paid_using'] == name]
        spend_by_cat = card_df.groupby('category')['amount'].sum().to_dict()
        card_spend = float(card_df['amount'].sum())
        # value this card earned on its own transactions (cap-aware)
        earned = sum(capped_category_value(card, cat, amt) for cat, amt in spend_by_cat.items())
        # value the user's ENTIRE spend would earn on this card (potential)
        total_by_cat = user_df.groupby('category')['amount'].sum().to_dict()
        potential = sum(capped_category_value(card, cat, amt) for cat, amt in total_by_cat.items())
        fee = float(card.get("annual_fee", 0) or 0)
        waiver = float(card.get("fee_waiver_spend", 0) or 0)
        roi = earned - fee
        if fee == 0:
            verdict = "Free card — pure upside. Keep it."
        elif waiver > 0 and card_spend >= waiver:
            verdict = f"Fee waived (spend ₹{int(card_spend):,} ≥ ₹{int(waiver):,}). Everything it earns is profit."
        elif roi > 0:
            verdict = f"Earning ₹{int(earned):,} vs ₹{int(fee):,} fee — paying for itself."
        elif waiver > 0:
            gap = waiver - card_spend
            verdict = f"Spend ₹{int(gap):,} more on this card to waive the ₹{int(fee):,} fee."
        else:
            verdict = f"Costing you ₹{int(-roi):,}/yr — consider replacing it."
        cards_out.append({
            "card_name": name, "bank_name": card.get("bank_name"),
            "spend_on_card": int(card_spend),
            "rewards_value_earned": int(earned),
            "annual_fee": int(fee),
            "fee_waiver_spend": int(waiver),
            "waiver_progress_pct": min(100, round(card_spend / waiver * 100)) if waiver > 0 else 100,
            "roi": int(roi),
            "reward_type": card.get("reward_type", "points"),
            "point_value_rs": card.get("point_value_rs", 1.0),
            "potential_if_all_spend": int(potential),
            "verdict": verdict,
        })
    cards_out.sort(key=lambda c: c["roi"], reverse=True)
    return {"cards": cards_out}

@app.get("/card-details/{card_name}")
def card_details(card_name: str):
    """Full catalog record for one card — powers the detail modal."""
    for card in get_all_cards():
        if str(card.get("card_name", "")).lower() == card_name.lower():
            cat_keys = ["dining", "travel", "shopping", "fuel", "utility", "others"]
            return {
                **card,
                "rates": {k: get_rate(card, k) * 100 for k in cat_keys},
                "caps": {k: get_cap(card, k) for k in cat_keys},
                "highlight_list": [h for h in str(card.get("highlights", "")).split("|") if h],
            }
    return {"error": f"Card '{card_name}' not found"}

@app.get("/model-info")
def get_model_info():
    """Returns metadata about the loaded AI models — useful for the demo."""
    nlp_trained = nlp_classifier is not None
    kmeans_trained = kmeans_model is not None
    from_disk = os.path.exists(p("nlp_model.joblib"))

    return {
        "nlp_model": {
            "loaded": nlp_trained,
            "source": "Trained (joblib)" if from_disk else "Inline fallback",
            "algorithm": "TF-IDF (char n-gram) + Logistic Regression",
            "categories": CATEGORIES,
        },
        "clustering_model": {
            "loaded": kmeans_trained,
            "algorithm": "K-Means (k=6) with RFM-weighted persona centers",
            "personas": PERSONAS,
            "n_clusters": 6,
        },
        "keyword_rules": len(KEYWORD_OVERRIDES),
        "prediction_pipeline": "Keyword Rules → ML (conf ≥ 0.55) → Gemini API",
    }

@app.post("/api/advisor/chat")
def advisor_chat(req: ChatRequest):
    meta = get_users_meta().get(req.user_id, {})
    persona = meta.get("persona", "User")
    pref = meta.get("redemption_pref", "Cashback")
    points = 0
    if os.path.exists(p('transactions.csv')):
        df = pd.read_csv(p('transactions.csv'))
        user_df = df[df['user'].str.lower() == req.user_id.lower()]
        if not user_df.empty:
            points = int(user_df['points_earned'].sum())

    context = (
        f"You are NeuroCard AI, a friendly smart finance assistant. "
        f"User is a '{persona}' spender with {points} reward points preferring '{pref}'. "
        f"Keep answers short, practical, casual — like a knowledgeable friend. Max 3 sentences."
    )
    if not gemini_client:
        return {"reply": f"Your {points} points are worth ~₹{int(points * 0.25)} in cashback. For {pref}, best to redeem during sale seasons for up to 2x value. (Add GEMINI_API_KEY to .env for real AI.)"}

    # Use the shared generate() helper (correct models + caching)
    result = generate(f"{context}\n\nUser: {req.message}")
    if result:
        return {"reply": result}
    return {"reply": "AI servers are busy — try again in a moment."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
