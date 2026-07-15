"""
NeuroCard — LLM Card Extractor (v2 data pipeline)
==================================================
Turns the messy text of a card page (PaisaBazaar / 1finance / bank site)
into a validated row for NeuroCard_Data.csv.

Why this design survives website redesigns: we never parse HTML structure.
We feed the PAGE TEXT to Gemini and ask for our exact schema. A redesign
changes the layout, not the words, so the pipeline keeps working.

Safety rules (the "never wreck the file" guarantee):
  1. Gemini output must parse as JSON and pass Pydantic validation.
  2. Numbers must pass sanity checks (rates 0–35%, fees 0–60k, caps >= 0).
  3. Anything that fails is printed for human review — NEVER auto-saved.
  4. Passing rows are UPSERTED into the CSV (update by card_name, else append).

Usage:
  python extract_card.py page.txt                 # text file you saved
  python extract_card.py page.txt --name "Axis Select"
  (Save page text via browser: Ctrl+A, Ctrl+C into a .txt file — beats
   fighting Cloudflare with requests.)

Uses GEMINI_API_KEY from .env — one API call per card. Even refreshing
50 cards monthly is ~50 calls: nowhere near free-tier limits.
"""

import sys
import os
import json
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from google import genai

from cards_data import CARD_CSV_COLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CSV_PATH = os.path.join(BASE_DIR, "NeuroCard_Data.csv")
MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


# ── VALIDATION SCHEMA ────────────────────────────────────────────────────────
class CardRow(BaseModel):
    card_name: str = Field(min_length=3, max_length=60)
    bank_name: str = Field(min_length=2, max_length=30)
    card_category: str
    network_type: str
    annual_fee: int = Field(ge=0, le=60000)
    joining_fee: int = Field(ge=0, le=60000)
    fee_waiver_spend: int = Field(ge=0, le=5000000)
    min_income: int = Field(ge=0, le=10000000)
    reward_type: str
    point_value_rs: float = Field(ge=0.05, le=3.0)
    reward_points_dining: float = Field(ge=0, le=35)
    reward_points_travel: float = Field(ge=0, le=35)
    reward_points_shopping: float = Field(ge=0, le=35)
    reward_points_fuel: float = Field(ge=0, le=35)
    reward_points_utilities: float = Field(ge=0, le=35)
    reward_points_others: float = Field(ge=0, le=35)
    cap_dining: int = Field(ge=0, le=100000)
    cap_travel: int = Field(ge=0, le=100000)
    cap_shopping: int = Field(ge=0, le=100000)
    cap_fuel: int = Field(ge=0, le=100000)
    cap_utility: int = Field(ge=0, le=100000)
    cap_others: int = Field(ge=0, le=100000)
    fuel_surcharge_waiver: int = Field(ge=0, le=1)
    forex_markup_pct: float = Field(ge=0, le=5)
    lounge_access: int = Field(ge=0, le=1)
    lounge_domestic_yr: int = Field(ge=0, le=99)
    lounge_intl_yr: int = Field(ge=0, le=99)
    lounge_type: str = ""
    card_tier: str
    eligibility_level: str
    max_benefit_category: str
    welcome_benefit: str = ""
    milestone_benefit: str = ""
    exclusions: str = ""
    highlights: str = ""
    source_url: str = ""
    data_updated: str = ""
    caps_verified: int = Field(ge=0, le=1, default=1)

    @field_validator("reward_type")
    @classmethod
    def check_reward_type(cls, v):
        allowed = {"cashback", "points", "miles", "neucoins", "coins"}
        if v.lower() not in allowed:
            raise ValueError(f"reward_type must be one of {allowed}")
        return v.lower()

    @field_validator("card_tier")
    @classmethod
    def check_tier(cls, v):
        allowed = {"Basic", "Mid", "Premium", "Super_Premium"}
        if v not in allowed:
            raise ValueError(f"card_tier must be one of {allowed}")
        return v


# ── THE PROMPT ───────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are a credit card data analyst. Below is the raw text of an
Indian credit card's information page. Extract its data into EXACTLY this JSON schema.

CRITICAL RULES for reward_points_* fields (the hard part):
- These are EFFECTIVE VALUE-BACK PERCENTAGES per category, as floats.
- Compute them as: (points earned per Rs.100 spent) x (rupee value of 1 point).
  Example: "5 reward points per Rs.200 spent" with "1 RP = Rs.0.50" means
  2.5 points per Rs.100 x 0.50 = 1.25% -> use 1.25.
- For plain cashback cards the percentage is stated directly.
- If a category earns NOTHING (excluded, e.g. fuel on most cards), use 0.
- Map the card's benefits onto exactly these 6 categories:
  dining (restaurants/food delivery), travel (flights/hotels/cabs/trains),
  shopping (e-commerce/retail/fashion/electronics), fuel (petrol/CNG),
  utilities (electricity/telecom/streaming/bills), others (everything else --
  this is also the card's BASE rate on uncategorised spends).

CAP RULES:
- cap_<category> = maximum reward VALUE in rupees earnable PER MONTH in that
  category. 0 means uncapped.
- Quarterly caps: divide by 3 and round. "Capped at 5,000 points/month" with
  1 point = Rs.0.50 -> cap is 2500 (rupee VALUE, not points).
- If one cap is shared across several categories, repeat it for each and
  mention the sharing in "highlights".

OTHER FIELDS:
- min_income: annual income requirement in rupees (monthly stated? x12).
- point_value_rs: rupee value of 1 point at the card's best realistic redemption.
- highlights: 3-5 short selling points separated by "|".
- exclusions: comma-separated spend types that earn nothing.
- card_tier: Basic / Mid / Premium / Super_Premium.
- eligibility_level: Low / Medium / High / Elite.
- If a value is genuinely not in the text, use 0 for numbers, "" for strings.
  DO NOT GUESS numbers that are not supported by the text.

Return ONLY the JSON object, no markdown fences, no commentary.

Schema (fill every key):
{schema}

PAGE TEXT:
---
{page_text}
---"""

SCHEMA_EXAMPLE = json.dumps({k: ("" if k in (
    "card_name", "bank_name", "card_category", "network_type", "reward_type",
    "lounge_type", "card_tier", "eligibility_level", "max_benefit_category",
    "welcome_benefit", "milestone_benefit", "exclusions", "highlights",
    "source_url", "data_updated") else 0) for k in CARD_CSV_COLS}, indent=2)


def extract(page_text: str, source_url: str = "", hint_name: str = "") -> dict | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY missing in .env")
        return None
    client = genai.Client(api_key=api_key)

    prompt = EXTRACTION_PROMPT.format(schema=SCHEMA_EXAMPLE, page_text=page_text[:24000])
    if hint_name:
        prompt += f"\n\n(The card being extracted is: {hint_name})"

    raw = None
    for model in MODELS:
        try:
            r = client.models.generate_content(model=model, contents=prompt)
            raw = r.text.strip()
            break
        except Exception as e:
            print(f"{model} failed: {e}")
    if not raw:
        return None

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.index("{"):raw.rindex("}") + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("Gemini returned invalid JSON — NOT saved. Raw output:\n", raw[:2000])
        return None

    if source_url:
        data["source_url"] = source_url
    data.setdefault("data_updated", pd.Timestamp.now().strftime("%Y-%m-%d"))

    try:
        row = CardRow(**data)
    except Exception as e:
        print("Validation FAILED — NOT saved. Review manually:")
        print(json.dumps(data, indent=2)[:2000])
        print("Errors:", e)
        return None

    return row.model_dump()


def upsert(row: dict):
    """Update-or-append into the CSV. Existing good data is never lost."""
    df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=CARD_CSV_COLS)
    for col in CARD_CSV_COLS:
        if col not in df.columns:
            df[col] = 0 if col.startswith(("cap", "reward_", "lounge_", "joining", "fuel_", "forex")) else ""
    mask = df["card_name"].str.lower() == row["card_name"].lower()
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
        action = "updated"
    else:
        df = pd.concat([df, pd.DataFrame([row])[df.columns.intersection(row.keys())]], ignore_index=True)
        action = "added"
    df[CARD_CSV_COLS].to_csv(CSV_PATH, index=False)
    print(f"✓ {row['card_name']} {action}. Catalog now has {len(df)} cards.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    text = open(sys.argv[1], encoding="utf-8").read()
    hint = ""
    if "--name" in sys.argv:
        hint = sys.argv[sys.argv.index("--name") + 1]
    result = extract(text, hint_name=hint)
    if result:
        print(json.dumps(result, indent=2))
        if input("Save to catalog? [y/N] ").strip().lower() == "y":
            upsert(result)
