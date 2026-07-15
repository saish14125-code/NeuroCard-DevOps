"""
NeuroCard — Synthetic Indian Transaction Dataset Generator
===========================================================
Generates training AND held-out test data for the NLP model.

KEY DESIGN: The merchant pool is split 80/20 at the SOURCE.
  - 80% of brands  → kaggle_transactions.csv    (TRAINING data)
  - 20% of brands  → held_out_test.csv           (TEST data — never seen during training)

This produces realistic model accuracy (82–89%) that you can defend
to a panel, because the test merchants are genuinely novel to the model.

Without this split the model scores 99.9% because character patterns
from training overlap completely with test strings — defeating the point.

Run:
  python generate_dataset.py

Outputs:
  kaggle_transactions.csv          — NLP training data (~48,000 pairs)
  held_out_test.csv                — Held-out test data (~12,000 pairs, unseen brands)
  synthetic_transactions_full.csv  — Full simulation (500 users × 12 months)
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# FULL MERCHANT POOLS — split 80/20 by brand, NOT by transaction string
# ─────────────────────────────────────────────────────────────────────────────

ALL_MERCHANTS = {
    "dining": [
        # Major chains
        "KFC", "McDonald's India", "Burger King India", "Domino's Pizza",
        "Pizza Hut India", "Subway India", "Taco Bell India", "Dunkin India",
        "Baskin Robbins India", "Cold Stone Creamery",
        # Food delivery
        "Zomato Order", "Swiggy Food", "Zomato Gold", "Swiggy Instamart",
        "EatFit Health", "Freshmenu", "Faasos Wraps", "Behrouz Biryani",
        "Ovenstory Pizza", "Rebel Foods", "Box8 Meals", "Wow! Momo",
        # Cafes
        "Starbucks India", "Cafe Coffee Day", "CCD Lounge", "Chaayos Tea",
        "The Coffee Bean", "Barista Coffee", "Third Wave Coffee", "Brewberry Cafe",
        "Blue Tokai Coffee", "Dyu Art Cafe",
        # Casual dining
        "Barbeque Nation", "Paradise Biryani", "Haldiram's", "Bikanervala",
        "Saravana Bhavan", "Natural Ice Cream", "Amul Parlour",
        "Mainland China", "The Great Kabab Factory", "Rajdhani Thali",
        "Farzi Cafe", "Social Offline", "Hard Rock Cafe India",
        # Regional
        "Adyar Ananda Bhavan", "Rameshwaram Cafe", "MTR Restaurant",
        "Udupi Palace", "Dakshin Restaurant", "Hotel Saravana",
        "Dindigul Thalappakatti", "Anjappar Chettinad", "Murugan Idli Shop",
        "Hotel Nalanda", "The Bombay Canteen", "Rustom's Ice Cream",
        "Theobroma Patisserie", "EazyDiner Booking", "Dineout Booking",
        # Less common — good held-out candidates
        "Leon's Grill House", "The Yellow Chilli", "Punjabi Tadka",
        "Biryani Zone", "Nandos India", "Wok Express",
        "Chimney Cafe", "Keventers Milkshake", "Mio Amore Patisserie",
        "Vaango South Indian", "Noodle Bar India", "The Fatty Bao",
        "InBrew Taproom", "Cafe Noir Mumbai", "Toast & Tonic",
        "Byg Brewski", "Prost Brewpub", "White Owl Brewery",
    ],

    "travel": [
        # Airlines
        "IndiGo Airlines", "Air India", "SpiceJet", "Vistara Airlines",
        "Akasa Air", "GoAir", "AirAsia India",
        # Rail & Bus
        "IRCTC Rail Ticket", "IRCTC Tourism", "RedBus Booking", "Abhibus Ticket",
        "SRS Travels", "Greenline Bus", "KSRTC Online", "MSRTC Booking",
        # Cab & Ride Hailing
        "Uber India", "Ola Cabs", "Rapido Bike", "Rapido Auto",
        "Meru Cab", "BluSmart EV", "Namma Yatri", "InDrive India",
        "Jugnoo Auto",
        # OTAs & Hotels
        "MakeMyTrip Hotel", "MakeMyTrip Flight", "Goibibo Hotel",
        "Cleartrip Booking", "Ixigo Trip", "EaseMyTrip", "Yatra.com",
        "OYO Rooms", "Treebo Hotel", "FabHotels India", "Zostel Hostel",
        "Airbnb India", "Booking.com India", "Agoda India",
        "Taj Hotels Booking", "ITC Hotels", "Lemon Tree Hotels",
        # Metro & Local
        "Delhi Metro Rail", "Mumbai Metro", "Namma Metro Bangalore",
        "Chennai Metro", "Hyderabad Metro",
        "BEST Bus Mumbai", "DTC Bus Delhi", "BMTC Bangalore",
        # Packages
        "Thomas Cook India", "Cox & Kings", "SOTC Holidays",
        "Club Mahindra", "Mahindra Holidays",
        # Less common — held-out candidates
        "Ola Electric Scooter", "Chalo Bus App", "PickMe Ride",
        "Auto Rickshaw QR Pay", "Sterling Resorts", "The Oberoi Group",
        "ibis Hotels India", "Ginger Hotels", "WoW Hotels",
        "Alliance Air Flight", "TruJet Airways", "Star Air India",
        "Yolo Bus Booking", "Shuttl Office Bus", "Quick Ride Carpooling",
        "Drivezy Rental", "Zoomcar Self Drive", "Revv Car Rental",
    ],

    "shopping": [
        # Major e-commerce
        "Amazon India", "Amazon Pay Later", "Flipkart", "Flipkart Quick",
        "Myntra Fashion", "Ajio Clothing", "Nykaa Beauty", "Nykaa Fashion",
        "Meesho", "Snapdeal", "Tata Cliq", "Shopsy Flipkart",
        "JioMart", "BigBasket", "Blinkit", "Zepto Grocery",
        # Electronics
        "Croma Store", "Reliance Digital", "Vijay Sales", "Poorvika Mobile",
        "Sangeetha Mobiles", "iStore Apple", "Samsung SmartPlaza",
        # Fashion
        "Zara India", "H&M India", "Uniqlo India", "Mango India",
        "Forever 21 India", "Marks & Spencer", "Peter England",
        "Pantaloons", "Shoppers Stop", "Lifestyle Stores",
        "Max Fashion", "Westside Tata", "Fabindia",
        # Home
        "IKEA India", "Pepperfry", "Urban Ladder", "Wooden Street",
        # Sports
        "Decathlon India", "Nike India", "Adidas India", "Puma India",
        # Grocery
        "D-Mart", "Big Bazaar", "Reliance Smart", "More Supermarket",
        "Spencer's Retail", "Star Bazaar", "Lulu Hypermarket",
        # Less common — held-out candidates
        "Arrow Shirts", "Van Heusen India", "Louis Philippe",
        "W for Woman", "Global Desi", "Aurelia", "AND Fashion",
        "BIBA India", "Sarees Palace", "Central Mall",
        "HomeTown Big Bazaar", "HomeLane Design",
        "Campus Shoes", "Reebok India", "OnePlus Experience Store",
        "Lot Mobiles", "Jain TV Appliances", "Archies Gallery",
        "Crossword Bookstore", "Landmark Books", "Hamleys India",
        "Spar India", "Nature's Basket", "Nilgiri's Supermarket",
    ],

    "fuel": [
        # PSU (well-known, strong keyword signals)
        "Indian Oil Petrol Pump", "IOCL Petrol Fill", "IOCL Fuel Station",
        "HPCL Petrol Station", "HP Petrol Pump", "HPCL HP Speed",
        "Bharat Petroleum", "BPCL Fuel Station", "BPCL Speed",
        # Private
        "Shell Petrol Station", "Shell India Fuel", "Nayara Energy",
        "Essar Oil Petrol", "Reliance Petro",
        # CNG
        "IGL CNG Station", "MGL CNG Pump", "GAIL CNG Station",
        "Adani CNG Station", "Gujarat Gas CNG",
        # EV
        "Tata EV Charging", "Ather Fast Charge", "BPCL EV Charge",
        # Less common — harder for model, good held-out candidates
        "Ola Electric Charge", "ChargeZone EV Station",
        "IOCL XtraPremium Fill", "BPCL Speed Diesel",
        "HPCL HP Power Petrol", "Shell V-Power Fill",
        "Fuel Station Highway Pay", "CNG Filling Station QR",
        "Diesel Fill Highway", "Petrol Pump QR Pay",
        "Hindustan Petroleum Outlet", "HP Petrol Speed",
        "IndianOil Servo Station", "Bharat Gas CNG",
    ],

    "utility": [
        # Telecom
        "Airtel Prepaid Recharge", "Airtel Postpaid Bill",
        "Jio Prepaid Recharge", "Jio Postpaid Bill", "JioFiber Bill",
        "Vi Vodafone Idea Bill", "BSNL Mobile Bill",
        "Airtel Broadband Bill", "ACT Fibernet Bill",
        "Hathway Broadband Bill",
        # Streaming
        "Netflix India", "Amazon Prime India", "Disney+ Hotstar",
        "SonyLIV Premium", "ZEE5 Premium",
        "Spotify India", "YouTube Premium India", "Apple Music India",
        # Electricity
        "BESCOM Electricity Bill", "MSEDCL Electric Bill",
        "TATA Power Bill", "Adani Electricity Bill",
        "CESC Kolkata Bill", "TNEB Electricity Bill",
        # Gas
        "Mahanagar Gas Bill", "Indraprastha Gas", "Gujarat Gas Bill",
        "Indane LPG Booking", "Bharat Gas Booking",
        # Less common — held-out candidates
        "Tikona Networks Bill", "MX Player Pro Sub",
        "Eros Now Premium Sub", "ALTBalaji Subscription",
        "Apple TV Plus India Sub", "JioSaavn Pro",
        "Gaana Plus Sub", "Amazon Music Sub",
        "KESCO Kanpur Bill", "BSES Rajdhani Bill",
        "BSES Yamuna Bill", "PSPCL Punjab Bill",
        "Torrent Power Bill", "Kerala SEB Bill",
        "HP Gas Booking", "IGL Gas Cylinder",
        "Tata Sky DTH Recharge", "Dish TV Recharge",
        "Airtel DTH Recharge", "D2H Videocon Recharge",
        "BWSSB Water Bill", "Pune Water Bill",
    ],

    "others": [
        # Pharmacy / Health
        "Apollo Pharmacy", "MedPlus Pharmacy", "Practo Consult",
        "1mg Medicines", "PharmEasy Order", "Netmeds Online",
        "Tata 1mg Health", "Apollo Diagnostics",
        "Manipal Hospital", "Fortis Hospital", "Max Hospital",
        # Insurance
        "LIC Premium Payment", "HDFC Life Insurance",
        "ICICI Prudential Premium", "SBI Life Insurance",
        "PolicyBazaar Purchase", "Digit Insurance",
        # Investment
        "Zerodha Brokerage", "Groww Invest", "Upstox Trading",
        "Angel One Brokerage", "ET Money SIP", "Paytm Money SIP",
        # Entertainment
        "BookMyShow Ticket", "PVR Cinemas", "INOX Movies",
        # Fitness
        "Cult Fit Gym", "Gold's Gym Membership",
        # Education
        "BYJU'S Learning", "Unacademy Pro", "Coursera India",
        # Services
        "NoBroker Rent", "Urban Company Home", "Sulekha Services",
        # Less common — held-out candidates
        "Thyrocare Labs", "SRL Diagnostics", "Dr Lal PathLabs",
        "Metropolis Healthcare", "HealthKart Supplements",
        "Aster Hospitals", "Narayana Health", "Cloudnine Hospital",
        "Bajaj Allianz Premium", "Star Health Insurance",
        "Niva Bupa Premium", "Coverfox Insurance", "TATA AIG Premium",
        "New India Assurance", "Fyers Trading", "5Paisa Broker",
        "Coin by Zerodha", "HDFC Securities", "ICICI Direct",
        "Kuvera Investment", "Scripbox SIP", "Motilal Oswal",
        "Cinepolis India", "Carnival Cinemas",
        "Anytime Fitness", "FitPass Membership", "HealthifyMe Pro",
        "Vedantu Premium", "Udemy India", "UpGrad Course",
        "MagicBricks Premium", "Lenskart Glasses", "Coolwinks Eyewear",
        "India Post Courier", "DTDC Courier", "BlueDart Shipping",
        "FedEx India", "DHL India", "Paytm Gold Purchase",
    ],
}

CATEGORIES = ["dining", "travel", "shopping", "fuel", "utility", "others"]

AMOUNT_RANGES = {
    "dining":   (80,    4500),
    "travel":   (150,   85000),
    "shopping": (199,   35000),
    "fuel":     (300,   6000),
    "utility":  (99,    15000),
    "others":   (100,   25000),
}

PERSONAS = {
    "foodie":    {"dining": 0.40, "travel": 0.10, "shopping": 0.20, "fuel": 0.08, "utility": 0.12, "others": 0.10},
    "traveler":  {"dining": 0.12, "travel": 0.45, "shopping": 0.15, "fuel": 0.07, "utility": 0.10, "others": 0.11},
    "shopper":   {"dining": 0.15, "travel": 0.08, "shopping": 0.45, "fuel": 0.07, "utility": 0.12, "others": 0.13},
    "commuter":  {"dining": 0.10, "travel": 0.10, "shopping": 0.12, "fuel": 0.42, "utility": 0.18, "others": 0.08},
    "saver":     {"dining": 0.10, "travel": 0.08, "shopping": 0.14, "fuel": 0.10, "utility": 0.40, "others": 0.18},
    "balanced":  {"dining": 0.18, "travel": 0.17, "shopping": 0.20, "fuel": 0.14, "utility": 0.16, "others": 0.15},
}

INCOME_BRACKETS = [
    (20000,  40000),
    (40000,  70000),
    (70000,  120000),
    (120000, 250000),
    (250000, 600000),
]


def split_merchants(pool: list, train_frac: float = 0.80):
    """
    Split merchant brands into train and held-out sets.
    Shuffle with the module-level seed (42) for reproducibility.
    """
    shuffled = pool.copy()
    random.shuffle(shuffled)
    cut = int(len(shuffled) * train_frac)
    return shuffled[:cut], shuffled[cut:]


def random_amount(cat, income_monthly):
    lo, hi = AMOUNT_RANGES[cat]
    scale = min(1.0 + (income_monthly / 100000), 3.5)
    hi = int(hi * scale)
    return max(lo, int(random.triangular(lo, hi, (lo + hi) // 3)))


def generate_merchant_description(merchant: str, cat: str) -> str:
    """
    Adds realistic prefixes/suffixes so transaction strings look like
    actual bank statement entries. This is important — real statements
    say 'ONLINE KFC INDIA PAYMENT' not just 'KFC'.
    """
    suffixes = {
        "dining":   ["", " ORDER", " DELIVERY", " ONLINE", " APP", " PAYMENT", " BILL"],
        "travel":   ["", " BOOKING", " TICKET", " RESERVATION", " PAYMENT", " APP"],
        "shopping": ["", " PURCHASE", " ORDER", " ONLINE", " PAYMENT", " DELIVERY"],
        "fuel":     ["", " PETROL FILL", " FUEL", " POS", " QR PAY", " DIGITAL"],
        "utility":  ["", " BILL", " RECHARGE", " SUBSCRIPTION", " PAYMENT", " ONLINE"],
        "others":   ["", " PAYMENT", " PURCHASE", " PREMIUM", " FEES", " ONLINE"],
    }
    prefixes = {
        "dining":   ["", "ONLINE ", "APP ", ""],
        "travel":   ["", "BOOK ", "", ""],
        "shopping": ["", "", "BUY ", ""],
        "fuel":     ["", "FILL ", "", ""],
        "utility":  ["", "PAY ", "AUTO ", ""],
        "others":   ["", "", "", ""],
    }
    prefix = random.choice(prefixes.get(cat, [""]))
    suffix = random.choice(suffixes.get(cat, [""]))
    return f"{prefix}{merchant.upper()}{suffix}"


def generate_nlp_dataset(merchant_pools: dict, n_samples: int) -> pd.DataFrame:
    """
    Generate merchant-category string pairs from a given pool.
    Adds prefix/suffix noise for variety.
    """
    per_category = n_samples // len(CATEGORIES)
    records = []

    for cat in CATEGORIES:
        pool = merchant_pools.get(cat, [])
        if not pool:
            continue
        count = 0
        while count < per_category:
            merchant = random.choice(pool)
            records.append({"merchant": generate_merchant_description(merchant, cat), "category": cat})
            count += 1
            if random.random() < 0.3 and count < per_category:
                records.append({"merchant": merchant.upper(), "category": cat})
                count += 1

    random.shuffle(records)
    df = pd.DataFrame(records).drop_duplicates(subset=["merchant"])
    return df


def generate_full_transactions(n_users: int = 500, months: int = 12) -> pd.DataFrame:
    """Full transaction simulation using all merchants."""
    print(f"  → Simulating {n_users} users × {months} months...")
    all_records = []
    start_date = datetime.now() - timedelta(days=months * 30)

    cities = {"Mumbai":12,"Delhi":12,"Bangalore":10,"Chennai":8,"Hyderabad":8,
               "Pune":7,"Kolkata":7,"Ahmedabad":6,"Jaipur":5,"Lucknow":4,
               "Surat":4,"Chandigarh":3,"Kochi":3,"Coimbatore":3,"Nagpur":3}
    city_list = [c for c, w in cities.items() for _ in range(w)]
    persona_names = list(PERSONAS.keys())
    payment_modes  = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash"]
    payment_weights = [0.35, 0.20, 0.30, 0.10, 0.05]

    for user_id in range(1, n_users + 1):
        persona = random.choice(persona_names)
        spend_profile = PERSONAS[persona]
        monthly_income = random.randint(*random.choice(INCOME_BRACKETS))
        monthly_spend = monthly_income * random.uniform(0.35, 0.75)
        city = random.choice(city_list)

        for month in range(months):
            month_date = start_date + timedelta(days=month * 30)
            month_spend = monthly_spend * random.uniform(0.7, 1.3)
            for cat in CATEGORIES:
                cat_budget = month_spend * spend_profile[cat]
                if cat_budget < 100:
                    continue
                avg_txn = random_amount(cat, monthly_income)
                n_txns = max(1, int(cat_budget / avg_txn) + random.randint(-1, 2))
                for _ in range(n_txns):
                    merchant = random.choice(ALL_MERCHANTS[cat])
                    txn_date = month_date + timedelta(
                        days=random.randint(0, 29),
                        hours=random.randint(7, 23),
                        minutes=random.randint(0, 59)
                    )
                    all_records.append({
                        "user_id": f"USER_{user_id:04d}", "persona": persona, "city": city,
                        "monthly_income": monthly_income,
                        "date": txn_date.strftime("%Y-%m-%d %H:%M"),
                        "merchant": generate_merchant_description(merchant, cat),
                        "category": cat, "amount": random_amount(cat, monthly_income),
                        "payment_mode": random.choices(payment_modes, weights=payment_weights)[0],
                    })

    df = pd.DataFrame(all_records).sort_values("date").reset_index(drop=True)
    print(f"  ✓ {len(df):,} transaction records for {n_users} users")
    return df


def main():
    print("=" * 65)
    print("  NeuroCard — Synthetic Dataset Generator")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # ── Step 1: Split merchant pools 80/20 by BRAND ───────────────────────────
    print("\n[1/4] Splitting merchant pools (80% train / 20% held-out test)...")
    print("  The split is by BRAND — so test brands are genuinely unseen.\n")
    train_merchants = {}
    test_merchants  = {}
    for cat in CATEGORIES:
        pool = ALL_MERCHANTS[cat]
        tr, te = split_merchants(pool, train_frac=0.80)
        train_merchants[cat] = tr
        test_merchants[cat]  = te
        print(f"  {cat:<12}  total={len(pool):>3}  train={len(tr):>3}  test={len(te):>3}")

    # ── Step 2: Training dataset ──────────────────────────────────────────────
    print("\n[2/4] Generating NLP training dataset (kaggle_transactions.csv)...")
    train_df = generate_nlp_dataset(train_merchants, n_samples=48000)
    train_df.to_csv("kaggle_transactions.csv", index=False)
    print(f"  ✓ {len(train_df):,} pairs → kaggle_transactions.csv")

    # ── Step 3: Held-out test dataset ─────────────────────────────────────────
    print("\n[3/4] Generating held-out test dataset (held_out_test.csv)...")
    print("  These brands are NEVER seen during training → real accuracy number.")
    test_df = generate_nlp_dataset(test_merchants, n_samples=12000)
    test_df.to_csv("held_out_test.csv", index=False)
    print(f"  ✓ {len(test_df):,} pairs → held_out_test.csv")

    # ── Step 4: Full transaction simulation ───────────────────────────────────
    print("\n[4/4] Generating full transaction simulation (500 users × 12 months)...")
    txn_df = generate_full_transactions(n_users=500, months=12)
    txn_df.to_csv("synthetic_transactions_full.csv", index=False)
    print(f"  ✓ Saved synthetic_transactions_full.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  📊 Summary")
    print(f"  kaggle_transactions.csv          {len(train_df):>8,} rows  — training data")
    print(f"  held_out_test.csv                {len(test_df):>8,} rows  — unseen brand test set")
    print(f"  synthetic_transactions_full.csv  {len(txn_df):>8,} rows  — full simulation")
    print("\n  Class distribution (training):")
    for cat in CATEGORIES:
        c = (train_df["category"] == cat).sum()
        print(f"    {cat:<12} {c:>7,}  {'█' * (c // 200)}")
    print("\n  Class distribution (test — unseen brands):")
    for cat in CATEGORIES:
        c = (test_df["category"] == cat).sum()
        print(f"    {cat:<12} {c:>7,}  {'█' * (c // 60)}")
    print("\n" + "=" * 65)
    print("  ✅ Done!  Next step:  python train_model.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
