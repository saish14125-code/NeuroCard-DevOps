# NeuroCard — AI-Powered Credit Card Recommendation System

NeuroCard analyzes your real spending and tells you, per transaction, which Indian credit
card earns you the most — using **real card economics** (effective value-back rates *and*
monthly reward caps) for 50+ popular Indian credit cards.

> Most comparison sites ignore caps. A "5% cashback" card capped at ₹500/month is worse
> than a flat 2% card for a heavy spender — NeuroCard's engine knows that.

## Features

- **📄 Real statement upload** — export your bank/credit-card statement (CSV/Excel) from
  netbanking and upload it. The parser auto-detects HDFC/ICICI/SBI/Axis formats, strips UPI
  narration junk (`UPI-SWIGGY LIMITED-swiggy8@ybl-...` → `SWIGGY LIMITED`), classifies every
  transaction, and rebuilds your whole dashboard from real data.
- **Hybrid merchant classifier** — keyword rules → char-level TF-IDF + Logistic Regression →
  Gemini as last resort. Evaluated on a brand-disjoint held-out set (~85%+ on unseen brands).
- **Cap-aware recommendation engine** — annual savings = Σ min(monthly spend × rate, cap) × 12,
  per category, per card. Income-filtered.
- **Card ROI Audit** — per card in your wallet: rewards earned vs annual fee, fee-waiver
  progress, and a keep/replace verdict.
- **Smart-Swipe** — type any merchant, get the best card from *your* wallet in real time.
- **K-Means spending personas** (6 archetypes, RFM-inspired normalized spend vectors).
- **AI Strategy Lab** — spend sliders + unlimited multi-card comparison.
- **Gemini advisor chat** with response caching to respect free-tier quotas.

## Data pipeline (the interesting part)

Card data comes from a two-source scraping pipeline (`backend/scraper_v2.py`):

1. **1finance.co.in** — server-rendered pages parsed *deterministically* (regex on a fixed
   format). These numbers are ground truth; no LLM can hallucinate them.
2. **PaisaBazaar** — prose pages fetched as context for caps/fees/benefits.
3. **Gemini merge** — the LLM maps both texts into the 38-column schema but is forbidden to
   contradict the parsed numbers; a cross-check enforces this after the fact.
4. **Pydantic validation** — every row must pass range checks (rates 0–35%, sane fees/caps).
   Failures go to `review_queue.jsonl` for human review — they never touch the catalog.
5. **Verified-only serving** — the app only serves rows with `caps_verified=1`
   (hand-curated or Gemini-checked). Auto-parsed rows wait in the CSV until
   `python scraper_v2.py --enrich` verifies them (~20/day on the Gemini free tier).
6. **Self-healing storage** — the backend keeps a rolling backup of the catalog and
   quarantines + restores automatically if the CSV is ever corrupted.

A site redesign can only break one scraper adapter — never the running app, which always
serves from the last-known-good local catalog.

## Stack

FastAPI · scikit-learn · pandas · Gemini API · React 19 · Tailwind CSS · Playwright-free
scraping (requests + BeautifulSoup)

## Run it

```bash
# Backend
cd backend
pip install fastapi uvicorn pandas scikit-learn joblib python-dotenv google-genai \
            requests beautifulsoup4 pydantic python-multipart openpyxl
cp .env.example .env          # add your free Gemini key (optional — app degrades gracefully)
python train_model.py         # train the NLP + K-Means models
python -m uvicorn main:app --port 8000

# Frontend
cd credit-ai-ui
npm install
npm start                     # opens http://localhost:3000
```

## Honest limitations

- Reward rates are *effective value-back estimates* (points/₹100 × ₹/point at a sane
  redemption) — the same editorial call every comparison expert makes, documented per card
  with a source link and verification date.
- Shared caps (e.g. Axis Ace's combined ₹500/month across its 4–5% categories) are
  approximated as per-category caps.
- Statement PDF parsing is not supported yet — export CSV/Excel from netbanking.
- Login is a demo gate (PBKDF2-hashed passwords, no sessions/JWT).
