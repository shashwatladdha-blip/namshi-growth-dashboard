"""
Namshi synthetic data simulator.

All data produced here is SYNTHETIC. It is built to look like a GCC online
fashion retailer (countries, seasons, payment mix, categories), but every
customer, order, review and campaign result is invented.

The same "ground truth" functions are reused by:
  * build_data.py  - writes the historical CSVs (Jan 2024 - Oct 2026)
  * analysis.py    - simulates the A/B test outcomes for Blog 9
  * app.py         - streams live orders and reviews into the dashboard
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-10-31")

COUNTRIES = ["UAE", "KSA", "Kuwait", "Qatar", "Bahrain", "Oman"]
COUNTRY_P = [0.42, 0.38, 0.07, 0.06, 0.04, 0.03]
COUNTRY_DELIVERY = {"UAE": 1.6, "KSA": 3.2, "Kuwait": 3.0, "Qatar": 2.6, "Bahrain": 2.8, "Oman": 3.4}
COUNTRY_COD = {"UAE": 0.22, "KSA": 0.45, "Kuwait": 0.35, "Qatar": 0.30, "Bahrain": 0.33, "Oman": 0.42}

CATEGORIES = ["Women's Fashion", "Men's Fashion", "Footwear", "Beauty",
              "Modest Wear", "Kids", "Sports", "Accessories"]
CAT_AOV = {"Women's Fashion": 1.0, "Men's Fashion": 0.95, "Footwear": 1.25, "Beauty": 0.55,
           "Modest Wear": 1.15, "Kids": 0.6, "Sports": 0.9, "Accessories": 0.7}
CAT_RETURN = {"Women's Fashion": 0.22, "Men's Fashion": 0.17, "Footwear": 0.20, "Beauty": 0.04,
              "Modest Wear": 0.15, "Kids": 0.10, "Sports": 0.12, "Accessories": 0.06}
CHANNELS = ["Instagram", "Google Search", "TikTok", "Referral", "Organic", "noon cross-sell"]
CHANNEL_P = [0.24, 0.22, 0.16, 0.08, 0.20, 0.10]

# Hidden customer "types" (not given to the models - they must infer them).
TYPES = ["premium", "regular", "bargain"]
TYPE_P = [0.16, 0.50, 0.34]
TYPE_GAP = {"premium": 24, "regular": 40, "bargain": 58}      # mean days between orders
TYPE_AOV = {"premium": 560, "regular": 270, "bargain": 150}   # AED
TYPE_CHURN = {"premium": -0.9, "regular": -0.2, "bargain": 0.35}

# GCC retail calendar (approximate dates).
EVENTS = [
    ("Ramadan", "2024-03-11", "2024-04-09", 1.45), ("Eid al-Fitr", "2024-04-01", "2024-04-12", 1.9),
    ("Eid al-Adha", "2024-06-08", "2024-06-18", 1.6), ("Back to School", "2024-08-15", "2024-09-05", 1.35),
    ("White Friday", "2024-11-22", "2024-12-02", 2.4), ("DSF", "2024-12-13", "2025-01-12", 1.3),
    ("Ramadan", "2025-03-01", "2025-03-29", 1.45), ("Eid al-Fitr", "2025-03-21", "2025-04-01", 1.9),
    ("Eid al-Adha", "2025-05-29", "2025-06-08", 1.6), ("Back to School", "2025-08-15", "2025-09-05", 1.35),
    ("White Friday", "2025-11-21", "2025-12-01", 2.4), ("DSF", "2025-12-05", "2026-01-11", 1.3),
    ("Ramadan", "2026-02-18", "2026-03-19", 1.45), ("Eid al-Fitr", "2026-03-11", "2026-03-22", 1.9),
    ("Eid al-Adha", "2026-05-19", "2026-05-29", 1.6), ("Back to School", "2026-08-15", "2026-09-05", 1.35),
    ("White Friday", "2026-11-20", "2026-11-30", 2.4), ("DSF", "2026-12-04", "2027-01-10", 1.3),
]
_EV = [(n, pd.Timestamp(a), pd.Timestamp(b), m) for n, a, b, m in EVENTS]


def season_weight(d: pd.Timestamp) -> float:
    """Demand multiplier for a date: events lift it, summer dips it, slow growth over time."""
    w = 1.0
    for _, a, b, m in _EV:
        if a <= d <= b:
            w = max(w, m)
    if d.month in (7,) and w == 1.0:
        w = 0.78
    growth = 1.0 + 0.10 * ((d - START).days / 365.0)
    return w * growth


def event_name(d: pd.Timestamp) -> str:
    for n, a, b, _ in _EV:
        if a <= d <= b:
            return n
    return ""


# ---------------------------------------------------------------- reviews ---
POS = ["love it", "perfect fit", "great quality", "fast delivery", "exactly like the pics",
       "will order again", "amazing price", "super comfy", "my daughter loved it", "very elegant abaya",
       "packaging was beautiful", "true to size", "best app for fashion in uae", "colour is gorgeous"]
NEU = ["it's ok", "average quality", "arrived on time", "as expected", "fine for the price",
       "nothing special", "size slightly big", "decent", "ok but took a while"]
NEG = ["late delivery", "wrong size sent", "poor quality", "return took forever", "colour faded after 1 wash",
       "not like the picture", "refund still pending", "courier never called", "cheap material",
       "item damaged", "customer service didn't reply", "cod order cancelled without reason"]
AR_POS = ["ممتاز", "جودة رائعة", "توصيل سريع", "يجنن", "انصح فيه"]
AR_NEU = ["عادي", "مقبول"]
AR_NEG = ["سيء", "التوصيل متأخر", "المقاس غلط", "ما انصح"]
EMO_POS = ["😍", "🔥", "👌", "💯", "❤️", "✨"]
EMO_NEG = ["😡", "👎", "😤", "🙄", "💔"]
SPAM = ["click here for free coupons!!! www.fake-deals.xyz", "BUY FOLLOWERS cheap", "asdfgh", ".", "??"]


def _typo(s: str, rng) -> str:
    if len(s) < 5 or rng.random() > 0.35:
        return s
    i = rng.integers(1, len(s) - 2)
    op = rng.integers(0, 3)
    if op == 0:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == 1:
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def make_review(sentiment: str, rng) -> str:
    """Build a noisy review text for a true sentiment ('pos' / 'neu' / 'neg')."""
    r = rng.random()
    if r < 0.03:
        return ""                                  # blank review
    if r < 0.05:
        return str(rng.choice(SPAM))               # spam / junk
    bank = {"pos": (POS, AR_POS, EMO_POS), "neu": (NEU, AR_NEU, []), "neg": (NEG, AR_NEG, EMO_NEG)}[sentiment]
    en, ar, emo = bank
    parts = list(rng.choice(en, size=rng.integers(1, 3), replace=False))
    if rng.random() < 0.18:
        parts.append(str(rng.choice(ar)))          # Arabic / mixed language
    if sentiment == "neg" and rng.random() < 0.25:
        parts.insert(0, "nice design but")         # mixed signal
    if sentiment == "pos" and rng.random() < 0.15:
        parts.append("delivery a bit slow tho")
    text = ", ".join(_typo(p, rng) for p in parts)
    if emo and rng.random() < 0.4:
        text += " " + "".join(rng.choice(emo, size=rng.integers(1, 3)))
    if rng.random() < 0.15:
        text = text.upper()
    if rng.random() < 0.2:
        text += "!" * int(rng.integers(1, 4))
    return text


def rating_for(sentiment: str, rng) -> int:
    """Star rating, with noise (some people give 5 stars to a complaint and vice versa)."""
    if rng.random() < 0.07:
        return int(rng.integers(1, 6))
    return int({"pos": rng.choice([4, 5, 5]), "neu": rng.choice([3, 3, 4, 2]), "neg": rng.choice([1, 1, 2])}[sentiment])


# ------------------------------------------------------------ one order -----
def draw_order(cust: dict, date: pd.Timestamp, rng) -> dict:
    """Simulate one order for a customer on a date (shared by history and live stream)."""
    ev = event_name(date)
    w = np.array([0.24, 0.17, 0.15, 0.12, 0.08, 0.10, 0.08, 0.06])
    if cust["gender"] == "M":
        w = np.array([0.06, 0.38, 0.20, 0.06, 0.02, 0.10, 0.13, 0.05])
    if ev in ("Ramadan", "Eid al-Fitr", "Eid al-Adha"):
        w[4] *= 3.0
        w[5] *= 1.6
    if ev == "Back to School":
        w[5] *= 2.5
    w = w / w.sum()
    cat = str(rng.choice(CATEGORIES, p=w))
    items = int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
    t = cust["ctype"]
    disc_p = {"premium": 0.18, "regular": 0.45, "bargain": 0.82}[t]
    if ev == "White Friday":
        disc_p = min(0.97, disc_p + 0.35)
    discount = float(rng.choice([0.1, 0.15, 0.2, 0.3, 0.5])) if rng.random() < disc_p else 0.0
    gross = TYPE_AOV[t] * CAT_AOV[cat] * rng.lognormal(0, 0.35) * (0.75 + 0.25 * items)
    gross = round(gross, 2)
    net = round(gross * (1 - discount), 2)
    payment = "COD" if rng.random() < COUNTRY_COD[cust["country"]] + (0.12 if t == "bargain" else 0) else \
        str(rng.choice(["Card", "Card", "Apple Pay", "Tabby"]))
    peak = season_weight(date) > 1.5
    delivery_days = max(1, int(round(rng.gamma(4, (COUNTRY_DELIVERY[cust["country"]] + (1.4 if peak else 0)) / 4))))
    promised = 2 if cust["country"] == "UAE" else 4
    late = int(delivery_days > promised + 1)
    returned = int(rng.random() < CAT_RETURN[cat] * (1.35 if t == "bargain" else 1.0) * (1.5 if late else 1.0))
    cancelled = int(payment == "COD" and rng.random() < 0.09)
    # sentiment of the experience
    s = 1.2 - 1.6 * late - 1.3 * returned - 2.0 * cancelled + rng.normal(0, 0.9)
    sentiment = "pos" if s > 0.6 else ("neu" if s > -0.4 else "neg")
    has_review = rng.random() < 0.32
    return dict(order_date=date, category=cat, items=items, gross_value=gross, discount_pct=discount,
                net_revenue=0.0 if cancelled else net, payment=payment, delivery_days=delivery_days,
                late_delivery=late, returned=returned, cancelled=cancelled, event=ev,
                review_rating=rating_for(sentiment, rng) if has_review else np.nan,
                review_text=make_review(sentiment, rng) if has_review else None,
                review_sentiment=sentiment if has_review else None, _sent=s)


def churn_prob(cust: dict, o: dict, n_orders: int) -> float:
    z = (-2.9 + TYPE_CHURN[cust["ctype"]] + 1.05 * o["late_delivery"] + 0.9 * o["returned"]
         + 1.4 * o["cancelled"] + 0.45 * (o["_sent"] < -0.4) - 0.32 * np.log1p(n_orders)
         + 0.35 * (o["payment"] == "COD") - 0.45 * cust["app_user"])
    return 1 / (1 + np.exp(-z))


def new_customer(cid: int, signup: pd.Timestamp, rng) -> dict:
    country = str(rng.choice(COUNTRIES, p=COUNTRY_P))
    return dict(customer_id=cid, signup_date=signup, country=country,
                gender=str(rng.choice(["F", "M"], p=[0.68, 0.32])),
                age=int(np.clip(rng.normal(31, 8), 18, 65)),
                channel=str(rng.choice(CHANNELS, p=CHANNEL_P)),
                app_user=int(rng.random() < 0.62),
                ctype=str(rng.choice(TYPES, p=TYPE_P)))


# ------------------------------------------------ campaign ground truth ------
OFFERS = ["No Offer", "Free Delivery", "20% Discount"]
DELIVERY_COST = 18.0   # AED Namshi pays per order when it waives delivery


def campaign_probs(df: pd.DataFrame) -> pd.DataFrame:
    """True 14-day purchase probability under each offer (heterogeneous treatment effects).

    df needs: ctype, recency, frequency, country, payment_cod_share, app_user.
    """
    rec = df["recency"].clip(0, 400).to_numpy()
    freq = np.log1p(df["frequency"].to_numpy())
    t = df["ctype"].to_numpy()
    base_z = -2.5 + 0.55 * freq - 0.006 * rec + 0.3 * df["app_user"].to_numpy() \
        + np.where(t == "premium", 0.6, np.where(t == "bargain", -0.3, 0.0))
    far = df["country"].isin(["KSA", "Oman", "Kuwait", "Bahrain"]).to_numpy().astype(float)
    cod = df["payment_cod_share"].to_numpy()
    # discount: bargain hunters & lapsed customers react; frequent premium buyers don't need it
    # and some are put off by "cheap" messaging (sleeping dogs).
    disc_eff = np.where(t == "bargain", 1.05, np.where(t == "regular", 0.55, -0.35)) + 0.002 * rec
    # free delivery: matters for far-away & COD customers
    fd_eff = 0.15 + 0.75 * far + 0.6 * cod + np.where(t == "premium", -0.2, 0.0)
    sig = lambda z: 1 / (1 + np.exp(-z))
    return pd.DataFrame({"No Offer": sig(base_z), "Free Delivery": sig(base_z + fd_eff),
                         "20% Discount": sig(base_z + disc_eff)}, index=df.index)


def campaign_outcome(df: pd.DataFrame, offer: pd.Series, rng) -> pd.DataFrame:
    """Draw purchase, revenue and net margin contribution for an assigned offer per customer."""
    p = campaign_probs(df)
    pr = np.array([p.loc[i, o] for i, o in zip(df.index, offer)])
    conv = (rng.random(len(df)) < pr).astype(int)
    aov = df["avg_order_value"].to_numpy() * rng.lognormal(0, 0.25, len(df))
    disc = np.where(offer.to_numpy() == "20% Discount", 0.20, 0.0)
    revenue = conv * aov * (1 - disc)
    deliv = np.where(offer.to_numpy() == "Free Delivery", DELIVERY_COST, 0.0) * conv
    gross_margin = 0.45  # fashion gross margin on full price
    profit = conv * aov * gross_margin - conv * aov * disc - deliv
    return pd.DataFrame({"converted": conv, "revenue": revenue.round(2), "profit": profit.round(2)}, index=df.index)
