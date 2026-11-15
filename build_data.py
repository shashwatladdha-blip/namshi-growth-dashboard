"""Build the synthetic Namshi dataset (Jan 2024 - Oct 2026). Run once: python build_data.py"""
import os
import numpy as np
import pandas as pd
from namshi_sim import (START, END, new_customer, draw_order, churn_prob, season_weight,
                        TYPE_GAP, OFFERS, campaign_outcome)

rng = np.random.default_rng(42)
N_CUST = 22000
os.makedirs("data", exist_ok=True)

# signup dates: 22% are existing customers at Jan 2024, the rest arrive with growing intensity
days = (END - START).days
existing = int(N_CUST * 0.22)
u = rng.random(N_CUST - existing)
new_days = (days * u ** 0.8).astype(int)          # more signups later (growth)
signup_offsets = np.concatenate([rng.integers(0, 45, existing), new_days])
max_w = 2.4 * 1.3

customers, orders = [], []
oid = 100000
for cid, off in enumerate(signup_offsets, start=1):
    c = new_customer(cid, START + pd.Timedelta(days=int(off)), rng)
    if cid <= existing:
        c["signup_date"] = START - pd.Timedelta(days=int(rng.integers(30, 1500)))
    customers.append(c)
    d = START + pd.Timedelta(days=int(off))
    n = 0
    gap_mean = TYPE_GAP[c["ctype"]] * rng.lognormal(0, 0.3)
    while d <= END:
        o = draw_order(c, d, rng)
        n += 1
        oid += 1
        o.update(order_id=oid, customer_id=cid, country=c["country"])
        orders.append(o)
        if rng.random() < churn_prob(c, o, n):
            break
        # next order: gamma gap, thinned by seasonality so peaks attract orders
        while True:
            d = d + pd.Timedelta(days=max(1, int(rng.gamma(1.6, gap_mean / 1.6 / 1.8))))
            if d > END or rng.random() < season_weight(d) / max_w:
                break

cust = pd.DataFrame(customers)
ords = pd.DataFrame(orders).drop(columns=["_sent"])
ords["order_date"] = pd.to_datetime(ords["order_date"])
# a little dirty data, like real exports
ords.loc[ords.sample(frac=0.004, random_state=1).index, "category"] = None
ords.loc[ords.sample(frac=0.002, random_state=2).index, "delivery_days"] = np.nan

truth = cust[["customer_id", "ctype"]]
cust.drop(columns=["ctype"]).to_csv("data/customers.csv", index=False)
truth.to_csv("data/_hidden_customer_types.csv", index=False)
ords.to_csv("data/orders.csv", index=False)

reviews = ords[ords["review_text"].notna()][["order_id", "customer_id", "order_date", "country", "category",
                                             "review_rating", "review_text", "review_sentiment"]]
reviews.to_csv("data/reviews.csv", index=False)

# ---------------- Sep-2026 promotion experiment (Blogs 7, 8) ----------------
cut = pd.Timestamp("2026-09-01")
hist = ords[(ords["order_date"] < cut) & (ords["cancelled"] == 0)]
f = hist.groupby("customer_id").agg(recency=("order_date", lambda s: (cut - s.max()).days),
                                    frequency=("order_id", "count"),
                                    avg_order_value=("net_revenue", "mean"),
                                    total_revenue=("net_revenue", "sum"),
                                    avg_discount=("discount_pct", "mean"),
                                    return_rate=("returned", "mean"),
                                    late_rate=("late_delivery", "mean"),
                                    payment_cod_share=("payment", lambda s: (s == "COD").mean()))
f = f.join(cust.set_index("customer_id")[["country", "gender", "age", "channel", "app_user", "ctype"]])
f = f[f["recency"] <= 540].copy()
f["offer"] = rng.choice(OFFERS, size=len(f))
out = campaign_outcome(f, f["offer"], rng)
camp = f.join(out).reset_index()
camp.drop(columns=["ctype"]).to_csv("data/campaign_sep2026.csv", index=False)

print(f"customers {len(cust):,} | orders {len(ords):,} | reviews {len(reviews):,} | campaign {len(camp):,}")
print(ords.groupby(ords.order_date.dt.year)["net_revenue"].sum().round(0))
print(camp.groupby("offer")["converted"].mean().round(4))
