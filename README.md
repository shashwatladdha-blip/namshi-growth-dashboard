# 🛍️ Namshi Growth Command Centre

A live Streamlit dashboard applying the nine *Data Driven Growth with Python* techniques to **Namshi.com**
(GCC online fashion retailer).

> ⚠️ **All data is synthetic.** It mimics a GCC fashion e-commerce business (countries, Ramadan / Eid /
> White Friday seasons, COD refusals, returns, noisy multilingual reviews), but no real Namshi data is used.

## What's inside

| Tab | Blog | Technique |
|---|---|---|
| 🔴 Live | – | Simulated live stream of orders and noisy reviews, AI sentiment, alerts, rescue list, scenario injection |
| 📊 Metrics | 1 | North Star Metric, active customers, AOV, retention, cohorts |
| 👥 Segments & CLV | 2, 3 | RFM + k-means (elbow), CLV groups with tuned XGBoost |
| ⚠️ Churn & Next Buy | 4, 5 | Churn XGBoost (tuned), decile lift, value × risk matrix, next-purchase-day model comparison |
| 📈 Sales Forecast | 6 | LSTM (PyTorch) with hyperparameter search vs current buying method |
| 🎯 Promotions & Uplift | 7, 8 | Market response model, conversion/order uplift, uplift score, cost-optimised offers |
| 🧪 A/B Test | 9 | Stratified A/B test, Welch t-test, bootstrap |
| 💬 Reviews | – | TF-IDF + logistic regression sentiment on noisy text |
| 🚀 Impact | – | Predicted improvement for the three business problems |

## Files

- `namshi_sim.py` – the simulator (ground truth for history, live stream and A/B test)
- `build_data.py` – writes `data/` (customers, orders, reviews, Sep-2026 campaign)
- `analysis.py` – runs all nine analyses with hyperparameter tuning, writes `outputs/`
- `app.py` – the dashboard (reads `data/` and `outputs/`; needs no XGBoost or PyTorch)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the data and models (needs `xgboost` and `torch` too):

```bash
pip install xgboost torch
python build_data.py && python analysis.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Choose this repo, branch `main`, main file `app.py`, then **Deploy**.
