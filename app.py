"""
Namshi Growth Command Centre - Streamlit dashboard.
Run locally:  streamlit run app.py
All data is SYNTHETIC (see README).
"""
import json
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from namshi_sim import draw_order, season_weight, event_name, COUNTRIES

st.set_page_config(page_title="Namshi Growth Command Centre", page_icon="🛍️", layout="wide")

# ------------------------------------------------------------------ theme ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEG_COL = {"High-Value": SERIES[0], "Mid-Value": SERIES[1], "Low-Value": SERIES[2]}
SENT_COL = {"pos": GOOD, "neu": "#9a9890", "neg": CRIT}

tpl = go.layout.Template(pio.templates["plotly_white"])
tpl.layout.colorway = SERIES
tpl.layout.font = dict(family="Inter, Segoe UI, sans-serif", size=13, color="#2b2a27")
tpl.layout.xaxis.gridcolor = tpl.layout.yaxis.gridcolor = "#ecebe7"
tpl.layout.hoverlabel = dict(bgcolor="white", font_size=12)
tpl.layout.margin = dict(l=10, r=10, t=50, b=10)
tpl.layout.legend = dict(orientation="h", yanchor="top", y=-0.14, x=0, title=dict(text=""))
tpl.data.bar = [go.Bar(marker_line_width=0)]
pio.templates["namshi"] = tpl
pio.templates.default = "namshi"

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.hero {background: linear-gradient(110deg,#6250d6 0%,#2a78d6 35%,#1baf7a 70%,#eda100 100%);
       border-radius: 18px; padding: 22px 28px; color: white; margin-bottom: 14px;}
.hero h1 {color: white; margin: 0; font-size: 2.0rem; letter-spacing: -0.5px;}
.hero p {margin: 4px 0 0 0; opacity: .92; font-size: 1.0rem;}
.pill {display:inline-block; background: rgba(255,255,255,.22); padding: 3px 10px; border-radius: 999px;
       font-size: .8rem; margin-right: 6px; margin-top: 8px;}
.kpi {border-radius: 14px; padding: 14px 16px; background: #fff; border: 1px solid #ecebe7;
      border-top: 5px solid var(--c); box-shadow: 0 1px 3px rgba(0,0,0,.04); height: 100%;}
.kpi .l {font-size: .78rem; color: #52514e; text-transform: uppercase; letter-spacing: .4px;}
.kpi .v {font-size: 1.6rem; font-weight: 700; color: #0b0b0b; line-height: 1.25;}
.kpi .d {font-size: .82rem; color: #52514e;}
.feed {font-size: .88rem; padding: 7px 10px; border-radius: 10px; margin-bottom: 6px; background: #fafaf8;
       border-left: 5px solid var(--c);}
.formula {background:#f4f2ff; border-left: 5px solid #6250d6; padding: 10px 14px; border-radius: 10px;
          font-family: ui-monospace, Menlo, monospace; font-size: .9rem; margin: 6px 0;}
.alert {padding: 10px 14px; border-radius: 10px; margin-bottom: 8px; font-size: .92rem;}
@media (prefers-color-scheme: dark) {
  .kpi {background:#1f1f1e; border-color:#383835;} .kpi .v{color:#f0efec;} .kpi .l,.kpi .d{color:#c3c2b7;}
  .feed {background:#232322; color:#f0efec;} .formula{background:#26233a; color:#f0efec;}
}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, delta="", color=SERIES[0]):
    col.markdown(f"<div class='kpi' style='--c:{color}'><div class='l'>{label}</div>"
                 f"<div class='v'>{value}</div><div class='d'>{delta}</div></div>", unsafe_allow_html=True)


def aed(x):
    return f"AED {x/1e6:,.2f}M" if abs(x) >= 1e6 else (f"AED {x/1e3:,.1f}K" if abs(x) >= 1e4 else f"AED {x:,.0f}")


# ------------------------------------------------------------------- data ---
@st.cache_data
def load():
    R = json.load(open("outputs/results.json"))
    m = pd.read_csv("outputs/monthly_metrics.csv", parse_dates=["month"])
    coh = pd.read_csv("outputs/cohort_retention.csv", index_col=0)
    sc = pd.read_csv("outputs/customer_scores.csv")
    wk = pd.read_csv("outputs/weekly_sales.csv", parse_dates=["week"])
    fct = pd.read_csv("outputs/forecast_test.csv", parse_dates=["week"])
    fcf = pd.read_csv("outputs/forecast_future.csv", parse_dates=["week"])
    camp = pd.read_csv("outputs/campaign_scored.csv")
    ab = pd.read_csv("outputs/ab_test.csv")
    cust = pd.read_csv("data/customers.csv").merge(pd.read_csv("data/_hidden_customer_types.csv"), on="customer_id")
    rv = pd.read_csv("data/reviews.csv", parse_dates=["order_date"])
    return R, m, coh, sc, wk, fct, fcf, camp, ab, cust, rv


R, M, COH, SC, WK, FCT, FCF, CAMP, AB, CUST, RV = load()


@st.cache_resource
def sentiment_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    bp = R["reviews"]["best_params"]
    C = float(bp.get("clf__C", 1))
    ng = tuple(int(x) for x in bp.get("feats__c__ngram_range", "(2, 4)").strip("()").split(","))
    d = RV.sample(min(30000, len(RV)), random_state=1)
    pipe = Pipeline([("feats", FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                                             ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=ng, min_df=3))])),
                     ("clf", LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))])
    pipe.fit(d["review_text"].fillna(""), d["review_sentiment"])
    return pipe


SENT = sentiment_model()
SC_IDX = SC.set_index("customer_id")
CUST_REC = CUST.set_index("customer_id")

# ---------------------------------------------------------------- header ----
now = dt.datetime.now(ZoneInfo("Asia/Dubai"))
ev_today = event_name(pd.Timestamp(now.date()))
st.markdown(f"""
<div class='hero'><h1>🛍️ Namshi Growth Command Centre</h1>
<p>Data-driven growth: 9 techniques, one dashboard, live order and review stream.</p>
<span class='pill'>⚠️ Synthetic data, 22,000-customer sample</span>
<span class='pill'>📅 History Jan 2024 to Oct 2026</span>
<span class='pill'>🕒 Dubai {now:%a %d %b %Y, %H:%M}</span>
{f"<span class='pill'>🎉 Season: {ev_today}</span>" if ev_today else ""}
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ Live simulator")
    running = st.toggle("Stream live data", value=True)
    speed = st.slider("Orders per refresh (speed)", 1, 30, 8)
    refresh = st.select_slider("Refresh every", ["2s", "3s", "5s", "10s"], value="3s")
    st.subheader("🌪️ Inject a scenario")
    surge = st.toggle("White Friday surge (×3 traffic)")
    strike = st.toggle("Courier disruption (late deliveries)")
    bad_batch = st.toggle("Bad product batch (quality complaints)")
    if st.button("🔄 Reset live feed"):
        for k in ("live", "tick"):
            st.session_state.pop(k, None)
    st.caption("The stream is generated by the same simulator that built the history, so it behaves like the "
               "real data: seasonal demand, COD refusals, late deliveries, returns and messy reviews.")

tabs = st.tabs(["🔴 Live", "📊 Metrics", "👥 Segments & CLV", "⚠️ Churn & Next Buy", "📈 Sales Forecast",
                "🎯 Promotions & Uplift", "🧪 A/B Test", "💬 Reviews", "🚀 Impact"])

# ============================================================== LIVE TAB ===
CUST_IDS = CUST["customer_id"].to_numpy()


def gen_events(n, rng, when):
    rows = []
    for _ in range(n):
        cid = int(rng.choice(CUST_IDS))
        c = CUST_REC.loc[cid].to_dict()
        c["customer_id"] = cid
        if strike:
            c["country"] = c["country"]
        o = draw_order(c, pd.Timestamp(when.date()), rng)
        if strike and rng.random() < 0.55:
            o["delivery_days"] += int(rng.integers(3, 7))
            o["late_delivery"] = 1
        if (strike or bad_batch) and o["review_text"] is not None and rng.random() < 0.6:
            from namshi_sim import make_review, rating_for
            o["review_sentiment"] = "neg"
            o["review_text"] = make_review("neg", rng) if not bad_batch else \
                str(rng.choice(["poor quality", "cheap material", "colour faded after 1 wash", "item damaged",
                                "stitching came off 😡", "smells weird, returning"]))
            o["review_rating"] = rating_for("neg", rng)
        o.update(customer_id=cid, country=c["country"], ts=when)
        rows.append(o)
    return rows


def tick():
    rng = np.random.default_rng()
    t = dt.datetime.now(ZoneInfo("Asia/Dubai"))
    hour_f = 0.35 + 0.65 * np.exp(-((t.hour - 21) % 24) ** 2 / 30)   # evening peak
    lam = speed * hour_f * season_weight(pd.Timestamp(t.date())) * (3 if surge else 1)
    n = int(rng.poisson(max(lam, 0.5)))
    ev = gen_events(n, rng, t)
    for e in ev:
        txt = e.get("review_text")
        e["pred_sentiment"] = SENT.predict([txt])[0] if txt is not None else None
    return ev


if "live" not in st.session_state:
    rng0 = np.random.default_rng()
    base_t = dt.datetime.now(ZoneInfo("Asia/Dubai"))
    hist = []
    for k in range(60, 0, -1):     # backfill the last hour so the screen is never empty
        when = base_t - dt.timedelta(minutes=k)
        hist += gen_events(int(rng0.poisson(speed * 0.6)), rng0, when)
    for e in hist:
        e["pred_sentiment"] = SENT.predict([e["review_text"]])[0] if e.get("review_text") is not None else None
    st.session_state["live"] = hist
    st.session_state["tick"] = 0


@st.fragment(run_every=refresh if running else None)
def live_panel():
    if running:
        st.session_state["live"] = (st.session_state["live"] + tick())[-4000:]
        st.session_state["tick"] += 1
    L = pd.DataFrame(st.session_state["live"])
    L["ts"] = pd.to_datetime(L["ts"])
    L = L.join(SC_IDX[["Segment", "value_tier", "churn_prob", "churn_risk"]], on="customer_id")
    rev = L["net_revenue"].sum()
    n_orders = len(L)
    reviewed = L[L["review_text"].notna()]
    neg_share = (reviewed["pred_sentiment"] == "neg").mean() if len(reviewed) else 0
    late = L["late_delivery"].mean()
    cod_ref = L["cancelled"].mean()
    c = st.columns(6)
    kpi(c[0], "Live revenue", aed(rev), f"{n_orders:,} orders in feed", SERIES[0])
    kpi(c[1], "Avg order value", aed(rev / max(n_orders - L.cancelled.sum(), 1)), "net of discounts", SERIES[1])
    kpi(c[2], "Avg rating", f"{reviewed['review_rating'].mean():.2f} ★" if len(reviewed) else "–",
        f"{len(reviewed):,} reviews", SERIES[3])
    kpi(c[3], "Negative reviews", f"{neg_share:.0%}", "AI sentiment, whole feed", CRIT if neg_share > .3 else GOOD)
    kpi(c[4], "Late deliveries", f"{late:.0%}", "vs promise, whole feed", CRIT if late > .3 else (WARN if late > .2 else GOOD))
    kpi(c[5], "COD refusals", f"{cod_ref:.1%}", "orders cancelled", SERIOUS if cod_ref > .05 else GOOD)

    # alerts look at the last 5 minutes only, so a new problem shows up fast
    st.write("")
    recent = L[L["ts"] >= L["ts"].max() - pd.Timedelta(minutes=5)]
    rrev = recent[recent["review_text"].notna()]
    late = recent["late_delivery"].mean()
    neg_share = (rrev["pred_sentiment"] == "neg").mean() if len(rrev) else 0
    alerts = []
    if late > .3:
        alerts.append((CRIT, "🚚 Critical: late deliveries above 30%. Churn risk rises (top churn driver in Blog 4 model)."))
    if neg_share > .3:
        alerts.append((CRIT, "💬 Critical: negative review share above 30%. Check the complaint themes below."))
    if surge:
        alerts.append((WARN, "🔥 Warning: traffic surge. Check the Forecast tab for stock cover on fast movers."))
    rescue = recent[(recent["pred_sentiment"] == "neg") & (recent["value_tier"].isin(["High value", "Mid value"]))]
    if len(rescue):
        alerts.append((SERIOUS, f"🛟 Serious: {rescue.customer_id.nunique()} valuable customer(s) just left a negative review. "
                                f"Rescue list below."))
    if not alerts:
        alerts.append((GOOD, "✅ Good: all live signals within normal range."))
    for colr, msg in alerts:
        st.markdown(f"<div class='alert' style='background:{colr}22;border-left:5px solid {colr}'>{msg}</div>",
                    unsafe_allow_html=True)

    a, b = st.columns([3, 2])
    per_min = L.set_index("ts").resample("1min")["net_revenue"].sum().reset_index()
    fig = px.area(per_min, x="ts", y="net_revenue", title="Revenue per minute (live)")
    fig.update_traces(line=dict(width=2, color=SERIES[0]), fillcolor="rgba(42,120,214,.15)")
    fig.update_layout(height=320, xaxis_title=None, yaxis_title="AED")
    a.plotly_chart(fig, width="stretch", key="live_rev")
    by_c = L.groupby("country")["net_revenue"].sum().reindex(COUNTRIES).fillna(0).reset_index()
    fig = px.bar(by_c, x="net_revenue", y="country", orientation="h", title="Live revenue by country",
                 color_discrete_sequence=[SERIES[6]])
    fig.update_layout(height=320, yaxis=dict(autorange="reversed", title=None), xaxis_title="AED")
    b.plotly_chart(fig, width="stretch", key="live_cty")

    a, b, d = st.columns([2, 2, 3])
    cat = L.groupby("category")["net_revenue"].sum().sort_values().reset_index()
    fig = px.bar(cat, x="net_revenue", y="category", orientation="h", title="Hot categories right now",
                 color_discrete_sequence=[SERIES[1]])
    fig.update_layout(height=340, yaxis_title=None, xaxis_title="AED")
    a.plotly_chart(fig, width="stretch", key="live_cat")
    seg = L.groupby("Segment")["net_revenue"].sum().reindex(["High-Value", "Mid-Value", "Low-Value"]).fillna(0).reset_index()
    fig = px.bar(seg, x="Segment", y="net_revenue", color="Segment", color_discrete_map=SEG_COL,
                 title="Who is buying (RFM segment)")
    fig.update_layout(height=340, showlegend=False, xaxis_title=None, yaxis_title="AED")
    b.plotly_chart(fig, width="stretch", key="live_seg")
    with d:
        st.markdown("**🗨️ Live review feed** (raw, noisy text → AI sentiment)")
        feed = reviewed.sort_values("ts", ascending=False).head(9)
        for _, r in feed.iterrows():
            colr = SENT_COL.get(r["pred_sentiment"], "#999")
            txt = r["review_text"] if r["review_text"] else "<i>(blank review)</i>"
            stars = "★" * int(r["review_rating"]) if pd.notna(r["review_rating"]) else ""
            st.markdown(f"<div class='feed' style='--c:{colr}'><b>{r['pred_sentiment'].upper()}</b> · {stars} · "
                        f"{r['category']} · {r['country']}<br>{txt}</div>", unsafe_allow_html=True)
    if len(rescue):
        st.markdown("**🛟 Rescue list: valuable customers with a negative experience just now**")
        show = rescue.sort_values("churn_prob", ascending=False)[
            ["ts", "customer_id", "Segment", "value_tier", "churn_prob", "category", "late_delivery", "review_text"]].head(10)
        show["churn_prob"] = (show["churn_prob"] * 100).round(0).astype(int).astype(str) + "%"
        show["ts"] = show["ts"].dt.strftime("%H:%M:%S")
        st.dataframe(show, hide_index=True, width="stretch")
    st.caption(f"Refresh #{st.session_state['tick']} · {dt.datetime.now(ZoneInfo('Asia/Dubai')):%H:%M:%S} Dubai · "
               "synthetic live stream")


with tabs[0]:
    live_panel()

# ============================================================ METRICS TAB ===
with tabs[1]:
    b1 = R["blog1"]
    c = st.columns(5)
    prev = M[M.month == "2025-10-01"].iloc[0]
    kpi(c[0], "⭐ North Star · Oct revenue", aed(b1["oct_2026_revenue"]),
        f"{b1['oct_2026_revenue']/prev.revenue-1:+.0%} vs Oct 2025", SERIES[0])
    kpi(c[1], "Active customers", f"{b1['oct_2026_active']:,}", f"{b1['oct_2026_active']/prev.active_customers-1:+.0%} YoY", SERIES[1])
    kpi(c[2], "Orders", f"{b1['oct_2026_orders']:,}", "October 2026", SERIES[2])
    kpi(c[3], "Avg order value", aed(b1["oct_2026_aov"]), "October 2026", SERIES[3])
    kpi(c[4], "Monthly retention", f"{b1['avg_retention_2026']:.0%}", f"2026 avg (2024: {b1['avg_retention_2024']:.0%})", SERIES[6])
    st.markdown("<div class='formula'>Revenue = Active Customers × Orders per Customer × Average Order Value</div>",
                unsafe_allow_html=True)
    a, b = st.columns(2)
    nm = M.melt("month", ["new_customer_revenue", "existing_customer_revenue"], var_name="type", value_name="AED")
    nm["type"] = nm["type"].map({"new_customer_revenue": "New customers", "existing_customer_revenue": "Existing customers"})
    fig = px.bar(nm, x="month", y="AED", color="type", title="Monthly revenue: new vs existing customers",
                 color_discrete_sequence=[SERIES[1], SERIES[0]])
    fig.update_layout(barmode="stack", height=380, xaxis_title=None, bargap=0.15)
    a.plotly_chart(fig, width="stretch")
    fig = px.line(M, x="month", y="active_customers", title="Monthly active customers", markers=True,
                  color_discrete_sequence=[SERIES[2]])
    fig.update_layout(height=380, xaxis_title=None, yaxis_title="Customers")
    b.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    fig = px.line(M.dropna(subset=["retention_rate"]), x="month", y="retention_rate", markers=True,
                  title="Monthly retention rate", color_discrete_sequence=[SERIES[6]])
    fig.update_layout(height=360, yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    fig = px.line(M, x="month", y="aov", markers=True, title="Average order value (AED)", color_discrete_sequence=[SERIES[3]])
    fig.update_layout(height=360, xaxis_title=None, yaxis_title="AED")
    b.plotly_chart(fig, width="stretch")
    ch = COH.iloc[-24:, :13]
    fig = px.imshow(ch.values, x=[f"M+{c}" for c in ch.columns], y=ch.index, color_continuous_scale=SEQ,
                    text_auto=".0%", aspect="auto", title="Cohort retention: share of each first-purchase cohort active k months later")
    fig.update_layout(height=620, coloraxis_colorbar=dict(tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")

# =========================================================== SEGMENTS TAB ===
with tabs[2]:
    b2, b3 = R["blog2"], R["blog3"]
    seg = pd.DataFrame(b2["segments"]).set_index("Segment").reindex(["High-Value", "Mid-Value", "Low-Value"])
    c = st.columns(3)
    for col, (s, r) in zip(c, seg.iterrows()):
        kpi(col, s, f"{int(r.customers):,} customers", f"{r.revenue_share:.0%} of revenue · avg {aed(r.avg_revenue)}", SEG_COL[s])
    st.markdown("<div class='formula'>RFM: k-means (k=4, elbow) on Recency, Frequency, Revenue → Overall Score = "
                "R cluster + F cluster + M cluster → 0-2 Low · 3-4 Mid · 5+ High</div>", unsafe_allow_html=True)
    a, b = st.columns([3, 2])
    smp = SC.dropna(subset=["Segment"]).sample(4000, random_state=1)
    fig = px.scatter(smp, x="Frequency", y="Revenue", color="Segment", color_discrete_map=SEG_COL, log_y=True,
                     opacity=.7, title="Frequency vs revenue by segment (4,000-customer sample)",
                     hover_data=["Recency", "country"])
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_layout(height=420)
    a.plotly_chart(fig, width="stretch")
    el = pd.DataFrame(b2["elbow_inertia"])
    el["k"] = range(1, 10)
    el = el.melt("k", var_name="RFM dimension", value_name="inertia")
    el["inertia (indexed, k=1 = 100)"] = el.groupby("RFM dimension")["inertia"].transform(lambda s: 100 * s / s.iloc[0])
    fig = px.line(el, x="k", y="inertia (indexed, k=1 = 100)", color="RFM dimension", markers=True,
                  title="Elbow method: why k = 4")
    fig.add_vline(x=4, line_dash="dot", line_color="#9a9890")
    fig.update_layout(height=420)
    b.plotly_chart(fig, width="stretch")
    st.subheader("Customer Lifetime Value prediction (Blog 3)")
    st.caption(f"Features: {b3['feature_window']} behaviour · Target: revenue in {b3['target_window']} · "
               f"{b3['customers']:,} customers · XGBoost tuned with 30-trial randomized search, 4-fold CV")
    g = pd.DataFrame(b3["ltv_groups"])
    c = st.columns(3)
    for col, (_, r), nm, colr in zip(c, g.iterrows(), ["Low CLV", "Mid CLV", "High CLV"], [SERIES[2], SERIES[1], SERIES[0]]):
        kpi(col, nm, f"avg {aed(r['mean'])}", f"6-month revenue {aed(r['min'])} to {aed(r['max'])} · {int(r['count']):,} cust.", colr)
    a, b = st.columns(2)
    mc = pd.DataFrame({"Model": ["Logistic regression", "XGBoost default", "XGBoost tuned"],
                       "Accuracy": [b3["baseline_logreg"]["accuracy"], b3["xgb_default"]["accuracy"], b3["xgb_tuned"]["accuracy"]],
                       "Macro F1": [b3["baseline_logreg"]["f1_macro"], b3["xgb_default"]["f1_macro"], b3["xgb_tuned"]["f1_macro"]]})
    fig = px.bar(mc.melt("Model", var_name="metric"), x="Model", y="value", color="metric", barmode="group",
                 text_auto=".1%", title="CLV model: test-set scores")
    fig.update_layout(height=380, yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None, bargap=.25)
    a.plotly_chart(fig, width="stretch")
    fi = pd.Series(b3["top_features"]).sort_values()
    fig = px.bar(fi, orientation="h", title="What drives CLV (feature importance)", color_discrete_sequence=[SERIES[6]])
    fig.update_layout(height=380, showlegend=False, xaxis_title="importance", yaxis_title=None)
    b.plotly_chart(fig, width="stretch")

# ============================================================== CHURN TAB ===
with tabs[3]:
    b4, b5 = R["blog4"], R["blog5"]
    t = b4["xgb_tuned_best_threshold"]
    c = st.columns(5)
    kpi(c[0], "Churn rate (6 mo)", f"{b4['churn_rate']:.0%}", f"{b4['customers']:,} active customers", CRIT)
    kpi(c[1], "ROC-AUC", f"{t['roc_auc']:.3f}", f"logistic baseline {b4['baseline_logreg']['roc_auc']:.3f}", SERIES[0])
    kpi(c[2], "Accuracy", f"{t['accuracy']:.1%}", f"threshold {b4['threshold']}", SERIES[2])
    kpi(c[3], "Recall (churners caught)", f"{t['recall']:.1%}", f"precision {t['precision']:.1%}", SERIES[1])
    kpi(c[4], "Top-10% risk group", f"{b4['top_decile_churn_rate']:.0%} churn", f"vs {b4['churn_rate']:.0%} average", SERIES[6])
    a, b = st.columns(2)
    dl = pd.DataFrame({"Risk decile (1 = riskiest)": range(1, 11), "Actual churn rate": b4["decile_churn_rate"]})
    fig = px.bar(dl, x="Risk decile (1 = riskiest)", y="Actual churn rate", title="Churn model lift: actual churn by predicted-risk decile",
                 color_discrete_sequence=[CRIT])
    fig.add_hline(y=b4["churn_rate"], line_dash="dot", annotation_text="average", line_color="#52514e")
    fig.update_layout(height=380, yaxis_tickformat=".0%", bargap=.25)
    a.plotly_chart(fig, width="stretch")
    fi = pd.Series(b4["top_features"]).sort_values()
    fig = px.bar(fi, orientation="h", title="Top churn drivers", color_discrete_sequence=[SERIES[1]])
    fig.update_layout(height=380, showlegend=False, xaxis_title="importance", yaxis_title=None)
    b.plotly_chart(fig, width="stretch")
    st.subheader("Today's action matrix: value × risk")
    W = SC[SC.winnable == 1]
    mat = W.pivot_table(index="value_tier", columns="churn_risk", values="customer_id", aggfunc="count", observed=False)
    mat = mat.reindex(["High value", "Mid value", "Low value"])[["High", "Medium", "Low"]].fillna(0).astype(int)
    fig = px.imshow(mat.values, x=["High risk", "Medium risk", "Low risk"], y=mat.index, text_auto=True,
                    color_continuous_scale=[[0, "#fde9e3"], [1, CRIT]], aspect="auto",
                    title="Customers bought in the last 12 months, by value tier and churn risk (scored 1 Nov 2026)")
    fig.update_layout(height=320, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Value tier = predicted CLV group (Blog 3) for customers who ordered in the last 3 months; "
               "RFM segment (Blog 2) for those who have gone quiet. Lapsed over a year = already lost, excluded.")
    st.markdown("🔴 **High value × High risk:** stylist message + free express delivery · 🟠 **Mid value × High risk:** "
                "personalised picks + points · 🟢 **Low value:** push/email only · ⚪ **Low risk:** no spend")
    st.subheader("Next purchase day (Blog 5)")
    a, b = st.columns([3, 2])
    cmpd = pd.Series(b5["model_comparison_cv_accuracy"]).sort_values().reset_index()
    cmpd.columns = ["Model", "CV accuracy"]
    cmpd["col"] = np.where(cmpd.Model == "XGBoost", "XGBoost", "Other")
    fig = px.bar(cmpd, x="CV accuracy", y="Model", orientation="h", color="col",
                 color_discrete_map={"XGBoost": SERIES[0], "Other": "#b7d3f6"}, text_auto=".1%",
                 title="7 models compared (4-fold CV accuracy)")
    fig.update_layout(height=380, showlegend=False, xaxis_tickformat=".0%", yaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    with b:
        x5 = b5["xgb_tuned_test"]
        kpi(st, "Tuned XGBoost · test accuracy", f"{x5['accuracy']:.1%}",
            f"majority-class guess: {b5['majority_class_baseline']:.1%} · macro F1 {x5['f1_macro']:.2f}", SERIES[0])
        st.write("")
        st.markdown("Classes: **0-20 days**, **21-49 days**, **50+ days** until the next order. "
                    "A customer who passes their predicted window without buying gets an early-warning nudge.")
        st.json(b5["best_params"], expanded=False)

# =========================================================== FORECAST TAB ===
with tabs[4]:
    b6 = R["blog6"]
    c = st.columns(4)
    kpi(c[0], "LSTM forecast accuracy", f"{b6['lstm']['accuracy_1_minus_mape']:.1%}", f"1 − MAPE, last {b6['test_weeks']} weeks", SERIES[0])
    kpi(c[1], "Current method (last year + growth)", f"{b6['last_year_plus_growth']['accuracy_1_minus_mape']:.1%}", "same weeks", SERIES[1])
    kpi(c[2], "Forecast error cut", f"{1 - b6['lstm']['mape'] / b6['last_year_plus_growth']['mape']:.0%}", "MAPE reduction", GOOD)
    kpi(c[3], "Next 9 weeks (forecast)", aed(FCF.forecast.sum()), "Nov-Dec 2026 incl. White Friday", SERIES[6])
    fig = go.Figure()
    hist = WK[WK.week >= "2025-06-01"]
    fig.add_trace(go.Scatter(x=hist.week, y=hist.sales, name="Actual", line=dict(color="#52514e", width=2)))
    fig.add_trace(go.Scatter(x=FCT.week, y=FCT.lstm, name="LSTM (test)", line=dict(color=SERIES[0], width=2.5)))
    fig.add_trace(go.Scatter(x=FCT.week, y=FCT.last_year_method, name="Last year + growth (test)",
                             line=dict(color=SERIES[1], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=FCF.week, y=FCF.forecast, name="LSTM forecast (future)", mode="lines+markers",
                             line=dict(color=SERIES[6], width=2.5), marker=dict(size=8)))
    fig.update_layout(title="Weekly net revenue: actual, back-test and forward forecast", height=460, yaxis_title="AED")
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    errs = pd.DataFrame({"Method": ["LSTM (tuned, 5-seed ensemble)", "Linear regression", "Last year + growth", "Naive (last week)"],
                         "MAPE": [b6["lstm"]["mape"], b6["linear_regression"]["mape"], b6["last_year_plus_growth"]["mape"],
                                  b6["naive_last_week"]["mape"]]}).sort_values("MAPE")
    fig = px.bar(errs, x="MAPE", y="Method", orientation="h", text_auto=".1%", title="Forecast error by method (lower is better)",
                 color="Method", color_discrete_sequence=SERIES)
    fig.update_layout(height=340, showlegend=False, xaxis_tickformat=".0%", yaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**LSTM hyperparameter search** (validation RMSE, best first)")
        st.dataframe(pd.DataFrame(b6["search"]).assign(val_rmse=lambda d: d.val_rmse.round(0).astype(int)), hide_index=True, width="stretch", height=300)
    share = pd.read_csv("outputs/category_share_novdec.csv", index_col=0).iloc[:, 0]
    plan = (share * FCF.forecast.sum()).sort_values().reset_index()
    plan.columns = ["category", "AED"]
    fig = px.bar(plan, x="AED", y="category", orientation="h", title="Buying plan: forecast Nov-Dec demand by category",
                 color_discrete_sequence=[SERIES[2]])
    fig.update_layout(height=360, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

# ========================================================= PROMOTIONS TAB ===
with tabs[5]:
    b7, b8 = R["blog7"], R["blog8"]
    st.markdown("<div class='formula'>Conversion uplift = Conversion(test) − Conversion(control)<br>"
                "Order uplift = Conversion uplift × Customers in test group<br>"
                "Revenue uplift = Order uplift × Average order value<br>"
                "Uplift score = P(TR) + P(CN) − P(TN) − P(CR)</div>", unsafe_allow_html=True)
    c = st.columns(4)
    for col, off, colr in zip(c[:2], ["20% Discount", "Free Delivery"], [SERIES[1], SERIES[2]]):
        r = b7["by_offer"][off]
        kpi(col, f"{off}: conversion uplift", f"+{r['conversion_uplift_pp']:.1f} pp",
            f"{r['conversion_rate']:.1%} vs {r['control_conversion']:.1%} control · +{r['order_uplift']:,} orders", colr)
    d, f = b7["by_offer"]["20% Discount"], b7["by_offer"]["Free Delivery"]
    kpi(c[2], "Profit / customer", f"{aed(d['profit_per_customer'])} vs {aed(f['profit_per_customer'])}",
        f"discount vs free delivery · control {aed(d['control_profit_per_customer'])}", SERIES[0])
    kpi(c[3], "Market response model", f"AUC {b7['model']['test_auc']:.3f}", "tuned XGBoost, holdout 30%", SERIES[6])
    a, b = st.columns(2)
    cs = CAMP.groupby("offer").agg(conversion=("converted", "mean"), profit=("profit", "mean")).reindex(
        ["No Offer", "Free Delivery", "20% Discount"]).reset_index()
    fig = px.bar(cs, x="offer", y="conversion", color="offer", text_auto=".1%", title="Sep-2026 campaign: conversion by offer",
                 color_discrete_sequence=["#9a9890", SERIES[2], SERIES[1]])
    fig.update_layout(height=380, showlegend=False, yaxis_tickformat=".0%", xaxis_title=None, bargap=.3)
    a.plotly_chart(fig, width="stretch")
    us = pd.DataFrame(b7["uplift_pp_by_segment"]).reindex(["High-Value", "Mid-Value", "Low-Value"]).reset_index().melt("index", var_name="offer", value_name="uplift (pp)")
    fig = px.bar(us, x="index", y="uplift (pp)", color="offer", barmode="group", title="Conversion uplift by RFM segment",
                 color_discrete_map={"20% Discount": SERIES[1], "Free Delivery": SERIES[2]}, text_auto=".1f")
    fig.update_layout(height=380, xaxis_title=None, bargap=.25)
    b.plotly_chart(fig, width="stretch")
    st.subheader("Uplift modelling (Blog 8): who should get an offer?")
    a, b = st.columns(2)
    dd = pd.DataFrame({"decile": list(range(1, 11)) * 2,
                       "uplift (pp)": b8["20% Discount"]["actual_uplift_pp_by_decile"] + b8["Free Delivery"]["actual_uplift_pp_by_decile"],
                       "offer": ["20% Discount"] * 10 + ["Free Delivery"] * 10})
    fig = px.bar(dd, x="decile", y="uplift (pp)", color="offer", barmode="group",
                 color_discrete_map={"20% Discount": SERIES[1], "Free Delivery": SERIES[2]},
                 title="Actual uplift by predicted uplift-score decile (holdout, 1 = best)")
    fig.add_hline(y=0, line_color="#52514e")
    fig.update_layout(height=400, bargap=.2)
    a.plotly_chart(fig, width="stretch")
    pm = pd.Series(b8["policy_mix"]).reindex(["No Offer", "Free Delivery", "20% Discount"]).fillna(0).reset_index()
    pm.columns = ["offer", "share"]
    fig = px.bar(pm, x="offer", y="share", color="offer", text_auto=".0%", title="Cost-optimised policy: offer chosen per customer",
                 color_discrete_sequence=["#9a9890", SERIES[2], SERIES[1]])
    fig.update_layout(height=400, showlegend=False, yaxis_tickformat=".0%", xaxis_title=None, bargap=.3)
    b.plotly_chart(fig, width="stretch")
    st.info(f"Top-3 deciles see **+{b8['20% Discount']['top3_deciles_uplift_pp']:.1f} pp** uplift from the discount; "
            f"bottom-3 deciles **{b8['20% Discount']['bottom3_deciles_uplift_pp']:+.1f} pp**. Sending everyone the same code "
            "pays the bottom deciles for sales they would have made anyway.")

# ============================================================= A/B TAB ======
with tabs[6]:
    b9 = R["blog9"]
    pr = b9["profit_per_customer"]
    sig = pr["p_value"] < 0.05
    c = st.columns(4)
    kpi(c[0], "A: blanket 20% discount", aed(pr["A_mean"]), f"profit / customer · n = {b9['n_A']:,}", SERIES[1])
    kpi(c[1], "B: uplift-targeted", aed(pr["B_mean"]), f"profit / customer · n = {b9['n_B']:,}", SERIES[0])
    kpi(c[2], "Lift", f"{pr['lift_pct']:+.0%}", "B vs A", GOOD if pr["lift_pct"] > 0 else CRIT)
    kpi(c[3], "p-value (Welch t-test)", f"{pr['p_value']:.2g}", "✅ significant at 5%" if sig else "❌ not significant",
        GOOD if sig else CRIT)
    st.markdown(f"<div class='formula'>H0: mean profit per customer(B) = mean profit(A) · t = {pr['t_stat']} · "
                f"stratified random split by RFM segment · reject H0 if p &lt; 0.05</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    rows = []
    for k, lab in [("profit_per_customer", "Profit / customer (AED)"), ("revenue_per_customer", "Revenue / customer (AED)")]:
        rows += [{"metric": lab, "group": "A: blanket", "value": b9[k]["A_mean"]},
                 {"metric": lab, "group": "B: targeted", "value": b9[k]["B_mean"]}]
    fig = px.bar(pd.DataFrame(rows), x="metric", y="value", color="group", barmode="group", text_auto=".1f",
                 color_discrete_sequence=[SERIES[1], SERIES[0]], title="A vs B: profit and revenue per customer")
    fig.update_layout(height=400, xaxis_title=None, yaxis_title="AED", bargap=.3)
    a.plotly_chart(fig, width="stretch")
    # bootstrap distribution of the difference
    rng = np.random.default_rng(0)
    A_ = AB[AB.group.str.startswith("A")].profit_ab.to_numpy()
    B_ = AB[AB.group.str.startswith("B")].profit_ab.to_numpy()
    boots = [rng.choice(B_, len(B_)).mean() - rng.choice(A_, len(A_)).mean() for _ in range(1500)]
    fig = px.histogram(x=boots, nbins=50, title="Bootstrap: difference in profit per customer (B − A)",
                       color_discrete_sequence=[SERIES[6]])
    fig.add_vline(x=0, line_color=CRIT, line_width=2)
    fig.update_layout(height=400, xaxis_title="AED", yaxis_title="resamples", bargap=.05)
    b.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(b9["stratification"]), width="stretch")

# ========================================================== REVIEWS TAB =====
with tabs[7]:
    rr = R["reviews"]
    c = st.columns(4)
    kpi(c[0], "Reviews analysed", f"{rr['n_reviews']:,}", "English, Arabic, emoji, typos", SERIES[0])
    kpi(c[1], "Sentiment model", f"{rr['model']['accuracy']:.1%}", f"macro F1 {rr['model']['f1_macro']:.2f}", SERIES[2])
    kpi(c[2], "Star-rating rule", f"{rr['star_rating_rule_baseline']['accuracy']:.1%}", "stars alone mislead", SERIES[1])
    kpi(c[3], "Blank / spam", f"{rr['blank_or_spam_share']:.1%}", "filtered as noise", "#9a9890")
    a, b = st.columns(2)
    cmp = pd.Series(rr["top_complaints_share_of_negative"]).sort_values().reset_index()
    cmp.columns = ["theme", "share"]
    fig = px.bar(cmp, x="share", y="theme", orientation="h", text_auto=".0%", title="Complaint themes (share of negative reviews)",
                 color_discrete_sequence=[CRIT])
    fig.update_layout(height=360, xaxis_tickformat=".0%", yaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    mth = RV.assign(month=RV.order_date.dt.to_period("M").dt.to_timestamp()).groupby(["month", "review_sentiment"]).size() \
        .unstack().pipe(lambda d: d.div(d.sum(axis=1), axis=0)).reset_index().melt("month", var_name="sentiment", value_name="share")
    mth["sentiment"] = mth["sentiment"].map({"pos": "Positive", "neu": "Neutral", "neg": "Negative"})
    fig = px.area(mth, x="month", y="share", color="sentiment", title="Review sentiment mix over time",
                  color_discrete_map={"Positive": GOOD, "Neutral": "#c3c2b7", "Negative": CRIT})
    fig.update_layout(height=360, yaxis_tickformat=".0%", xaxis_title=None)
    b.plotly_chart(fig, width="stretch")
    st.markdown("**🧪 Try the sentiment model on your own review**")
    txt = st.text_input("Type a review (English, Arabic or mixed)", "nice design but courier never called 😡")
    if txt:
        p = SENT.predict_proba([txt])[0]
        cls = SENT.classes_
        st.markdown(" · ".join(f"**{c_}**: {v:.0%}" for c_, v in zip(cls, p)))
    st.markdown("**Sample of raw reviews (noise included)**")
    st.dataframe(RV.sample(12, random_state=int(dt.datetime.now().minute))[["order_date", "country", "category", "review_rating", "review_text", "review_sentiment"]],
                 hide_index=True, width="stretch")

# =========================================================== IMPACT TAB =====
with tabs[8]:
    I = R["improvement"]
    st.subheader("Predicted improvement after running the techniques")
    c = st.columns(3)
    p1, p2, p3 = I["P1_churn"], I["P2_promotions"], I["P3_inventory"]
    kpi(c[0], "Problem 1 · churn", aed(p1["revenue_protected_6m_aed"]),
        f"6-month revenue protected · {p1['targeted_customers']:,} customers targeted · ROI {p1['roi']:.1f}×", SERIES[0])
    kpi(c[1], "Problem 2 · promotions", f"{p2['profit_per_customer_lift']:+.0%}",
        f"profit per customer (A/B tested) · discount spend {p2['discount_spend_change_pct']:+.0%}", SERIES[1])
    kpi(c[2], "Problem 3 · inventory", f"−{p3['error_reduction_pct']:.0%}",
        f"forecast error (MAPE {p3['forecast_error_old_mape']:.1%} → {p3['forecast_error_new_mape']:.1%})", SERIES[2])
    st.write("")
    st.markdown("""
| Problem | Technique (blogs) | Evidence | How sure |
|---|---|---|---|
| 1 · Churn | Metrics → RFM → CLV → churn → next purchase (1-5) | Tuned churn model ranks risk; value × risk matrix | Estimate: assumes 20% of contacted at-risk customers are saved, AED 25 per contact |
| 2 · Promotions | Market response + uplift + cost optimisation (7, 8) | Stratified A/B test, Welch t-test (9) | Measured in the A/B simulation |
| 3 · Inventory | LSTM forecast + segment timing (2, 5, 6) | 16-week back-test vs current method | Measured forecast error; stock savings scale with it |
""")
    st.caption("All figures come from a synthetic 22,000-customer sample. Percentages transfer better than absolute AED.")
