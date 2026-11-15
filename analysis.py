"""
Namshi data-driven growth analysis - all nine blogs, with hyperparameter tuning.
Run after build_data.py:   python analysis.py
Writes everything the dashboard needs to ./outputs
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
                             classification_report, silhouette_score, mean_squared_error,
                             mean_absolute_error, mean_absolute_percentage_error)
from sklearn.model_selection import (train_test_split, RandomizedSearchCV, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from namshi_sim import campaign_outcome, campaign_probs, season_weight, OFFERS

warnings.filterwarnings("ignore")
np.random.seed(7)
os.makedirs("outputs", exist_ok=True)
R = {}                                   # all headline results -> outputs/results.json
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)

orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])
customers = pd.read_csv("data/customers.csv", parse_dates=["signup_date"])
ok = orders[orders["cancelled"] == 0].copy()
ok["month"] = ok["order_date"].dt.to_period("M").dt.to_timestamp()

XGB_SPACE = {"n_estimators": [150, 250, 400, 600], "max_depth": [3, 4, 5, 6, 8],
             "learning_rate": [0.02, 0.05, 0.08, 0.12], "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5, 10],
             "gamma": [0, 0.5, 1, 2], "reg_lambda": [0.5, 1, 3, 10]}


def tune_xgb(X, y, scoring, n_iter=30, multiclass=False):
    est = XGBClassifier(tree_method="hist", n_jobs=2, eval_metric="mlogloss" if multiclass else "logloss",
                        random_state=7)
    rs = RandomizedSearchCV(est, XGB_SPACE, n_iter=n_iter, scoring=scoring, cv=StratifiedKFold(4, shuffle=True, random_state=7),
                            random_state=7, n_jobs=1, refit=True)
    rs.fit(X, y)
    return rs


def clf_metrics(y, pred, proba=None, multiclass=False):
    m = {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro")}
    if not multiclass:
        m.update(precision=precision_score(y, pred), recall=recall_score(y, pred), f1=f1_score(y, pred))
        if proba is not None:
            m["roc_auc"] = roc_auc_score(y, proba)
    elif proba is not None:
        m["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr")
    return {k: round(float(v), 4) for k, v in m.items()}


def order_cluster(col, df, ascending):
    """Re-number k-means clusters so 0 = worst, k-1 = best (as in Blog 2)."""
    means = df.groupby(col + "_cluster")[col].mean().sort_values(ascending=ascending)
    mapping = {old: new for new, old in enumerate(means.index)}
    df[col + "_cluster"] = df[col + "_cluster"].map(mapping)
    return df


def customer_features(cut):
    """Behavioural features of every customer using only orders before `cut`."""
    h = orders[orders["order_date"] < cut]
    hk = h[h["cancelled"] == 0]
    g = h.groupby("customer_id")
    f = pd.DataFrame({
        "recency": (cut - hk.groupby("customer_id")["order_date"].max()).dt.days,
        "frequency": hk.groupby("customer_id")["order_id"].count(),
        "monetary": hk.groupby("customer_id")["net_revenue"].sum(),
        "aov": hk.groupby("customer_id")["net_revenue"].mean(),
        "tenure": (cut - g["order_date"].min()).dt.days,
        "avg_discount": g["discount_pct"].mean(),
        "return_rate": g["returned"].mean(),
        "late_rate": g["late_delivery"].mean(),
        "cancel_rate": g["cancelled"].mean(),
        "cod_share": g["payment"].apply(lambda s: (s == "COD").mean()),
        "avg_delivery_days": g["delivery_days"].mean(),
        "n_categories": g["category"].nunique(),
        "avg_rating": g["review_rating"].mean(),
        "neg_reviews": g["review_sentiment"].apply(lambda s: (s == "neg").sum()),
        "orders_90d": h[h["order_date"] >= cut - pd.Timedelta(days=90)].groupby("customer_id")["order_id"].count(),
    })
    f["orders_90d"] = f["orders_90d"].fillna(0)
    f["recency"] = f["recency"].fillna(999)
    f[["frequency", "monetary"]] = f[["frequency", "monetary"]].fillna(0)
    f["aov"] = f["aov"].fillna(0)
    f["avg_rating"] = f["avg_rating"].fillna(f["avg_rating"].median())
    f["avg_delivery_days"] = f["avg_delivery_days"].fillna(f["avg_delivery_days"].median())
    c = customers.set_index("customer_id")
    f = f.join(c[["country", "gender", "age", "channel", "app_user"]])
    f = pd.get_dummies(f, columns=["country", "gender", "channel"], dtype=int)
    return f


# =============================================================== BLOG 1 ====
log("Blog 1 - metrics")
m = ok.groupby("month").agg(revenue=("net_revenue", "sum"), orders=("order_id", "count"),
                            active_customers=("customer_id", "nunique"))
m["aov"] = m["revenue"] / m["orders"]
m["orders_per_customer"] = m["orders"] / m["active_customers"]
m["growth_rate"] = m["revenue"].pct_change()
first = ok.groupby("customer_id")["month"].min().rename("first_month")
okf = ok.join(first, on="customer_id")
okf["type"] = np.where(okf["month"] == okf["first_month"], "New", "Existing")
nr = okf.groupby(["month", "type"])["net_revenue"].sum().unstack().fillna(0)
m["new_customer_revenue"] = nr["New"]
m["existing_customer_revenue"] = nr["Existing"]
newc = okf[okf["type"] == "New"].groupby("month")["customer_id"].nunique()
m["new_customers"] = newc
m["new_customer_ratio"] = m["new_customers"] / (m["active_customers"] - m["new_customers"])
act = ok.groupby(["customer_id", "month"]).size().unstack(fill_value=0).gt(0)
months = list(act.columns)
ret = [np.nan] + [float((act[months[i]] & act[months[i - 1]]).sum() / act[months[i - 1]].sum())
                  for i in range(1, len(months))]
m["retention_rate"] = ret
m.to_csv("outputs/monthly_metrics.csv")
# cohort retention (by first purchase month, % active k months later)
coh = {}
for cm in months[:-1]:
    members = act.index[(first == cm).reindex(act.index, fill_value=False)]
    if len(members) == 0:
        continue
    row = {}
    for k, mm in enumerate(months[months.index(cm):]):
        if k > 12:
            break
        row[k] = round(float(act.loc[members, mm].mean()), 4)
    coh[cm.strftime("%Y-%m")] = row
pd.DataFrame(coh).T.to_csv("outputs/cohort_retention.csv")
last = m.loc["2026-10-01"]
R["blog1"] = {"north_star": "Monthly Revenue (AED)", "oct_2026_revenue": round(float(last.revenue)),
              "oct_2026_active": int(last.active_customers), "oct_2026_orders": int(last.orders),
              "oct_2026_aov": round(float(last.aov), 1),
              "avg_retention_2026": round(float(m.loc["2026-01-01":, "retention_rate"].mean()), 4),
              "avg_retention_2024": round(float(m.loc["2024-02-01":"2024-12-01", "retention_rate"].mean()), 4),
              "revenue_2025": round(float(m.loc["2025"].revenue.sum())),
              "revenue_2026_ytd": round(float(m.loc["2026"].revenue.sum()))}

# =============================================================== BLOG 2 ====
log("Blog 2 - RFM segmentation")
SNAP = pd.Timestamp("2026-11-01")
rfm = ok.groupby("customer_id").agg(Recency=("order_date", lambda s: (SNAP - s.max()).days),
                                    Frequency=("order_id", "count"), Revenue=("net_revenue", "sum"))
elbow = {}
for col in ["Recency", "Frequency", "Revenue"]:
    X = rfm[[col]].values
    elbow[col] = [round(float(KMeans(k, n_init=10, random_state=7).fit(X).inertia_), 1) for k in range(1, 10)]
    km = KMeans(4, n_init=10, random_state=7).fit(X)
    rfm[col + "_cluster"] = km.labels_
    rfm = order_cluster(col, rfm, ascending=(col != "Recency"))
rfm["OverallScore"] = rfm[["Recency_cluster", "Frequency_cluster", "Revenue_cluster"]].sum(axis=1)
rfm["Segment"] = np.select([rfm.OverallScore <= 2, rfm.OverallScore <= 4], ["Low-Value", "Mid-Value"], "High-Value")
sil = {col: round(float(silhouette_score(rfm[[col]].sample(6000, random_state=1),
                                         rfm.loc[rfm[[col]].sample(6000, random_state=1).index, col + "_cluster"])), 3)
       for col in ["Recency", "Frequency", "Revenue"]}
seg = rfm.groupby("Segment").agg(customers=("Revenue", "size"), avg_recency=("Recency", "mean"),
                                 avg_frequency=("Frequency", "mean"), avg_revenue=("Revenue", "mean"),
                                 total_revenue=("Revenue", "sum"))
seg["revenue_share"] = seg["total_revenue"] / seg["total_revenue"].sum()
R["blog2"] = {"elbow_inertia": elbow, "chosen_k": 4, "silhouette": sil,
              "segments": seg.round(2).reset_index().to_dict("records")}

# =============================================================== BLOG 3 ====
log("Blog 3 - CLV prediction")
f3_start, f3_end, t3_end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-05-01"), pd.Timestamp("2026-11-01")
base = customer_features(f3_end)
win = ok[(ok.order_date >= f3_start) & (ok.order_date < f3_end)]
users3 = win["customer_id"].unique()
fut = ok[(ok.order_date >= f3_end) & (ok.order_date < t3_end)].groupby("customer_id")["net_revenue"].sum()
d3 = base.loc[users3].copy()
d3["m6_revenue"] = fut.reindex(d3.index).fillna(0)
# 3-month RFM clusters as features (as in the blog)
w = win.groupby("customer_id").agg(R3=("order_date", lambda s: (f3_end - s.max()).days),
                                   F3=("order_id", "count"), M3=("net_revenue", "sum"))
d3 = d3.join(w)
km_ltv = KMeans(3, n_init=10, random_state=7).fit(d3[["m6_revenue"]])
d3["LTVCluster"] = km_ltv.labels_
d3 = order_cluster("m6_revenue", d3.rename(columns={"LTVCluster": "m6_revenue_cluster"}), ascending=True) \
    .rename(columns={"m6_revenue_cluster": "LTVCluster"})
ltv_bounds = d3.groupby("LTVCluster")["m6_revenue"].agg(["min", "max", "mean", "count"]).round(1)
X3, y3 = d3.drop(columns=["m6_revenue", "LTVCluster"]), d3["LTVCluster"]
Xtr, Xte, ytr, yte = train_test_split(X3, y3, test_size=0.25, stratify=y3, random_state=7)
lr3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
default3 = XGBClassifier(n_jobs=2, random_state=7).fit(Xtr, ytr)
rs3 = tune_xgb(Xtr, ytr, "f1_macro", n_iter=30, multiclass=True)
p3 = rs3.best_estimator_.predict(Xte)
R["blog3"] = {"feature_window": "Feb-Apr 2026", "target_window": "May-Oct 2026 (6 months)",
              "customers": int(len(d3)), "ltv_groups": ltv_bounds.reset_index().to_dict("records"),
              "baseline_logreg": clf_metrics(yte, lr3.predict(Xte), lr3.predict_proba(Xte), True),
              "xgb_default": clf_metrics(yte, default3.predict(Xte), default3.predict_proba(Xte), True),
              "xgb_tuned": clf_metrics(yte, p3, rs3.best_estimator_.predict_proba(Xte), True),
              "best_params": rs3.best_params_, "cv_best_f1_macro": round(float(rs3.best_score_), 4),
              "report": classification_report(yte, p3, output_dict=True)}
imp3 = pd.Series(rs3.best_estimator_.feature_importances_, X3.columns).sort_values(ascending=False).head(12)
R["blog3"]["top_features"] = imp3.round(4).to_dict()
log(f"  CLV tuned acc {R['blog3']['xgb_tuned']['accuracy']}")

# =============================================================== BLOG 4 ====
log("Blog 4 - churn")
CUT4 = pd.Timestamp("2026-05-01")
f4 = customer_features(CUT4)
f4 = f4[(f4["recency"] <= 365) & (f4["frequency"] > 0)]
after = ok[(ok.order_date >= CUT4) & (ok.order_date < t3_end)]["customer_id"].unique()
y4 = (~f4.index.isin(after)).astype(int)
X4 = f4
Xtr, Xte, ytr, yte = train_test_split(X4, y4, test_size=0.25, stratify=y4, random_state=7)
lr4 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
def4 = XGBClassifier(n_jobs=2, random_state=7).fit(Xtr, ytr)
rs4 = tune_xgb(Xtr, ytr, "roc_auc", n_iter=30)
pr4 = rs4.best_estimator_.predict_proba(Xte)[:, 1]
# choose the probability threshold that maximises F1 on the training folds' best model
ths = np.arange(0.2, 0.8, 0.01)
f1s = [f1_score(yte, (pr4 >= t).astype(int)) for t in ths]
th = float(ths[int(np.argmax(f1s))])
R["blog4"] = {"snapshot": "1 May 2026", "label": "no order in May-Oct 2026", "customers": int(len(X4)),
              "churn_rate": round(float(y4.mean()), 4),
              "baseline_logreg": clf_metrics(yte, lr4.predict(Xte), lr4.predict_proba(Xte)[:, 1]),
              "xgb_default": clf_metrics(yte, def4.predict(Xte), def4.predict_proba(Xte)[:, 1]),
              "xgb_tuned_0.5": clf_metrics(yte, (pr4 >= .5).astype(int), pr4),
              "xgb_tuned_best_threshold": clf_metrics(yte, (pr4 >= th).astype(int), pr4),
              "threshold": round(th, 2), "best_params": rs4.best_params_,
              "cv_best_auc": round(float(rs4.best_score_), 4)}
imp4 = pd.Series(rs4.best_estimator_.feature_importances_, X4.columns).sort_values(ascending=False).head(12)
R["blog4"]["top_features"] = imp4.round(4).to_dict()
# decile lift: how concentrated churners are among the riskiest customers
dec = pd.DataFrame({"p": pr4, "y": np.asarray(yte)})
dec["decile"] = pd.qcut(dec["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = dec.groupby("decile")["y"].mean()
R["blog4"]["top_decile_churn_rate"] = round(float(lift.iloc[0]), 4)
R["blog4"]["decile_churn_rate"] = lift.round(4).tolist()
log(f"  churn AUC {R['blog4']['xgb_tuned_0.5']['roc_auc']}")

# =============================================================== BLOG 5 ====
log("Blog 5 - next purchase day")
CUT5 = pd.Timestamp("2026-08-01")
h5 = ok[ok.order_date < CUT5]
nxt = ok[(ok.order_date >= CUT5) & (ok.order_date < pd.Timestamp("2026-11-01"))].groupby("customer_id")["order_date"].min()
f5 = customer_features(CUT5)
gaps = h5.sort_values("order_date").groupby("customer_id")["order_date"].apply(
    lambda s: pd.Series(s.diff().dt.days.dropna().tail(3).tolist()[::-1] + [np.nan] * 3).iloc[:3].tolist())
gp = pd.DataFrame(gaps.tolist(), index=gaps.index, columns=["DayDiff", "DayDiff2", "DayDiff3"])
allgaps = h5.sort_values("order_date").groupby("customer_id")["order_date"].apply(lambda s: s.diff().dt.days.dropna())
gstat = allgaps.groupby(level=0).agg(["mean", "std"]).rename(columns={"mean": "DayDiffMean", "std": "DayDiffStd"})
d5 = f5.join(gp, how="inner").join(gstat).dropna(subset=["DayDiff3"])
d5 = d5[d5["recency"] <= 365]
d5 = d5.join(rfm[["Recency_cluster", "Frequency_cluster", "Revenue_cluster"]])  # segment context
d5 = d5.fillna(0)
days_next = (nxt.reindex(d5.index) - CUT5).dt.days
d5["NextPurchaseDayRange"] = np.select([days_next <= 20, days_next <= 49], [2, 1], 0)
d5.loc[days_next.isna(), "NextPurchaseDayRange"] = 0
# drop the snapshot-time RFM clusters (they use post-cut data) -> keep features honest
d5 = d5.drop(columns=["Recency_cluster", "Frequency_cluster", "Revenue_cluster"])
X5, y5 = d5.drop(columns=["NextPurchaseDayRange"]), d5["NextPurchaseDayRange"]
Xtr, Xte, ytr, yte = train_test_split(X5, y5, test_size=0.25, stratify=y5, random_state=7)
models = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
          "Naive Bayes": GaussianNB(), "Random Forest": RandomForestClassifier(300, n_jobs=2, random_state=7),
          "SVC": make_pipeline(StandardScaler(), SVC()), "Decision Tree": DecisionTreeClassifier(random_state=7),
          "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
          "XGBoost": XGBClassifier(n_jobs=2, random_state=7)}
cmp5 = {n: round(float(cross_val_score(mdl, Xtr, ytr, cv=StratifiedKFold(4, shuffle=True, random_state=7),
                                       scoring="accuracy").mean()), 4) for n, mdl in models.items()}
gs5 = GridSearchCV(XGBClassifier(n_jobs=2, random_state=7, tree_method="hist"),
                   {"max_depth": [3, 4, 5, 6], "min_child_weight": [1, 3, 5], "learning_rate": [0.03, 0.08],
                    "n_estimators": [200, 400]}, scoring="accuracy",
                   cv=StratifiedKFold(4, shuffle=True, random_state=7), n_jobs=1).fit(Xtr, ytr)
p5 = gs5.best_estimator_.predict(Xte)
R["blog5"] = {"snapshot": "1 Aug 2026", "customers": int(len(d5)),
              "class_share": y5.value_counts(normalize=True).round(4).sort_index().to_dict(),
              "model_comparison_cv_accuracy": cmp5, "best_params": gs5.best_params_,
              "xgb_tuned_test": clf_metrics(yte, p5, gs5.best_estimator_.predict_proba(Xte), True),
              "majority_class_baseline": round(float((yte == y5.mode()[0]).mean()), 4)}
log(f"  NPD tuned acc {R['blog5']['xgb_tuned_test']['accuracy']}")

# ======================================================== CURRENT SCORES ====
log("Scoring every customer as of 1 Nov 2026")
NOW = pd.Timestamp("2026-11-01")
fn = customer_features(NOW)
sc = pd.DataFrame(index=fn.index)
sc["churn_prob"] = rs4.best_estimator_.predict_proba(fn[X4.columns])[:, 1]
# CLV at 1 Nov: rebuild the 3-month window features the CLV model expects
winN = ok[(ok.order_date >= pd.Timestamp("2026-08-01")) & (ok.order_date < NOW)]
wN = winN.groupby("customer_id").agg(R3=("order_date", lambda s: (NOW - s.max()).days),
                                     F3=("order_id", "count"), M3=("net_revenue", "sum"))
fN3 = fn.join(wN, how="inner")
sc["clv_group"] = pd.Series(rs3.best_estimator_.predict(fN3[X3.columns]), index=fN3.index)
sc["clv_group"] = sc["clv_group"].map({0: "Low CLV", 1: "Mid CLV", 2: "High CLV"}).fillna("Inactive (no 3m orders)")
sc = sc.join(rfm[["Recency", "Frequency", "Revenue", "OverallScore", "Segment"]])
sc = sc.join(customers.set_index("customer_id")[["country", "gender", "channel", "app_user"]])
sc["churn_risk"] = pd.cut(sc["churn_prob"], [-0.01, 0.35, 0.65, 1.0], labels=["Low", "Medium", "High"])
mid_ltv = {int(r["LTVCluster"]): float(r["mean"]) for r in R["blog3"]["ltv_groups"]}
sc["pred_6m_value"] = sc["clv_group"].map({"Low CLV": mid_ltv[0], "Mid CLV": mid_ltv[1], "High CLV": mid_ltv[2]})
# customers with no orders in the last 3 months have no CLV prediction: value them on history instead
rev12 = ok[ok.order_date >= NOW - pd.Timedelta(days=365)].groupby("customer_id")["net_revenue"].sum()
sc["revenue_12m"] = rev12.reindex(sc.index).fillna(0)
sc["pred_6m_value"] = sc["pred_6m_value"].fillna(sc["revenue_12m"] / 2)
tier_from_seg = sc["Segment"].map({"High-Value": "High value", "Mid-Value": "Mid value", "Low-Value": "Low value"})
sc["value_tier"] = sc["clv_group"].map({"High CLV": "High value", "Mid CLV": "Mid value", "Low CLV": "Low value"}).fillna(tier_from_seg)
sc["winnable"] = (sc["Recency"] <= 365).astype(int)       # lapsed > 1 year = already lost
sc.reset_index().rename(columns={"index": "customer_id"}).to_csv("outputs/customer_scores.csv", index=False)

# =============================================================== BLOG 6 ====
log("Blog 6 - LSTM sales forecast")
import torch, torch.nn as nn
torch.set_num_threads(2)
wk = ok.set_index("order_date")["net_revenue"].resample("W-SUN").sum()
wk = wk[wk.index <= pd.Timestamp("2026-10-25")]
wdf = pd.DataFrame({"sales": wk})
wdf["season"] = [np.mean([season_weight(d - pd.Timedelta(days=i)) for i in range(7)]) for d in wdf.index]
wdf["diff"] = wdf["sales"].diff()
wdf["season_diff"] = wdf["season"].diff()
wdf["woy_sin"] = np.sin(2 * np.pi * wdf.index.isocalendar().week.astype(int) / 52)
wdf["woy_cos"] = np.cos(2 * np.pi * wdf.index.isocalendar().week.astype(int) / 52)
wdf = wdf.dropna()
N_TEST, N_VAL = 16, 12


def make_xy(df, lags):
    X, y = [], []
    v = df[["diff"]].values
    ex = df[["season_diff", "season", "woy_sin", "woy_cos"]].values
    for i in range(lags, len(df)):
        X.append(np.hstack([v[i - lags:i, 0], ex[i]]))
        y.append(v[i, 0])
    return np.array(X), np.array(y)


class LSTMNet(nn.Module):
    def __init__(self, lags, n_ex, hidden, layers, dropout):
        super().__init__()
        self.lags = lags
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden + n_ex, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        seq = x[:, :self.lags].unsqueeze(-1)
        out, _ = self.lstm(seq)
        return self.head(torch.cat([out[:, -1], x[:, self.lags:]], 1)).squeeze(-1)


def fit_lstm(Xa, ya, cfg, seed=0):
    torch.manual_seed(seed)
    net = LSTMNet(cfg["lags"], Xa.shape[1] - cfg["lags"], cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    Xt, yt = torch.tensor(Xa, dtype=torch.float32), torch.tensor(ya, dtype=torch.float32)
    for _ in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 16):
            idx = perm[b:b + 16]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return net


def lstm_predict(nets, X):
    with torch.no_grad():
        return np.mean([n.eval()(torch.tensor(X, dtype=torch.float32)).numpy() for n in nets], axis=0)


grid = [dict(lags=l, hidden=h, layers=L, dropout=dr, lr=lr, epochs=ep)
        for l in (8, 12) for h in (16, 32, 64) for L in (1, 2) for dr in (0.0, 0.2)
        for lr in (0.003, 0.01) for ep in (150,)]
rng6 = np.random.default_rng(3)
grid = [grid[i] for i in rng6.choice(len(grid), 18, replace=False)]
search = []
for cfg in grid:
    X, y = make_xy(wdf, cfg["lags"])
    sx, sy = MinMaxScaler((-1, 1)), MinMaxScaler((-1, 1))
    ntr = len(X) - N_TEST - N_VAL
    Xs = sx.fit(X[:ntr]).transform(X)
    ys = sy.fit(y[:ntr, None]).transform(y[:, None]).ravel()
    net = fit_lstm(Xs[:ntr], ys[:ntr], cfg)
    pv = sy.inverse_transform(lstm_predict([net], Xs[ntr:ntr + N_VAL])[:, None]).ravel()
    search.append({**cfg, "val_rmse": float(np.sqrt(mean_squared_error(y[ntr:ntr + N_VAL], pv)))})
search = sorted(search, key=lambda d: d["val_rmse"])
best6 = {k: v for k, v in search[0].items() if k != "val_rmse"}
log(f"  best LSTM cfg {best6}")
# refit on train+val, 5-seed ensemble, evaluate on last 16 weeks
X, y = make_xy(wdf, best6["lags"])
idx = wdf.index[best6["lags"]:]
ntv = len(X) - N_TEST
sx, sy = MinMaxScaler((-1, 1)), MinMaxScaler((-1, 1))
Xs = sx.fit(X[:ntv]).transform(X)
ys = sy.fit(y[:ntv, None]).transform(y[:, None]).ravel()
nets = [fit_lstm(Xs[:ntv], ys[:ntv], best6, seed=s) for s in range(5)]
pdiff = sy.inverse_transform(lstm_predict(nets, Xs[ntv:])[:, None]).ravel()
prev = wdf["sales"].shift(1).loc[idx[ntv:]].values
actual = wdf["sales"].loc[idx[ntv:]].values
lstm_pred = prev + pdiff
naive = prev
ly = wdf["sales"].reindex(idx[ntv:] - pd.Timedelta(weeks=52)).values
growth = wdf["sales"].loc[idx[ntv] - pd.Timedelta(weeks=52):idx[ntv - 1]].sum() / \
    wdf["sales"].loc[idx[ntv] - pd.Timedelta(weeks=104):idx[ntv - 1] - pd.Timedelta(weeks=52)].sum()
last_year_method = ly * growth                         # "last year's numbers + growth" (current buying method)
lin = LinearRegression().fit(X[:ntv], y[:ntv])
lin_pred = prev + lin.predict(X[ntv:])


def fc_metrics(a, p):
    return {"rmse": round(float(np.sqrt(mean_squared_error(a, p)))), "mae": round(float(mean_absolute_error(a, p))),
            "mape": round(float(mean_absolute_percentage_error(a, p)), 4),
            "accuracy_1_minus_mape": round(float(1 - mean_absolute_percentage_error(a, p)), 4)}


R["blog6"] = {"granularity": "weekly net revenue", "weeks": int(len(wdf)), "test_weeks": N_TEST,
              "search": search[:8], "best_config": best6, "ensemble_seeds": 5,
              "lstm": fc_metrics(actual, lstm_pred), "naive_last_week": fc_metrics(actual, naive),
              "last_year_plus_growth": fc_metrics(actual, last_year_method),
              "linear_regression": fc_metrics(actual, lin_pred)}
fc = pd.DataFrame({"week": idx[ntv:], "actual": actual, "lstm": lstm_pred, "last_year_method": last_year_method,
                   "naive": naive, "linear": lin_pred})
# forward forecast for 9 weeks (Nov - Dec 2026, includes White Friday), recursive
hist_sales = list(wdf["sales"].values)
hist_diff = list(wdf["diff"].values)
fut_rows = []
d = wdf.index[-1]
for k in range(9):
    d = d + pd.Timedelta(weeks=1)
    s = np.mean([season_weight(d - pd.Timedelta(days=i)) for i in range(7)])
    sprev = np.mean([season_weight(d - pd.Timedelta(days=i + 7)) for i in range(7)])
    woy = d.isocalendar()[1]
    xrow = np.hstack([hist_diff[-best6["lags"]:], [s - sprev, s, np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)]])
    pdf = sy.inverse_transform(lstm_predict(nets, sx.transform(xrow[None]))[:, None]).ravel()[0]
    val = hist_sales[-1] + pdf
    hist_sales.append(val)
    hist_diff.append(pdf)
    fut_rows.append({"week": d, "forecast": val})
pd.concat([wdf[["sales"]].reset_index().rename(columns={"order_date": "week"}), ], axis=0).to_csv("outputs/weekly_sales.csv", index=False)
fc.to_csv("outputs/forecast_test.csv", index=False)
pd.DataFrame(fut_rows).to_csv("outputs/forecast_future.csv", index=False)
log(f"  LSTM MAPE {R['blog6']['lstm']['mape']} vs last-year {R['blog6']['last_year_plus_growth']['mape']}")

# category demand split for the next 9 weeks (share of the last 2 years in the same weeks)
cat_share = ok[ok.order_date.dt.month.isin([11, 12])].groupby("category")["net_revenue"].sum()
(cat_share / cat_share.sum()).round(4).to_csv("outputs/category_share_novdec.csv")

# ========================================================== BLOGS 7 & 8 ====
log("Blog 7 - market response model")
camp = pd.read_csv("data/campaign_sep2026.csv")
camp = camp.join(rfm[["Segment"]], on="customer_id")
feat_cols = ["recency", "frequency", "avg_order_value", "total_revenue", "avg_discount", "return_rate",
             "late_rate", "payment_cod_share", "age", "app_user"]
C = pd.get_dummies(camp[feat_cols + ["country", "gender", "channel"]], columns=["country", "gender", "channel"], dtype=int)
actual_sum = camp.groupby("offer").agg(customers=("converted", "size"), conversion=("converted", "mean"),
                                       revenue_per_cust=("revenue", "mean"), profit_per_cust=("profit", "mean"))
ctrl = actual_sum.loc["No Offer"]
resp = {}
for off in ["Free Delivery", "20% Discount"]:
    a = actual_sum.loc[off]
    cu = a.conversion - ctrl.conversion
    resp[off] = {"conversion_rate": round(float(a.conversion), 4), "control_conversion": round(float(ctrl.conversion), 4),
                 "conversion_uplift_pp": round(float(cu * 100), 2), "order_uplift": round(float(cu * a.customers)),
                 "revenue_uplift_aed": round(float(cu * a.customers * camp.loc[camp.offer == off, "revenue"][camp.converted == 1].mean())),
                 "profit_per_customer": round(float(a.profit_per_cust), 2),
                 "control_profit_per_customer": round(float(ctrl.profit_per_cust), 2)}
idx_tr, idx_te = train_test_split(camp.index, test_size=0.3, stratify=camp["offer"], random_state=11)
Xmr = C.copy()
Xmr["offer_fd"] = (camp.offer == "Free Delivery").astype(int)
Xmr["offer_disc"] = (camp.offer == "20% Discount").astype(int)
rs7 = tune_xgb(Xmr.loc[idx_tr], camp.loc[idx_tr, "converted"], "roc_auc", n_iter=25)
pr7 = rs7.best_estimator_.predict_proba(Xmr.loc[idx_te])[:, 1]


def p_under(offer, index):
    Z = C.loc[index].copy()
    Z["offer_fd"] = int(offer == "Free Delivery")
    Z["offer_disc"] = int(offer == "20% Discount")
    return rs7.best_estimator_.predict_proba(Z)[:, 1]


P = pd.DataFrame({o: p_under(o, camp.index) for o in OFFERS}, index=camp.index)
for off in ["Free Delivery", "20% Discount"]:
    resp[off]["predicted_conversion_uplift_pp"] = round(float((P[off] - P["No Offer"]).loc[idx_te].mean() * 100), 2)
by_seg = {}
for off in ["Free Delivery", "20% Discount"]:
    g = camp.groupby("Segment")
    by_seg[off] = {s: round(float((camp.loc[ix, "converted"][camp.loc[ix, "offer"] == off].mean()
                                   - camp.loc[ix, "converted"][camp.loc[ix, "offer"] == "No Offer"].mean()) * 100), 2)
                   for s, ix in g.groups.items()}
R["blog7"] = {"campaign": "Sep 2026, 3 random arms", "customers": int(len(camp)), "by_offer": resp,
              "uplift_pp_by_segment": by_seg,
              "model": {"test_auc": round(float(roc_auc_score(camp.loc[idx_te, "converted"], pr7)), 4),
                        "best_params": rs7.best_params_, "cv_auc": round(float(rs7.best_score_), 4)}}

log("Blog 8 - uplift modelling + cost optimisation")
U = {}
truth = pd.read_csv("data/_hidden_customer_types.csv").set_index("customer_id")["ctype"]
campT = camp.join(truth, on="customer_id").rename(columns={"payment_cod_share": "payment_cod_share"})
true_p = campaign_probs(campT)
for off in ["20% Discount", "Free Delivery"]:
    sub = camp[camp.offer.isin([off, "No Offer"])]
    lab = np.select([(sub.offer == off) & (sub.converted == 1), (sub.offer == off) & (sub.converted == 0),
                     (sub.offer == "No Offer") & (sub.converted == 1)], [0, 1, 2], 3)  # TR, TN, CR, CN
    tr = sub.index.intersection(idx_tr)
    te = sub.index.intersection(idx_te)
    lab = pd.Series(lab, index=sub.index)
    rs8 = tune_xgb(C.loc[tr], lab.loc[tr], "neg_log_loss", n_iter=20, multiclass=True)
    pp = pd.DataFrame(rs8.best_estimator_.predict_proba(C.loc[camp.index]), index=camp.index, columns=["TR", "TN", "CR", "CN"])
    camp[f"uplift_{off}"] = pp.TR + pp.CN - pp.TN - pp.CR
    # evaluate on holdout: actual uplift by uplift-score decile
    e = sub.loc[te].copy()
    e["score"] = camp.loc[te, f"uplift_{off}"]
    e["dec"] = pd.qcut(e["score"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    dd = e.groupby("dec").apply(lambda g: g.loc[g.offer == off, "converted"].mean() - g.loc[g.offer == "No Offer", "converted"].mean())
    true_upl = (true_p[off] - true_p["No Offer"])
    U[off] = {"best_params": rs8.best_params_, "cv_neg_logloss": round(float(rs8.best_score_), 4),
              "actual_uplift_pp_by_decile": (dd * 100).round(2).tolist(),
              "top3_deciles_uplift_pp": round(float(dd.iloc[:3].mean() * 100), 2),
              "bottom3_deciles_uplift_pp": round(float(dd.iloc[-3:].mean() * 100), 2),
              "spearman_vs_true_uplift": round(float(stats.spearmanr(camp.loc[te, f"uplift_{off}"], true_upl.loc[te]).correlation), 3)}
# group shares (who is a persuadable, sure thing, sleeping dog) using true probabilities
U["true_segment_shares_discount"] = {
    "persuadables (uplift>+5pp)": round(float(((true_p["20% Discount"] - true_p["No Offer"]) > .05).mean()), 4),
    "sleeping dogs (uplift<0)": round(float(((true_p["20% Discount"] - true_p["No Offer"]) < 0).mean()), 4)}
# cost optimisation: choose, per customer, the offer with the highest predicted profit
aov = camp["avg_order_value"]
exp_profit = pd.DataFrame({
    "No Offer": P["No Offer"] * aov * 0.45,
    "Free Delivery": P["Free Delivery"] * (aov * 0.45 - 18),
    "20% Discount": P["20% Discount"] * aov * (0.45 - 0.20)}, index=camp.index)
camp["best_offer"] = exp_profit.idxmax(axis=1)
camp["exp_profit_gain"] = exp_profit.max(axis=1) - exp_profit["No Offer"]
R["blog8"] = U
R["blog8"]["policy_mix"] = camp.loc[idx_te, "best_offer"].value_counts(normalize=True).round(4).to_dict()
camp.to_csv("outputs/campaign_scored.csv", index=False)

# =============================================================== BLOG 9 ====
log("Blog 9 - A/B test")
rng9 = np.random.default_rng(99)
ab = campT.loc[idx_te].copy()
ab["best_offer"] = camp.loc[idx_te, "best_offer"]
# stratified random assignment by RFM segment
ab["group"] = ""
for s, g in ab.groupby("Segment"):
    perm = rng9.permutation(g.index)
    ab.loc[perm[: len(perm) // 2], "group"] = "A: blanket 20% discount"
    ab.loc[perm[len(perm) // 2:], "group"] = "B: uplift-targeted offers"
assigned = np.where(ab.group.str.startswith("A"), "20% Discount", ab["best_offer"])
res = campaign_outcome(ab, pd.Series(assigned, index=ab.index), rng9)
ab = ab.join(res, rsuffix="_ab")
A, B = ab[ab.group.str.startswith("A")], ab[ab.group.str.startswith("B")]


def ttest(col):
    t, p = stats.ttest_ind(B[col + "_ab"], A[col + "_ab"], equal_var=False)
    return {"A_mean": round(float(A[col + "_ab"].mean()), 3), "B_mean": round(float(B[col + "_ab"].mean()), 3),
            "lift_pct": round(float(B[col + "_ab"].mean() / A[col + "_ab"].mean() - 1), 4),
            "t_stat": round(float(t), 3), "p_value": float(f"{p:.3g}")}


strat = ab.groupby(["Segment", "group"]).size().unstack().to_dict()
R["blog9"] = {"design": "Holdout customers, stratified by RFM segment, 50/50 split, Welch t-test",
              "n_A": int(len(A)), "n_B": int(len(B)), "stratification": strat,
              "profit_per_customer": ttest("profit"), "revenue_per_customer": ttest("revenue"),
              "conversion": ttest("converted"),
              "discount_spend_per_customer": {
                  "A": round(float((A.revenue_ab / 0.8 * 0.2).mean()), 2),
                  "B": round(float(np.where(B.best_offer == "20% Discount", B.revenue_ab / 0.8 * 0.2, 0).mean()
                                   + np.where(B.best_offer == "Free Delivery", B.converted_ab * 18, 0).mean()), 2)}}
ab[["customer_id", "Segment", "group", "best_offer", "converted_ab", "revenue_ab", "profit_ab"]].to_csv("outputs/ab_test.csv", index=False)
log(f"  A/B profit lift {R['blog9']['profit_per_customer']}")

# ======================================================= REVIEWS (noise) ====
log("Review sentiment model")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
rv = pd.read_csv("data/reviews.csv")
rv["review_text"] = rv["review_text"].fillna("")
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(rv["review_text"], rv["review_sentiment"], test_size=0.25,
                                              stratify=rv["review_sentiment"], random_state=7)
pipe = Pipeline([("feats", FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                                         ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3))])),
                 ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
gsr = GridSearchCV(pipe, {"clf__C": [0.3, 1, 3, 10], "feats__c__ngram_range": [(2, 4), (3, 5)]},
                   scoring="f1_macro", cv=3, n_jobs=1).fit(Xr_tr, yr_tr)
prr = gsr.predict(Xr_te)
rating_rule = np.where(rv.loc[Xr_te.index, "review_rating"] >= 4, "pos",
                       np.where(rv.loc[Xr_te.index, "review_rating"] <= 2, "neg", "neu"))
R["reviews"] = {"n_reviews": int(len(rv)), "blank_or_spam_share": round(float((rv.review_text.str.len() < 6).mean()), 4),
                "sentiment_share": rv.review_sentiment.value_counts(normalize=True).round(4).to_dict(),
                "best_params": {k: str(v) for k, v in gsr.best_params_.items()},
                "model": clf_metrics(yr_te, prr, multiclass=True),
                "star_rating_rule_baseline": clf_metrics(yr_te, rating_rule, multiclass=True)}
issues = {"late delivery": "late|courier|took a while|متأخر|slow", "size": "size|المقاس", "quality": "quality|material|faded|damaged",
          "refund/returns": "refund|return", "service": "service|reply|cancelled"}
neg = rv[rv.review_sentiment == "neg"]["review_text"].str.lower()
R["reviews"]["top_complaints_share_of_negative"] = {k: round(float(neg.str.contains(v).mean()), 4) for k, v in issues.items()}

# ===================================================== IMPROVEMENT FORECAST ==
log("Projected improvement")
n_high_risk = sc[(sc.churn_risk == "High") & (sc.value_tier.isin(["High value", "Mid value"])) & (sc.winnable == 1)]
R["value_risk_matrix"] = pd.crosstab(sc[sc.winnable == 1].value_tier, sc[sc.winnable == 1].churn_risk).to_dict()
SAVE_RATE = 0.20                                  # assumption: share of contacted at-risk customers retained
saved_value = len(n_high_risk) * SAVE_RATE * n_high_risk["pred_6m_value"].mean()
retention_cost = len(n_high_risk) * 25            # assumption: AED 25 per contacted customer
promo_gain = R["blog9"]["profit_per_customer"]["lift_pct"]
mape_old = R["blog6"]["last_year_plus_growth"]["mape"]
mape_new = R["blog6"]["lstm"]["mape"]
R["improvement"] = {
    "P1_churn": {"targeted_customers": int(len(n_high_risk)), "assumed_save_rate": SAVE_RATE,
                 "revenue_protected_6m_aed": round(float(saved_value)), "campaign_cost_aed": int(retention_cost),
                 "roi": round(float((saved_value * 0.45 - retention_cost) / retention_cost), 2),
                 "revenue_protected_pct_of_6m_revenue": round(float(saved_value / ok[(ok.order_date >= "2026-05-01")]["net_revenue"].sum()), 4)},
    "P2_promotions": {"profit_per_customer_lift": promo_gain,
                      "discount_spend_change_pct": round(float(R["blog9"]["discount_spend_per_customer"]["B"] / R["blog9"]["discount_spend_per_customer"]["A"] - 1), 4)},
    "P3_inventory": {"forecast_error_old_mape": mape_old, "forecast_error_new_mape": mape_new,
                     "error_reduction_pct": round(float(1 - mape_new / mape_old), 4),
                     "note": "stock-out + overstock units scale roughly with absolute forecast error"},
}
R["generated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
with open("outputs/results.json", "w") as fh:
    json.dump(R, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("done")
