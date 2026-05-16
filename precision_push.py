"""
How high can precision actually go?

Two levers we haven't pulled yet:
  1. Stop optimizing F1. Sweep the threshold and read precision off the PR curve.
  2. Restrict the population to age >= 50, where stroke prevalence is ~10x higher.
     A model on a higher-prior subgroup has a much higher precision ceiling.

Also: report "top-K precision" — what real screening tools use. If you flag
only the top 5% riskiest patients, how many are actual strokes?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

RNG = 42
CAT = ["gender","ever_married","smoking_status"]
NUM = ["age","hypertension","heart_disease","avg_glucose_level","bmi"]
FEATS = CAT + NUM

# ---------- data ----------
kaggle = pd.read_csv("healthcare-dataset-stroke-data.csv")
kaggle["bmi"] = pd.to_numeric(kaggle["bmi"], errors="coerce")
kaggle["bmi"] = kaggle["bmi"].fillna(kaggle["bmi"].median())
kaggle = kaggle[FEATS + ["stroke"]].copy()

nh = pd.read_csv("nhanes_stroke_combined.csv")
nh["bmi"] = nh["bmi"].fillna(nh["bmi"].median())
nh["avg_glucose_level"] = nh["avg_glucose_level"].fillna(nh["avg_glucose_level"].median())
nh_pos = nh[nh.stroke == 1][FEATS + ["stroke"]]
nh_neg = nh[nh.stroke == 0][FEATS + ["stroke"]].sample(3000, random_state=RNG)

def pipeline(model):
    return Pipeline([
        ("pre", ColumnTransformer([
            ("c", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
            ("n", StandardScaler(), NUM),
        ])),
        ("m", model),
    ])

def fit_and_score(train_df, test_df):
    pipe = pipeline(GradientBoostingClassifier(random_state=RNG))
    pipe.fit(train_df[FEATS], train_df["stroke"])
    return pipe, pipe.predict_proba(test_df[FEATS])[:, 1]

def threshold_sweep(y, proba, points=(0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90)):
    rows = []
    for t in points:
        pred = (proba >= t).astype(int)
        flagged = pred.sum()
        tp = int(((pred==1)&(y==1)).sum())
        fp = int(((pred==1)&(y==0)).sum())
        fn = int(((pred==0)&(y==1)).sum())
        p = tp/flagged if flagged else 0.0
        r = tp/y.sum() if y.sum() else 0.0
        rows.append({"threshold":t,"flagged":flagged,"tp":tp,"fp":fp,"fn":fn,
                     "precision":p,"recall":r})
    return pd.DataFrame(rows)

def topk_precision(y, proba, k_pcts=(0.01,0.02,0.05,0.10,0.20)):
    rows = []
    order = np.argsort(-proba)
    for k in k_pcts:
        n = max(1, int(len(y)*k))
        idx = order[:n]
        tp = int(y.iloc[idx].sum())
        rows.append({"top_pct":k,"flagged":n,"tp":tp,
                     "precision":tp/n,"recall":tp/y.sum()})
    return pd.DataFrame(rows)

# ============================================================
# Strategy 1: full population, threshold sweep
# ============================================================
k_train, k_test = train_test_split(kaggle, test_size=0.20, random_state=RNG, stratify=kaggle["stroke"])
train_full = pd.concat([k_train, nh_pos, nh_neg], ignore_index=True)
_, proba_full = fit_and_score(train_full, k_test)

print("="*70)
print("STRATEGY 1: Variant B + GradientBoosting, threshold sweep")
print(f"Test set: {len(k_test)} patients, {k_test.stroke.sum()} positives ({k_test.stroke.mean()*100:.1f}% base rate)")
print("="*70)
sweep = threshold_sweep(k_test["stroke"].values, proba_full)
print(sweep.to_string(index=False, formatters={"precision":"{:.3f}".format,"recall":"{:.3f}".format}))

print(f"\nTop-K precision (rank patients by risk, flag the top K%):")
print(topk_precision(k_test["stroke"].reset_index(drop=True), proba_full).to_string(
    index=False, formatters={"precision":"{:.3f}".format,"recall":"{:.3f}".format,"top_pct":"{:.0%}".format}))

# ============================================================
# Strategy 2: age >= 50 only
# ============================================================
kaggle_50 = kaggle[kaggle.age >= 50].copy()
nh_pos_50 = nh_pos[nh_pos.age >= 50]
nh_neg_50 = nh_neg[nh_neg.age >= 50].sample(min(2000, len(nh_neg[nh_neg.age >= 50])), random_state=RNG)

k50_train, k50_test = train_test_split(kaggle_50, test_size=0.20, random_state=RNG, stratify=kaggle_50["stroke"])
train_50 = pd.concat([k50_train, nh_pos_50, nh_neg_50], ignore_index=True)
_, proba_50 = fit_and_score(train_50, k50_test)

print("\n" + "="*70)
print("STRATEGY 2: Restrict to age >= 50 (higher base rate)")
print(f"Test set: {len(k50_test)} patients, {k50_test.stroke.sum()} positives ({k50_test.stroke.mean()*100:.1f}% base rate)")
print(f"Train: {len(train_50)} rows, {train_50.stroke.sum()} positives")
print("="*70)
sweep50 = threshold_sweep(k50_test["stroke"].values, proba_50)
print(sweep50.to_string(index=False, formatters={"precision":"{:.3f}".format,"recall":"{:.3f}".format}))
print(f"\nTop-K precision (age 50+ subgroup):")
print(topk_precision(k50_test["stroke"].reset_index(drop=True), proba_50).to_string(
    index=False, formatters={"precision":"{:.3f}".format,"recall":"{:.3f}".format,"top_pct":"{:.0%}".format}))

# ============================================================
# Strategy 3: age >= 60 only (even higher prior)
# ============================================================
kaggle_60 = kaggle[kaggle.age >= 60].copy()
nh_pos_60 = nh_pos[nh_pos.age >= 60]
nh_neg_60 = nh_neg[nh_neg.age >= 60].sample(min(1500, len(nh_neg[nh_neg.age >= 60])), random_state=RNG)

k60_train, k60_test = train_test_split(kaggle_60, test_size=0.20, random_state=RNG, stratify=kaggle_60["stroke"])
train_60 = pd.concat([k60_train, nh_pos_60, nh_neg_60], ignore_index=True)
_, proba_60 = fit_and_score(train_60, k60_test)

print("\n" + "="*70)
print("STRATEGY 3: Restrict to age >= 60")
print(f"Test set: {len(k60_test)} patients, {k60_test.stroke.sum()} positives ({k60_test.stroke.mean()*100:.1f}% base rate)")
print("="*70)
sweep60 = threshold_sweep(k60_test["stroke"].values, proba_60)
print(sweep60.to_string(index=False, formatters={"precision":"{:.3f}".format,"recall":"{:.3f}".format}))
print(f"\nTop-K precision (age 60+ subgroup):")
print(topk_precision(k60_test["stroke"].reset_index(drop=True), proba_60).to_string(
    index=False, formatters={"precision":"{:.3f}".format,"recall":"{:.3f}".format,"top_pct":"{:.0%}".format}))
