"""
Compare three training strategies on a *Kaggle-only* test set.

  Baseline : Kaggle data only (the original setup)
  Variant A: Kaggle + 898 NHANES stroke-positives
  Variant B: Kaggle + 898 NHANES positives + 3,000 random NHANES negatives

Mitigations applied:
  1. `source` column is DROPPED before training (never given to the model)
  2. The test set is sampled from KAGGLE ONLY, so all variants are compared
     against the same real-world population
  3. Variant B re-introduces NHANES negatives to dilute the easy
     "is-this-row-from-NHANES" signal that would otherwise inflate metrics

For each variant, three models (LogReg, RandomForest, GradientBoosting) are
trained with class_weight=balanced (no SMOTE), thresholds tuned via the
precision-recall curve to maximize F1, and full metrics + a PR curve printed.
"""
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (precision_recall_curve, roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score, accuracy_score,
                             confusion_matrix)

RNG = 42
FEATURES_CAT = ["gender","ever_married","smoking_status"]
FEATURES_NUM = ["age","hypertension","heart_disease","avg_glucose_level","bmi"]
FEATURES = FEATURES_CAT + FEATURES_NUM

# ---------- load ----------
kaggle = pd.read_csv("healthcare-dataset-stroke-data.csv")
kaggle["bmi"] = pd.to_numeric(kaggle["bmi"], errors="coerce")
kaggle["bmi"] = kaggle["bmi"].fillna(kaggle["bmi"].median())
kaggle = kaggle[FEATURES + ["stroke"]].copy()

nh_combined = pd.read_csv("nhanes_stroke_combined.csv")
nh_combined["bmi"] = nh_combined["bmi"].fillna(nh_combined["bmi"].median())
nh_combined["avg_glucose_level"] = nh_combined["avg_glucose_level"].fillna(nh_combined["avg_glucose_level"].median())
nh_pos = nh_combined[nh_combined["stroke"]==1][FEATURES + ["stroke"]].copy()
nh_neg = nh_combined[nh_combined["stroke"]==0][FEATURES + ["stroke"]].copy()

print(f"Kaggle: {len(kaggle)} rows ({kaggle.stroke.sum()} positives)")
print(f"NHANES positives available: {len(nh_pos)}")
print(f"NHANES negatives available: {len(nh_neg)}")

# ---------- splits (Kaggle-only test set, same for all variants) ----------
k_train, k_test = train_test_split(
    kaggle, test_size=0.20, random_state=RNG, stratify=kaggle["stroke"])
print(f"\nKaggle train: {len(k_train)} ({k_train.stroke.sum()} pos)")
print(f"Kaggle test : {len(k_test)} ({k_test.stroke.sum()} pos)  <-- evaluation set for all variants")

# ---------- build the three training sets ----------
nh_neg_sample = nh_neg.sample(n=3000, random_state=RNG)

train_sets = {
    "Baseline (Kaggle only)"        : k_train,
    "Variant A (+ NHANES positives)": pd.concat([k_train, nh_pos], ignore_index=True),
    "Variant B (+ pos + 3k negs)"   : pd.concat([k_train, nh_pos, nh_neg_sample], ignore_index=True),
}

# ---------- preprocessor ----------
def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), FEATURES_CAT),
        ("num", StandardScaler(), FEATURES_NUM),
    ])

# ---------- models ----------
def models():
    return {
        "LogReg": LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1, random_state=RNG),
        "GradBoost": GradientBoostingClassifier(random_state=RNG),
    }

# ---------- threshold tuning: maximize F1 on the training set's holdout slice ----------
def best_f1_threshold(y_true, proba):
    p, r, thr = precision_recall_curve(y_true, proba)
    f1 = 2*p*r / (p+r+1e-9)
    i = f1[:-1].argmax()
    return float(thr[i]), float(p[i]), float(r[i]), float(f1[i])

# ---------- evaluate ----------
X_test, y_test = k_test[FEATURES], k_test["stroke"].values
results = []

for variant_name, train_df in train_sets.items():
    # internal val split for threshold tuning
    tr, va = train_test_split(train_df, test_size=0.20, random_state=RNG,
                              stratify=train_df["stroke"])
    X_tr, y_tr = tr[FEATURES], tr["stroke"].values
    X_va, y_va = va[FEATURES], va["stroke"].values

    print(f"\n========== {variant_name}  (train rows={len(tr)}, pos={y_tr.sum()}) ==========")
    for mname, model in models().items():
        pipe = Pipeline([("pre", make_preprocessor()), ("clf", model)])
        pipe.fit(X_tr, y_tr)

        # tune threshold on validation
        va_proba = pipe.predict_proba(X_va)[:, 1]
        thr, *_ = best_f1_threshold(y_va, va_proba)

        # evaluate on KAGGLE TEST SET
        te_proba = pipe.predict_proba(X_test)[:, 1]
        te_pred  = (te_proba >= thr).astype(int)

        acc  = accuracy_score(y_test, te_pred)
        prec = precision_score(y_test, te_pred, zero_division=0)
        rec  = recall_score(y_test, te_pred, zero_division=0)
        f1   = f1_score(y_test, te_pred, zero_division=0)
        auc  = roc_auc_score(y_test, te_proba)
        ap   = average_precision_score(y_test, te_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, te_pred).ravel()

        results.append({
            "variant": variant_name, "model": mname,
            "threshold": round(thr, 4),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": auc, "pr_auc": ap,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
        print(f"  {mname:13s} thr={thr:.3f}  acc={acc:.3f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}  (TP={tp} FP={fp} FN={fn} TN={tn})")

# ---------- summary ----------
df = pd.DataFrame(results)
df.to_csv("training_comparison.csv", index=False)
print("\n\n================= SUMMARY (sorted by F1 on Kaggle test set) =================")
print(df.sort_values("f1", ascending=False)[
    ["variant","model","threshold","precision","recall","f1","roc_auc","pr_auc"]
].round(3).to_string(index=False))

print("\nBest by F1:")
best = df.sort_values("f1", ascending=False).iloc[0]
print(f"  {best['variant']} / {best['model']}")
print(f"  Precision={best['precision']:.3f}  Recall={best['recall']:.3f}  F1={best['f1']:.3f}  PR-AUC={best['pr_auc']:.3f}")
print("\nBest by Precision (with Recall ≥ 0.4):")
sub = df[df["recall"] >= 0.4]
if len(sub):
    best_p = sub.sort_values("precision", ascending=False).iloc[0]
    print(f"  {best_p['variant']} / {best_p['model']}")
    print(f"  Precision={best_p['precision']:.3f}  Recall={best_p['recall']:.3f}  F1={best_p['f1']:.3f}")
