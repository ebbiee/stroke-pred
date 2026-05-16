"""
Download NHANES cycles 2013-2014, 2015-2016, 2017-2018 and assemble a CSV
that matches the Kaggle stroke dataset columns where possible.

Columns produced:
    gender, age, hypertension, heart_disease, ever_married,
    avg_glucose_level, bmi, smoking_status, stroke

NHANES has no Residence_type or work_type analog in public files, so those
columns are omitted. Join key is SEQN (respondent sequence number).
"""
import io, os, sys
import pandas as pd
import requests

# (path_year_prefix, file_suffix). 2017-2020 is a combined pre-pandemic cycle
# that uses a "P_" prefix and no letter suffix.
CYCLES = {
    "2013-2014":      ("2013", "H",  ""),
    "2015-2016":      ("2015", "I",  ""),
    "2017-2020 (P_)": ("2017", "",   "P_"),
}
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{prefix}{name}{suffix}.xpt"
FILES = ["DEMO", "MCQ", "BPQ", "BMX", "SMQ", "GLU"]

os.makedirs("nhanes_raw", exist_ok=True)

def fetch(year, name, suffix, prefix):
    fname = f"{prefix}{name}{('_'+suffix) if suffix else ''}.xpt"
    url = BASE.format(year=year, name=name, suffix=('_'+suffix) if suffix else '', prefix=prefix)
    local = f"nhanes_raw/{fname}"
    if not os.path.exists(local) or os.path.getsize(local) < 5000:
        r = requests.get(url, timeout=120)
        ct = r.headers.get("content-type", "")
        if r.status_code != 200 or "html" in ct.lower():
            print(f"  [skip] {url} -> HTTP {r.status_code} {ct}")
            return None
        with open(local, "wb") as f:
            f.write(r.content)
    return pd.read_sas(local, format="xport")

def process_cycle(label, year, suffix, prefix):
    print(f"=== Cycle {label} ===")
    parts = {}
    for name in FILES:
        df = fetch(year, name, suffix, prefix)
        if df is None:
            print(f"  missing {prefix}{name}{suffix}")
            continue
        df.columns = [c.upper() for c in df.columns]
        parts[name] = df
        print(f"  {prefix}{name}{('_'+suffix) if suffix else ''}: {df.shape}")

    demo_cols = ["SEQN", "RIAGENDR", "RIDAGEYR"]
    if "DMDMARTL" in parts["DEMO"].columns:
        demo = parts["DEMO"][demo_cols + ["DMDMARTL"]].copy()
    elif "DMDMARTZ" in parts["DEMO"].columns:
        # 2017-2020 collapses categories: 1=Married/Living w partner, 2=Widowed/Divorced/Separated, 3=Never married
        demo = parts["DEMO"][demo_cols + ["DMDMARTZ"]].copy()
        demo["DMDMARTL"] = demo["DMDMARTZ"].map({1: 1, 2: 2, 3: 5})
    else:
        demo = parts["DEMO"][demo_cols].copy()
        demo["DMDMARTL"] = None
    mcq  = parts["MCQ"][["SEQN", "MCQ160F"] + (["MCQ160B"] if "MCQ160B" in parts["MCQ"].columns else []) + (["MCQ160C"] if "MCQ160C" in parts["MCQ"].columns else [])].copy()
    bpq  = parts["BPQ"][["SEQN", "BPQ020"]].copy()
    bmx  = parts["BMX"][["SEQN", "BMXBMI"]].copy()
    smq  = parts["SMQ"][["SEQN", "SMQ020"] + (["SMQ040"] if "SMQ040" in parts["SMQ"].columns else [])].copy()
    glu  = parts["GLU"][["SEQN", "LBXGLU"]].copy() if "GLU" in parts else pd.DataFrame({"SEQN": [], "LBXGLU": []})

    df = demo.merge(mcq, on="SEQN", how="left") \
             .merge(bpq, on="SEQN", how="left") \
             .merge(bmx, on="SEQN", how="left") \
             .merge(smq, on="SEQN", how="left") \
             .merge(glu, on="SEQN", how="left")

    # gender: 1=Male, 2=Female
    df["gender"] = df["RIAGENDR"].map({1: "Male", 2: "Female"})
    df["age"] = df["RIDAGEYR"]
    # marital: 1=Married,2=Widowed,3=Divorced,4=Separated,5=Never married,6=Living w partner
    df["ever_married"] = df["DMDMARTL"].map(
        {1: "Yes", 2: "Yes", 3: "Yes", 4: "Yes", 5: "No", 6: "Yes"})
    # MCQ160F = stroke
    df["stroke"] = (df["MCQ160F"] == 1).astype(int)
    # heart_disease: CHF (MCQ160B) OR CHD (MCQ160C)
    hd = pd.Series(0, index=df.index)
    if "MCQ160B" in df.columns:
        hd = hd | (df["MCQ160B"] == 1).astype(int)
    if "MCQ160C" in df.columns:
        hd = hd | (df["MCQ160C"] == 1).astype(int)
    df["heart_disease"] = hd
    # hypertension: BPQ020 == 1 (ever told had high BP)
    df["hypertension"] = (df["BPQ020"] == 1).astype(int)
    df["avg_glucose_level"] = df["LBXGLU"]
    df["bmi"] = df["BMXBMI"]
    # smoking: SMQ020 ever smoked 100 cigs; SMQ040 now (1/2 = some, 3 = not at all)
    def smoke(row):
        ever = row.get("SMQ020")
        now  = row.get("SMQ040")
        if ever == 2: return "never smoked"
        if ever == 1 and now in (1, 2): return "smokes"
        if ever == 1 and now == 3: return "formerly smoked"
        return "Unknown"
    df["smoking_status"] = df.apply(smoke, axis=1)
    df["cycle"] = label

    out = df[["gender","age","hypertension","heart_disease","ever_married",
              "avg_glucose_level","bmi","smoking_status","stroke","cycle"]]
    # MCQ160F is only asked of 20+; restrict so stroke=0 means actually answered "No"
    answered = df["MCQ160F"].isin([1.0, 2.0])
    out = out.loc[answered].reset_index(drop=True)
    print(f"  cycle rows (answered stroke Q): {len(out)} ; positives: {out['stroke'].sum()}")
    return out

frames = []
for label, (year, suffix, prefix) in CYCLES.items():
    try:
        frames.append(process_cycle(label, year, suffix, prefix))
    except Exception as e:
        print(f"  ERROR {label}: {e}")

big = pd.concat(frames, ignore_index=True)
big.to_csv("nhanes_stroke_combined.csv", index=False)
print(f"\nSaved nhanes_stroke_combined.csv  rows={len(big)}  positives={big['stroke'].sum()}  ({big['stroke'].mean()*100:.2f}%)")

# Stroke-positive only file (the people-with-stroke data the user asked for)
pos = big[big["stroke"] == 1].copy()
pos.to_csv("nhanes_stroke_positives.csv", index=False)
print(f"Saved nhanes_stroke_positives.csv  rows={len(pos)}")
print("\nPreview of positives:")
print(pos.head(10).to_string(index=False))
