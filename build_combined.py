"""Combine Kaggle (all rows) + NHANES stroke-positives only."""
import pandas as pd

kaggle = pd.read_csv("healthcare-dataset-stroke-data.csv")
nhanes_pos = pd.read_csv("nhanes_stroke_positives.csv")

# Align Kaggle to NHANES's 9-feature schema (drop id, work_type, Residence_type)
shared = ["gender","age","hypertension","heart_disease","ever_married",
          "avg_glucose_level","bmi","smoking_status","stroke"]

kaggle["bmi"] = pd.to_numeric(kaggle["bmi"], errors="coerce")  # "N/A" -> NaN
kaggle["source"] = "kaggle"
nhanes_pos["source"] = "nhanes"

combined = pd.concat([kaggle[shared + ["source"]],
                      nhanes_pos[shared + ["source"]]],
                     ignore_index=True)

combined.to_csv("combined_dataset.csv", index=False)

print(f"Rows: {len(combined)}")
print(f"Stroke positives: {combined['stroke'].sum()}  ({combined['stroke'].mean()*100:.2f}%)")
print(f"Stroke negatives: {(combined['stroke']==0).sum()}")
print("\nBy source × stroke:")
print(combined.groupby(['source','stroke']).size())
