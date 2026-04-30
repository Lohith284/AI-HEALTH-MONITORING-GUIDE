
"""
PyCaret ALL models for patient_dataset_with_risk_labels.xlsx
Target: Risk Label
Uses:
- today's input columns already present in dataset
- 3-day and 5-day average features computed per patient
- medical-guideline-inspired flag features
- 70 / 20 / 10 patient-wise split
- evaluates all available PyCaret classification models on TRAIN / VALIDATION / TEST separately

Run:
    py -3.11 pycaret_risklabel_70_20_10_all_models.py
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pycaret.classification import ClassificationExperiment

warnings.filterwarnings("ignore")

FILE_PATH = Path("patient_dataset_with_risk_labels.xlsx")
SHEET_NAME = "Daily Readings + Risk Label"
TARGET_COL = "Risk Label"
GROUP_COL = "Patient ID"
DATE_COL = "Date"

TRAIN_OUT = Path("risk_train_70.csv")
VAL_OUT = Path("risk_validation_20.csv")
TEST_OUT = Path("risk_test_10.csv")

CV_LEADERBOARD_OUT = Path("pycaret_risk_cv_leaderboard_70_20_10.csv")
TRAIN_RESULTS_OUT = Path("pycaret_risk_train_results.csv")
VAL_RESULTS_OUT = Path("pycaret_risk_validation_results.csv")
TEST_RESULTS_OUT = Path("pycaret_risk_test_results.csv")
BEST_MODEL_INFO_OUT = Path("pycaret_risk_best_model_info.json")
MODEL_IDS_OUT = Path("pycaret_risk_model_ids_used.json")

TOP_N_MODELS_TO_TEST = 10


def clean_column_name(col: str) -> str:
    col = str(col).replace("\n", " ")
    col = " ".join(col.split())
    return col.strip()


def load_data() -> pd.DataFrame:
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File not found: {FILE_PATH.resolve()}")

    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=1)
    df.columns = [clean_column_name(c) for c in df.columns]
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace({
        "": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "na": np.nan,
        "null": np.nan,
        "None": np.nan,
    })

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found. Columns: {df.columns.tolist()}")

    df = df.dropna(subset=[TARGET_COL, GROUP_COL])
    return df


def add_history_and_guideline_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Try to detect likely column names
    candidates = {c.lower(): c for c in df.columns}

    def find_col(options):
        for opt in options:
            if opt.lower() in candidates:
                return candidates[opt.lower()]
        return None

    fasting_glucose_col = find_col(["Fasting Glucose", "Fasting Blood Sugar", "Glucose", "Blood Sugar"])
    postmeal_glucose_col = find_col(["Post Meal Glucose", "Post-Meal Glucose", "Postprandial Glucose"])
    systolic_col = find_col(["Systolic BP", "Systolic", "BP Systolic"])
    diastolic_col = find_col(["Diastolic BP", "Diastolic", "BP Diastolic"])
    cholesterol_col = find_col(["Cholesterol", "Total Cholesterol"])
    pulse_col = find_col(["Pulse", "Heart Rate"])
    medicine_col = find_col(["Medicine Taken", "Medication Taken", "Med Taken", "Took Medicine"])
    symptoms_col = find_col(["Symptoms", "Symptom"])

    # convert likely numeric columns
    for col in [fasting_glucose_col, postmeal_glucose_col, systolic_col, diastolic_col, cholesterol_col, pulse_col, medicine_col]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if DATE_COL in df.columns:
        df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    else:
        df = df.sort_values([GROUP_COL]).reset_index(drop=True)

    g = df.groupby(GROUP_COL, group_keys=False)

    # history averages
    for col in [fasting_glucose_col, postmeal_glucose_col, systolic_col, diastolic_col, pulse_col, cholesterol_col]:
        if col is not None:
            df[f"{col}_avg_3d"] = g[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            df[f"{col}_avg_5d"] = g[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            df[f"{col}_diff_vs_3d"] = df[col] - df[f"{col}_avg_3d"]

    # medical-guideline-inspired features
    if fasting_glucose_col is not None:
        df["glucose_flag"] = pd.cut(
            df[fasting_glucose_col],
            bins=[-np.inf, 99, 125, np.inf],
            labels=["normal", "prediabetes", "diabetes"]
        ).astype(str)

    if systolic_col is not None and diastolic_col is not None:
        def bp_category(row):
            s = row[systolic_col]
            d = row[diastolic_col]
            if pd.isna(s) or pd.isna(d):
                return np.nan
            if s > 180 or d > 120:
                return "crisis"
            elif s >= 140 or d >= 90:
                return "stage2"
            elif s >= 130 or d >= 80:
                return "stage1"
            elif s < 120 and d < 80:
                return "normal"
            else:
                return "elevated"
        df["bp_flag"] = df.apply(bp_category, axis=1)

    if cholesterol_col is not None:
        df["cholesterol_flag"] = pd.cut(
            df[cholesterol_col],
            bins=[-np.inf, 199, 239, np.inf],
            labels=["desirable", "borderline_high", "high"]
        ).astype(str)

    if medicine_col is not None:
        df["missed_meds_last_5"] = g[medicine_col].transform(lambda s: (s.shift(1).fillna(1).eq(0)).rolling(5, min_periods=1).sum())

    if symptoms_col is not None:
        df[symptoms_col] = df[symptoms_col].fillna("none").astype(str)

    if DATE_COL in df.columns:
        df["day_of_week"] = df[DATE_COL].dt.dayofweek
        df["day_of_month"] = df[DATE_COL].dt.day
        df["month"] = df[DATE_COL].dt.month

    return df


def split_70_20_10(df: pd.DataFrame):
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df[GROUP_COL]))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, train_size=(20/30), random_state=42)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df[GROUP_COL]))
    val_df = temp_df.iloc[val_idx_rel].copy()
    test_df = temp_df.iloc[test_idx_rel].copy()

    return train_df, val_df, test_df


def get_pred_col(pred_df: pd.DataFrame) -> str:
    for c in ["prediction_label", "Label", "prediction"]:
        if c in pred_df.columns:
            return c
    raise ValueError(f"Prediction column not found. Columns: {pred_df.columns.tolist()}")


def evaluate_split(exp: ClassificationExperiment, model, split_df: pd.DataFrame):
    pred_df = exp.predict_model(model, data=split_df, verbose=False)
    pred_col = get_pred_col(pred_df)

    y_true = split_df[TARGET_COL].astype(str)
    y_pred = pred_df[pred_col].astype(str)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics, pred_df


print("Loading data...")
df = load_data()
print("Original shape:", df.shape)

df = add_history_and_guideline_features(df)
print("Shape after feature engineering:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df[TARGET_COL].value_counts(dropna=False))

train_df, val_df, test_df = split_70_20_10(df)

train_patients = set(train_df[GROUP_COL].astype(str).unique())
val_patients = set(val_df[GROUP_COL].astype(str).unique())
test_patients = set(test_df[GROUP_COL].astype(str).unique())

print("\nSplit sizes:")
print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))

print("\nPatient counts:")
print("Train patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Test patients:", len(test_patients))

print("\nPatient overlap:")
print("Train-Val:", len(train_patients.intersection(val_patients)))
print("Train-Test:", len(train_patients.intersection(test_patients)))
print("Val-Test:", len(val_patients.intersection(test_patients)))

train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

ignore_features = [GROUP_COL]
if DATE_COL in train_df.columns:
    ignore_features.append(DATE_COL)

exp = ClassificationExperiment()
exp.setup(
    data=train_df,
    target=TARGET_COL,
    session_id=42,
    train_size=0.999,
    fold=5,
    fold_strategy="stratifiedkfold",
    imputation_type="simple",
    numeric_imputation="median",
    categorical_imputation="mode",
    ignore_features=ignore_features,
    remove_multicollinearity=False,
    normalize=False,
    transformation=False,
    verbose=True,
    html=False,
    use_gpu=False,
)

models_table = exp.models()
model_ids = models_table.index.tolist()

with open(MODEL_IDS_OUT, "w", encoding="utf-8") as f:
    json.dump(model_ids, f, indent=2)

print("\nRunning compare_models...")
best_model = exp.compare_models(
    include=model_ids,
    sort="F1",
    turbo=False,
    n_select=1,
    cross_validation=True,
    errors="ignore",
)

cv_leaderboard = exp.get_leaderboard()
cv_leaderboard.to_csv(CV_LEADERBOARD_OUT, index=False)

train_results = []
val_results = []
test_results = []

best_test_f1 = -1.0
best_test_model_name = None

for model_id in model_ids:
    try:
        print(f"\nTraining model: {model_id}")
        model = exp.create_model(model_id, cross_validation=False, verbose=False)

        train_metrics, _ = evaluate_split(exp, model, train_df)
        val_metrics, _ = evaluate_split(exp, model, val_df)
        test_metrics, _ = evaluate_split(exp, model, test_df)

        train_results.append({"ModelID": model_id, **{k: round(v, 6) for k, v in train_metrics.items()}})
        val_results.append({"ModelID": model_id, **{k: round(v, 6) for k, v in val_metrics.items()}})
        test_results.append({"ModelID": model_id, **{k: round(v, 6) for k, v in test_metrics.items()}})

        if test_metrics["F1_weighted"] > best_test_f1:
            best_test_f1 = test_metrics["F1_weighted"]
            best_test_model_name = model_id

    except Exception as e:
        print(f"Skipping {model_id}: {e}")

pd.DataFrame(train_results).sort_values(by=["F1_weighted", "Accuracy"], ascending=False).to_csv(TRAIN_RESULTS_OUT, index=False)
pd.DataFrame(val_results).sort_values(by=["F1_weighted", "Accuracy"], ascending=False).to_csv(VAL_RESULTS_OUT, index=False)
pd.DataFrame(test_results).sort_values(by=["F1_weighted", "Accuracy"], ascending=False).to_csv(TEST_RESULTS_OUT, index=False)

best_info = {
    "target": TARGET_COL,
    "best_model_on_test_f1": best_test_model_name,
    "best_test_f1_weighted": best_test_f1,
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "train_patients": int(len(train_patients)),
    "validation_patients": int(len(val_patients)),
    "test_patients": int(len(test_patients)),
    "ignored_features": ignore_features,
}

with open(BEST_MODEL_INFO_OUT, "w", encoding="utf-8") as f:
    json.dump(best_info, f, indent=2)

print("\nSaved outputs:")
print(TRAIN_OUT.resolve())
print(VAL_OUT.resolve())
print(TEST_OUT.resolve())
print(CV_LEADERBOARD_OUT.resolve())
print(TRAIN_RESULTS_OUT.resolve())
print(VAL_RESULTS_OUT.resolve())
print(TEST_RESULTS_OUT.resolve())
print(BEST_MODEL_INFO_OUT.resolve())
print(MODEL_IDS_OUT.resolve())
print("\nDone.")
