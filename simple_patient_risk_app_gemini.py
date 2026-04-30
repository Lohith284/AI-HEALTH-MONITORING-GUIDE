import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

try:
    from google import genai
except Exception:
    genai = None

APP_TITLE = "Patient Risk Predictor (Gemini)"
DATASET_CANDIDATES = [
    "daily_monitoring_dataset_v2(2).csv",
    "daily_monitoring_dataset_v2.csv",
]
MODEL_CANDIDATES = [
    "random_forest_today_plus_past5_model.joblib",
    "daily_monitoring_dataset_v2_rf_today_plus_past5.joblib",
    "daily_monitoring_dataset_v2(2)_rf_today_plus_past5.joblib",
]

def find_existing_file(candidates):
    for name in candidates:
        if os.path.exists(name):
            return name
        p = f"/mnt/data/{name}"
        if os.path.exists(p):
            return p
    return None

def build_training_matrix(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["patient_id", "date"]).reset_index(drop=True)

    gender_le = LabelEncoder()
    symptom_le = LabelEncoder()
    risk_le = LabelEncoder()

    df["gender_enc"] = gender_le.fit_transform(df["gender"].astype(str))
    df["symptoms_enc"] = symptom_le.fit_transform(df["symptoms"].astype(str))
    df["risk_enc"] = risk_le.fit_transform(df["risk_label"].astype(str))

    g = df.groupby("patient_id", group_keys=False)

    for col in ["fasting_glucose", "post_meal_glucose", "systolic_bp", "diastolic_bp", "pulse"]:
        df[f"{col}_avg_3d_prev"] = g[col].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
        df[f"{col}_avg_5d_prev"] = g[col].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())

    df["fasting_glucose_yesterday"] = g["fasting_glucose"].shift(1)
    df["systolic_bp_yesterday"] = g["systolic_bp"].shift(1)

    df["missed_medicine_count_5d"] = g["medicine_taken"].transform(lambda s: (s.shift(1).eq(0)).rolling(5, min_periods=5).sum())
    df["high_glucose_days_5d"] = g["fasting_glucose"].transform(lambda s: (s.shift(1).ge(145)).rolling(5, min_periods=5).sum())
    df["high_bp_days_5d"] = g["systolic_bp"].transform(lambda s: (s.shift(1).ge(145)).rolling(5, min_periods=5).sum())

    def symptom_repeat_flag(series):
        vals = series.to_numpy()
        out = [None] * len(vals)
        for i in range(5, len(vals)):
            out[i] = 1 if len(set(vals[i-5:i])) <= 2 else 0
        return pd.Series(out, index=series.index)

    df["symptom_repeat_flag_5d"] = g["symptoms_enc"].apply(symptom_repeat_flag).reset_index(level=0, drop=True)

    df["glucose_diff_vs_3d_avg"] = df["fasting_glucose"] - df["fasting_glucose_avg_3d_prev"]
    df["glucose_diff_vs_yesterday"] = df["fasting_glucose"] - df["fasting_glucose_yesterday"]
    df["bp_diff_vs_3d_avg"] = df["systolic_bp"] - df["systolic_bp_avg_3d_prev"]
    df["bp_diff_vs_yesterday"] = df["systolic_bp"] - df["systolic_bp_yesterday"]

    model_df = df.dropna(subset=[
        "fasting_glucose_avg_3d_prev", "fasting_glucose_avg_5d_prev",
        "post_meal_glucose_avg_3d_prev", "post_meal_glucose_avg_5d_prev",
        "systolic_bp_avg_3d_prev", "systolic_bp_avg_5d_prev",
        "diastolic_bp_avg_3d_prev", "diastolic_bp_avg_5d_prev",
        "pulse_avg_3d_prev", "pulse_avg_5d_prev",
        "glucose_diff_vs_3d_avg", "glucose_diff_vs_yesterday",
        "bp_diff_vs_3d_avg", "bp_diff_vs_yesterday",
        "missed_medicine_count_5d", "high_glucose_days_5d", "high_bp_days_5d",
        "symptom_repeat_flag_5d",
    ]).copy()

    feature_cols = [
        "age", "gender_enc", "bmi", "diabetes", "hypertension", "heart_disease", "smoker",
        "exercise_mins", "medicine_taken", "fasting_glucose", "post_meal_glucose",
        "systolic_bp", "diastolic_bp", "pulse", "cholesterol", "symptoms_enc",
        "fasting_glucose_avg_3d_prev", "post_meal_glucose_avg_3d_prev",
        "systolic_bp_avg_3d_prev", "diastolic_bp_avg_3d_prev", "pulse_avg_3d_prev",
        "fasting_glucose_avg_5d_prev", "post_meal_glucose_avg_5d_prev",
        "systolic_bp_avg_5d_prev", "diastolic_bp_avg_5d_prev", "pulse_avg_5d_prev",
        "glucose_diff_vs_3d_avg", "glucose_diff_vs_yesterday",
        "bp_diff_vs_3d_avg", "bp_diff_vs_yesterday",
        "missed_medicine_count_5d", "high_glucose_days_5d", "high_bp_days_5d", "symptom_repeat_flag_5d",
    ]

    X = model_df[feature_cols].copy()
    y = model_df["risk_enc"].to_numpy()
    groups = model_df["patient_id"].to_numpy()

    metadata = {
        "gender_encoder_classes": list(gender_le.classes_),
        "symptom_encoder_classes": list(symptom_le.classes_),
        "risk_encoder_classes": list(risk_le.classes_),
        "feature_columns": feature_cols,
        "history_window_days": 5,
    }
    return X, y, groups, metadata

def train_and_save_model(df, model_path):
    X, y, groups, metadata = build_training_matrix(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(gss.split(X, y, groups))
    X_train, y_train = X.iloc[train_idx], y[train_idx]

    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=14,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    artifact = {"model": model, "metadata": metadata}
    joblib.dump(artifact, model_path)
    return artifact

def load_resources():
    dataset_path = find_existing_file(DATASET_CANDIDATES)
    if not dataset_path:
        raise FileNotFoundError("Dataset CSV not found.")

    model_path = find_existing_file(MODEL_CANDIDATES)
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["patient_id", "date"]).reset_index(drop=True)

    if not model_path:
        model_path = os.path.join(os.path.dirname(dataset_path) or ".", "random_forest_today_plus_past5_model.joblib")
        artifact = train_and_save_model(df, model_path)
    else:
        artifact = joblib.load(model_path)

    return df, artifact, dataset_path, model_path

def build_feature_row(past_5_days, today_input, metadata):
    past = past_5_days.tail(5).copy()
    if len(past) < 5:
        raise ValueError("Patient must have at least 5 past records.")

    gender_map = {label: idx for idx, label in enumerate(metadata["gender_encoder_classes"])}
    symptom_map = {label: idx for idx, label in enumerate(metadata["symptom_encoder_classes"])}

    if today_input["gender"] not in gender_map:
        raise ValueError(f"Unknown gender: {today_input['gender']}")
    if today_input["symptoms"] not in symptom_map:
        raise ValueError(f"Unknown symptom: {today_input['symptoms']}")

    def m(col, n=None):
        s = past[col].tail(n) if n else past[col]
        return float(pd.Series(s).mean())

    row = {
        "age": today_input["age"],
        "gender_enc": gender_map[today_input["gender"]],
        "bmi": today_input["bmi"],
        "diabetes": today_input["diabetes"],
        "hypertension": today_input["hypertension"],
        "heart_disease": today_input["heart_disease"],
        "smoker": today_input["smoker"],

        "exercise_mins": today_input["exercise_mins"],
        "medicine_taken": today_input["medicine_taken"],
        "fasting_glucose": today_input["fasting_glucose"],
        "post_meal_glucose": today_input["post_meal_glucose"],
        "systolic_bp": today_input["systolic_bp"],
        "diastolic_bp": today_input["diastolic_bp"],
        "pulse": today_input["pulse"],
        "cholesterol": today_input["cholesterol"],
        "symptoms_enc": symptom_map[today_input["symptoms"]],

        "fasting_glucose_avg_3d_prev": m("fasting_glucose", 3),
        "post_meal_glucose_avg_3d_prev": m("post_meal_glucose", 3),
        "systolic_bp_avg_3d_prev": m("systolic_bp", 3),
        "diastolic_bp_avg_3d_prev": m("diastolic_bp", 3),
        "pulse_avg_3d_prev": m("pulse", 3),

        "fasting_glucose_avg_5d_prev": m("fasting_glucose"),
        "post_meal_glucose_avg_5d_prev": m("post_meal_glucose"),
        "systolic_bp_avg_5d_prev": m("systolic_bp"),
        "diastolic_bp_avg_5d_prev": m("diastolic_bp"),
        "pulse_avg_5d_prev": m("pulse"),

        "glucose_diff_vs_3d_avg": today_input["fasting_glucose"] - m("fasting_glucose", 3),
        "glucose_diff_vs_yesterday": today_input["fasting_glucose"] - float(past.iloc[-1]["fasting_glucose"]),
        "bp_diff_vs_3d_avg": today_input["systolic_bp"] - m("systolic_bp", 3),
        "bp_diff_vs_yesterday": today_input["systolic_bp"] - float(past.iloc[-1]["systolic_bp"]),

        "missed_medicine_count_5d": int((past["medicine_taken"] == 0).sum()),
        "high_glucose_days_5d": int((past["fasting_glucose"] >= 145).sum()),
        "high_bp_days_5d": int((past["systolic_bp"] >= 145).sum()),
        "symptom_repeat_flag_5d": int(past["symptoms"].nunique() <= 2),
    }

    X_new = pd.DataFrame([row])[metadata["feature_columns"]]
    return X_new, row

def gemini_explanation(today_input, predicted_risk, probabilities, past_5_days):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY first.")
    if genai is None:
        raise RuntimeError("Install google-genai first: py -m pip install google-genai")

    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a careful health-monitoring assistant.
Do not diagnose disease.
Do not prescribe medication changes.
Explain the result in simple English.

Today's input:
{today_input}

Past 5 days:
{past_5_days.to_dict(orient="records")}

Predicted risk:
{predicted_risk}

Probabilities:
{probabilities}

Write:
1. A short explanation of the result.
2. One diet/lifestyle suggestion.
3. One safety note about when to contact a doctor.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    try:
        df, artifact, dataset_path, model_path = load_resources()
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.caption(f"Dataset: {dataset_path}")
    st.caption(f"Model: {model_path}")

    patient_ids = sorted(df["patient_id"].unique().tolist())
    patient_id = st.selectbox("Patient ID", patient_ids)

    patient_df = df[df["patient_id"] == patient_id].sort_values("date").reset_index(drop=True)
    if len(patient_df) < 5:
        st.error("This patient has fewer than 5 past records.")
        st.stop()

    past_5_days = patient_df.tail(5)
    latest = patient_df.iloc[-1]

    st.subheader("Past 5 Days")
    st.dataframe(
        past_5_days[[
            "date", "fasting_glucose", "post_meal_glucose",
            "systolic_bp", "diastolic_bp", "pulse",
            "medicine_taken", "exercise_mins", "symptoms", "risk_label"
        ]],
        use_container_width=True
    )

    st.subheader("Enter Today's Input")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=46, max_value=100, value=int(latest["age"]))
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if str(latest["gender"]) == "Female" else 1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(latest["bmi"]), step=0.1)
        diabetes = st.selectbox("Diabetes", [0, 1], index=int(latest["diabetes"]))
        hypertension = st.selectbox("Hypertension", [0, 1], index=int(latest["hypertension"]))
        heart_disease = st.selectbox("Heart disease", [0, 1], index=int(latest["heart_disease"]))
        smoker = st.selectbox("Smoker", [0, 1], index=int(latest["smoker"]))

    with c2:
        exercise_mins = st.number_input("Exercise mins", min_value=0, max_value=180, value=int(latest["exercise_mins"]))
        medicine_taken = st.selectbox("Medicine taken", [0, 1], index=int(latest["medicine_taken"]))
        fasting_glucose = st.number_input("Fasting glucose", min_value=50, max_value=400, value=int(latest["fasting_glucose"]))
        post_meal_glucose = st.number_input("Post-meal glucose", min_value=50, max_value=450, value=int(latest["post_meal_glucose"]))
        systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=260, value=int(latest["systolic_bp"]))
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=160, value=int(latest["diastolic_bp"]))

    with c3:
        pulse = st.number_input("Pulse", min_value=35, max_value=180, value=int(latest["pulse"]))
        cholesterol = st.number_input("Cholesterol", min_value=80, max_value=400, value=int(latest["cholesterol"]))
        symptom_options = sorted(df["symptoms"].astype(str).unique().tolist())
        default_symptom = str(latest["symptoms"])
        symptom_index = symptom_options.index(default_symptom) if default_symptom in symptom_options else 0
        symptoms = st.selectbox("Symptoms", symptom_options, index=symptom_index)

    if st.button("Get Risk Label"):
        today_input = {
            "age": int(age),
            "gender": gender,
            "bmi": float(bmi),
            "diabetes": int(diabetes),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "smoker": int(smoker),
            "exercise_mins": int(exercise_mins),
            "medicine_taken": int(medicine_taken),
            "fasting_glucose": int(fasting_glucose),
            "post_meal_glucose": int(post_meal_glucose),
            "systolic_bp": int(systolic_bp),
            "diastolic_bp": int(diastolic_bp),
            "pulse": int(pulse),
            "cholesterol": int(cholesterol),
            "symptoms": symptoms,
        }

        X_new, feature_row = build_feature_row(past_5_days, today_input, artifact["metadata"])
        model = artifact["model"]
        pred_idx = int(model.predict(X_new)[0])
        pred_label = artifact["metadata"]["risk_encoder_classes"][pred_idx]

        probs = None
        if hasattr(model, "predict_proba"):
            pr = model.predict_proba(X_new)[0]
            classes = artifact["metadata"]["risk_encoder_classes"]
            probs = {classes[i]: float(pr[i]) for i in range(len(classes))}

        st.success(f"Risk Label: {pred_label}")
        if probs:
            st.json({k: round(v, 4) for k, v in probs.items()})

        try:
            explanation = gemini_explanation(today_input, pred_label, probs, past_5_days)
            st.subheader("Gemini Explanation")
            st.write(explanation)
        except Exception as e:
            st.error(f"Gemini error: {e}")

        with st.expander("Feature Row Used"):
            st.json(feature_row)

if __name__ == "__main__":
    main()
