import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lightgbm import LGBMClassifier

try:
    from google import genai
except Exception:
    genai = None

APP_TITLE = "AI Health Monitoring Guide (LightGBM + Gemini + RAG)"
EXCEL_CANDIDATES = ["patient_dataset_with_risk_labels.xlsx"]
SHEET_NAME = "Daily Readings + Risk Label"
MODEL_CANDIDATES = ["health_risk_lightgbm_model.joblib", "lightgbm_risk_label_history_model.joblib"]

RAG_DOCS = [
    {"id": "bp_rule", "text": "Blood pressure can be grouped into stages. Normal is below 120 over 80. Stage 1 begins at systolic 130 to 139 or diastolic 80 to 89. Stage 2 begins at systolic 140 or diastolic 90. Crisis risk is higher above 180 over 120."},
    {"id": "glucose_rule", "text": "Blood glucose readings matter over time. A fasting-style glucose below 100 is generally normal. Between 100 and 125 may indicate elevated risk. At 126 and above risk is higher, and sustained values matter more than one isolated reading."},
    {"id": "cholesterol_rule", "text": "Total cholesterol below 200 is generally desirable. Between 200 and 239 is borderline high. At 240 and above risk is higher, especially if repeated over several days."},
    {"id": "trend_rule", "text": "Recent trend is important. If the current reading is worse than the previous 3-day or 5-day average, or if multiple recent days were elevated, the patient may be moving into a higher risk state."},
    {"id": "diet_bp", "text": "For elevated blood pressure, suggest less salt, fewer packaged foods, soups, dal, vegetables, curd, fruit in moderation, and home-cooked meals. Avoid chips, pickles, very salty snacks, and heavily processed food."},
    {"id": "diet_glucose", "text": "For elevated blood glucose, suggest controlled portions, more fiber, vegetables, dal, eggs, paneer, sprouts, and unsweetened curd. Avoid sweets, sugary drinks, desserts, and large refined-carb meals."},
    {"id": "diet_cholesterol", "text": "For high cholesterol, suggest lighter meals, less fried food, more vegetables, oats, soups, and reduced oily or creamy foods."},
    {"id": "safety", "text": "If the patient has severe dizziness, fainting, chest pain, breathlessness, or very extreme readings, they should seek urgent medical care rather than rely only on monitoring."},
    {"id": "tone", "text": "Explain results in simple English for a non-technical patient or caregiver. Be supportive, practical, and avoid sounding like a formal diagnosis."},
]

def find_existing_file(candidates):
    for name in candidates:
        if os.path.exists(name):
            return name
        p = f"/mnt/data/{name}"
        if os.path.exists(p):
            return p
    return None

def load_excel_dataset():
    path = find_existing_file(EXCEL_CANDIDATES)
    if not path:
        raise FileNotFoundError("patient_dataset_with_risk_labels.xlsx not found.")
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values(["Patient ID", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, path

def add_medical_guideline_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bp_stage"] = 0
    df.loc[(df["Systolic BP (mmHg)"] >= 120) & (df["Systolic BP (mmHg)"] <= 129) & (df["Diastolic BP (mmHg)"] < 80), "bp_stage"] = 1
    df.loc[((df["Systolic BP (mmHg)"] >= 130) & (df["Systolic BP (mmHg)"] <= 139)) | ((df["Diastolic BP (mmHg)"] >= 80) & (df["Diastolic BP (mmHg)"] <= 89)), "bp_stage"] = 2
    df.loc[(df["Systolic BP (mmHg)"] >= 140) | (df["Diastolic BP (mmHg)"] >= 90), "bp_stage"] = 3
    df.loc[(df["Systolic BP (mmHg)"] > 180) | (df["Diastolic BP (mmHg)"] > 120), "bp_stage"] = 4

    df["glucose_band"] = 0
    df.loc[(df["Blood Glucose (mg/dL)"] >= 100) & (df["Blood Glucose (mg/dL)"] <= 125), "glucose_band"] = 1
    df.loc[(df["Blood Glucose (mg/dL)"] >= 126) & (df["Blood Glucose (mg/dL)"] < 200), "glucose_band"] = 2
    df.loc[df["Blood Glucose (mg/dL)"] >= 200, "glucose_band"] = 3

    df["cholesterol_band"] = 0
    df.loc[(df["Cholesterol (mg/dL)"] >= 200) & (df["Cholesterol (mg/dL)"] <= 239), "cholesterol_band"] = 1
    df.loc[df["Cholesterol (mg/dL)"] >= 240, "cholesterol_band"] = 2

    df["comorbidity_count"] = df[["Diabetes", "Hypertension", "Heart Disease"]].sum(axis=1)
    df["rule_risk_score"] = df["bp_stage"] + df["glucose_band"] + df["cholesterol_band"] + df["comorbidity_count"]
    return df

def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Patient ID", group_keys=False)
    for col in ["Blood Glucose (mg/dL)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "Pulse (bpm)", "Cholesterol (mg/dL)"]:
        safe_col = col.replace(" (mg/dL)", "").replace(" (mmHg)", "").replace(" (bpm)", "").replace(" ", "_").replace("/", "_")
        df[f"{safe_col}_avg_3d_prev"] = g[col].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
        df[f"{safe_col}_avg_5d_prev"] = g[col].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
        df[f"{safe_col}_yesterday"] = g[col].shift(1)

    df["high_glucose_days_5d"] = g["Blood Glucose (mg/dL)"].transform(lambda s: (s.shift(1).ge(126)).rolling(5, min_periods=5).sum())
    df["high_bp_days_5d"] = g["Systolic BP (mmHg)"].transform(lambda s: (s.shift(1).ge(140)).rolling(5, min_periods=5).sum())
    df["high_cholesterol_days_5d"] = g["Cholesterol (mg/dL)"].transform(lambda s: (s.shift(1).ge(240)).rolling(5, min_periods=5).sum())

    df["glucose_diff_vs_3d_avg"] = df["Blood Glucose (mg/dL)"] - df["Blood_Glucose_avg_3d_prev"]
    df["glucose_diff_vs_yesterday"] = df["Blood Glucose (mg/dL)"] - df["Blood_Glucose_yesterday"]
    df["systolic_diff_vs_3d_avg"] = df["Systolic BP (mmHg)"] - df["Systolic_BP_avg_3d_prev"]
    df["systolic_diff_vs_yesterday"] = df["Systolic BP (mmHg)"] - df["Systolic_BP_yesterday"]
    df["cholesterol_diff_vs_3d_avg"] = df["Cholesterol (mg/dL)"] - df["Cholesterol_avg_3d_prev"]
    df["cholesterol_diff_vs_yesterday"] = df["Cholesterol (mg/dL)"] - df["Cholesterol_yesterday"]
    return df

def build_model_df(df: pd.DataFrame):
    df = add_medical_guideline_features(df)
    df = add_history_features(df)
    model_df = df.dropna(subset=[
        "Blood_Glucose_avg_3d_prev", "Blood_Glucose_avg_5d_prev",
        "Systolic_BP_avg_3d_prev", "Systolic_BP_avg_5d_prev",
        "Diastolic_BP_avg_3d_prev", "Diastolic_BP_avg_5d_prev",
        "Pulse_avg_3d_prev", "Pulse_avg_5d_prev",
        "Cholesterol_avg_3d_prev", "Cholesterol_avg_5d_prev",
        "glucose_diff_vs_3d_avg", "glucose_diff_vs_yesterday",
        "systolic_diff_vs_3d_avg", "systolic_diff_vs_yesterday",
        "cholesterol_diff_vs_3d_avg", "cholesterol_diff_vs_yesterday",
        "high_glucose_days_5d", "high_bp_days_5d", "high_cholesterol_days_5d"
    ]).copy()

    feature_cols = [
        "Day", "Age", "Gender", "BMI",
        "Blood Glucose (mg/dL)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)",
        "Pulse (bpm)", "Cholesterol (mg/dL)",
        "Diabetes", "Hypertension", "Heart Disease",
        "bp_stage", "glucose_band", "cholesterol_band", "comorbidity_count", "rule_risk_score",
        "Blood_Glucose_avg_3d_prev", "Blood_Glucose_avg_5d_prev",
        "Systolic_BP_avg_3d_prev", "Systolic_BP_avg_5d_prev",
        "Diastolic_BP_avg_3d_prev", "Diastolic_BP_avg_5d_prev",
        "Pulse_avg_3d_prev", "Pulse_avg_5d_prev",
        "Cholesterol_avg_3d_prev", "Cholesterol_avg_5d_prev",
        "glucose_diff_vs_3d_avg", "glucose_diff_vs_yesterday",
        "systolic_diff_vs_3d_avg", "systolic_diff_vs_yesterday",
        "cholesterol_diff_vs_3d_avg", "cholesterol_diff_vs_yesterday",
        "high_glucose_days_5d", "high_bp_days_5d", "high_cholesterol_days_5d"
    ]

    gender_le = LabelEncoder()
    model_df["Gender"] = gender_le.fit_transform(model_df["Gender"])

    target_le = LabelEncoder()
    model_df["Risk_Label_Encoded"] = target_le.fit_transform(model_df["Risk Label"])

    metadata = {"feature_cols": feature_cols, "gender_classes": list(gender_le.classes_), "target_classes": list(target_le.classes_)}
    return model_df, gender_le, target_le, metadata

def train_lightgbm_model(model_df, target_le, metadata, out_path):
    X = model_df[metadata["feature_cols"]].copy()
    y = target_le.transform(model_df["Risk Label"])
    groups = model_df["Patient ID"].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, _ = next(gss.split(X, y, groups=groups))
    X_train = X.iloc[train_idx].copy()
    y_train = y[train_idx]

    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9,
        colsample_bytree=0.9, class_weight="balanced", random_state=42,
        objective="multiclass", verbose=-1
    )
    model.fit(X_train, y_train)
    artifact = {"model": model, "metadata": metadata, "target_classes": metadata["target_classes"]}
    joblib.dump(artifact, out_path)
    return artifact

def load_resources():
    raw_df, dataset_path = load_excel_dataset()
    model_df, gender_le, target_le, metadata = build_model_df(raw_df)
    model_path = find_existing_file(MODEL_CANDIDATES)
    if not model_path:
        model_path = os.path.join(os.path.dirname(dataset_path) or ".", "health_risk_lightgbm_model.joblib")
        artifact = train_lightgbm_model(model_df, target_le, metadata, model_path)
    else:
        artifact = joblib.load(model_path)
    return raw_df, model_df, artifact, dataset_path, model_path

@st.cache_resource
def build_rag_index():
    texts = [d["text"] for d in RAG_DOCS]
    vectorizer = TfidfVectorizer(stop_words="english")
    mat = vectorizer.fit_transform(texts)
    return vectorizer, mat

def retrieve_rag(query, top_k=4):
    vectorizer, mat = build_rag_index()
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, mat).flatten()
    idxs = np.argsort(sims)[::-1][:top_k]
    return [dict(RAG_DOCS[i], score=float(sims[i])) for i in idxs]

def rag_query_from_context(today_input, pred_label, past_5_days, user_question=None):
    parts = [
        f"risk {pred_label}",
        f"glucose {today_input['Blood Glucose (mg/dL)']}",
        f"systolic {today_input['Systolic BP (mmHg)']}",
        f"diastolic {today_input['Diastolic BP (mmHg)']}",
        f"cholesterol {today_input['Cholesterol (mg/dL)']}",
        f"diabetes {today_input['Diabetes']}",
        f"hypertension {today_input['Hypertension']}",
        f"heart disease {today_input['Heart Disease']}",
        f"recent avg glucose {past_5_days['Blood Glucose (mg/dL)'].mean():.1f}",
        f"recent avg systolic {past_5_days['Systolic BP (mmHg)'].mean():.1f}",
        f"recent avg cholesterol {past_5_days['Cholesterol (mg/dL)'].mean():.1f}",
    ]
    if user_question:
        parts.append(user_question)
    return " | ".join(parts)

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY first.")
    if genai is None:
        raise RuntimeError("Install google-genai first: py -m pip install google-genai")
    return genai.Client(api_key=api_key)

def generate_gemini_text(prompt):
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = get_gemini_client()
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text

def snippets_text(snippets):
    return "\n".join([f"- {x['text']}" for x in snippets])

def gemini_explanation(today_input, pred_label, probs, past_5_days, rag_snippets):
    prompt = f"""You are a careful health-monitoring assistant.

Use:
1. today's patient readings
2. the last 5 days of readings
3. the predicted risk label
4. the retrieved guidance snippets

Retrieved guidance:
{snippets_text(rag_snippets)}

Today's input:
{today_input}

Past 5 days:
{past_5_days.to_dict(orient="records")}

Predicted risk:
{pred_label}

Probabilities:
{probs}

Rules:
- Explain in simple English.
- Do not diagnose disease.
- Do not change medication.
- Give practical diet and lifestyle suggestions.
- Mention a safety note if severe symptoms or very risky readings are present.

Write:
1. Explanation of result
2. Foods/meals recommended today
3. Foods to reduce or avoid
4. Safety note
"""
    return generate_gemini_text(prompt)

def gemini_qa(question, context, history, rag_snippets):
    prompt = f"""You are a helpful health-monitoring Q&A assistant.

Retrieved guidance:
{snippets_text(rag_snippets)}

Patient context:
{context}

Chat history:
{history}

User question:
{question}

Rules:
- Use the retrieved guidance and patient context.
- Do not diagnose disease.
- Do not prescribe medication changes.
- For diet questions, give direct practical suggestions.
- Keep the answer simple and supportive.
- If severe symptoms are described, advise prompt medical help.

Answer in 4-8 sentences.
"""
    return generate_gemini_text(prompt)

def build_prediction_row(past_5_days, today_input, metadata):
    if len(past_5_days) < 5:
        raise ValueError("Patient must have at least 5 previous daily records.")
    gender_map = {label: idx for idx, label in enumerate(metadata["gender_classes"])}
    row = {
        "Day": today_input["Day"], "Age": today_input["Age"], "Gender": gender_map[today_input["Gender"]],
        "BMI": today_input["BMI"], "Blood Glucose (mg/dL)": today_input["Blood Glucose (mg/dL)"],
        "Systolic BP (mmHg)": today_input["Systolic BP (mmHg)"], "Diastolic BP (mmHg)": today_input["Diastolic BP (mmHg)"],
        "Pulse (bpm)": today_input["Pulse (bpm)"], "Cholesterol (mg/dL)": today_input["Cholesterol (mg/dL)"],
        "Diabetes": today_input["Diabetes"], "Hypertension": today_input["Hypertension"], "Heart Disease": today_input["Heart Disease"],
    }

    s, d = today_input["Systolic BP (mmHg)"], today_input["Diastolic BP (mmHg)"]
    if s > 180 or d > 120: bp_stage = 4
    elif s >= 140 or d >= 90: bp_stage = 3
    elif s >= 130 or d >= 80: bp_stage = 2
    elif 120 <= s <= 129 and d < 80: bp_stage = 1
    else: bp_stage = 0

    g = today_input["Blood Glucose (mg/dL)"]
    if g >= 200: glucose_band = 3
    elif g >= 126: glucose_band = 2
    elif g >= 100: glucose_band = 1
    else: glucose_band = 0

    c = today_input["Cholesterol (mg/dL)"]
    if c >= 240: cholesterol_band = 2
    elif c >= 200: cholesterol_band = 1
    else: cholesterol_band = 0

    comorbidity_count = today_input["Diabetes"] + today_input["Hypertension"] + today_input["Heart Disease"]
    row.update({
        "bp_stage": bp_stage, "glucose_band": glucose_band, "cholesterol_band": cholesterol_band,
        "comorbidity_count": comorbidity_count, "rule_risk_score": bp_stage + glucose_band + cholesterol_band + comorbidity_count,
    })

    hist_map = {
        "Blood Glucose (mg/dL)": "Blood_Glucose",
        "Systolic BP (mmHg)": "Systolic_BP",
        "Diastolic BP (mmHg)": "Diastolic_BP",
        "Pulse (bpm)": "Pulse",
        "Cholesterol (mg/dL)": "Cholesterol",
    }
    for original, safe in hist_map.items():
        row[f"{safe}_avg_3d_prev"] = float(past_5_days[original].tail(3).mean())
        row[f"{safe}_avg_5d_prev"] = float(past_5_days[original].tail(5).mean())

    row["glucose_diff_vs_3d_avg"] = row["Blood Glucose (mg/dL)"] - row["Blood_Glucose_avg_3d_prev"]
    row["glucose_diff_vs_yesterday"] = row["Blood Glucose (mg/dL)"] - float(past_5_days.iloc[-1]["Blood Glucose (mg/dL)"])
    row["systolic_diff_vs_3d_avg"] = row["Systolic BP (mmHg)"] - row["Systolic_BP_avg_3d_prev"]
    row["systolic_diff_vs_yesterday"] = row["Systolic BP (mmHg)"] - float(past_5_days.iloc[-1]["Systolic BP (mmHg)"])
    row["cholesterol_diff_vs_3d_avg"] = row["Cholesterol (mg/dL)"] - row["Cholesterol_avg_3d_prev"]
    row["cholesterol_diff_vs_yesterday"] = row["Cholesterol (mg/dL)"] - float(past_5_days.iloc[-1]["Cholesterol (mg/dL)"])
    row["high_glucose_days_5d"] = int((past_5_days["Blood Glucose (mg/dL)"] >= 126).sum())
    row["high_bp_days_5d"] = int((past_5_days["Systolic BP (mmHg)"] >= 140).sum())
    row["high_cholesterol_days_5d"] = int((past_5_days["Cholesterol (mg/dL)"] >= 240).sum())

    X_new = pd.DataFrame([row])[metadata["feature_cols"]]
    return X_new, row

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "prediction_context" not in st.session_state:
        st.session_state.prediction_context = None

    try:
        raw_df, model_df, artifact, dataset_path, model_path = load_resources()
    except Exception as e:
        st.error(str(e))
        st.stop()

    metadata = artifact["metadata"]
    model = artifact["model"]

    st.caption(f"Dataset: {dataset_path}")
    st.caption(f"Model: {model_path}")

    patient_ids = sorted(raw_df["Patient ID"].astype(str).unique().tolist())
    patient_id = st.selectbox("Patient ID", patient_ids)

    patient_df = raw_df[raw_df["Patient ID"].astype(str) == patient_id].sort_values("Date").reset_index(drop=True)
    if len(patient_df) < 6:
        st.error("This patient does not have enough records for 3-5 day history-based prediction.")
        st.stop()

    past_5_days = patient_df.tail(5).copy()
    latest = patient_df.iloc[-1]

    st.subheader("Past 5 Days")
    st.dataframe(
        past_5_days[["Date", "Day", "Age", "Gender", "BMI", "Blood Glucose (mg/dL)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "Pulse (bpm)", "Cholesterol (mg/dL)", "Diabetes", "Hypertension", "Heart Disease", "Risk Label"]],
        use_container_width=True
    )

    st.subheader("Enter Today's Input")
    c1, c2, c3 = st.columns(3)
    with c1:
        day = st.number_input("Day", min_value=1, max_value=366, value=int(latest["Day"]))
        age = st.number_input("Age", min_value=1, max_value=120, value=int(latest["Age"]))
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if str(latest["Gender"]) == "Female" else 1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(latest["BMI"]), step=0.1)
    with c2:
        glucose = st.number_input("Blood Glucose (mg/dL)", min_value=40.0, max_value=500.0, value=float(latest["Blood Glucose (mg/dL)"]), step=0.1)
        systolic = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=260.0, value=float(latest["Systolic BP (mmHg)"]), step=0.1)
        diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=160.0, value=float(latest["Diastolic BP (mmHg)"]), step=0.1)
        pulse = st.number_input("Pulse (bpm)", min_value=30.0, max_value=220.0, value=float(latest["Pulse (bpm)"]), step=0.1)
    with c3:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80.0, max_value=500.0, value=float(latest["Cholesterol (mg/dL)"]), step=0.1)
        diabetes = st.selectbox("Diabetes", [0, 1], index=int(latest["Diabetes"]))
        hypertension = st.selectbox("Hypertension", [0, 1], index=int(latest["Hypertension"]))
        heart_disease = st.selectbox("Heart Disease", [0, 1], index=int(latest["Heart Disease"]))

    if st.button("Get Risk Label"):
        today_input = {
            "Day": int(day), "Age": int(age), "Gender": gender, "BMI": float(bmi),
            "Blood Glucose (mg/dL)": float(glucose), "Systolic BP (mmHg)": float(systolic),
            "Diastolic BP (mmHg)": float(diastolic), "Pulse (bpm)": float(pulse),
            "Cholesterol (mg/dL)": float(cholesterol), "Diabetes": int(diabetes),
            "Hypertension": int(hypertension), "Heart Disease": int(heart_disease),
        }

        X_new, feature_row = build_prediction_row(past_5_days, today_input, metadata)
        pred_idx = int(model.predict(X_new)[0])
        pred_label = artifact["target_classes"][pred_idx]

        probs = None
        if hasattr(model, "predict_proba"):
            pr = model.predict_proba(X_new)[0]
            probs = {artifact["target_classes"][i]: float(pr[i]) for i in range(len(artifact["target_classes"]))}

        st.success(f"Risk Label: {pred_label}")
        if probs:
            st.json({k: round(v, 4) for k, v in probs.items()})

        rag_snippets = retrieve_rag(rag_query_from_context(today_input, pred_label, past_5_days), top_k=4)
        with st.expander("RAG snippets used"):
            for item in rag_snippets:
                st.write(f"- {item['text']}")

        try:
            explanation = gemini_explanation(today_input, pred_label, probs, past_5_days, rag_snippets)
            st.subheader("Gemini Explanation")
            st.write(explanation)
        except Exception as e:
            st.error(f"Gemini error: {e}")
            explanation = None

        st.session_state.prediction_context = {
            "patient_id": patient_id,
            "today_input": today_input,
            "past_5_days": past_5_days.to_dict(orient="records"),
            "predicted_risk": pred_label,
            "probabilities": probs,
            "feature_row": feature_row,
            "explanation": explanation,
        }
        st.session_state.qa_history = []

        with st.expander("Feature Row Used"):
            st.json(feature_row)

    if st.session_state.prediction_context is not None:
        st.markdown("---")
        st.subheader("Ask a Question About This Patient")
        for item in st.session_state.qa_history:
            with st.chat_message(item["role"]):
                st.write(item["content"])

        q = st.chat_input("Ask about risk, diet, trends, or what changed in the last 5 days")
        if q:
            st.session_state.qa_history.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.write(q)

            ctx = st.session_state.prediction_context
            rag_snippets = retrieve_rag(rag_query_from_context(ctx["today_input"], ctx["predicted_risk"], pd.DataFrame(ctx["past_5_days"]), q), top_k=4)

            try:
                answer = gemini_qa(q, {
                    "patient_id": ctx["patient_id"],
                    "today_input": ctx["today_input"],
                    "past_5_days": ctx["past_5_days"],
                    "predicted_risk": ctx["predicted_risk"],
                    "probabilities": ctx["probabilities"],
                    "initial_explanation": ctx["explanation"],
                }, st.session_state.qa_history[:-1], rag_snippets)
            except Exception as e:
                answer = f"Gemini error: {e}"

            st.session_state.qa_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

if __name__ == "__main__":
    main()
