import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lightgbm import LGBMClassifier
from gtts import gTTS
from pypdf import PdfReader

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

APP_TITLE = "AI Health Monitoring Guide"
EXCEL_CANDIDATES = ["patient_dataset.xlsx", "patient_dataset_with_risk_labels.xlsx"]
SHEET_CANDIDATES = ["Daily Patient Readings", "Daily Readings + Risk Label"]
MODEL_CANDIDATES = ["health_risk_lightgbm_from_patient_dataset.joblib", "health_risk_lightgbm_model.joblib"]

RAG_DOCS = [
    {"id":"bp_rule","text":"Blood pressure can be grouped into stages. Normal is below 120 over 80. Stage 1 begins at systolic 130 to 139 or diastolic 80 to 89. Stage 2 begins at systolic 140 or diastolic 90. Crisis risk is higher above 180 over 120."},
    {"id":"glucose_rule","text":"Blood glucose below 100 is generally normal. Between 100 and 125 may indicate elevated risk. At 126 and above risk is higher. Higher sustained readings over multiple days matter more than one isolated reading."},
    {"id":"cholesterol_rule","text":"Total cholesterol below 200 is generally desirable. Between 200 and 239 is borderline high. At 240 and above risk is higher, especially if repeated over several days."},
    {"id":"trend_rule","text":"Recent trend matters. If the current reading is worse than the previous 3-day or 5-day average, or if multiple recent days were elevated, the patient may be moving into a higher risk state."},
    {"id":"weekly_summary","text":"A weekly insight should highlight what improved, what worsened, which metric was most unstable, and what pattern needs attention next week."},
    {"id":"voice","text":"Voice assistant responses should be short, clear, supportive, and easy to hear for older adults."},
    {"id":"safety","text":"If the patient has severe dizziness, fainting, chest pain, breathlessness, or very extreme readings, they should seek urgent medical care rather than rely only on monitoring."}
]

def ensure_state():
    for k, v in {"prediction_context": None, "qa_history": [], "voice_audio": None, "voice_text": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_patient_state():
    st.session_state["prediction_context"] = None
    st.session_state["qa_history"] = []
    st.session_state["voice_audio"] = None
    st.session_state["voice_text"] = None

def find_existing_file(candidates):
    for name in candidates:
        if os.path.exists(name):
            return name
        p = f"/mnt/data/{name}"
        if os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def load_excel_dataset_cached():
    path = find_existing_file(EXCEL_CANDIDATES)
    if not path:
        raise FileNotFoundError("No supported patient Excel file found.")
    last_err = None
    for sheet in SHEET_CANDIDATES:
        try:
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=1)
                if "Patient ID" not in df.columns:
                    raise ValueError("header=1 did not produce expected columns")
            except Exception:
                df = pd.read_excel(path, sheet_name=sheet)
            df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
            df = df.drop_duplicates().copy()
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).str.strip()
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.sort_values(["Patient ID", "Date"]).reset_index(drop=True)
            return df, path, sheet
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load a supported sheet from {path}: {last_err}")

def add_medical_guideline_features(df):
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

    if "Risk Label" not in df.columns:
        df["Risk Label"] = "Low"
        df.loc[df["rule_risk_score"] >= 3, "Risk Label"] = "Medium"
        df.loc[df["rule_risk_score"] >= 5, "Risk Label"] = "High"
        df.loc[(df["bp_stage"] >= 4) | (df["glucose_band"] >= 3) | ((df["rule_risk_score"] >= 7) & (df["Heart Disease"] == 1)), "Risk Label"] = "Critical"
    return df

def add_history_features(df):
    df = df.copy()
    g = df.groupby("Patient ID", group_keys=False)
    for col in ["Blood Glucose (mg/dL)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "Pulse (bpm)", "Cholesterol (mg/dL)"]:
        safe = col.replace(" (mg/dL)", "").replace(" (mmHg)", "").replace(" (bpm)", "").replace(" ", "_").replace("/", "_")
        df[f"{safe}_avg_3d_prev"] = g[col].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
        df[f"{safe}_avg_5d_prev"] = g[col].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
        df[f"{safe}_yesterday"] = g[col].shift(1)

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

@st.cache_data(show_spinner=False)
def build_model_df_cached():
    raw_df, _, _ = load_excel_dataset_cached()
    df = add_medical_guideline_features(raw_df)
    df = add_history_features(df)
    model_df = df.dropna().copy()
    feature_cols = ["Day","Age","Gender","BMI","Blood Glucose (mg/dL)","Systolic BP (mmHg)","Diastolic BP (mmHg)","Pulse (bpm)","Cholesterol (mg/dL)","Diabetes","Hypertension","Heart Disease","bp_stage","glucose_band","cholesterol_band","comorbidity_count","rule_risk_score","Blood_Glucose_avg_3d_prev","Blood_Glucose_avg_5d_prev","Systolic_BP_avg_3d_prev","Systolic_BP_avg_5d_prev","Diastolic_BP_avg_3d_prev","Diastolic_BP_avg_5d_prev","Pulse_avg_3d_prev","Pulse_avg_5d_prev","Cholesterol_avg_3d_prev","Cholesterol_avg_5d_prev","glucose_diff_vs_3d_avg","glucose_diff_vs_yesterday","systolic_diff_vs_3d_avg","systolic_diff_vs_yesterday","cholesterol_diff_vs_3d_avg","cholesterol_diff_vs_yesterday","high_glucose_days_5d","high_bp_days_5d","high_cholesterol_days_5d"]
    gender_le = LabelEncoder()
    model_df["Gender"] = gender_le.fit_transform(model_df["Gender"])
    target_le = LabelEncoder()
    target_le.fit(model_df["Risk Label"])
    metadata = {"feature_cols": feature_cols, "gender_classes": list(gender_le.classes_), "target_classes": list(target_le.classes_)}
    return model_df, target_le, metadata

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    raw_df, dataset_path, sheet_name = load_excel_dataset_cached()
    model_df, target_le, metadata = build_model_df_cached()
    model_path = find_existing_file(MODEL_CANDIDATES)
    if not model_path:
        model_path = os.path.join(os.path.dirname(dataset_path) or ".", "health_risk_lightgbm_from_patient_dataset.joblib")
        X = model_df[metadata["feature_cols"]].copy()
        y = target_le.transform(model_df["Risk Label"])
        groups = model_df["Patient ID"].copy()
        train_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(X, y, groups=groups))
        model = LGBMClassifier(n_estimators=220, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, class_weight="balanced", random_state=42, objective="multiclass", verbose=-1)
        model.fit(X.iloc[train_idx], y[train_idx])
        joblib.dump({"model": model, "metadata": metadata, "target_classes": metadata["target_classes"]}, model_path)
    return raw_df, joblib.load(model_path), dataset_path, model_path, sheet_name

@st.cache_resource(show_spinner=False)
def build_rag_index():
    texts = [d["text"] for d in RAG_DOCS]
    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer, vectorizer.fit_transform(texts)

def retrieve_rag(query, top_k=4):
    vectorizer, mat = build_rag_index()
    sims = cosine_similarity(vectorizer.transform([query]), mat).flatten()
    idxs = np.argsort(sims)[::-1][:top_k]
    return [dict(RAG_DOCS[i], score=float(sims[i])) for i in idxs]

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY first.")
    if genai is None:
        raise RuntimeError("Install google-genai first: py -m pip install google-genai")
    return genai.Client(api_key=api_key)

def generate_gemini_text(prompt):
    client = get_gemini_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text

def snippets_text(snippets):
    return "\n".join([f"- {x['text']}" for x in snippets])

def rag_query_from_context(today_input, pred_label, history_df, user_question=None):
    parts = [f"risk {pred_label}", f"glucose {today_input['Blood Glucose (mg/dL)']}", f"systolic {today_input['Systolic BP (mmHg)']}", f"diastolic {today_input['Diastolic BP (mmHg)']}", f"cholesterol {today_input['Cholesterol (mg/dL)']}"]
    if len(history_df) > 0:
        parts += [f"recent avg glucose {history_df['Blood Glucose (mg/dL)'].mean():.1f}", f"recent avg systolic {history_df['Systolic BP (mmHg)'].mean():.1f}", f"recent avg cholesterol {history_df['Cholesterol (mg/dL)'].mean():.1f}"]
    if user_question:
        parts.append(user_question)
    return " | ".join(parts)

def gemini_explanation(today_input, pred_label, probs, past_5_days, rag_snippets):
    return generate_gemini_text(f"""You are a careful health-monitoring assistant.
Retrieved guidance:
{snippets_text(rag_snippets)}

Today's input:
{today_input}

Past 5 days:
{past_5_days.to_dict(orient='records')}

Predicted risk:
{pred_label}

Probabilities:
{probs}

Write:
1. Explanation of result
2. Foods/meals recommended today
3. Foods to reduce or avoid
4. Safety note

Use simple English. Do not diagnose disease or change medication.""")

def generate_weekly_trend_insight(recent_7_days, rag_snippets):
    return generate_gemini_text(f"""Generate a weekly health trend insight.
Retrieved guidance:
{snippets_text(rag_snippets)}

Recent 7 days:
{recent_7_days.to_dict(orient='records')}

Write 5 short bullet points covering:
1. overall trend
2. what improved
3. what worsened
4. most unstable metric
5. what to watch next week""")

def gemini_voice_brief(today_input, pred_label, weekly_summary, rag_snippets):
    return generate_gemini_text(f"""Create a short voice-friendly health summary.
Retrieved guidance:
{snippets_text(rag_snippets)}

Today's input:
{today_input}

Predicted risk:
{pred_label}

Weekly summary:
{weekly_summary}

Rules:
- Make it short and easy to hear
- Use simple English
- Maximum 8 lines
- Sound supportive and clear for an older adult""")

def gemini_qa(question, context, history, rag_snippets):
    return generate_gemini_text(f"""You are a helpful health-monitoring Q&A assistant.
Retrieved guidance:
{snippets_text(rag_snippets)}

Patient context:
{context}

Chat history:
{history}

User question:
{question}

Rules:
- Use patient context and retrieved guidance
- Do not diagnose disease
- Do not prescribe medication changes
- Give practical answers in simple English""")


def summarize_health_text(report_text):
    clean_text = (report_text or "").strip()
    if not clean_text:
        return "No readable text was found in the uploaded document."
    clean_text = clean_text[:12000]
    prompt = f"""You are a health report summarization assistant.

Summarize the uploaded health report in simple English.

Report text:
{clean_text}

Write the output under these headings:
1. Key findings
2. Important abnormal values or observations
3. What this means in simple words
4. What to monitor next
5. Safety note

Rules:
- Keep it easy to understand
- Do not diagnose disease
- Do not change medication
- Mention when a doctor review may be helpful
"""
    return generate_gemini_text(prompt)

def summarize_health_image(file_bytes, mime_type):
    if types is None:
        raise RuntimeError("google-genai types support is not available.")
    prompt = """Summarize this uploaded health-related image or report in simple English.

Write the output under these headings:
1. Key findings
2. Important abnormal values or observations
3. What this means in simple words
4. What to monitor next
5. Safety note

Rules:
- Keep it easy to understand
- Do not diagnose disease
- Do not change medication
- Mention when a doctor review may be helpful
"""
    client = get_gemini_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = client.models.generate_content(
        model=model_name,
        contents=[
            prompt,
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
        ],
    )
    return response.text

def summarize_uploaded_health_report(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type or ""

    if mime_type.startswith("image/"):
        return summarize_health_image(file_bytes, mime_type)

    if mime_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    pass
            return summarize_health_text("\n".join(text_parts))
        except Exception as e:
            raise RuntimeError(f"Could not read PDF: {e}")

    if mime_type.startswith("text/"):
        try:
            return summarize_health_text(file_bytes.decode("utf-8", errors="ignore"))
        except Exception as e:
            raise RuntimeError(f"Could not read text file: {e}")

    raise RuntimeError("Unsupported file type. Please upload a PNG, JPG, JPEG, PDF, or TXT file.")

def text_to_speech_bytes(text, lang='en'):
    bio = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(bio)
    bio.seek(0)
    return bio.read()

def build_prediction_row(past_5_days, today_input, metadata):
    gender_map = {label: idx for idx, label in enumerate(metadata["gender_classes"])}
    row = {"Day": today_input["Day"], "Age": today_input["Age"], "Gender": gender_map[today_input["Gender"]], "BMI": today_input["BMI"], "Blood Glucose (mg/dL)": today_input["Blood Glucose (mg/dL)"], "Systolic BP (mmHg)": today_input["Systolic BP (mmHg)"], "Diastolic BP (mmHg)": today_input["Diastolic BP (mmHg)"], "Pulse (bpm)": today_input["Pulse (bpm)"], "Cholesterol (mg/dL)": today_input["Cholesterol (mg/dL)"], "Diabetes": today_input["Diabetes"], "Hypertension": today_input["Hypertension"], "Heart Disease": today_input["Heart Disease"]}
    s, d = today_input["Systolic BP (mmHg)"], today_input["Diastolic BP (mmHg)"]
    bp_stage = 4 if (s > 180 or d > 120) else 3 if (s >= 140 or d >= 90) else 2 if (s >= 130 or d >= 80) else 1 if (120 <= s <= 129 and d < 80) else 0
    g = today_input["Blood Glucose (mg/dL)"]
    glucose_band = 3 if g >= 200 else 2 if g >= 126 else 1 if g >= 100 else 0
    c = today_input["Cholesterol (mg/dL)"]
    cholesterol_band = 2 if c >= 240 else 1 if c >= 200 else 0
    comorbidity_count = today_input["Diabetes"] + today_input["Hypertension"] + today_input["Heart Disease"]
    row.update({"bp_stage": bp_stage, "glucose_band": glucose_band, "cholesterol_band": cholesterol_band, "comorbidity_count": comorbidity_count, "rule_risk_score": bp_stage + glucose_band + cholesterol_band + comorbidity_count})
    hist_map = {"Blood Glucose (mg/dL)": "Blood_Glucose", "Systolic BP (mmHg)": "Systolic_BP", "Diastolic BP (mmHg)": "Diastolic_BP", "Pulse (bpm)": "Pulse", "Cholesterol (mg/dL)": "Cholesterol"}
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
    return pd.DataFrame([row])[metadata["feature_cols"]], row

def render_prediction_context(ctx):
    st.markdown("### Current Result")
    feature_row = ctx.get("feature_row", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Label", ctx["predicted_risk"])
    c2.metric("Health Risk Score", str(feature_row.get("rule_risk_score", "-")))
    c3.metric("Comorbidity Count", str(feature_row.get("comorbidity_count", "-")))

    band_text = []
    glucose_band = feature_row.get("glucose_band", 0)
    bp_stage = feature_row.get("bp_stage", 0)
    cholesterol_band = feature_row.get("cholesterol_band", 0)

    glucose_map = {0: "Glucose: Normal", 1: "Glucose: Borderline", 2: "Glucose: High", 3: "Glucose: Very High"}
    bp_map = {0: "BP: Normal", 1: "BP: Slightly Elevated", 2: "BP: Stage 1", 3: "BP: Stage 2", 4: "BP: Crisis Risk"}
    chol_map = {0: "Cholesterol: Normal", 1: "Cholesterol: Borderline", 2: "Cholesterol: High"}

    band_text.append(glucose_map.get(glucose_band, "Glucose: -"))
    band_text.append(bp_map.get(bp_stage, "BP: -"))
    band_text.append(chol_map.get(cholesterol_band, "Cholesterol: -"))

    st.caption(" | ".join(band_text))

    tabs = st.tabs(["Explanation", "Weekly Trend", "Voice Summary"])
    with tabs[0]:
        st.write(ctx.get("explanation") or "No explanation generated yet.")
    with tabs[1]:
        st.write(ctx.get("weekly_summary") or "No weekly trend generated yet.")
    with tabs[2]:
        if st.session_state.get("voice_text"):
            st.write(st.session_state["voice_text"])
        if st.session_state.get("voice_audio"):
            st.audio(st.session_state["voice_audio"], format="audio/mp3")
        if not st.session_state.get("voice_text"):
            st.info("Voice summary not generated yet.")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_state()
    st.markdown("""<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
    div[data-testid="stMetric"] {
        background: #1e293b;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    div[data-testid="stMetric"] label {
        color: #cbd5e1 !important;
    }
    div[data-testid="stMetric"] div {
        color: white !important;
    }
    </style>""", unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption("Faster patient switching, cleaner layout, and stable Q&A state.")
    raw_df, artifact, dataset_path, model_path, sheet_name = load_or_train_model()
    metadata = artifact["metadata"]
    model = artifact["model"]

    with st.sidebar:
        st.markdown("### App Status")
        st.write(f"Dataset: `{os.path.basename(dataset_path)}`")
        st.write(f"Sheet: `{sheet_name}`")
        st.write(f"Model: `{os.path.basename(model_path)}`")
        patient_ids = sorted(raw_df["Patient ID"].astype(str).unique().tolist())
        st.selectbox("Patient ID", patient_ids, key="selected_patient_id", on_change=reset_patient_state)

    patient_id = st.session_state["selected_patient_id"]
    patient_df = raw_df[raw_df["Patient ID"].astype(str) == patient_id].sort_values("Date").reset_index(drop=True)
    if len(patient_df) < 8:
        st.error("This patient does not have enough records for history-based prediction and weekly insight.")
        st.stop()

    past_5_days = patient_df.tail(5).copy()
    recent_7_days = patient_df.tail(7).copy()
    latest = patient_df.iloc[-1]

    top1, top2 = st.columns([1.2, 1])
    with top1:
        st.subheader(f"Patient {patient_id}")
        st.dataframe(past_5_days[["Date","Day","Age","Gender","BMI","Blood Glucose (mg/dL)","Systolic BP (mmHg)","Diastolic BP (mmHg)","Pulse (bpm)","Cholesterol (mg/dL)","Diabetes","Hypertension","Heart Disease"]], use_container_width=True, hide_index=True)
    with top2:
        st.subheader("Quick Snapshot")
        a, b = st.columns(2)
        a.metric("Latest Glucose", f"{latest['Blood Glucose (mg/dL)']:.1f}")
        b.metric("Latest BP", f"{latest['Systolic BP (mmHg)']:.0f}/{latest['Diastolic BP (mmHg)']:.0f}")
        a.metric("Latest Pulse", f"{latest['Pulse (bpm)']:.1f}")
        b.metric("Latest Chol.", f"{latest['Cholesterol (mg/dL)']:.1f}")
        st.info("Changing patient ID now refreshes the visible history immediately.")

    st.markdown("### Enter Today's Input")
    with st.form("prediction_form", clear_on_submit=False):
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
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        today_input = {"Day": int(day), "Age": int(age), "Gender": gender, "BMI": float(bmi), "Blood Glucose (mg/dL)": float(glucose), "Systolic BP (mmHg)": float(systolic), "Diastolic BP (mmHg)": float(diastolic), "Pulse (bpm)": float(pulse), "Cholesterol (mg/dL)": float(cholesterol), "Diabetes": int(diabetes), "Hypertension": int(hypertension), "Heart Disease": int(heart_disease)}
        with st.spinner("Generating result..."):
            X_new, feature_row = build_prediction_row(past_5_days, today_input, metadata)
            pred_idx = int(model.predict(X_new)[0])
            pred_label = artifact["target_classes"][pred_idx]
            probs = None
            if hasattr(model, "predict_proba"):
                pr = model.predict_proba(X_new)[0]
                probs = {artifact["target_classes"][i]: float(pr[i]) for i in range(len(artifact["target_classes"]))}
            rag_snippets = retrieve_rag(rag_query_from_context(today_input, pred_label, past_5_days), top_k=4)
            try:
                explanation = gemini_explanation(today_input, pred_label, probs, past_5_days, rag_snippets)
            except Exception as e:
                explanation = f"Gemini explanation error: {e}"
            weekly_rag = retrieve_rag(rag_query_from_context(today_input, pred_label, recent_7_days, "weekly trend summary"), top_k=4)
            try:
                weekly_summary = generate_weekly_trend_insight(recent_7_days, weekly_rag)
            except Exception as e:
                weekly_summary = f"Weekly trend generation error: {e}"
            st.session_state["prediction_context"] = {"patient_id": patient_id, "today_input": today_input, "past_5_days": past_5_days.to_dict(orient="records"), "recent_7_days": recent_7_days.to_dict(orient="records"), "predicted_risk": pred_label, "probabilities": probs, "feature_row": feature_row, "explanation": explanation, "weekly_summary": weekly_summary}
            st.session_state["qa_history"] = []
            st.session_state["voice_audio"] = None
            st.session_state["voice_text"] = None

    if st.session_state["prediction_context"] is not None:
        render_prediction_context(st.session_state["prediction_context"])
        a, b = st.columns([1,1])
        with a:
            if st.button("Generate Voice Summary", use_container_width=True):
                with st.spinner("Preparing voice summary..."):
                    ctx = st.session_state["prediction_context"]
                    rag_snippets = retrieve_rag(rag_query_from_context(ctx["today_input"], ctx["predicted_risk"], pd.DataFrame(ctx["past_5_days"]), "voice summary"), top_k=4)
                    try:
                        st.session_state["voice_text"] = gemini_voice_brief(ctx["today_input"], ctx["predicted_risk"], ctx["weekly_summary"], rag_snippets)
                        st.session_state["voice_audio"] = text_to_speech_bytes(st.session_state["voice_text"])
                    except Exception as e:
                        st.session_state["voice_text"] = f"Voice summary error: {e}"
                        st.session_state["voice_audio"] = None
        with b:
            st.info("Voice summary is generated only when you click the button, so the app feels faster.")

        st.markdown("---")
        st.subheader("Q&A Assistant")
        for item in st.session_state["qa_history"]:
            with st.chat_message(item["role"]):
                st.write(item["content"])

        q = st.chat_input("Ask about risk, diet, weekly trend, or next steps")
        if q:
            st.session_state["qa_history"].append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.write(q)
            ctx = st.session_state["prediction_context"]
            with st.spinner("Generating answer..."):
                rag_snippets = retrieve_rag(rag_query_from_context(ctx["today_input"], ctx["predicted_risk"], pd.DataFrame(ctx["past_5_days"]), q), top_k=4)
                try:
                    answer = gemini_qa(q, {"patient_id": ctx["patient_id"], "today_input": ctx["today_input"], "past_5_days": ctx["past_5_days"], "recent_7_days": ctx["recent_7_days"], "predicted_risk": ctx["predicted_risk"], "probabilities": ctx["probabilities"], "initial_explanation": ctx["explanation"], "weekly_summary": ctx["weekly_summary"], "voice_text": st.session_state.get("voice_text")}, st.session_state["qa_history"][:-1], rag_snippets)
                except Exception as e:
                    answer = f"Gemini Q&A error: {e}"
            st.session_state["qa_history"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)


    st.markdown("---")
    st.subheader("Health Report Summarizer")
    st.caption("Upload a health-related image or document to get a simple summary.")
    uploaded_report = st.file_uploader(
        "Upload PNG, JPG, JPEG, PDF, or TXT",
        type=["png", "jpg", "jpeg", "pdf", "txt"],
        key="health_report_uploader"
    )

    if uploaded_report is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Summarize Uploaded Report", use_container_width=True):
                with st.spinner("Summarizing uploaded report..."):
                    try:
                        st.session_state["report_summary"] = summarize_uploaded_health_report(uploaded_report)
                    except Exception as e:
                        msg = str(e)
                        if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "quota" in msg.lower():
                            st.session_state["report_summary"] = "Report summarization is temporarily unavailable because the Gemini quota is exhausted. Please wait a bit and try again."
                        else:
                            st.session_state["report_summary"] = f"Report summarization error: {e}"
        with c2:
            st.info("This feature is useful for uploaded lab reports, scanned readings, and health-related images.")

    if st.session_state.get("report_summary"):
        st.markdown("### Uploaded Report Summary")
        st.write(st.session_state["report_summary"])

if __name__ == "__main__":
    main()
