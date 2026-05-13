Run this app

1. Keep these files in the same folder:
   - health_monitor_FINALAPP.py
   - patient_dataset_with_risk_labels.xlsx
requirements_health_monitor_lightgbm_voice_weekly_v2_fixed.txt

2. Install dependencies:
   py -m pip install -r requirements_health_monitor_lightgbm_voice_weekly_v2_fixed.txt

3. Set your Gemini API key:
   $env:GEMINI_API_KEY="your_real_gemini_api_key_here"
   $env:GEMINI_MODEL="gemini-2.5-flash"

4. Run:
   py -m streamlit run health_monitor_FINALAPP.py
