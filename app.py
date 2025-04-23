# app.py
# Streamlit app for Heart Failure Survival Prediction - v7 (Re-Add Image)

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import warnings

# --- Page Config ---
st.set_page_config(
    page_title="HF Risk Assessor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Injection v7 ---
st.markdown("""
<style>
    /* Basic Dark Theme */
    body { color: #E0E0E0; background-color: #1E1E1E; }
    .main .block-container {
        padding-top: 2rem; padding-bottom: 3rem; padding-left: 3rem; padding-right: 3rem;
        background-color: #2C2C2C; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); color: #E0E0E0;
    }
    [data-testid="stSidebar"] {
        background-color: #1A1A1A; padding: 15px; border-right: 1px solid #444444;
    }
    [data-testid="stSidebar"] * { color: #CCCCCC; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #58A6FF; }
    [data-testid="stSidebar"] .stMarkdown a { color: #58A6FF; text-decoration: none; }
    [data-testid="stSidebar"] .stMarkdown a:hover { color: #82BFFF; text-decoration: underline; }
    [data-testid="stSidebar"] .stCaption { color: #AAAAAA; }
    [data-testid="stSidebar"] img { border-radius: 8px; } /* Style image in sidebar */
    h1, h2 { color: #E0E0E0; }
    h3 { color: #CCCCCC; }
    .stMarkdown { color: #E0E0E0; }
    div[data-testid="stButton"] > button:first-child { /* Main Button Style */
        background-image: linear-gradient(to right, #0d6efd 0%, #0a58ca 51%, #0d6efd 100%);
        color: white; font-weight: bold; border-radius: 8px; border: none; padding: 12px 28px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); transition: all 0.3s ease-in-out; background-size: 200% auto;
        display: block; margin: auto;
    }
    div[data-testid="stButton"] > button:first-child:hover { background-position: right center; box-shadow: 0 7px 14px rgba(0, 0, 0, 0.2); transform: translateY(-1px); }
    div[data-testid="stButton"] > button:first-child:active { transform: translateY(1px); box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1); }
    div[data-testid="stMetric"] { background-color: #333333; border: 1px solid #555555; border-radius: 8px; padding: 15px; text-align: center; }
    div[data-testid="stMetric"] > label { color: #AAAAAA; font-weight: 500; font-size: 0.95em;}
    div[data-testid="stMetric"] > div[data-testid="stMetricValue"] { color: #58A6FF; font-size: 1.8em; font-weight: 600;}
    .risk-category-box { border: 2px solid; border-radius: 8px; padding: 10px; text-align: center; margin-top: 5px; font-weight: bold; font-size: 1.1em; }
    .risk-very-low, .risk-low { border-color: #34C759; color: #34C759; }
    .risk-moderate { border-color: #FF9500; color: #FF9500; }
    .risk-high, .risk-very-high { border-color: #FF3B30; color: #FF3B30; }
    [data-testid="stExpander"] { border: 1px solid #555555; border-radius: 8px; background-color: #333333 !important; margin-top: 1rem; }
    [data-testid="stExpander"] * { color: #E0E0E0 !important; }
    [data-testid="stExpander"] summary { color: #E0E0E0 !important; font-weight: bold; }
    [data-testid="stExpander"] summary:hover { color: #58A6FF !important; }
    [data-testid="stExpander"] a { color: #58A6FF !important; }
    .disclaimer-box { background-color: rgba(255, 193, 7, 0.1); border-left: 6px solid #ffc107; padding: 1rem; border-radius: 8px; margin-top: 1.5rem; }
    .disclaimer-box h3 { color: #A07400; margin-top: 0; font-size: 1.1em; }
    .disclaimer-box p { color: #A07400; margin-bottom: 0; font-size: 0.9em; }

</style>
""", unsafe_allow_html=True)

# --- Configuration ---
MODEL_PATH = 'final_svc_model.joblib'
SCALER_PATH = 'scaler.joblib'
EXPECTED_FEATURES = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']

# --- Load Artifacts ---
@st.cache_resource
def load_model_scaler(model_path, scaler_path):
    model, scaler = None, None
    try: model = joblib.load(model_path)
    except Exception as e: st.sidebar.error(f"❌ Model Load Error: {e}")
    try: scaler = joblib.load(scaler_path)
    except Exception as e: st.sidebar.error(f"❌ Scaler Load Error: {e}")
    return model, scaler
model, scaler = load_model_scaler(MODEL_PATH, SCALER_PATH)

# --- Prediction Function ---
def predict_survival_app(patient_data, loaded_model, loaded_scaler):
    if loaded_model is None or loaded_scaler is None: return None
    try:
        input_data_filtered = {k: float(patient_data[k]) for k in EXPECTED_FEATURES}
        input_df = pd.DataFrame([input_data_filtered])[EXPECTED_FEATURES]
        input_scaled = loaded_scaler.transform(input_df)
        probability = loaded_model.predict_proba(input_scaled)[0]
        return float(probability[1]) # P(Death)
    except Exception: return None

# --- Streamlit App UI ---
# --- Sidebar ---
with st.sidebar:
    # Adding image back
    IMAGE_URL = "https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80"
    st.image(IMAGE_URL, use_container_width=True, caption="Image: Unsplash")
    st.divider()

    st.title("⚕️ Patient Data")
    st.caption("Adjust clinical parameters:")
    age = st.slider('Age (years)', 40, 95, 60, help="Patient's age.")
    cpk = st.number_input('CPK (mcg/L)', min_value=20, max_value=8000, value=580, step=10, help="Creatinine Phosphokinase level.")
    ef = st.slider('Ejection Fraction (%)', 10, 80, 35, help="Heart's pumping efficiency.")
    sc = st.number_input('Serum Creatinine (mg/dL)', min_value=0.5, max_value=10.0, value=1.4, step=0.1, format="%.1f", help="Kidney function indicator.")
    ss = st.slider('Serum Sodium (mEq/L)', 110, 150, 136, help="Blood sodium level.")
    st.markdown("---")
    st.subheader("Project Team Leaders")
    st.markdown(
        "*   [Dr. Shashank Neupane](https://www.linkedin.com/in/shashankneupane131)\n"        "*   [Dr. Prasamsa Pudasaini](https://www.linkedin.com/in/prasamsapudasaini77)\n\n"        "*In collaboration with the project team.*"
    )
    st.markdown("---")
    st.subheader("Feature Impact")
    st.caption("(Relative importance score)")
    importance_data = {'Feature': ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'creatinine_phosphokinase'], 'Importance': [abs(0.1480), abs(0.0472), abs(0.0366), abs(0.0269), abs(-0.0115)]}
    importance_df = pd.DataFrame(importance_data).rename(columns={'Importance':'Relative Impact'})
    importance_df = importance_df.sort_values(by='Relative Impact', ascending=False)
    st.bar_chart(importance_df, height=200)
    st.caption("Higher bar indicates greater average impact.")

# --- Main Page ---
st.title("Heart Failure Survival Risk Assessment")
st.subheader("Machine Learning-Based Predictive Tool")
st.markdown("Use the sidebar to input patient data, then click 'Assess Risk'.")
input_data = {'age': age, 'creatinine_phosphokinase': cpk, 'ejection_fraction': ef, 'serum_creatinine': sc, 'serum_sodium': ss}
with st.expander("View Current Input Values"):
    st.dataframe(pd.DataFrame([input_data]).T.rename(columns={0: 'Value'}))
predict_button = st.button('🩺 Assess Survival Risk', use_container_width=True, help="Calculate risk based on sidebar inputs")
st.divider()
prediction_placeholder = st.container()
if predict_button:
    if model is not None and scaler is not None:
        with st.spinner('⏳ Calculating Risk Score...'):
            time.sleep(0.6)
            prob_death = predict_survival_app(input_data, model, scaler)
        if prob_death is not None:
            with prediction_placeholder:
                st.subheader("📊 Risk Assessment Profile")
                if prob_death < 0.1: risk_level = "Very Low"; icon = "✅"
                elif prob_death < 0.3: risk_level = "Low"; icon = "✅"
                elif prob_death < 0.5: risk_level = "Moderate"; icon = "⚠️"
                elif prob_death < 0.7: risk_level = "High"; icon = "🚨"
                else: risk_level = "Very High"; icon = "🚨"
                res_col1, res_col2 = st.columns([1, 1])
                with res_col1: st.metric(label="Estimated Probability of Mortality*", value=f"{prob_death:.1%}")
                with res_col2:
                    css_class_map = {"Very Low": "risk-very-low", "Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high", "Very High": "risk-very-high"}
                    st.markdown("**Risk Category:**")
                    st.markdown(f"<div class='risk-category-box {css_class_map[risk_level]}'>{icon} {risk_level}</div>", unsafe_allow_html=True)
                if risk_level in ["Very Low", "Low"]: st.balloons()
                st.markdown("**Interpretation:**")
                interp_text_base = f"The model estimates a **{prob_death:.1%} probability** of a death event (relative to the study's follow-up). This corresponds to a **'{risk_level}'** risk category based on the learned data patterns. "
                interp_text_detail = {"Very Low": "Suggests a risk substantially lower than the study average.","Low": "Suggests a risk lower than the study average.","Moderate": "Suggests risk near the study average. Standard clinical considerations apply.","High": "Suggests risk higher than the study average. Enhanced clinical vigilance may be warranted.","Very High": "Suggests risk substantially higher than the study average. Strongly indicates potential need for closer clinical review."}.get(risk_level, "")
                if risk_level in ["Very Low", "Low"]: st.success(f"{interp_text_base} {interp_text_detail}")
                elif risk_level == "Moderate": st.warning(f"{interp_text_base} {interp_text_detail}")
                else: st.error(f"{interp_text_base} {interp_text_detail}")
                st.caption("*Relative probability based on study data. Risk categories: Very Low (<10%), Low (10-30%), Moderate (30-50%), High (50-70%), Very High (>=70%).")
    else:
        with prediction_placeholder: st.error("⛔ Cannot predict. Model or scaler failed to load.")
st.divider()
st.markdown(
    '<div class="disclaimer-box">'
    '<h3><span style="font-size: 1.3em;">⚠️</span> Important Disclaimer</h3>'
    '<p>This tool provides risk estimations using a machine learning model trained on historical data. It is for informational purposes only and <strong>is NOT a substitute</strong> for professional medical judgment, diagnosis, or treatment. Clinical decisions must <strong>NEVER</strong> be based solely on this tool\'s output. <strong>Always consult a qualified healthcare professional</strong> for patient care. Model ROC AUC 0.825 indicates general ranking ability, not perfect individual prediction.</p>'
    '</div>',
    unsafe_allow_html=True
)
st.divider()
with st.expander("ℹ️ Learn More: Model Details, Data Source, and Limitations"):
    st.subheader("Model & Development")
    st.markdown(
        "*   **Model Type:** Support Vector Classifier (SVC), RBF kernel (Test ROC AUC: 0.825). Chosen for performance on complex data patterns.\n"        "*   **Features Used:** `Age`, `CPK`, `Ejection Fraction`, `Serum Creatinine`, `Serum Sodium`.\n"        "*   **Functionality:** Analyzes combined input patterns for risk stratification. Feature impacts are interactive and non-linear.\n"        "*   **Limitations:** Based on a specific study dataset (N=299); performance may vary. Provides risk estimation, not definitive prognosis."
    )
    st.subheader("Dataset Information & License")
    st.markdown(
        "*   **Source:** [UCI ML Repository: Heart Failure Clinical Records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)\n"        "*   **Citation:** Chicco D, Jurman G. *BMC Med Inform Decis Mak* 20, 16 (2020). [DOI Link](https://doi.org/10.1186/s12911-020-1023-5)\n"        "*   **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"
    )