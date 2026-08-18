import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
EDA_DIR = BASE_DIR / "Outputs" / "EDA_Plots"
SHAP_DIR = BASE_DIR / "Outputs" / "SHAP_Plots"

st.set_page_config(
    page_title="SmartCare Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f2f7ff;
            --panel: #ffffff;
            --panel-alt: #eef6ff;
            --primary: #0d6efd;
            --secondary: #1ea1b6;
            --success: #1f9d69;
            --warning: #f0ad4e;
            --danger: #d9534f;
            --text: #132238;
            --muted: #55708d;
        }

        .stApp {
            background: linear-gradient(135deg, #f5f9ff 0%, #eef7f8 100%);
            color: var(--text);
        }

        div[data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, #0d1b2a 0%, #143a5a 100%);
            padding: 1.2rem 0.9rem;
        }

        .stSidebar .stRadio > div {
            gap: 0.5rem;
        }

        .stSidebar .stRadio [role="radio"] {
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.55rem 0.65rem;
            color: #ebf6ff;
        }

        .stSidebar .stRadio [role="radio"] > div {
            color: #ebf6ff;
        }

        section[data-testid="stSidebar"] {
            color: #ebf6ff;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-box {
            background: linear-gradient(135deg, rgba(13,110,253,0.12), rgba(30,161,182,0.08));
            border: 1px solid rgba(13,110,253,0.18);
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(13,110,253,0.08), rgba(30,161,182,0.05));
            border: 1px solid rgba(13,110,253,0.12);
            border-radius: 14px;
            padding: 0.9rem;
        }

        .prediction-result {
            background: rgba(255,255,255,0.72);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            border: 1px solid rgba(19, 34, 56, 0.1);
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.7rem 1.2rem;
            box-shadow: 0 8px 18px rgba(13,110,253,0.15);
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 2. LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# =========================================================

@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "lr_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_columns_path = MODELS_DIR / "feature_columns.pkl"

    missing = [
        str(path) for path in [model_path, scaler_path, feature_columns_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing))

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_columns_path)
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as exc:
    st.error(
        "Required model files were not found. Please ensure the repository includes the trained artifacts in the Models folder."
    )
    st.code(str(exc), language="text")
    st.stop()

# =========================================================
# 3. CONSTANTS
# =========================================================

TRUE_NUMERIC = [
    "age",
    "admitted",
    "length_of_stay_days",
    "previous_admissions",
    "systolic_bp",
    "diastolic_bp",
    "blood_sugar_mg_dl",
    "cholesterol_mg_dl",
    "bmi",
    "lab_tests_count",
    "treatments_count",
    "previous_appointments",
    "missed_previous_appointments",
    "pulse_pressure",
    "chronic_diagnosis_flag",
    "care_intensity",
]

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# =========================================================
# 4. HELPER FUNCTION
# =========================================================

def create_model_input(
    age,
    gender,
    blood_group,
    department,
    diagnosis,
    admitted,
    room_type,
    length_of_stay_days,
    previous_admissions,
    systolic_bp,
    diastolic_bp,
    blood_sugar_mg_dl,
    cholesterol_mg_dl,
    bmi,
    lab_tests_count,
    treatments_count,
    previous_appointments,
    missed_previous_appointments,
):
    admitted_value = 1 if admitted == "Yes" else 0
    pulse_pressure = systolic_bp - diastolic_bp
    chronic_diagnosis_flag = int(diagnosis in ["Diabetes", "Hypertension"])
    care_intensity = lab_tests_count + treatments_count

    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    if age < 18:
        age_group = "Child"
    elif age < 40:
        age_group = "Adult"
    elif age < 60:
        age_group = "Middle-aged"
    else:
        age_group = "Senior"

    row = pd.DataFrame(
        [{
            "age": age,
            "gender": gender,
            "blood_group": blood_group,
            "department": department,
            "diagnosis": diagnosis,
            "previous_appointments": previous_appointments,
            "missed_previous_appointments": missed_previous_appointments,
            "admitted": admitted_value,
            "room_type": room_type,
            "length_of_stay_days": length_of_stay_days,
            "previous_admissions": previous_admissions,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "blood_sugar_mg_dl": blood_sugar_mg_dl,
            "cholesterol_mg_dl": cholesterol_mg_dl,
            "bmi": bmi,
            "lab_tests_count": lab_tests_count,
            "treatments_count": treatments_count,
            "pulse_pressure": pulse_pressure,
            "bmi_category": bmi_category,
            "age_group": age_group,
            "chronic_diagnosis_flag": chronic_diagnosis_flag,
            "care_intensity": care_intensity,
        }]
    )

    nominal_cols = [
        "gender",
        "blood_group",
        "department",
        "diagnosis",
        "room_type",
        "bmi_category",
        "age_group",
    ]

    row_encoded = pd.get_dummies(row, columns=nominal_cols, drop_first=True)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    return row_encoded

# =========================================================
# 5. SIDEBAR
# =========================================================

st.sidebar.title("SmartCare AI")
st.sidebar.caption("Clinical Decision Support System")

page = st.sidebar.radio(
    "Navigation",
    ["Patient Predictor", "Model Analytics"],
)

# =========================================================
# 6. PATIENT PREDICTOR
# =========================================================

if page == "Patient Predictor":
    st.markdown(
        """
        <div class="hero-box">
            <h2 style='margin:0 0 0.4rem 0;'>SmartCare Disease Risk Predictor</h2>
            <p style='margin:0; color:#3f5978;'>AI-assisted disease risk assessment for hospital patients.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Enter patient information below to generate an AI-based disease risk prediction.")
    st.divider()

    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=90, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_group = st.selectbox(
            "Blood Group",
            ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"],
        )

    with col2:
        department = st.selectbox(
            "Department",
            [
                "General Medicine",
                "Cardiology",
                "Neurology",
                "Orthopedics",
                "Pediatrics",
                "Laboratory Services",
            ],
        )
        diagnosis = st.selectbox(
            "Diagnosis",
            [
                "Asthma",
                "Back Pain",
                "Chest Pain",
                "Diabetes",
                "Fever",
                "Fracture",
                "Hypertension",
                "Kidney Disease",
                "Migraine",
                "Pneumonia",
            ],
        )
        admitted = st.selectbox("Admitted?", ["No", "Yes"])

    with col3:
        room_type = st.selectbox(
            "Room Type",
            ["Not Admitted", "General Ward", "Private Room", "ICU"],
        )
        previous_admissions = st.number_input(
            "Previous Admissions",
            min_value=0,
            max_value=5,
            value=0,
        )
        length_of_stay_days = st.number_input(
            "Length of Stay (days)",
            min_value=0,
            max_value=9,
            value=0,
        )

    st.subheader("Clinical Measurements")
    col1, col2, col3 = st.columns(3)

    with col1:
        systolic_bp = st.number_input("Systolic BP", min_value=85, max_value=178, value=128)
        diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=111, value=79)

    with col2:
        blood_sugar_mg_dl = st.number_input(
            "Blood Sugar (mg/dL)",
            min_value=65,
            max_value=201,
            value=115,
        )
        cholesterol_mg_dl = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=330,
            value=205,
        )

    with col3:
        bmi = st.number_input("BMI", min_value=14.0, max_value=38.8, value=25.5, step=0.1)

    st.subheader("Patient History and Treatment")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        previous_appointments = st.number_input(
            "Previous Appointments",
            min_value=0,
            max_value=10,
            value=3,
        )

    with col2:
        missed_previous_appointments = st.number_input(
            "Missed Previous Appointments",
            min_value=0,
            max_value=4,
            value=0,
        )

    with col3:
        lab_tests_count = st.number_input("Lab Tests", min_value=0, max_value=10, value=2)

    with col4:
        treatments_count = st.number_input("Treatments", min_value=0, max_value=10, value=1)

    st.divider()

    if st.button("Generate Disease Risk Prediction", type="primary", use_container_width=True):
        try:
            model_input = create_model_input(
                age=age,
                gender=gender,
                blood_group=blood_group,
                department=department,
                diagnosis=diagnosis,
                admitted=admitted,
                room_type=room_type,
                length_of_stay_days=length_of_stay_days,
                previous_admissions=previous_admissions,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                blood_sugar_mg_dl=blood_sugar_mg_dl,
                cholesterol_mg_dl=cholesterol_mg_dl,
                bmi=bmi,
                lab_tests_count=lab_tests_count,
                treatments_count=treatments_count,
                previous_appointments=previous_appointments,
                missed_previous_appointments=missed_previous_appointments,
            )

            probabilities = model.predict_proba(model_input)[0]
            predicted_class = int(np.argmax(probabilities))
            prediction = RISK_LABELS[predicted_class]
            confidence = probabilities[predicted_class]

            st.subheader("Prediction Result")
            with st.container():
                if prediction == "Low":
                    st.success(f"Disease Risk Level: {prediction}")
                elif prediction == "Medium":
                    st.warning(f"Disease Risk Level: {prediction}")
                else:
                    st.error(f"Disease Risk Level: {prediction}")

                st.metric("Prediction Confidence", f"{confidence:.1%}")

                st.subheader("Risk Probability")
                probability_df = pd.DataFrame(
                    {"Probability": probabilities},
                    index=["Low", "Medium", "High"],
                )
                st.bar_chart(probability_df)

                with st.expander("View Processed Patient Data"):
                    display_row = pd.DataFrame({
                        "Feature": model_input.columns,
                        "Value": model_input.iloc[0].values,
                    })
                    st.dataframe(display_row, use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction failed: {str(exc)}")

# =========================================================
# 7. MODEL ANALYTICS
# =========================================================

elif page == "Model Analytics":
    st.markdown(
        """
        <div class="hero-box">
            <h2 style='margin:0 0 0.4rem 0;'>Model Analytics and Explainable AI</h2>
            <p style='margin:0; color:#3f5978;'>Exploratory charts and SHAP insights created from the trained predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("1. Clinical Feature Correlations")
    heatmap_path = EDA_DIR / "correlation_heatmap.png"
    if heatmap_path.exists():
        st.image(Image.open(heatmap_path), caption="Clinical Feature Correlation Heatmap", use_container_width=True)
    else:
        st.info("Correlation heatmap not found.")

    st.header("2. Disease Risk Class Distribution")
    class_path = EDA_DIR / "class_distribution.png"
    if class_path.exists():
        st.image(Image.open(class_path), caption="Disease Risk Class Distribution", use_container_width=True)
    else:
        st.info("Class distribution plot not found.")

    st.header("3. Clinical Features by Risk Level")
    boxplot_path = EDA_DIR / "boxplots_by_risk.png"
    if boxplot_path.exists():
        st.image(Image.open(boxplot_path), caption="Clinical Features by Disease Risk", use_container_width=True)
    else:
        st.info("Boxplot not found.")

    st.header("4. Explainable AI (SHAP)")
    st.write("SHAP visualizations show which features contribute to the model's predictions.")

    shap_global = SHAP_DIR / "shap_overall_importance.png"
    if shap_global.exists():
        st.image(Image.open(shap_global), caption="Global SHAP Feature Importance", use_container_width=True)
    else:
        st.info("Global SHAP plot not found.")

    st.subheader("Risk-Specific SHAP Drivers")
    c1, c2, c3 = st.columns(3)

    with c1:
        path = SHAP_DIR / "shap_summary_low_risk.png"
        if path.exists():
            st.image(Image.open(path), caption="Low Risk Drivers", use_container_width=True)

    with c2:
        path = SHAP_DIR / "shap_summary_medium_risk.png"
        if path.exists():
            st.image(Image.open(path), caption="Medium Risk Drivers", use_container_width=True)

    with c3:
        path = SHAP_DIR / "shap_summary_high_risk.png"
        if path.exists():
            st.image(Image.open(path), caption="High Risk Drivers", use_container_width=True)

    st.divider()
    st.caption("Prototype for academic demonstration. Predictions are model outputs and are not a medical diagnosis.")
