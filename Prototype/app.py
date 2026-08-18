
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE = "/content/drive/MyDrive/AI Prediction Problems"

MODELS_DIR = f"{BASE}/Models"
EDA_DIR = f"{BASE}/Outputs/EDA_Plots"
SHAP_DIR = f"{BASE}/Outputs/SHAP_Plots"

st.set_page_config(
    page_title="SmartCare Disease Risk Predictor",
    page_icon="\U0001F3E5",
    layout="wide"
)

# =========================================================
# 2. LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# =========================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(
        f"{MODELS_DIR}/lr_model.pkl"
    )
    scaler = joblib.load(
        f"{MODELS_DIR}/scaler.pkl"
    )
    feature_columns = joblib.load(
        f"{MODELS_DIR}/feature_columns.pkl"
    )
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Required model files were not found. "
        "Please run Task 05 before starting Task 08."
    )
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
    "care_intensity"
]

RISK_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High"
}

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
    missed_previous_appointments
):
    # Convert Yes/No to the same numeric format
    admitted_value = 1 if admitted == "Yes" else 0

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------
    pulse_pressure = systolic_bp - diastolic_bp

    chronic_diagnosis_flag = int(
        diagnosis in ["Diabetes", "Hypertension"]
    )

    care_intensity = (
        lab_tests_count +
        treatments_count
    )

    # BMI categories used during Task 03/05
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # Age groups used during Task 03/05
    if age < 18:
        age_group = "Child"
    elif age < 40:
        age_group = "Adult"
    elif age < 60:
        age_group = "Middle-aged"
    else:
        age_group = "Senior"

    # -----------------------------------------------------
    # Create raw patient record
    # -----------------------------------------------------
    row = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "blood_group": blood_group,
        "department": department,
        "diagnosis": diagnosis,
        "previous_appointments":
            previous_appointments,
        "missed_previous_appointments":
            missed_previous_appointments,
        "admitted":
            admitted_value,
        "room_type":
            room_type,
        "length_of_stay_days":
            length_of_stay_days,
        "previous_admissions":
            previous_admissions,
        "systolic_bp":
            systolic_bp,
        "diastolic_bp":
            diastolic_bp,
        "blood_sugar_mg_dl":
            blood_sugar_mg_dl,
        "cholesterol_mg_dl":
            cholesterol_mg_dl,
        "bmi":
            bmi,
        "lab_tests_count":
            lab_tests_count,
        "treatments_count":
            treatments_count,
        "pulse_pressure":
            pulse_pressure,
        "bmi_category":
            bmi_category,
        "age_group":
            age_group,
        "chronic_diagnosis_flag":
            chronic_diagnosis_flag,
        "care_intensity":
            care_intensity
    }])

    # -----------------------------------------------------
    # One-hot encoding
    # -----------------------------------------------------
    nominal_cols = [
        "gender",
        "blood_group",
        "department",
        "diagnosis",
        "room_type",
        "bmi_category",
        "age_group"
    ]

    row_encoded = pd.get_dummies(
        row,
        columns=nominal_cols,
        drop_first=True
    )

    # -----------------------------------------------------
    # Ensure exactly the same columns as training
    # -----------------------------------------------------
    row_encoded = row_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # -----------------------------------------------------
    # Apply the SAME scaler used during training
    # -----------------------------------------------------
    row_encoded[TRUE_NUMERIC] = scaler.transform(
        row_encoded[TRUE_NUMERIC]
    )

    return row_encoded

# =========================================================
# 5. SIDEBAR
# =========================================================

st.sidebar.title("SmartCare AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Patient Predictor",
        "Model Analytics"
    ]
)

# =========================================================
# 6. PATIENT PREDICTOR
# =========================================================

if page == "Patient Predictor":

    st.title("SmartCare Disease Risk Predictor")
    st.write(
        "Enter patient information below to generate "
        "an AI-based disease risk prediction."
    )

    st.divider()

    # -----------------------------------------------------
    # Patient information
    # -----------------------------------------------------
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=90,
            value=45
        )
        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )
        blood_group = st.selectbox(
            "Blood Group",
            [
                "A+",
                "A-",
                "AB+",
                "AB-",
                "B+",
                "B-",
                "O+",
                "O-"
            ]
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
                "Laboratory Services"
            ]
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
                "Pneumonia"
            ]
        )
        admitted = st.selectbox(
            "Admitted?",
            [
                "No",
                "Yes"
            ]
        )

    with col3:
        room_type = st.selectbox(
            "Room Type",
            [
                "Not Admitted",
                "General Ward",
                "Private Room",
                "ICU"
            ]
        )
        previous_admissions = st.number_input(
            "Previous Admissions",
            min_value=0,
            max_value=5,
            value=0
        )
        length_of_stay_days = st.number_input(
            "Length of Stay (days)",
            min_value=0,
            max_value=9,
            value=0
        )

    # -----------------------------------------------------
    # Clinical information
    # -----------------------------------------------------
    st.subheader("Clinical Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        systolic_bp = st.number_input(
            "Systolic BP",
            min_value=85,
            max_value=178,
            value=128
        )
        diastolic_bp = st.number_input(
            "Diastolic BP",
            min_value=50,
            max_value=111,
            value=79
        )

    with col2:
        blood_sugar_mg_dl = st.number_input(
            "Blood Sugar (mg/dL)",
            min_value=65,
            max_value=201,
            value=115
        )
        cholesterol_mg_dl = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=330,
            value=205
        )

    with col3:
        bmi = st.number_input(
            "BMI",
            min_value=14.0,
            max_value=38.8,
            value=25.5,
            step=0.1
        )

    # -----------------------------------------------------
    # Patient history / treatment
    # -----------------------------------------------------
    st.subheader("Patient History and Treatment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        previous_appointments = st.number_input(
            "Previous Appointments",
            min_value=0,
            max_value=10,
            value=3
        )

    with col2:
        missed_previous_appointments = st.number_input(
            "Missed Previous Appointments",
            min_value=0,
            max_value=4,
            value=0
        )

    with col3:
        lab_tests_count = st.number_input(
            "Lab Tests",
            min_value=0,
            max_value=10,
            value=2
        )

    with col4:
        treatments_count = st.number_input(
            "Treatments",
            min_value=0,
            max_value=10,
            value=1
        )

    st.divider()

    # =====================================================
    # 7. PREDICTION
    # =====================================================

    if st.button(
        "Generate Disease Risk Prediction",
        type="primary",
        use_container_width=True
    ):
        try:
            # Create model-ready input
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
                missed_previous_appointments=missed_previous_appointments
            )

            # -------------------------------------------------
            # Generate prediction
            # -------------------------------------------------
            probabilities = model.predict_proba(
                model_input
            )[0]

            predicted_class = int(
                np.argmax(probabilities)
            )

            prediction = RISK_LABELS[
                predicted_class
            ]

            confidence = probabilities[
                predicted_class
            ]

            # -------------------------------------------------
            # Display result
            # -------------------------------------------------
            st.subheader("Prediction Result")

            if prediction == "Low":
                st.success(
                    f"Disease Risk Level: {prediction}"
                )
            elif prediction == "Medium":
                st.warning(
                    f"Disease Risk Level: {prediction}"
                )
            else:
                st.error(
                    f"Disease Risk Level: {prediction}"
                )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.1%}"
            )

            # -------------------------------------------------
            # Probability chart
            # -------------------------------------------------
            st.subheader("Risk Probability")

            probability_df = pd.DataFrame(
                {
                    "Probability": probabilities
                },
                index=[
                    "Low",
                    "Medium",
                    "High"
                ]
            )

            st.bar_chart(
                probability_df
            )

            # -------------------------------------------------
            # Show processed patient information
            # -------------------------------------------------
            with st.expander(
                "View Processed Patient Data"
            ):
                display_row = pd.DataFrame({
                    "Feature": model_input.columns,
                    "Value": model_input.iloc[0].values
                })
                st.dataframe(
                    display_row,
                    use_container_width=True
                )

        except Exception as e:
            st.error(
                f"Prediction failed: {str(e)}"
            )

# =========================================================
# 8. MODEL ANALYTICS
# =========================================================

elif page == "Model Analytics":

    st.title(
        "Model Analytics and Explainable AI"
    )
    st.write(
        "This section displays the analysis and "
        "interpretability outputs generated during "
        "the previous tasks."
    )

    # -----------------------------------------------------
    # Correlation heatmap
    # -----------------------------------------------------
    st.header(
        "1. Clinical Feature Correlations"
    )

    heatmap_path = (
        f"{EDA_DIR}/correlation_heatmap.png"
    )

    if os.path.exists(heatmap_path):
        st.image(
            Image.open(heatmap_path),
            caption="Clinical Feature Correlation Heatmap",
            use_container_width=True
        )
    else:
        st.info(
            "Correlation heatmap not found."
        )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------
    st.header(
        "2. Disease Risk Class Distribution"
    )

    class_path = (
        f"{EDA_DIR}/class_distribution.png"
    )

    if os.path.exists(class_path):
        st.image(
            Image.open(class_path),
            caption="Disease Risk Class Distribution",
            use_container_width=True
        )
    else:
        st.info(
            "Class distribution plot not found."
        )

    # -----------------------------------------------------
    # Boxplots
    # -----------------------------------------------------
    st.header(
        "3. Clinical Features by Risk Level"
    )

    boxplot_path = (
        f"{EDA_DIR}/boxplots_by_risk.png"
    )

    if os.path.exists(boxplot_path):
        st.image(
            Image.open(boxplot_path),
            caption="Clinical Features by Disease Risk",
            use_container_width=True
        )
    else:
        st.info(
            "Boxplot not found."
        )

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------
    st.header(
        "4. Explainable AI (SHAP)"
    )
    st.write(
        "SHAP visualizations show which features "
        "contribute to the model's predictions."
    )

    shap_global = (
        f"{SHAP_DIR}/shap_overall_importance.png"
    )

    if os.path.exists(shap_global):
        st.image(
            Image.open(shap_global),
            caption="Global SHAP Feature Importance",
            use_container_width=True
        )
    else:
        st.info(
            "Global SHAP plot not found."
        )

    st.subheader(
        "Risk-Specific SHAP Drivers"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        path = (
            f"{SHAP_DIR}/shap_summary_low_risk.png"
        )
        if os.path.exists(path):
            st.image(
                Image.open(path),
                caption="Low Risk Drivers",
                use_container_width=True
            )

    with c2:
        path = (
            f"{SHAP_DIR}/shap_summary_medium_risk.png"
        )
        if os.path.exists(path):
            st.image(
                Image.open(path),
                caption="Medium Risk Drivers",
                use_container_width=True
            )

    with c3:
        path = (
            f"{SHAP_DIR}/shap_summary_high_risk.png"
        )
        if os.path.exists(path):
            st.image(
                Image.open(path),
                caption="High Risk Drivers",
                use_container_width=True
            )

    st.divider()
    st.caption(
        "Prototype for academic demonstration. "
        "Predictions are model outputs and are not "
        "a medical diagnosis."
    )
