"""
Streamlit dashboard for churn prediction.
Sends requests to the FastAPI backend.
"""

import streamlit as st
import requests

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("RetainML - Churn Predictor")

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.success("API connected, model loaded")
    else:
        st.warning("API connected but model not loaded")
except requests.ConnectionError:
    st.error("Cannot connect to API. Start it with: python scripts/run_api.py")
    st.stop()

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

with col2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])

with st.expander("Additional Services"):
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict Churn", type="primary"):
    payload = {
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": str(total_charges),
        "gender": gender,
    }

    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

    if resp.status_code == 200:
        result = resp.json()

        st.header("Prediction Result")

        risk = result["risk_level"]
        if risk == "high":
            st.error(f"Risk Level: HIGH")
        elif risk == "medium":
            st.warning(f"Risk Level: MEDIUM")
        else:
            st.success(f"Risk Level: LOW")

        col_a, col_b = st.columns(2)
        col_a.metric("Churn Probability", f"{result['churn_probability']:.1%}")
        col_b.metric("Prediction", "Will Churn" if result["prediction"] == 1 else "Will Stay")

        # Key Risk Factors
        if result.get("concerns"):
            st.header("Key Risk Factors")
            for c in result["concerns"]:
                st.markdown(f"- **{c['feature']}** (impact: {c['impact_score']:.3f}) — {c['description']}")

        # Retention Plan
        if result.get("retention_plan"):
            st.header("Retention Plan")
            for a in result["retention_plan"]:
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(a["priority"], "⚪")
                st.markdown(f"{icon} **[{a['priority'].upper()}]** {a['action']}")
                st.caption(a["reason"])
    else:
        st.error(f"API error: {resp.text}")

# --- Prediction History ---
with st.expander("Prediction History", expanded=False):
    try:
        import pandas as pd

        hist_resp = requests.get(f"{API_URL}/history", params={"n": 50}, timeout=5)
        if hist_resp.status_code == 200:
            records = hist_resp.json()
            if not records:
                st.info("No predictions logged yet.")
            else:
                rows = []
                for rec in records:
                    rows.append({
                        "Timestamp": rec["timestamp"],
                        "Risk Level": rec["risk_level"],
                        "Churn Probability": f"{rec['churn_probability']:.1%}",
                        "Prediction": "Will Churn" if rec["prediction"] == 1 else "Will Stay",
                        "Model": rec["model_name"],
                        "Tenure": rec["inputs"].get("tenure"),
                        "Monthly Charges": rec["inputs"].get("MonthlyCharges"),
                        "Contract": rec["inputs"].get("Contract"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                st.subheader("Risk Level Distribution")
                risk_counts = pd.Series(
                    [rec["risk_level"] for rec in records]
                ).value_counts()
                st.bar_chart(risk_counts)
        else:
            st.warning(f"Could not load history: {hist_resp.status_code}")
    except requests.ConnectionError:
        st.warning("History unavailable: cannot reach API.")
    except Exception as e:
        st.warning(f"History error: {e}")
