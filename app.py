import streamlit as st
import pandas as pd
import os
import joblib
from modules import data_handler, audit_engine, mitigation_engine, explainability, report_generator
from io import BytesIO
import numpy as np
from pathlib import Path
import sys

# Add the project's root directory to the system path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Ensure 'outputs' directory exists for saving files
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="EthixAI", layout="wide")
st.title("🧠 EthixAI - Ethical Auditing and Monitoring System")
st.markdown("A comprehensive tool for auditing, correcting, and monitoring AI models for fairness, transparency, and privacy.")

# --- Session State Management ---
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None
if 'audited_results' not in st.session_state:
    st.session_state['audited_results'] = None
if 'mitigation_results' not in st.session_state:
    st.session_state['mitigation_results'] = None
if 'ethical_score' not in st.session_state:
    st.session_state['ethical_score'] = None
if 'cleaned_data' not in st.session_state:
    st.session_state['cleaned_data'] = None
if 'preprocessed_data' not in st.session_state:
    st.session_state['preprocessed_data'] = None
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
if 'sensitive_cols' not in st.session_state:
    st.session_state['sensitive_cols'] = None

# --- Main UI Flow ---
st.sidebar.title("App Navigation")
page = st.sidebar.radio("Go to", ["Upload & Audit", "Ethical Scorecard", "Simulation Sandbox", "Community Repository"])

# --- Page: Upload & Audit ---
if page == "Upload & Audit":
    st.header("1. Upload Your Data 📂")
    uploaded_file = st.file_uploader("Upload a CSV file to begin the audit", type=["csv"])

    if uploaded_file:
        df = data_handler.load_data(uploaded_file)
        st.subheader("Data Preview (Original)")
        st.dataframe(df.head())
        
        # FIX: Drop the 'Name' column before any preprocessing
        if 'Name' in df.columns:
            df = df.drop(columns=['Name'])
            
        df_cleaned = df.dropna()
        
        # Identify categorical columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
        
        # Let user select sensitive columns from original column names
        sensitive_cols_raw = st.multiselect("Select all sensitive columns:", df_cleaned.columns.tolist(), default=['Gender', 'Age'] if 'Gender' in df_cleaned.columns and 'Age' in df_cleaned.columns else [])
        
        # Manually encode sensitive columns
        df_preprocessed = df_cleaned.copy()
        for col in sensitive_cols_raw:
            if col in categorical_cols:
                df_preprocessed[col] = df_preprocessed[col].astype('category').cat.codes
        
        # Apply one-hot encoding to all other non-sensitive categorical columns
        non_sensitive_categorical_cols = [c for c in categorical_cols if c not in sensitive_cols_raw]
        df_preprocessed = pd.get_dummies(df_preprocessed, columns=non_sensitive_categorical_cols, drop_first=True)
        
        st.session_state['uploaded_data'] = df_preprocessed
        st.session_state['sensitive_cols'] = sensitive_cols_raw
        
        st.subheader("Data Preview (Preprocessed for Audit)")
        st.dataframe(df_preprocessed.head())
        
        # Let user select the target column from the preprocessed data
        target_col_options = df_preprocessed.columns.tolist()
        if 'Shortlisted' in target_col_options:
            default_target_index = target_col_options.index('Shortlisted')
        else:
            default_target_index = 0
            
        target_col = st.selectbox("Select the target column:", target_col_options, index=default_target_index)
        st.session_state['target_col'] = target_col
        
        if st.button("Run Ethical Audit"):
            with st.spinner("Running comprehensive ethical audit..."):
                st.session_state['audited_results'] = audit_engine.run_audit(df_preprocessed, target_col, sensitive_cols_raw)
                st.success("✅ Audit complete! See results below.")
                st.balloons()
        
    if st.session_state['audited_results']:
        st.header("2. Audit Results & Bias Mitigation ⚖️")
        results = st.session_state['audited_results']
        
        st.subheader("Fairness Analysis")
        fairness_df = pd.DataFrame(list(results['fairness_metrics'].items()), columns=['Metric', 'Value'])
        st.dataframe(fairness_df)
        
        di_ratio = results['fairness_metrics']['Disparate Impact Ratio']
        
        # FIX: Correct logic for the fairness warning
        if not np.isnan(di_ratio) and di_ratio >= 0.8 and di_ratio <= 1.25:
            st.success(f"✅ Disparate Impact Ratio of {di_ratio:.2f} is within acceptable bounds (0.8-1.25).")
        elif not np.isnan(di_ratio):
             st.warning(f"🔴 Bias Detected: The disparate impact ratio of {di_ratio:.2f} is not close to 1.0. Mitigation is recommended.")
        else:
            st.warning("🔴 Bias Detected: Could not compute fairness metrics. Please check your data and selections.")
        
        st.subheader("Privacy Risk Analysis")
        if results['privacy_flags']:
            st.error("Potential privacy risks detected:")
            for flag in results['privacy_flags']:
                st.markdown(f"- {flag}")
        else:
            st.success("✅ No significant privacy risks detected.")

        if st.button("Mitigate Bias & Retrain Model"):
            with st.spinner("Applying bias mitigation and retraining the model..."):
                mitigation_res = mitigation_engine.retrain_model_with_mitigation(
                    st.session_state['uploaded_data'],
                    st.session_state['target_col'],
                    st.session_state['sensitive_cols']
                )
                st.session_state['mitigation_results'] = mitigation_res
                st.session_state['cleaned_data'] = mitigation_res.get('cleaned_data_df', None)
                st.success("✅ Retraining complete! Metrics have been updated.")
        
    if st.session_state['mitigation_results']:
        st.header("3. Before vs. After Comparison 📈")
        res = st.session_state['mitigation_results']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{res['accuracy_after']:.2f}", f"{res['accuracy_after'] - res['accuracy_before']:.2f}")
        
        # FIX: Calculate fairness change based on normalized score
        fairness_before_normalized = 1 - abs(1 - res['fairness_before']) if not np.isnan(res['fairness_before']) else 0.5
        fairness_after_normalized = 1 - abs(1 - res['fairness_after']) if not np.isnan(res['fairness_after']) else 0.5
        fairness_change = fairness_after_normalized - fairness_before_normalized

        col2.metric("Fairness Score", f"{fairness_after_normalized:.2f}", f"{fairness_change:.2f}")
        col3.metric("Privacy Score", f"{res['privacy_score']:.2f}", "0")
        
        st.subheader("Role-Based Explainability 🧐")
        role = st.selectbox("Select your role:", ["End-user", "Decision-maker (HR)", "Auditor"])
        
        explanation = explainability.get_explanation(
            role,
            res['model'],
            res['test_data'],
            res['feature_names']
        )
        st.markdown(explanation)
        
        st.subheader("Technical Explainability (SHAP Plot)")
        st.image("outputs/shap_summary.png", caption="SHAP Feature Importance Plot")

        st.header("4. Download Outputs 📥")
        col1, col2, col3 = st.columns(3)

        report_buffer = report_generator.generate_pdf_report(st.session_state['audited_results'], st.session_state['mitigation_results'])
        col1.download_button("📄 Download PDF Audit Report", report_buffer, "EthixAI_Report.pdf")

        model_buffer = BytesIO()
        joblib.dump(res['model'], model_buffer)
        col2.download_button("📦 Download Retrained Model", model_buffer.getvalue(), "ethixai_model.pkl")
        
        if 'cleaned_data' in st.session_state and st.session_state['cleaned_data'] is not None:
            csv = st.session_state['cleaned_data'].to_csv(index=False).encode('utf-8')
            col3.download_button("🗂️ Download Cleaned Dataset", csv, "cleaned_dataset.csv", "text/csv")

# --- Page: Ethical Scorecard ---
elif page == "Ethical Scorecard":
    st.header("Ethical Scorecard 🎯")
    if st.session_state['mitigation_results']:
        res = st.session_state['mitigation_results']
        
        ethical_score = 0.5 * res['accuracy_after'] + 0.4 * (1 - abs(1 - res['fairness_after'])) + 0.1 * (res['privacy_score'] / 100)
        st.session_state['ethical_score'] = ethical_score
        
        st.success(f"# Final Ethical Score: {ethical_score:.2f} / 100")
        st.markdown("This score reflects a weighted average of accuracy, fairness, and privacy.")
    else:
        st.info("Please run an ethical audit and mitigation first.")

# --- Page: Simulation Sandbox ---
elif page == "Simulation Sandbox":
    st.header("Interactive Fairness Simulation Sandbox 🧪")
    st.markdown("Experiment with data features to see how they impact the model's prediction and fairness.")

    if 'mitigation_results' in st.session_state and st.session_state['mitigation_results']:
        res = st.session_state['mitigation_results']
        model = res['model']
        feature_names = res['feature_names']
        test_data = res['test_data']
        
        st.success("Model loaded. Use the sliders and selectors to change features and see the impact.")

        st.subheader("Create a Hypothetical Candidate")
        
        user_input = {}
        
        for feature_name in feature_names:
            if 'Age' in feature_name:
                user_input[feature_name] = st.slider("Age", min_value=18, max_value=70, value=30)
            elif 'Income' in feature_name:
                user_input[feature_name] = st.slider("Income", min_value=30000, max_value=200000, value=75000)
            elif 'Experience' in feature_name:
                user_input[feature_name] = st.slider("Experience", min_value=0, max_value=25, value=5)
            elif 'PythonScore' in feature_name:
                user_input[feature_name] = st.slider("PythonScore", min_value=50, max_value=100, value=85)
            elif 'Gender' in feature_name:
                user_input[feature_name] = st.radio("Gender", [0, 1], format_func=lambda x: "Female (0)" if x==0 else "Male (1)")
            else:
                user_input[feature_name] = 0

        # Create a DataFrame for the hypothetical candidate
        input_df = pd.DataFrame([user_input])

        if st.button("Get Prediction"):
            # Ensure the input DataFrame has the same columns as the training data
            for col in test_data.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[test_data.columns]

            prediction_prob = model.predict_proba(input_df)
            prediction = np.argmax(prediction_prob)

            st.subheader("Prediction Result")
            st.markdown(f"The model predicts the candidate will be **{'Shortlisted' if prediction == 1 else 'Not Shortlisted'}**.")
            st.markdown(f"Prediction Probability: **{prediction_prob[0][1]:.2f}**")
            
            # --- Counterfactual Fairness Simulation ---
            st.subheader("Counterfactual Fairness Analysis")
            st.markdown("What if we change a sensitive feature?")
            
            if 'Gender' in user_input:
                counterfactual_input = input_df.copy()
                counterfactual_input['Gender'] = 1 - counterfactual_input['Gender']
                
                counterfactual_prob = model.predict_proba(counterfactual_input)
                prob_change = counterfactual_prob[0][1] - prediction_prob[0][1]
                
                st.info(f"If the candidate's gender were changed, their shortlisting probability would change by **{prob_change:.2f}**.")
    
    else:
        st.warning("Please run an ethical audit and retrain the model first to use the simulation sandbox.")

# --- Page: Community Repository ---
elif page == "Community Repository":
    st.header("Community-Driven Ethical AI Repository 🤝")
    st.markdown("Browse, share, and collaborate on ethically-audited models and datasets.")

    st.subheader("Browse Submissions")
    
    repo_models = [
        {'name': 'Hiring Model v1.0', 'score': 91.5, 'domain': 'Hiring', 'author': 'EthixAI Team', 'verified': True},
        {'name': 'Loan Approval v2.3', 'score': 88.2, 'domain': 'Finance', 'author': 'User_A', 'verified': True},
    ]
    st.dataframe(pd.DataFrame(repo_models))
    
    if st.session_state['ethical_score']:
        st.subheader("Submit Your Own Model")
        model_name = st.text_input("Model Name", "My Ethically Audited Model")
        domain = st.selectbox("Domain", ['Hiring', 'Finance', 'Healthcare', 'Other'])
        if st.button("Submit to Repository"):
            st.success(f"🎉 Model '{model_name}' submitted with an Ethical Score of {st.session_state['ethical_score']:.2f}!")
            st.balloons()
    else:
        st.info("Complete an audit and get a final score to submit your model.")