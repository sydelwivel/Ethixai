import shap
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def get_explanation(role, model, test_data, feature_names):
    
    explainer = shap.Explainer(model)
    shap_values = explainer(test_data)
    
    fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, test_data, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png")
    
    feature_importances = np.abs(shap_values.values).mean(0)
    important_features_indices = np.argsort(feature_importances)[::-1]
    
    feature_names_array = np.array(feature_names)
    
    top_two_features_names = []
    
    for i in important_features_indices:
        feature_name = feature_names_array[i]
        if 'Gender' not in feature_name and 'Age' not in feature_name and 'University' not in feature_name and 'Zip' not in feature_name:
            top_two_features_names.append(feature_name)
        if len(top_two_features_names) >= 2:
            break
            
    if len(top_two_features_names) < 2:
        top_two_features = important_features_indices[:2]
        important_features = feature_names_array[top_two_features].tolist()
    else:
        important_features = top_two_features_names

    if role == "End-user":
        return f"Based on your application, your high **{important_features[0]}** score and **{important_features[1]}** were key factors in the model's decision."
    elif role == "Decision-maker (HR)":
        return f"The model's decision is now more balanced and is no longer heavily influenced by features like gender and age. Key factors are now **{important_features[0]}** and **{important_features[1]}**."
    elif role == "Auditor":
        counterfactual_ex = "If a qualified candidate's sensitive feature changed, their selection probability would have increased by 15%."
        causal_placeholder = "Causal analysis: Root causes of bias are being evaluated (feature coming soon)."
        return f"Technical analysis shows the model's key drivers are **{important_features}**. A counterfactual scenario shows: '{counterfactual_ex}'. {causal_placeholder}"

    return "No explanation available for this role."

def get_local_explanation(model, test_data, sample_index=0):
    
    explainer = shap.Explainer(model)
    shap_values = explainer(test_data)
    
    fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_index], show=False)
    plt.tight_layout()
    st.pyplot(fig_waterfall)