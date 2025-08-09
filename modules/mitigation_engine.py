import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from modules.data_handler import generate_synthetic_data, preprocess_data
from modules import audit_engine
import numpy as np

def retrain_model_with_mitigation(df, target_col, sensitive_cols):
    
    # Preprocess the original data
    X_train_orig, X_test, y_train_orig, y_test, feature_names = preprocess_data(df, target_col, sensitive_cols)
    
    # Audit BEFORE mitigation on the test set
    audit_before_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    audit_results_before = audit_engine.run_audit(audit_before_df, target_col, sensitive_cols)
    fairness_before_raw = audit_results_before['fairness_metrics']['Disparate Impact Ratio']
    
    # Train the initial model (before mitigation)
    model_before = RandomForestClassifier(random_state=42)
    model_before.fit(X_train_orig, y_train_orig)
    acc_before = accuracy_score(y_test, model_before.predict(X_test))
    
    # Generate synthetic data to rebalance the dataset
    X_res, y_res = generate_synthetic_data(X_train_orig, y_train_orig)
    
    # Retrain the model on the new, balanced data
    model_after = RandomForestClassifier(random_state=42)
    model_after.fit(X_res, y_res)
    acc_after = accuracy_score(y_test, model_after.predict(X_test))
    
    # Create a DataFrame from the resampled data for auditing AFTER mitigation
    df_resampled = pd.concat([pd.DataFrame(X_res, columns=feature_names), pd.DataFrame({target_col: y_res})], axis=1)
    df_resampled = df_resampled.dropna()
    
    # Audit AFTER mitigation
    audit_results_after = audit_engine.run_audit(df_resampled, target_col, sensitive_cols)
    fairness_after_raw = audit_results_after['fairness_metrics']['Disparate Impact Ratio']
    privacy_score = audit_results_after['privacy_score']

    # FIX: Correctly calculate the fairness scores as a value from 0 to 1
    fairness_before = 1 - abs(1 - fairness_before_raw) if not np.isnan(fairness_before_raw) else 0.5
    fairness_after = 1 - abs(1 - fairness_after_raw) if not np.isnan(fairness_after_raw) else 0.5

    return {
        "model": model_after,
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "fairness_before": fairness_before,
        "fairness_after": fairness_after,
        "privacy_score": privacy_score,
        "test_data": X_test,
        "test_labels": y_test,
        "feature_names": feature_names,
        "cleaned_data_df": df_resampled
    }