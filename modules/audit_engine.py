import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
import numpy as np

def run_audit(df, target_col, sensitive_cols):
    
    try:
        privileged_groups = [{col: 1} for col in sensitive_cols]
        unprivileged_groups = [{col: 0} for col in sensitive_cols]
        
        bld = BinaryLabelDataset(df=df, favorable_label=1, unfavorable_label=0,
                                 label_names=[target_col], 
                                 protected_attribute_names=sensitive_cols)

        metric = ClassificationMetric(bld, bld, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        
        fairness_metrics = {
            "Disparate Impact Ratio": metric.disparate_impact(),
            "Equal Opportunity Difference": metric.equal_opportunity_difference(),
            "Average Odds Difference": metric.average_odds_difference(),
        }
    except Exception as e:
        print(f"Error computing fairness metrics: {e}")
        fairness_metrics = {
            "Disparate Impact Ratio": np.nan,
            "Equal Opportunity Difference": np.nan,
            "Average Odds Difference": np.nan,
        }

    # --- Privacy Risk Analysis ---
    privacy_flags = []
    
    # Heuristic: Focus on a more realistic subset of potential identifiers
    potential_identifiers = [col for col in df.columns if col not in [target_col] and col not in sensitive_cols]
    
    # Exclude features from the check if they are one-hot encoded and not sensitive
    potential_identifiers = [col for col in potential_identifiers if df[col].nunique() > 1]
    
    # We will only check combinations of these specific features
    if len(potential_identifiers) > 1:
        for i in range(len(potential_identifiers)):
            for j in range(i + 1, len(potential_identifiers)):
                col1 = potential_identifiers[i]
                col2 = potential_identifiers[j]
                
                combined_feature = df[[col1, col2]].astype(str).agg(' '.join, axis=1)
                
                # Use a stricter threshold to avoid false positives
                if combined_feature.nunique() / len(df) > 0.99:
                    privacy_flags.append(f"High risk: '{col1}' and '{col2}' combined. Consider masking or generalizing.")
    
    privacy_score = max(0, 100 - len(privacy_flags) * 25)
    
    return {
        "target_col": target_col,
        "sensitive_cols": sensitive_cols,
        "fairness_metrics": fairness_metrics,
        "privacy_flags": privacy_flags,
        "privacy_score": privacy_score
    }