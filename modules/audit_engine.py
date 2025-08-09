import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
import numpy as np

def run_audit(df, target_col, sensitive_cols):
    # Drop NA values to avoid errors in aif360
    df_clean = df.dropna().copy()
    if df_clean.empty:
        print("Error: DataFrame is empty after dropping NA values.")
        fairness_metrics = {
            "Disparate Impact Ratio": np.nan,
            "Equal Opportunity Difference": np.nan,
            "Average Odds Difference": np.nan,
        }
        privacy_flags = []
        privacy_score = 0
        privacy_masking_suggestion = ""
        return {
            "target_col": target_col,
            "sensitive_cols": sensitive_cols,
            "fairness_metrics": fairness_metrics,
            "privacy_flags": privacy_flags,
            "privacy_score": privacy_score,
            "privacy_masking_suggestion": privacy_masking_suggestion
        }

    try:
        privileged_groups = [{col: 1} for col in sensitive_cols]
        unprivileged_groups = [{col: 0} for col in sensitive_cols]
        bld = BinaryLabelDataset(df=df_clean, favorable_label=1, unfavorable_label=0,
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
    # Ignore one-hot encoded columns (with underscores and only 0/1 values)
    def is_one_hot(col):
        return "_" in col and set(df_clean[col].unique()).issubset({0, 1})

    potential_identifiers = [
        col for col in df_clean.columns
        if col not in [target_col] and col not in sensitive_cols and not is_one_hot(col)
    ]
    potential_identifiers = [col for col in potential_identifiers if df_clean[col].nunique() > 1]

    # Limit combinations to avoid excessive flagging
    max_flags_considered = 4  # Only penalize up to 4 flags for privacy score

    if len(potential_identifiers) > 1:
        for i in range(len(potential_identifiers)):
            for j in range(i + 1, len(potential_identifiers)):
                col1 = potential_identifiers[i]
                col2 = potential_identifiers[j]
                combined_feature = df_clean[[col1, col2]].astype(str).agg(' '.join, axis=1)
                if combined_feature.nunique() / len(df_clean) > 0.99:
                    privacy_flags.append(f"High risk: '{col1}' and '{col2}' combined. Consider masking or generalizing.")

    # Privacy score: 100 if no flags, else decrease by 25 per flag (up to 4 flags, min 0)
    flags_for_score = min(len(privacy_flags), max_flags_considered)
    privacy_score = max(0, 100 - flags_for_score * 25)
    privacy_masking_suggestion = "Consider masking or generalizing flagged feature combinations." if privacy_flags else ""
    return {
        "target_col": target_col,
        "sensitive_cols": sensitive_cols,
        "fairness_metrics": fairness_metrics,
        "privacy_flags": privacy_flags,
        "privacy_score": privacy_score,
        "privacy_masking_suggestion": privacy_masking_suggestion
    }