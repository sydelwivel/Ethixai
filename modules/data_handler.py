import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np

def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def preprocess_data(df, target_col, sensitive_cols):
    df_copy = df.copy()
    
    # Separate features and target
    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]

    # The data is already preprocessed in app.py, so we just split it here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def generate_synthetic_data(X, y):
    smote = SMOTE(k_neighbors=1, random_state=42)
    return smote.fit_resample(X, y)