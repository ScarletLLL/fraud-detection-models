import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

def apply_data_processing(df, target_column='Class'):
    """
    Apply data processing steps similar to data_processing.py
    """
    print("\n" + "="*60)
    print("APPLYING DATA PROCESSING")
    print("="*60)
    
    df_processed = df.copy()
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    
    # Get numerical columns (excluding target)
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != target_column]
    
    if len(numerical_cols) > 0:
        df_processed[numerical_cols] = imputer.fit_transform(df_processed[numerical_cols])
        print(f"✓ Handled missing values for {len(numerical_cols)} numerical columns")
    
    # Handle outliers using IQR method
    outlier_count = 0
    for col in numerical_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = ((df_processed[col] < lower_bound) | 
                          (df_processed[col] > upper_bound)).sum()
        outlier_count += outliers_before
        
        df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
    
    print(f"✓ Handled {outlier_count} outliers across all numerical columns")
    
    # Scale features
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    print(f"✓ Scaled {len(numerical_cols)} numerical features")
    
    print(f"✓ Data processing completed")
    return df_processed