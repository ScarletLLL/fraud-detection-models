import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def apply_feature_engineering(df, target_column='Class'):
    """
    Apply feature engineering steps similar to feature_engineering.py
    """
    print("\n" + "="*60)
    print("APPLYING FEATURE ENGINEERING")
    print("="*60)
    
    df_engineered = df.copy()
    original_feature_count = len(df.columns) - 1
    
    # Time-based features
    if 'Time' in df_engineered.columns:
        print("Creating time-based features...")
        df_engineered['Time_hour'] = (df_engineered['Time'] / 3600) % 24
        df_engineered['Time_day'] = (df_engineered['Time'] / 86400) % 7
        df_engineered['is_weekend'] = (df_engineered['Time_day'] >= 5).astype(int)
        df_engineered['is_night'] = ((df_engineered['Time_hour'] >= 22) | 
                                   (df_engineered['Time_hour'] <= 6)).astype(int)
        df_engineered['is_business_hours'] = ((df_engineered['Time_hour'] >= 9) & 
                                            (df_engineered['Time_hour'] <= 17)).astype(int)
    
    # Amount-based features
    if 'Amount' in df_engineered.columns:
        print("Creating amount-based features...")
        df_engineered['Amount_log'] = np.log1p(df_engineered['Amount'])
        df_engineered['Amount_sqrt'] = np.sqrt(df_engineered['Amount'])
        
        # Amount categories
        amount_threshold_95 = df_engineered['Amount'].quantile(0.95)
        df_engineered['is_high_amount'] = (df_engineered['Amount'] > amount_threshold_95).astype(int)
        df_engineered['is_very_low_amount'] = (df_engineered['Amount'] < 1).astype(int)
        df_engineered['is_round_amount'] = (df_engineered['Amount'] % 1 == 0).astype(int)
    
    # PCA-based features
    v_cols = [col for col in df_engineered.columns if col.startswith('V') and col[1:].isdigit()]
    if len(v_cols) > 0:
        print(f"Creating PCA-based features from {len(v_cols)} V columns...")
        df_engineered['V_sum'] = df_engineered[v_cols].sum(axis=1)
        df_engineered['V_mean'] = df_engineered[v_cols].mean(axis=1)
        df_engineered['V_std'] = df_engineered[v_cols].std(axis=1)
        df_engineered['V_max'] = df_engineered[v_cols].max(axis=1)
        df_engineered['V_min'] = df_engineered[v_cols].min(axis=1)
        df_engineered['V_range'] = df_engineered['V_max'] - df_engineered['V_min']
        df_engineered['V_magnitude'] = np.sqrt(df_engineered[v_cols].pow(2).sum(axis=1))
        
        # Create interactions for top PCA components
        top_v_cols = v_cols[:5]
        for i, col1 in enumerate(top_v_cols):
            for j, col2 in enumerate(top_v_cols[i+1:], i+1):
                df_engineered[f'{col1}_{col2}_interaction'] = df_engineered[col1] * df_engineered[col2]
    
    # Risk score features
    if 'Amount_log' in df_engineered.columns and 'is_night' in df_engineered.columns:
        print("Creating risk score features...")
        df_engineered['simple_risk_score'] = (
            df_engineered['Amount_log'] * 0.3 +
            df_engineered['is_night'] * 0.2 +
            df_engineered['is_weekend'] * 0.1 +
            df_engineered['is_high_amount'] * 0.4
        )
    
    # Anomaly score based on PCA components
    if len(v_cols) > 0:
        df_engineered['pca_anomaly_score'] = np.abs(df_engineered[v_cols]).sum(axis=1)
    
    # Handle any NaN values
    df_engineered = df_engineered.fillna(0)
    
    new_feature_count = len(df_engineered.columns) - 1
    added_features = new_feature_count - original_feature_count
    
    print(f"✓ Feature engineering completed")
    print(f"✓ Original features: {original_feature_count}")
    print(f"✓ New features: {new_feature_count}")
    print(f"✓ Added features: {added_features}")
    
    return df_engineered