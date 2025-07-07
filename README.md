# Fraud Detection Models

A comprehensive machine learning project for detecting fraudulent credit card transactions using multiple algorithms and feature engineering techniques.

## 📊 Project Overview

This project implements various machine learning models to detect credit card fraud using anonymized transaction data. The system includes data preprocessing, feature engineering, model training, and evaluation components with a focus on handling imbalanced datasets.

## 🗂️ Project Structure

```
fraud-detection-models/
├── data/
│   ├── raw/
│   │   └── creditcard.csv                    # Original dataset
│   └── processed/
│       ├── creditcard_with_basic_features.csv # Dataset with engineered features
│       ├── feature_names.csv                 # Feature column names
│       ├── test_data.csv                     # Test dataset
│       ├── train_data.csv                    # Training dataset
│       ├── X_test.csv                        # Test features
│       ├── X_train.csv                       # Training features
│       ├── y_test.csv                        # Test labels
│       └── y_train.csv                       # Training labels
├── models/
│   ├── feature_selector.pkl                 # Feature selection model
│   ├── lasso_fraud_model.pkl                # Lasso regression model
│   ├── logistic_fraud_model.pkl             # Logistic regression model
│   ├── random_forest_fraud_model.pkl        # Random forest model
│   └── xgboost_fraud_model.pkl              # XGBoost model
├── notebooks/
│   ├── 01_data_exploration.ipynb            # Data exploration and analysis
│   ├── 02_feature_engineering.ipynb         # Feature engineering process
│   └── 03_model_training.ipynb              # Model training and evaluation
├── results/
│   ├── data_quality_report.json             # Data quality assessment
│   ├── exploration_summary.json             # EDA summary
│   ├── feature_importance.csv               # Feature importance rankings
│   ├── fraud_detection_report.txt           # Detailed analysis report
│   ├── fraud_detection_summary.json         # Summary statistics
│   ├── model_comparison_results.csv         # Model performance comparison
│   └── selected_features.csv                # Selected features for modeling
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py                # Data cleaning and preprocessing
│   ├── feature_engineering.py               # Feature creation and selection
│   ├── model_evaluation.py                  # Model evaluation metrics
│   └── utils.py                             # Utility functions
├── requirements.txt                          # Project dependencies
└── setup.py                                # Package setup configuration
```

## 📈 Dataset

The project uses the **Credit Card Fraud Detection Dataset** containing:
- **284,807** transactions
- **31 features** (V1-V28 are anonymized, plus Time, Amount, and Class)
- **0.17%** fraud rate (highly imbalanced)
- **European cardholders** transactions from September 2013

### Data Features:
- `Time`: Seconds elapsed between transactions
- `V1-V28`: Anonymized features (PCA transformed)
- `Amount`: Transaction amount
- `Class`: Target variable (0 = Normal, 1 = Fraud)


## 🤖 Models Implemented

| Model | Description | Key Features |
|-------|-------------|--------------|
| **Logistic Regression** | Linear classification baseline | Interpretable, fast training |
| **Lasso Regression** | L1 regularized linear model | Feature selection, sparse solutions |
| **Random Forest** | Ensemble of decision trees | Handles non-linearity, feature importance |
| **XGBoost** | Gradient boosting framework | High performance, handles imbalance |

## 📊 Model Performance

The models are evaluated using metrics appropriate for imbalanced datasets:

- **Precision**: Ability to avoid false positives
- **Recall**: Ability to detect actual fraud cases
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the Precision-Recall curve

## 🔍 Key Features

### Data Preprocessing
- Missing value handling
- Outlier detection and treatment
- Feature scaling and normalization
- Data quality assessment

### Feature Engineering
- Time-based features extraction
- Amount-based statistical features
- Feature selection using various techniques
- Handling class imbalance

### Model Training
- Cross-validation for robust evaluation
- Hyperparameter tuning
- Model persistence and loading
- Performance comparison and selection

## 📋 Results

Detailed results are available in the `results/` directory:
- **Model Comparison**: Performance metrics for all models
- **Feature Importance**: Top features contributing to fraud detection
- **Data Quality Report**: Assessment of data completeness and quality
- **Fraud Detection Summary**: Key insights and recommendations

## 🔗 References

- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Handling Imbalanced Datasets](https://imbalanced-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

