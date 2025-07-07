from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_models(y_test, y_pred_lasso, y_pred_proba_lasso, 
                   y_pred_logreg, y_pred_proba_logreg):
    """Comprehensive model evaluation and comparison"""
    print("\n" + "="*80)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*80)
    
    # Calculate metrics for both models
    auc_lasso = roc_auc_score(y_test, y_pred_proba_lasso)
    auc_logreg = roc_auc_score(y_test, y_pred_proba_logreg)
    
    print("LASSO REGRESSION RESULTS:")
    print("-" * 40)
    print(f"AUC Score: {auc_lasso:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lasso))
    
    print("\nLOGISTIC REGRESSION RESULTS:")
    print("-" * 40)
    print(f"AUC Score: {auc_logreg:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_logreg))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curves
    fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_pred_proba_lasso)
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_proba_logreg)
    
    axes[0, 0].plot(fpr_lasso, tpr_lasso, label=f'Lasso (AUC = {auc_lasso:.3f})')
    axes[0, 0].plot(fpr_logreg, tpr_logreg, label=f'Logistic Reg (AUC = {auc_logreg:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confusion Matrix - Lasso
    cm_lasso = confusion_matrix(y_test, y_pred_lasso)
    sns.heatmap(cm_lasso, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix - Lasso')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Confusion Matrix - Logistic Regression
    cm_logreg = confusion_matrix(y_test, y_pred_logreg)
    sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix - Logistic Regression')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Prediction Probability Distributions
    axes[1, 1].hist(y_pred_proba_lasso[y_test == 0], bins=50, alpha=0.7, 
                   label='Non-Fraud (Lasso)', density=True)
    axes[1, 1].hist(y_pred_proba_lasso[y_test == 1], bins=50, alpha=0.7, 
                   label='Fraud (Lasso)', density=True)
    axes[1, 1].hist(y_pred_proba_logreg[y_test == 0], bins=50, alpha=0.5, 
                   label='Non-Fraud (LogReg)', density=True)
    axes[1, 1].hist(y_pred_proba_logreg[y_test == 1], bins=50, alpha=0.5, 
                   label='Fraud (LogReg)', density=True)
    axes[1, 1].set_xlabel('Prediction Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Prediction Probability Distributions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Model comparison summary
    print("\nMODEL COMPARISON SUMMARY:")
    print("="*50)
    comparison_df = pd.DataFrame({
        'Model': ['Lasso Regression', 'Logistic Regression'],
        'AUC Score': [auc_lasso, auc_logreg],
        'Best Model': ['⭐' if auc_lasso > auc_logreg else '', '⭐' if auc_logreg > auc_lasso else '']
    })
    print(comparison_df.to_string(index=False))
    
    return comparison_df