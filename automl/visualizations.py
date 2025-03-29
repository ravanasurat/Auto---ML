import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix # type: ignore
from file_utils import plot_to_base64

def generate_visualizations(y_true, y_pred, task_type: str, model_name: str) -> Dict[str, str]:
    """
    Generate visualizations for model evaluation.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        task_type: Type of task
        model_name: Name of the model
    
    Returns:
        Dictionary mapping visualization names to base64-encoded image data
    """
    visualizations = {}
    
    if task_type == 'classification':
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        
        img = plot_to_base64(plt)
        plt.close()
        
        visualizations['confusion_matrix'] = img
        
       
        plt.figure(figsize=(10, 6))
        
        
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
       
        true_counts = pd.Series(y_true).value_counts().reindex(classes).fillna(0)
        pred_counts = pd.Series(y_pred).value_counts().reindex(classes).fillna(0)
       
        x = np.arange(len(classes))
        width = 0.35
        
        plt.bar(x - width/2, true_counts, width, label='True')
        plt.bar(x + width/2, pred_counts, width, label='Predicted')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(f'Class Distribution - {model_name}')
        plt.xticks(x, classes)
        plt.legend()
        plt.tight_layout()
        
        
        img = plot_to_base64(plt)
        plt.close()
        
        visualizations['class_distribution'] = img
    
    elif task_type == 'regression':
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title(f'Actual vs Predicted - {model_name}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        
        img = plot_to_base64(plt)
        plt.close()
        
        visualizations['actual_vs_predicted'] = img
        
        
        plt.figure(figsize=(8, 6))
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
        plt.title(f'Residual Plot - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        
        img = plot_to_base64(plt)
        plt.close()
        
        visualizations['residual_plot'] = img
        
        
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.title(f'Residual Distribution - {model_name}')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        
        img = plot_to_base64(plt)
        plt.close()
        
        visualizations['residual_distribution'] = img
    
    return visualizations

def generate_model_comparison_viz(evaluation_results: Dict[str, Any], task_type: str) -> Dict[str, str]:
    """
    Generate visualizations comparing all models.
    
    Args:
        evaluation_results: Dictionary with evaluation results
        task_type: Type of task
    
    Returns:
        Dictionary mapping visualization names to base64-encoded image data
    """
    comparison_viz = {}
    
    
    models = list(evaluation_results['models'].keys())
    
    if not models:
        return {}
    
    
    if task_type == 'classification':
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        title = 'Classification Metrics Comparison'
    else:
        metrics = ['rmse', 'mae', 'r2']
        title = 'Regression Metrics Comparison'
    
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        
        values = []
        for model in models:
            if metric in evaluation_results['models'][model]['metrics']:
               
                if metric == 'r2':
                    values.append(evaluation_results['models'][model]['metrics'][metric])
                elif metric in ['rmse', 'mae', 'mse']:
                    
                    error_val = evaluation_results['models'][model]['metrics'][metric]
                    
                    max_error = max([
                        evaluation_results['models'][m]['metrics'][metric]
                        for m in models
                        if metric in evaluation_results['models'][m]['metrics']
                    ])
                    if max_error > 0:
                        values.append(1 - (error_val / max_error))
                    else:
                        values.append(1)  # All errors are 0
                else:
                    values.append(evaluation_results['models'][model]['metrics'][metric])
            else:
                values.append(0)
        
       
        plt.bar(models, values, alpha=0.7)
        plt.title(f'{metric.upper()} Comparison')
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        
        img = plot_to_base64(plt)
        plt.close()
        
        comparison_viz[f'{metric}_comparison'] = img
    
    
    plt.figure(figsize=(10, 6))
    
    cv_means = []
    cv_stds = []
    
    for model in models:
        if 'cross_validation' in evaluation_results['models'][model]:
            cv_info = evaluation_results['models'][model]['cross_validation']
            if 'mean' in cv_info and 'std' in cv_info:
                cv_means.append(cv_info['mean'])
                cv_stds.append(cv_info['std'])
            else:
                cv_means.append(0)
                cv_stds.append(0)
        else:
            cv_means.append(0)
            cv_stds.append(0)
    
    plt.bar(models, cv_means, yerr=cv_stds, alpha=0.7, capsize=5)
    plt.title('Cross-Validation Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('CV Score (Higher is Better)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    
    img = plot_to_base64(plt)
    plt.close()
    
    comparison_viz['cv_comparison'] = img
    
    return comparison_viz