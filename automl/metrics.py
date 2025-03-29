import numpy as np
import pandas as pd # type: ignore
from typing import Dict, Any, Optional, List
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, KFold # type: ignore

def calculate_metrics(y_true, y_pred, task_type: str) -> Dict[str, float]:
    """
    Calculate evaluation metrics based on task type.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        task_type: Type of machine learning task
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    if task_type == 'classification':
      
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            
            metrics['precision'] = float(precision_score(y_true, y_pred, average='binary'))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='binary'))
            metrics['f1'] = float(f1_score(y_true, y_pred, average='binary'))
        else:
            
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
            metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted'))
        
       
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
    
    elif task_type == 'regression':
        
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))
    
    return metrics

def extract_feature_importance(model, feature_names):
    """
    Extract feature importance from a model if available.
    
    Args:
        model: Trained model
        feature_names: Names of features
    
    Returns:
        Dictionary with feature importance information or None
    """
    feature_names = list(feature_names)
    
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return {str(name): float(imp) for name, imp in zip(feature_names, importances)}
    
    
    elif hasattr(model, 'named_steps') and 'model' in model.named_steps:
        inner_model = model.named_steps['model']
        if hasattr(inner_model, 'feature_importances_'):
            
            importances = inner_model.feature_importances_
            
           
            if len(importances) == len(feature_names):
                return {str(name): float(imp) for name, imp in zip(feature_names, importances)}
            else:
                
                return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
    
    
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        if len(coefs.shape) == 1:  # Binary classification or regression
            if len(coefs) == len(feature_names):
                return {str(name): float(coef) for name, coef in zip(feature_names, coefs)}
            else:
                return {f"feature_{i}": float(coef) for i, coef in enumerate(coefs)}
        else:  # Multiclass
            
            avg_coefs = np.mean(np.abs(coefs), axis=0)
            if len(avg_coefs) == len(feature_names):
                return {str(name): float(coef) for name, coef in zip(feature_names, avg_coefs)}
            else:
                return {f"feature_{i}": float(coef) for i, coef in enumerate(avg_coefs)}
    
    
    elif hasattr(model, 'named_steps') and 'model' in model.named_steps:
        inner_model = model.named_steps['model']
        if hasattr(inner_model, 'coef_'):
            coefs = inner_model.coef_
            if len(coefs.shape) == 1:  # Binary classification or regression
                
                return {f"feature_{i}": float(coef) for i, coef in enumerate(coefs)}
            else:  # Multiclass
                
                avg_coefs = np.mean(np.abs(coefs), axis=0)
                return {f"feature_{i}": float(coef) for i, coef in enumerate(avg_coefs)}
    
    
    return None

def perform_cross_validation(
    model,
    X,
    y,
    task_type: str,
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Perform cross-validation and return scores.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        task_type: Type of task
        n_folds: Number of CV folds
    
    Returns:
        Dictionary with cross-validation results
    """
    try:
        
        if task_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_squared_error'
        
       
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
       
        if task_type == 'regression':
            
            cv_scores = np.sqrt(-cv_scores)
            metric_name = 'rmse'
        else:
            metric_name = 'accuracy'
        
        return {
            'metric': metric_name,
            'scores': cv_scores.tolist(),
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'n_folds': n_folds
        }
    
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        return {
            'error': str(e)
        }