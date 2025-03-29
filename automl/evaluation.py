import numpy as np
from typing import Dict
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def evaluate_predictions(y_true, y_pred, task_type: str) -> Dict[str, float]:
    """
    Evaluate predictions with appropriate metrics based on task type.
    
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
    
    elif task_type == 'regression':

        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))
    
    return metrics