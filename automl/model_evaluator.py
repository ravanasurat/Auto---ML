import pandas as pd # type: ignore
import numpy as np
from typing import Dict, List, Any, Optional

from file_utils import read_dataset
from supervised_eval import evaluate_supervised_models
from unsupervised_eval import evaluate_unsupervised_models
from prediction import predict_with_models

def evaluate_models(
    dataset_path: str,
    trained_models: Dict[str, Any],
    task_type: str,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate trained models and handle mixed label types.
    
    Args:
        dataset_path: Path to the dataset file
        trained_models: Dictionary of trained models
        task_type: Type of task (classification/regression/clustering)
        target_column: Name of the target column (for supervised learning)
    
    Returns:
        Dictionary containing evaluation results and visualizations
    """
    print(f"Evaluating models for task: {task_type}")
    
    try:
        df = read_dataset(dataset_path)
    except Exception as e:
        print(f"Error reading dataset for evaluation: {str(e)}")
        
        return {
            'error': f"Could not read dataset for evaluation: {str(e)}",
            'models': {
                model_name: {'error': 'Dataset could not be loaded for evaluation'}
                for model_name in trained_models.keys()
            }
        }
    
    
    if task_type in ('clustering', 'dimensionality_reduction') or target_column is None:
        print("Evaluating unsupervised models...")
        return evaluate_unsupervised_models(df, trained_models, task_type)
    
    
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found for evaluation")
        return {
            'error': f"Target column '{target_column}' not found for evaluation",
            'models': {
                model_name: {'error': 'Target column not found'}
                for model_name in trained_models.keys()
            }
        }
    
    
    print("Evaluating supervised models...")
    return evaluate_supervised_models(df, trained_models, task_type, target_column)

def make_predictions(
    trained_models: Dict[str, Any],
    input_data: Dict[str, Any],
    task_type: str
) -> Dict[str, Any]:
    """
    Make predictions with trained models.
    
    Args:
        trained_models: Dictionary of trained models
        input_data: Dictionary with input features
        task_type: Type of task
    
    Returns:
        Dictionary with predictions
    """
    return predict_with_models(trained_models, input_data, task_type)