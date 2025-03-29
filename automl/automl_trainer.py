import pandas as pd # type: ignore
from typing import Dict, List, Any, Optional, Tuple

from preprocessing import read_dataset
from model_training import train_supervised_models
from unsupervised import train_unsupervised_models

def train_models(
    dataset_path: str,
    selected_models: List[Dict[str, Any]],
    task_type: str,
    is_labeled: bool,
    target_column: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train selected models with handling for string target labels.
    
    Args:
        dataset_path: Path to the dataset file
        selected_models: List of selected model configurations
        task_type: Type of task (classification/regression/clustering)
        is_labeled: Whether the dataset is labeled
        target_column: Name of the target column for supervised learning
    
    Returns:
        Tuple of (trained_models_dict, evaluation_results_dict)
    """
    print(f"Training models for task: {task_type}, is_labeled: {is_labeled}")

    df = read_dataset(dataset_path)
    
    if not is_labeled or task_type in ('clustering', 'dimensionality_reduction'):
        print("Unsupervised learning task detected")
        return train_unsupervised_models(df, selected_models)
    
    return train_supervised_models(df, selected_models, task_type, target_column)