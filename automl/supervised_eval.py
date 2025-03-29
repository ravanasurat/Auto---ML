import numpy as np
import pandas as pd # type: ignore
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder # type: ignore
from metrics import calculate_metrics, extract_feature_importance
from visualizations import generate_visualizations, generate_model_comparison_viz
from file_utils import read_dataset

def evaluate_supervised_models(
    df: pd.DataFrame,
    trained_models: Dict[str, Any],
    task_type: str,
    target_column: str
) -> Dict[str, Any]:
    """
    Evaluate supervised models and generate metrics and visualizations.
    
    Args:
        df: DataFrame with dataset
        trained_models: Dictionary of trained models
        task_type: Type of task (classification/regression)
        target_column: Name of the target column
    
    Returns:
        Dictionary containing evaluation results and visualizations
    """
    
    X = df.drop(columns=[target_column])
    y_true = df[target_column]
    
    print(f"Prepared evaluation data. X shape: {X.shape}, y shape: {y_true.shape}")
    
    
    y_is_string = y_true.dtype == 'object' or pd.api.types.is_categorical_dtype(y_true)
    
    
    evaluation_results = {
        'models': {},
        'comparison': {},
        'visualizations': {}
    }
    
   
    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name}...")
        
        try:
            
            y_pred_raw = model.predict(X)
        
            if y_is_string and np.issubdtype(y_pred_raw.dtype, np.number):
                print(f"Converting numeric predictions to strings for {model_name}")
                
                class_mapping = None
                
                
                if hasattr(model, 'classes_'):
                    classes = model.classes_
                    class_mapping = {i: cls for i, cls in enumerate(classes)}
                elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'classes_'):
                    classes = model.named_steps['model'].classes_
                    class_mapping = {i: cls for i, cls in enumerate(classes)}
                
                
                if class_mapping:
                    y_pred = np.array([class_mapping.get(pred, str(pred)) for pred in y_pred_raw])
                else:
                  
                    y_pred = np.array([str(pred) for pred in y_pred_raw])
            else:
                y_pred = y_pred_raw
            
            
            if task_type == 'classification':
                
                from sklearn.preprocessing import LabelEncoder # type: ignore
                
                
                if y_true.dtype != y_pred.dtype:
                    print(f"Converting labels to same type for {model_name}")
                    
                    y_true_str = y_true.astype(str)
                    y_pred_str = np.array([str(p) for p in y_pred])
                    
                    
                    encoder = LabelEncoder()
                   
                    encoder.fit(np.union1d(y_true_str.unique(), np.unique(y_pred_str)))
                    
                   
                    y_true_enc = encoder.transform(y_true_str)
                    y_pred_enc = encoder.transform(y_pred_str)
                    
                   
                    metrics = calculate_metrics(y_true_enc, y_pred_enc, task_type)
                else:
                    
                    metrics = calculate_metrics(y_true, y_pred, task_type)
                
                print(f"Classification metrics for {model_name}: {metrics}")
                
            else:
                
                metrics = calculate_metrics(y_true, y_pred, task_type)
                print(f"Regression metrics for {model_name}: {metrics}")
            
            
            feature_importance = extract_feature_importance(model, X.columns)
            
           
            visualizations = generate_visualizations(y_true, y_pred, task_type, model_name)
            
            
            evaluation_results['models'][model_name] = {
                'metrics': metrics,
                'feature_importance': feature_importance
            }
            
            
            if visualizations:
                if 'per_model' not in evaluation_results['visualizations']:
                    evaluation_results['visualizations']['per_model'] = {}
                evaluation_results['visualizations']['per_model'][model_name] = visualizations
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            evaluation_results['models'][model_name] = {
                'error': str(e)
            }
    
    
    if task_type == 'classification':
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            evaluation_results['comparison'][metric] = {
                model_name: model_info['metrics'].get(metric)
                for model_name, model_info in evaluation_results['models'].items()
                if 'metrics' in model_info and metric in model_info['metrics']
            }
    else:
        for metric in ['rmse', 'mae', 'r2']:
            evaluation_results['comparison'][metric] = {
                model_name: model_info['metrics'].get(metric)
                for model_name, model_info in evaluation_results['models'].items()
                if 'metrics' in model_info and metric in model_info['metrics']
            }
    
    
    comparison_viz = generate_model_comparison_viz(evaluation_results, task_type)
    if comparison_viz:
        evaluation_results['visualizations']['comparison'] = comparison_viz
    
    return evaluation_results