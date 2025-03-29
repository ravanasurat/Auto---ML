import numpy as np
import pandas as pd # type: ignore
from typing import Dict, Any

def predict_with_models(
    trained_models: Dict[str, Any],
    input_data: Dict[str, Any],
    task_type: str
) -> Dict[str, Any]:
    """
    Make predictions with all trained models and handle string/numeric label conversion.
    
    Args:
        trained_models: Dictionary of trained models
        input_data: Dictionary with input features
        task_type: Type of task (classification/regression)
    
    Returns:
        Dictionary containing predictions from all models
    """
    
    input_df = pd.DataFrame([input_data])
    print(f"Making predictions for input: {input_data}")
    
    
    predictions = {}
    
    for model_name, model in trained_models.items():
        print(f"Predicting with model: {model_name}")
        try:
          
            raw_pred = model.predict(input_df)
            print(f"Raw prediction from {model_name}: {raw_pred}")
            
            
            if task_type == 'classification':
                
                if hasattr(model, 'classes_'):
                    class_labels = model.classes_
                    print(f"Found classes in model: {class_labels}")
                   
                    if np.issubdtype(raw_pred.dtype, np.number) and isinstance(class_labels[0], str):
                        pred_value = str(class_labels[raw_pred[0]])
                    else:
                        pred_value = str(raw_pred[0])
                
                
                elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'classes_'):
                    class_labels = model.named_steps['model'].classes_
                    print(f"Found classes in pipeline model: {class_labels}")
                    
                    if np.issubdtype(raw_pred.dtype, np.number) and isinstance(class_labels[0], str):
                        try:
                            pred_value = str(class_labels[raw_pred[0]])
                        except IndexError:
                            
                            pred_value = str(raw_pred[0])
                    else:
                        pred_value = str(raw_pred[0])
                
                else:
                    
                    pred_value = str(raw_pred[0])
            else:
                
                pred_value = float(raw_pred[0])
            
            
            probabilities = None
            if task_type == 'classification' and hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(input_df)
                    
                    
                    if hasattr(model, 'classes_'):
                        class_labels = model.classes_
                    elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'classes_'):
                        class_labels = model.named_steps['model'].classes_
                    else:
                        class_labels = [i for i in range(proba.shape[1])]
                    
                    
                    probabilities = {str(label): float(p) for label, p in zip(class_labels, proba[0])}
                    print(f"Probabilities from {model_name}: {probabilities}")
                except Exception as proba_error:
                    print(f"Error getting probabilities from {model_name}: {str(proba_error)}")
                    
                    pass
            
            
            if task_type == 'classification':
                predictions[model_name] = {
                    'prediction': pred_value,
                    'probabilities': probabilities
                }
            else:
                predictions[model_name] = {
                    'prediction': pred_value
                }
            
            print(f"Final prediction from {model_name}: {pred_value}")
            
        except Exception as e:
            print(f"Error predicting with {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            predictions[model_name] = {
                'error': str(e)
            }
    
    return predictions