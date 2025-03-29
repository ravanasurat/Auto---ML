import os
import shutil
import re
from typing import Dict, List, Any, Optional
import uuid
import pandas as pd # type: ignore
import numpy as np

def allowed_file(filename: str) -> bool:
    """
    Check if the file has an allowed extension.
    
    Args:
        filename: Name of the file
    
    Returns:
        True if file extension is allowed, False otherwise
    """
    ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
    
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names in a DataFrame to make them compatible with ML models.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with sanitized column names
    """
  
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col).lower() for col in df.columns]
    
    
    seen = set()
    for i, col in enumerate(df.columns):
        if col in seen:
            j = 1
            new_col = f"{col}_{j}"
            while new_col in seen:
                j += 1
                new_col = f"{col}_{j}"
            df.columns.values[i] = new_col
        seen.add(df.columns[i])
    
    return df

def preprocess_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Preprocess data for machine learning models.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        categorical_columns: List of categorical column names
        numeric_columns: List of numeric column names
    
    Returns:
        Dictionary with preprocessed data and preprocessing info
    """
   
    df = sanitize_column_names(df)
    
    
    if target_column:
        sanitized_target = re.sub(r'[^a-zA-Z0-9_]', '_', target_column).lower()
        if sanitized_target != target_column and sanitized_target in df.columns:
            target_column = sanitized_target
    
   
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    
    for col in numeric_columns:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    
    for col in categorical_columns:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
    else:
        X = df.copy()
    
    return {
        'X': X,
        'y': y,
        'categorical_columns': categorical_columns,
        'numeric_columns': numeric_columns,
        'target_column': target_column
    }

def create_temp_directory(base_dir: str) -> str:
    """
    Create a temporary directory with a unique name.
    
    Args:
        base_dir: Base directory path
    
    Returns:
        Path to the created directory
    """
   
    os.makedirs(base_dir, exist_ok=True)
    
    
    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(base_dir, unique_id)
    
    
    os.makedirs(temp_dir, exist_ok=True)
    
    return temp_dir

def clean_temp_directory(base_dir: str, max_age_hours: int = 24) -> None:
    """
    Clean up temporary directories older than the specified age.
    
    Args:
        base_dir: Base directory path
        max_age_hours: Maximum age in hours before cleaning
    """
    import time
    
    
    if not os.path.exists(base_dir):
        return
    
    
    current_time = time.time()
    
    
    max_age_seconds = max_age_hours * 60 * 60
    
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        
        if os.path.isdir(item_path):
            
            mod_time = os.path.getmtime(item_path)
            
           
            if current_time - mod_time > max_age_seconds:
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error removing directory {item_path}: {e}")

def format_model_info(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format model information for display.
    
    Args:
        model_info: Model information dictionary
    
    Returns:
        Formatted model information
    """
    
    if 'metrics' in model_info:
        for metric, value in model_info['metrics'].items():
            if isinstance(value, (float, np.float64, np.float32)):
                model_info['metrics'][metric] = round(value, 4)
    
    
    if 'cross_validation' in model_info and 'scores' in model_info['cross_validation']:
        model_info['cross_validation']['scores'] = [
            round(score, 4) for score in model_info['cross_validation']['scores']
        ]
        
        if 'mean' in model_info['cross_validation']:
            model_info['cross_validation']['mean'] = round(model_info['cross_validation']['mean'], 4)
        
        if 'std' in model_info['cross_validation']:
            model_info['cross_validation']['std'] = round(model_info['cross_validation']['std'], 4)
    
    
    if 'feature_importance' in model_info and model_info['feature_importance']:
        
        sorted_features = sorted(
            model_info['feature_importance'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        model_info['feature_importance'] = {
            feature: round(importance, 4)
            for feature, importance in sorted_features
        }
    
    return model_info

def create_sample_prediction_input(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a sample input for prediction based on dataset features.
    
    Args:
        X: Features DataFrame
    
    Returns:
        Sample input dictionary
    """
    sample_input = {}
    
    for col in X.columns:
        
        dtype = X[col].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            
            sample_input[col] = float(X[col].median())
        elif pd.api.types.is_datetime64_dtype(dtype):
            
            sample_input[col] = str(X[col].max())
        else:
            
            sample_input[col] = str(X[col].mode()[0])
    
    return sample_input

def convert_dict_types_for_json(d: Dict) -> Dict:
    """
    Convert dictionary with numpy and pandas types to JSON-serializable types.
    
    Args:
        d: Input dictionary
    
    Returns:
        Dictionary with JSON-serializable types
    """
    if not isinstance(d, dict):
        return d
    
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = convert_dict_types_for_json(v)
        elif isinstance(v, list):
            result[k] = [convert_dict_types_for_json(item) if isinstance(item, dict) else item for item in v]
        elif isinstance(v, (np.int64, np.int32)):
            result[k] = int(v)
        elif isinstance(v, (np.float64, np.float32)):
            result[k] = float(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, pd.DataFrame):
            result[k] = v.to_dict('records')
        elif isinstance(v, pd.Series):
            result[k] = v.tolist()
        else:
            result[k] = v
    
    return result