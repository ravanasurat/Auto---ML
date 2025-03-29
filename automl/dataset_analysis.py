import pandas as pd # type: ignore
import numpy as np
import os
from typing import Dict, List, Optional, Any, Tuple

def analyze_column_data(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze each column in the dataset and extract statistics.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with column information
    """
    column_info = {}
    for col in df.columns:
        print(f"Analyzing column: {col}")
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = 'numeric'
                col_stats = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'missing': int(df[col].isna().sum())
                }
            elif pd.api.types.is_datetime64_dtype(df[col]):
                col_type = 'datetime'
                col_stats = {
                    'min': str(df[col].min()),
                    'max': str(df[col].max()),
                    'missing': int(df[col].isna().sum())
                }
            else:
                col_type = 'categorical'
                value_counts = df[col].value_counts().head(10).to_dict()
                col_stats = {
                    'unique_values': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in value_counts.items()},
                    'missing': int(df[col].isna().sum())
                }
            
            column_info[col] = {
                'type': col_type,
                'stats': col_stats
            }
        except Exception as col_error:
            print(f"Error analyzing column {col}: {str(col_error)}")

            column_info[col] = {
                'type': 'unknown',
                'stats': {
                    'error': str(col_error),
                    'missing': int(df[col].isna().sum())
                }
            }
    
    return column_info

def analyze_target_column(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Analyze the target column with specific statistics.
    
    Args:
        df: DataFrame containing the data
        target_column: Name of the target column
    
    Returns:
        Dictionary with target column statistics
    """
    if target_column not in df.columns:
        return None
    
    print(f"Analyzing target column: {target_column}")
    try:
        if pd.api.types.is_numeric_dtype(df[target_column]):
            if df[target_column].nunique() <= 10:
      
                target_info = {
                    'type': 'classification',
                    'classes': [str(c) for c in df[target_column].unique().tolist()],
                    'distribution': {str(k): int(v) for k, v in df[target_column].value_counts().to_dict().items()}
                }
            else:

                target_info = {
                    'type': 'regression',
                    'min': float(df[target_column].min()),
                    'max': float(df[target_column].max()),
                    'mean': float(df[target_column].mean()),
                    'median': float(df[target_column].median())
                }
        else:
            
            target_info = {
                'type': 'classification',
                'classes': [str(c) for c in df[target_column].unique().tolist()],
                'distribution': {str(k): int(v) for k, v in df[target_column].value_counts().to_dict().items()}
            }
        return target_info
    except Exception as target_error:
        print(f"Error analyzing target column: {str(target_error)}")
        return {
            'type': 'unknown',
            'error': str(target_error)
        }

def calculate_correlations(df: pd.DataFrame) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: DataFrame containing the data
    
    Returns:
        Dictionary with correlation matrix or None if not applicable
    """
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) > 1 and len(numeric_cols) <= 20:
        try:
            corr_matrix = df[numeric_cols].corr()

            correlation = {
                str(idx): {str(col): float(val) for col, val in row.items()}
                for idx, row in corr_matrix.to_dict('index').items()
            }
            return correlation
        except Exception as corr_error:
            print(f"Error calculating correlation: {str(corr_error)}")
    
    return None

def analyze_dataset(file_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a dataset and extract key information.
    
    Args:
        file_path: Path to the dataset file
        target_column: Name of the target column for labeled data
    
    Returns:
        Dict containing dataset information
    """
    from dataset_loader import load_dataset
    
 
    df = load_dataset(file_path)
    

    num_rows, num_cols = df.shape

    column_info = analyze_column_data(df)
    
 
    target_info = None
    if target_column:
        target_info = analyze_target_column(df, target_column)

    correlation = calculate_correlations(df)
    

    missing_values = {str(col): int(val) for col, val in df.isna().sum().to_dict().items()}
    
    result = {
        'filename': os.path.basename(file_path),
        'num_rows': num_rows,
        'num_columns': num_cols,
        'columns': column_info,
        'target_info': target_info,
        'correlation': correlation,
        'missing_values': missing_values
    }
    
    print("Dataset analysis completed successfully")
    return result