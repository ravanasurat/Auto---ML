import pandas as pd # type: ignore
from typing import Dict, List, Optional, Any, Tuple
import os

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset with proper encoding handling.
    
    Args:
        file_path: Path to the dataset file
    
    Returns:
        Loaded pandas DataFrame
    """
    if file_path.endswith('.csv'):
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        
        for encoding in encodings_to_try:
            try:
                print(f"Trying to analyze CSV with encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                print(f"Failed with encoding: {encoding}")
                continue
        else:
 
            print("All standard encodings failed, trying with error handling")
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    print(f"Dataset loaded with shape: {df.shape}")
    return df