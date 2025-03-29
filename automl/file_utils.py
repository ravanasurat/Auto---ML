import pandas as pd # type: ignore
import io
import base64
import matplotlib.pyplot as plt # type: ignore
from typing import Optional

def read_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load a dataset with proper encoding handling.
    
    Args:
        dataset_path: Path to the dataset file
    
    Returns:
        DataFrame containing the dataset
    """
    if dataset_path.endswith('.csv'):

        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        
        for encoding in encodings_to_try:
            try:
                print(f"Evaluation - Trying encoding: {encoding}")
                df = pd.read_csv(dataset_path, encoding=encoding)
                print(f"Evaluation - Successfully read with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                print(f"Evaluation - Failed with encoding: {encoding}")
                continue
        else:
            
            print("Evaluation - All standard encodings failed, trying with error handling")
            df = pd.read_csv(dataset_path, encoding='latin1', on_bad_lines='skip')
    elif dataset_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    print(f"Dataset loaded for evaluation with shape: {df.shape}")
    return df

def plot_to_base64(plt) -> str:
    """
    Convert matplotlib plot to base64-encoded image.
    
    Args:
        plt: Matplotlib pyplot instance
    
    Returns:
        Base64-encoded image string
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str