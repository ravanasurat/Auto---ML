import pandas as pd # type: ignore
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore

def read_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Read dataset with handling for different file formats and encodings.
    
    Args:
        dataset_path: Path to the dataset file
    
    Returns:
        DataFrame containing the dataset
    """
    try:
        if dataset_path.endswith('.csv'):
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    print(f"Trying encoding: {encoding}")
                    df = pd.read_csv(dataset_path, encoding=encoding)
                    print(f"Successfully read with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    print(f"Failed with encoding: {encoding}")
                    continue
            else:
                print("All standard encodings failed, trying with error handling")
                df = pd.read_csv(dataset_path, encoding='latin1', on_bad_lines='skip')
        elif dataset_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(dataset_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        
        print(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise ValueError(f"Could not read dataset: {str(e)}")

def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns in a DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    print(f"Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
    return numeric_features, categorical_features

def create_preprocessor(numeric_features: List[str], categorical_features: List[str]):
    """
    Create a column transformer for preprocessing numeric and categorical features.
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
    
    Returns:
        ColumnTransformer for preprocessing
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
  
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop' 
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='drop'
        )
    
    return preprocessor