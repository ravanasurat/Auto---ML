import requests # type: ignore
from typing import Dict, List, Optional, Any

def generate_fallback_description(
    dataset_info: Dict[str, Any],
    user_description: str,
    task_type: str,
    is_labeled: bool,
    target_column: Optional[str] = None
) -> str:
    """
    Generate a fallback description when the LLM API is not available.
    
    Args:
        dataset_info: Dataset information dictionary
        user_description: User-provided description
        task_type: Type of task
        is_labeled: Whether the dataset is labeled
        target_column: Name of the target column
    
    Returns:
        Generated description
    """
    description = f"""
    ## Dataset Analysis Summary
    
    **Basic Information:**
    - Filename: {dataset_info['filename']}
    - Rows: {dataset_info['num_rows']}
    - Columns: {dataset_info['num_columns']}
    - Task Type: {task_type.capitalize()}
    - Labeled Data: {'Yes' if is_labeled else 'No'}
    {f'- Target Column: {target_column}' if target_column else ''}
    
    **Column Overview:**
    """
    
    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
    
    for col_name, col_info in dataset_info['columns'].items():
        if col_info['type'] == 'numeric':
            numeric_columns.append(col_name)
        elif col_info['type'] == 'categorical':
            categorical_columns.append(col_name)
        elif col_info['type'] == 'datetime':
            datetime_columns.append(col_name)
    
    description += f"- Numeric Columns: {len(numeric_columns)}\n"
    description += f"- Categorical Columns: {len(categorical_columns)}\n"
    description += f"- Datetime Columns: {len(datetime_columns)}\n"
    
    if dataset_info['target_info']:
        target_info = dataset_info['target_info']
        
        if target_info['type'] == 'classification':
            num_classes = len(target_info['classes'])
            description += f"\n**Target Information:**\n"
            description += f"- Classification task with {num_classes} classes\n"
            
            if num_classes <= 10:
                description += "- Class distribution:\n"
                for cls, count in target_info['distribution'].items():
                    description += f"  - {cls}: {count} samples\n"
        
        elif target_info['type'] == 'regression':
            description += f"\n**Target Information:**\n"
            description += f"- Regression task with target range: {target_info['min']} to {target_info['max']}\n"
            description += f"- Mean: {target_info['mean']:.2f}, Median: {target_info['median']:.2f}\n"
    

    description += f"\n**Recommendations:**\n"
    
    if task_type == 'classification':
        description += "- Consider handling class imbalance if present\n"
        description += "- Feature engineering may improve model performance\n"
        description += "- Try ensemble methods like Random Forest or Gradient Boosting\n"
    
    elif task_type == 'regression':
        description += "- Check for outliers in the target variable\n"
        description += "- Consider feature scaling for better model performance\n"
        description += "- Evaluate models using metrics like RMSE and R²\n"
    
    elif task_type == 'clustering':
        description += "- Standardize features before clustering\n"
        description += "- Try different numbers of clusters and evaluate\n"
        description += "- Consider dimensionality reduction for high-dimensional data\n"
    

    missing_values = sum(dataset_info['missing_values'].values())
    if missing_values > 0:
        description += "\n**Data Quality:**\n"
        description += f"- Missing values detected: {missing_values} total\n"
        description += "- Consider imputation strategies for missing data\n"
    
    if user_description:
        description += f"\n**User Description:**\n{user_description}\n"
    
    return description

def prepare_llm_prompt(dataset_info: Dict[str, Any], user_description: str, task_type: str, is_labeled: bool, target_column: Optional[str] = None) -> str:
    """
    Prepare the prompt for the LLM API.
    
    Args:
        dataset_info: Dataset information dictionary
        user_description: User-provided description
        task_type: Type of task
        is_labeled: Whether the dataset is labeled
        target_column: Name of the target column
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""
    I have a dataset with the following characteristics:
    - Filename: {dataset_info['filename']}
    - Number of rows: {dataset_info['num_rows']}
    - Number of columns: {dataset_info['num_columns']}
    
    Additional information:
    - Task type: {task_type}
    - Is labeled: {'Yes' if is_labeled else 'No'}
    {f'- Target column: {target_column}' if target_column else ''}
    
    User description: {user_description if user_description else 'Not provided'}
    
    Column information:
    """

    for col_name, col_info in dataset_info['columns'].items():
        prompt += f"\n- {col_name} ({col_info['type']})"
        
        if col_info['type'] == 'numeric':
            stats = col_info['stats']
            prompt += f": Range {stats['min']} to {stats['max']}, Mean {stats['mean']:.2f}, Median {stats['median']:.2f}"
        elif col_info['type'] == 'categorical':
            stats = col_info['stats']
            prompt += f": {stats['unique_values']} unique values"
            if stats['unique_values'] <= 10:
                top_values = ", ".join([f"{k} ({v})" for k, v in list(stats['top_values'].items())[:5]])
                prompt += f" - Top values: {top_values}"
    

    if dataset_info['target_info']:
        prompt += f"\n\nTarget information:"
        target_info = dataset_info['target_info']
        if target_info['type'] == 'classification':
            classes = ", ".join([str(c) for c in target_info['classes'][:10]])
            prompt += f"\n- Classification task with classes: {classes}"
            if len(target_info['classes']) > 10:
                prompt += f" and {len(target_info['classes']) - 10} more"
        else:
            prompt += f"\n- Regression task with target range: {target_info['min']} to {target_info['max']}"
    

    if dataset_info['correlation']:
        prompt += "\n\nThere is correlation information available for numeric columns."
    
    prompt += """
    
    Based on this information, please provide:
    1. A detailed description of this dataset
    2. The likely domain or field this data belongs to
    3. Key characteristics and patterns to look for
    4. Recommended preprocessing steps
    5. Potential challenges in modeling this data
    
    Format your response as a clear summary that could help someone understand this dataset quickly.
    """
    
    return prompt

def get_llm_description(
    dataset_info: Dict[str, Any],
    user_description: str,
    task_type: str,
    is_labeled: bool,
    target_column: Optional[str] = None
) -> str:
    """
    Get dataset description using the LLM API.
    
    Args:
        dataset_info: Dataset information dictionary
        user_description: User-provided description
        task_type: Type of task (classification/regression)
        is_labeled: Whether the dataset is labeled
        target_column: Name of the target column
    
    Returns:
        Generated description
    """
    try:

        api_key = "8418379e37ab69c2db94857405652801b3a7ae776e3cc0155a78a175d5d4668d"
   
        if not api_key:
            return generate_fallback_description(dataset_info, user_description, task_type, is_labeled, target_column)

        prompt = prepare_llm_prompt(dataset_info, user_description, task_type, is_labeled, target_column)

        try:
            response = requests.post(
                "https://api.together.xyz/v1/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "prompt": prompt,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                timeout=10 
            )
            
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['text'].strip()
            else:
                return generate_fallback_description(dataset_info, user_description, task_type, is_labeled, target_column)
        
        except Exception as e:
            print(f"Error calling Together API: {str(e)}")
            return generate_fallback_description(dataset_info, user_description, task_type, is_labeled, target_column)
    
    except Exception as e:
        print(f"Error in get_llm_description: {str(e)}")
        return (
            f"Dataset summary:\n"
            f"This dataset contains {dataset_info['num_rows']} rows and {dataset_info['num_columns']} columns. "
            f"It appears to be a {task_type} task. "
            f"{'The target variable is ' + target_column if target_column else 'No target variable was specified.'}"
        )