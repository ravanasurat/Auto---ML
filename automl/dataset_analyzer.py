
from typing import Dict, List, Optional, Any

from dataset_loader import load_dataset
from dataset_analysis import analyze_dataset
from description_generator import get_llm_description, generate_fallback_description

def analyze_and_describe_dataset(
    file_path: str,
    user_description: str = "",
    task_type: str = "unknown",
    is_labeled: bool = False,
    target_column: Optional[str] = None,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Complete function to analyze a dataset and generate a description.
    
    Args:
        file_path: Path to the dataset file
        user_description: User-provided description
        task_type: Type of task (classification/regression/clustering)
        is_labeled: Whether the dataset is labeled
        target_column: Name of the target column
        use_llm: Whether to use LLM for description generation
    
    Returns:
        Dictionary with analysis results and description
    """

    dataset_info = analyze_dataset(file_path, target_column)
    
    if use_llm:
        description = get_llm_description(
            dataset_info,
            user_description,
            task_type,
            is_labeled,
            target_column
        )
    else:
        description = generate_fallback_description(
            dataset_info,
            user_description,
            task_type,
            is_labeled,
            target_column
        )

    dataset_info['description'] = description
    
    return dataset_info

if __name__ == "__main__":
    result = analyze_and_describe_dataset(
        file_path="example.csv",
        task_type="classification",
        is_labeled=True,
        target_column="target"
    )
    print(result['description'])