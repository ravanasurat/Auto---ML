import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from typing import Dict, Any
from sklearn.decomposition import PCA # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score # type: ignore
from file_utils import plot_to_base64

def evaluate_unsupervised_models(
    df: pd.DataFrame,
    trained_models: Dict[str, Any],
    task_type: str
) -> Dict[str, Any]:
    """
    Evaluate unsupervised models (clustering, dimensionality reduction).
    
    Args:
        df: DataFrame with dataset
        trained_models: Dictionary of trained models
        task_type: Type of task
    
    Returns:
        Dictionary containing evaluation results and visualizations
    """
    evaluation_results = {
        'models': {},
        'visualizations': {}
    }
    
    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name}...")
        
        if task_type == 'clustering':
            
            try:
                
                cluster_labels = model.predict(df)
                
                
                unique_clusters = np.unique(cluster_labels)
                num_clusters = len(unique_clusters)
                if -1 in unique_clusters:  # For DBSCAN, -1 is noise
                    num_clusters -= 1
                
                
                silhouette_avg = None
                if num_clusters > 1:
                    silhouette_avg = silhouette_score(df, cluster_labels)
                
                
                db_index = None
                if num_clusters > 1:
                    db_index = davies_bouldin_score(df, cluster_labels)
                
               
                evaluation_results['models'][model_name] = {
                    'metrics': {
                        'num_clusters': num_clusters,
                        'silhouette_score': silhouette_avg,
                        'davies_bouldin_index': db_index
                    }
                }
                
                
                if df.shape[1] > 2:
                    pca = PCA(n_components=2)
                    df_pca = pca.fit_transform(df)
                    
                   
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, label='Cluster')
                    plt.title(f'Cluster Visualization - {model_name}')
                    plt.xlabel('PCA Component 1')
                    plt.ylabel('PCA Component 2')
                    
                    
                    img = plot_to_base64(plt)
                    plt.close()
                    
                    if 'cluster_visualization' not in evaluation_results['visualizations']:
                        evaluation_results['visualizations']['cluster_visualization'] = {}
                    
                    evaluation_results['visualizations']['cluster_visualization'][model_name] = img
            
            except Exception as e:
                print(f"Error evaluating clustering model {model_name}: {e}")
        
        elif task_type == 'dimensionality_reduction':
            
            try:
                
                X_transformed = model.transform(df)
                
                
                explained_variance = None
                cumulative_variance = None
                
                if hasattr(model, 'explained_variance_ratio_'):
                    explained_variance = model.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance)
                
                
                evaluation_results['models'][model_name] = {
                    'metrics': {
                        'n_components': X_transformed.shape[1],
                        'explained_variance': explained_variance.tolist() if explained_variance is not None else None,
                        'cumulative_variance': cumulative_variance.tolist() if cumulative_variance is not None else None
                    }
                }
                
               
                if explained_variance is not None:
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
                    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-')
                    plt.grid(True, alpha=0.3)
                    plt.title(f'Explained Variance - {model_name}')
                    plt.xlabel('Principal Components')
                    plt.ylabel('Explained Variance Ratio')
                    plt.tight_layout()
                    
                    
                    img = plot_to_base64(plt)
                    plt.close()
                    
                    if 'explained_variance' not in evaluation_results['visualizations']:
                        evaluation_results['visualizations']['explained_variance'] = {}
                    
                    evaluation_results['visualizations']['explained_variance'][model_name] = img
                
                
                if X_transformed.shape[1] >= 2:
                    plt.figure(figsize=(10, 8))
                    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
                    plt.title(f'2D Projection - {model_name}')
                    plt.xlabel('Component 1')
                    plt.ylabel('Component 2')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    
                    img = plot_to_base64(plt)
                    plt.close()
                    
                    if '2d_projection' not in evaluation_results['visualizations']:
                        evaluation_results['visualizations']['2d_projection'] = {}
                    
                    evaluation_results['visualizations']['2d_projection'][model_name] = img
            
            except Exception as e:
                print(f"Error evaluating dimensionality reduction model {model_name}: {e}")
    
    return evaluation_results