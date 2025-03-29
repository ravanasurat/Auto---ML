import pandas as pd # type: ignore
import importlib
import time
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import silhouette_score # type: ignore

from preprocessing import identify_column_types, create_preprocessor

def train_unsupervised_models(
    df: pd.DataFrame,
    selected_models: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train unsupervised models (clustering, dimensionality reduction).
    
    Args:
        df: DataFrame with dataset
        selected_models: List of selected model configurations
    
    Returns:
        Tuple of (trained_models_dict, evaluation_results_dict)
    """

    numeric_features, categorical_features = identify_column_types(df)
    
  
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    X_processed = preprocessor.fit_transform(df)
    
    trained_models = {}
    evaluation_results = {}
    
    for model_config in selected_models:
        model_name = model_config['name']
        model_class_path = model_config['library']
        model_class_name = model_config['class']
        hyperparameters = model_config.get('hyperparameters', {})
        model_type = model_config['type']
        
        print(f"Training {model_name}...")
        
   
        try:
            module_path, class_name = model_class_path, model_class_name
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"Error importing {model_class_path}.{model_class_name}: {e}")
            continue
     
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', ModelClass())
        ])
        
      
        if hyperparameters and len(hyperparameters) > 0:
            param_grid = {f'model__{param}': values for param, values in hyperparameters.items()}
            
            scoring = None
            if model_type == 'clustering':
                def silhouette_scorer(estimator, X):
                    if model_class_name == 'DBSCAN':
                        labels = estimator.fit_predict(X)
                        if len(set(labels)) <= 1 or -1 in labels:
                            return -1
                        return silhouette_score(X, labels)
                    
                    clusters = estimator.fit_predict(X)
                    if len(set(clusters)) <= 1:
                        return -1
                    return silhouette_score(X, clusters)
                
                scoring = silhouette_scorer
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,  
                scoring=scoring,
                n_jobs=-1
            )
            
            try:
                start_time = time.time()
                
                if scoring is not None:

                    grid_search.fit(X_processed)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    first_params = {k: v[0] for k, v in param_grid.items()}
                    for k, v in first_params.items():
                        setattr(pipeline.named_steps['model'], k.replace('model__', ''), v)
                    
                    pipeline.fit(X_processed)
                    best_model = pipeline
                    best_params = first_params
                
                train_time = time.time() - start_time
                
                trained_models[model_name] = best_model

                if model_type == 'clustering':
                    clusters = best_model.predict(X_processed)
                    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    
                    evaluation_results[model_name] = {
                        'metrics': {
                            'num_clusters': num_clusters,
                            'silhouette_score': silhouette_score(X_processed, clusters) if num_clusters > 1 else -1
                        },
                        'best_params': {k.replace('model__', ''): v for k, v in best_params.items()},
                        'training_time': train_time
                    }
                elif model_type == 'dimensionality_reduction':

                    explained_variance = None
                    if hasattr(best_model.named_steps['model'], 'explained_variance_ratio_'):
                        explained_variance = best_model.named_steps['model'].explained_variance_ratio_.sum()
                    
                    evaluation_results[model_name] = {
                        'metrics': {
                            'explained_variance': explained_variance,
                            'n_components': best_model.named_steps['model'].n_components
                        },
                        'best_params': {k.replace('model__', ''): v for k, v in best_params.items()},
                        'training_time': train_time
                    }
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        else:
  
            try:
                start_time = time.time()
                pipeline.fit(X_processed)
                train_time = time.time() - start_time
                
                trained_models[model_name] = pipeline
                
                if model_type == 'clustering':
                    clusters = pipeline.predict(X_processed)
                    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    
                    evaluation_results[model_name] = {
                        'metrics': {
                            'num_clusters': num_clusters,
                            'silhouette_score': silhouette_score(X_processed, clusters) if num_clusters > 1 else -1
                        },
                        'training_time': train_time
                    }
                elif model_type == 'dimensionality_reduction':
                    explained_variance = None
                    if hasattr(pipeline.named_steps['model'], 'explained_variance_ratio_'):
                        explained_variance = pipeline.named_steps['model'].explained_variance_ratio_.sum()
                    
                    evaluation_results[model_name] = {
                        'metrics': {
                            'explained_variance': explained_variance,
                            'n_components': pipeline.named_steps['model'].n_components
                        },
                        'training_time': train_time
                    }
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
    
    return trained_models, evaluation_results