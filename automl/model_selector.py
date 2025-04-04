import pandas as pd # type: ignore
import numpy as np
from typing import Dict, List, Any, Optional

def select_models(
    llm_description: str,
    dataset_info: Dict[str, Any],
    task_type: str,
    is_labeled: bool
) -> List[Dict[str, Any]]:
    """
    Select appropriate models based on dataset characteristics and LLM description.
    
    Args:
        llm_description: Description generated by LLM
        dataset_info: Dataset information dictionary
        task_type: Type of task (classification/regression)
        is_labeled: Whether the dataset is labeled
    
    Returns:
        List of selected models with configurations
    """
    selected_models = []
    

    if not is_labeled:

        selected_models.extend([
            {
                'name': 'kmeans',
                'display_name': 'K-Means Clustering',
                'type': 'clustering',
                'library': 'sklearn.cluster',
                'class': 'KMeans',
                'requires_scaling': True,
                'hyperparameters': {
                    'n_clusters': [3, 5, 8, 10, 15],
                    'init': ['k-means++', 'random'],
                    'n_init': [10, 15, 20]
                }
            },
            {
                'name': 'dbscan',
                'display_name': 'DBSCAN',
                'type': 'clustering',
                'library': 'sklearn.cluster',
                'class': 'DBSCAN',
                'requires_scaling': True,
                'hyperparameters': {
                    'eps': [0.1, 0.3, 0.5, 0.8, 1.0],
                    'min_samples': [5, 10, 15, 20],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            {
                'name': 'hierarchical',
                'display_name': 'Hierarchical Clustering',
                'type': 'clustering',
                'library': 'sklearn.cluster',
                'class': 'AgglomerativeClustering',
                'requires_scaling': True,
                'hyperparameters': {
                    'n_clusters': [3, 5, 8, 10],
                    'linkage': ['ward', 'complete', 'average'],
                    'affinity': ['euclidean', 'manhattan', 'cosine']
                }
            }
        ])

        selected_models.extend([
            {
                'name': 'pca',
                'display_name': 'Principal Component Analysis',
                'type': 'dimensionality_reduction',
                'library': 'sklearn.decomposition',
                'class': 'PCA',
                'requires_scaling': True,
                'hyperparameters': {
                    'n_components': [2, 3, 5, 10],
                    'svd_solver': ['auto', 'full']
                }
            }
        ])
        
        return selected_models
    

    target_info = dataset_info.get('target_info', {})
    target_type = target_info.get('type') if target_info else task_type

    if task_type == 'classification':
        num_classes = 0
        if target_info and 'classes' in target_info:
            num_classes = len(target_info['classes'])
        

        is_binary = num_classes == 2

        classification_models = [
            {
                'name': 'random_forest',
                'display_name': 'Random Forest',
                'type': 'classification',
                'library': 'sklearn.ensemble',
                'class': 'RandomForestClassifier',
                'requires_scaling': False,
                'hyperparameters': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
            },
            {
                'name': 'gradient_boosting',
                'display_name': 'Gradient Boosting',
                'type': 'classification',
                'library': 'sklearn.ensemble',
                'class': 'GradientBoostingClassifier',
                'requires_scaling': False,
                'hyperparameters': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            {
                'name': 'logistic_regression',
                'display_name': 'Logistic Regression',
                'type': 'classification',
                'library': 'sklearn.linear_model',
                'class': 'LogisticRegression',
                'requires_scaling': True,
                'hyperparameters': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'solver': ['saga', 'liblinear'],
                    'max_iter': [1000]
                }
            }
        ]
        

        classification_models.append({
            'name': 'xgboost',
            'display_name': 'XGBoost',
            'type': 'classification',
            'library': 'xgboost',
            'class': 'XGBClassifier',
            'requires_scaling': False,
            'hyperparameters': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        })
        

        dataset_rows = dataset_info.get('num_rows', 0)
        if dataset_rows > 1000:
            classification_models.append({
                'name': 'mlp',
                'display_name': 'Neural Network',
                'type': 'classification',
                'library': 'sklearn.neural_network',
                'class': 'MLPClassifier',
                'requires_scaling': True,
                'hyperparameters': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'sgd'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [1000]
                }
            })

        selected_models = [
            classification_models[0],  
            classification_models[1], 
            classification_models[3],  
        ]
    
    elif task_type == 'regression':
        regression_models = [
            {
                'name': 'random_forest_regressor',
                'display_name': 'Random Forest Regressor',
                'type': 'regression',
                'library': 'sklearn.ensemble',
                'class': 'RandomForestRegressor',
                'requires_scaling': False,
                'hyperparameters': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
            },
            {
                'name': 'gradient_boosting_regressor',
                'display_name': 'Gradient Boosting Regressor',
                'type': 'regression',
                'library': 'sklearn.ensemble',
                'class': 'GradientBoostingRegressor',
                'requires_scaling': False,
                'hyperparameters': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            {
                'name': 'linear_regression',
                'display_name': 'Linear Regression',
                'type': 'regression',
                'library': 'sklearn.linear_model',
                'class': 'LinearRegression',
                'requires_scaling': True,
                'hyperparameters': {} 
            },
            {
                'name': 'elastic_net',
                'display_name': 'Elastic Net',
                'type': 'regression',
                'library': 'sklearn.linear_model',
                'class': 'ElasticNet',
                'requires_scaling': True,
                'hyperparameters': {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [1000]
                }
            }
        ]
        

        regression_models.append({
            'name': 'xgboost_regressor',
            'display_name': 'XGBoost Regressor',
            'type': 'regression',
            'library': 'xgboost',
            'class': 'XGBRegressor',
            'requires_scaling': False,
            'hyperparameters': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        })
        
        dataset_rows = dataset_info.get('num_rows', 0)
        if dataset_rows > 1000:
            regression_models.append({
                'name': 'mlp_regressor',
                'display_name': 'Neural Network Regressor',
                'type': 'regression',
                'library': 'sklearn.neural_network',
                'class': 'MLPRegressor',
                'requires_scaling': True,
                'hyperparameters': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'sgd'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [1000]
                }
            })
        
        selected_models = [
            regression_models[0], 
            regression_models[1],  
            regression_models[4], 
        ]

    elif task_type == 'time_series':

        time_series_models = [
            {
                'name': 'arima',
                'display_name': 'ARIMA',
                'type': 'time_series',
                'library': 'statsmodels.tsa.arima.model',
                'class': 'ARIMA',
                'requires_scaling': False,
                'hyperparameters': {
                    'p': [1, 2, 3, 4, 5],
                    'd': [0, 1, 2],
                    'q': [0, 1, 2]
                }
            },
            {
                'name': 'prophet',
                'display_name': 'Prophet',
                'type': 'time_series',
                'library': 'prophet',
                'class': 'Prophet',
                'requires_scaling': False,
                'hyperparameters': {
                    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                    'seasonality_mode': ['additive', 'multiplicative']
                }
            },
            {
                'name': 'exponential_smoothing',
                'display_name': 'Exponential Smoothing',
                'type': 'time_series',
                'library': 'statsmodels.tsa.holtwinters',
                'class': 'ExponentialSmoothing',
                'requires_scaling': False,
                'hyperparameters': {
                    'trend': [None, 'add', 'mul'],
                    'seasonal': [None, 'add', 'mul'],
                    'seasonal_periods': [4, 7, 12, 52]
                }
            }
        ]
        

        selected_models = time_series_models[:3]

    num_columns = dataset_info.get('num_columns', 0)
    if num_columns > 50:
        high_dim_models = [
            {
                'name': 'ridge',
                'display_name': 'Ridge Regression',
                'type': 'regression' if task_type == 'regression' else 'classification',
                'library': 'sklearn.linear_model',
                'class': 'Ridge' if task_type == 'regression' else 'RidgeClassifier',
                'requires_scaling': True,
                'hyperparameters': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                }
            }
        ]
     
        if len(selected_models) > 0:
            selected_models[-1] = high_dim_models[0]
    if task_type == 'classification' and target_info and 'distribution' in target_info:
        distribution = target_info['distribution']
        values = list(distribution.values())
        if len(values) > 0:
            max_val = max(values)
            min_val = min(values)
            imbalance_ratio = max_val / min_val if min_val > 0 else float('inf')
            
            if imbalance_ratio > 10: 
                imbalanced_model = {
                    'name': 'balanced_random_forest',
                    'display_name': 'Balanced Random Forest',
                    'type': 'classification',
                    'library': 'imblearn.ensemble',
                    'class': 'BalancedRandomForestClassifier',
                    'requires_scaling': False,
                    'hyperparameters': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'sampling_strategy': ['auto', 'majority', 'not minority', 'not majority', 'all']
                    }
                }
                
                if len(selected_models) > 0:
                    selected_models[0] = imbalanced_model
    if len(selected_models) > 3:
        selected_models = selected_models[:3]
    
    return selected_models