import pandas as pd # type: ignore
import numpy as np
import importlib
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, RandomizedSearchCV # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.pipeline import Pipeline # type: ignore

from preprocessing import identify_column_types, create_preprocessor
from evaluation import evaluate_predictions

def train_single_model(model_config, preprocessor, X_train, y_train, X_test, y_test, task_type):
    """
    Train a single model with hyperparameter optimization.
    
    Args:
        model_config: Configuration for the model
        preprocessor: Preprocessor pipeline
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        task_type: Type of task (classification/regression)
    
    Returns:
        Dictionary with model training results
    """
    model_name = model_config['name']
    model_class_path = model_config['library']
    model_class_name = model_config['class']
    
    print(f"Starting training for {model_name}")
    
    try:
        module_path, class_name = model_class_path, model_class_name
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        print(f"Successfully imported {model_class_path}.{model_class_name}")
        
        if 'xgboost' in model_name:
            if task_type == 'classification':
                num_classes = len(np.unique(y_train))
                if num_classes == 2:
                    model_instance = ModelClass(objective='binary:logistic')
                else:
                    model_instance = ModelClass(objective='multi:softprob', num_class=num_classes)
            else:
                model_instance = ModelClass(objective='reg:squarederror')
        else:
            model_instance = ModelClass()
        

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_instance)
        ])
        
        model_search_spaces = {
            'random_forest': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5],
                'model__subsample': [0.8, 1.0]
            },
            'xgboost': {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5],
                'model__colsample_bytree': [0.8, 1.0]
            }
        }
        

        search_space = model_search_spaces.get(model_name, {})

        if not search_space and 'hyperparameters' in model_config:
            search_space = {f'model__{param}': values[:2] if len(values) > 2 else values 
                            for param, values in model_config['hyperparameters'].items()}

        if search_space:
            scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
            
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=search_space,
                n_iter=5, 
                cv=3,     
                scoring=scoring,
                n_jobs=-1, 
                random_state=42
            )
            
            print(f"Starting RandomizedSearchCV for {model_name} with {len(search_space)} parameters")
            start_time = time.time()
            search.fit(X_train, y_train)
            train_time = time.time() - start_time
    
            best_model = search.best_estimator_
            best_params = search.best_params_

            y_pred = best_model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, task_type)
            
            return {
                'name': model_name,
                'model': best_model,
                'best_params': {k.replace('model__', ''): v for k, v in best_params.items()},
                'metrics': metrics,
                'training_time': train_time
            }
        

        else:
            print(f"Training {model_name} with default parameters")
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = pipeline.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, task_type)
            
            return {
                'name': model_name,
                'model': pipeline,
                'metrics': metrics,
                'training_time': train_time
            }
            
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'name': model_name,
            'error': str(e)
        }

def train_supervised_models(
    df: pd.DataFrame,
    selected_models: List[Dict[str, Any]],
    task_type: str,
    target_column: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train supervised models with handling for string target labels.
    
    Args:
        df: DataFrame with dataset
        selected_models: List of selected model configurations
        task_type: Type of task (classification/regression)
        target_column: Name of the target column for supervised learning
    
    Returns:
        Tuple of (trained_models_dict, evaluation_results_dict)
    """

    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    

    X = df.drop(columns=[target_column])
    y_raw = df[target_column]

    print(f"Target data type: {y_raw.dtype}, unique values: {y_raw.unique()}")
    
    class_mapping = None
    if task_type == 'classification' and (y_raw.dtype == 'object' or pd.api.types.is_categorical_dtype(y_raw)):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)

        class_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
        print(f"Encoded target classes. Mapping: {class_mapping}")
    else:
        y = y_raw

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Split data. Training set: {X_train.shape}, Test set: {X_test.shape}")
    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        raise ValueError(f"Error splitting data: {str(e)}")

    numeric_features, categorical_features = identify_column_types(X)

    preprocessor = create_preprocessor(numeric_features, categorical_features)

    trained_models = {}
    evaluation_results = {}
    
    print(f"Training {len(selected_models)} models in parallel")
    with ThreadPoolExecutor(max_workers=min(len(selected_models), 3)) as executor:
        future_to_model = {
            executor.submit(
                train_single_model, 
                model_config, 
                preprocessor, 
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                task_type
            ): model_config['name'] 
            for model_config in selected_models
        }
        
       
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                if 'error' in result:
                    print(f"Model {model_name} failed: {result['error']}")
                else:
                    print(f"Model {model_name} completed successfully")
                    trained_models[model_name] = result['model']
                    evaluation_results[model_name] = {
                        'metrics': result['metrics'],
                        'training_time': result['training_time']
                    }
                    if 'best_params' in result:
                        evaluation_results[model_name]['best_params'] = result['best_params']
            except Exception as e:
                print(f"Exception handling result for {model_name}: {str(e)}")
    
    if not trained_models:
        raise ValueError("All models failed to train. Check the logs for details.")
    
    if class_mapping is not None:
        for model_name in evaluation_results:
            if model_name in trained_models:
                evaluation_results[model_name]['class_mapping'] = class_mapping
    
    print(f"Successfully trained {len(trained_models)} models")
    return trained_models, evaluation_results