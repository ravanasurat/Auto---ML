import os
import pandas as pd # type: ignore
import json
from flask import Flask, request, render_template, jsonify, send_file
import joblib # type: ignore
from werkzeug.utils import secure_filename
import numpy as np
from dataset_analyzer import analyze_dataset, get_llm_description
from model_selector import select_models
from automl_trainer import train_models
from model_evaluator import evaluate_models, predict_with_models
from utils import allowed_file, create_temp_directory, clean_temp_directory

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.getcwd(), 'models')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    

os.makedirs('templates', exist_ok=True)


session_data = {
    'dataset_path': None,
    'dataset_description': None,
    'llm_description': None,
    'task_type': None,
    'is_labeled': None,
    'target_column': None,
    'selected_models': None,
    'trained_models': None,
    'evaluation_results': None
}

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'datafile' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['datafile']
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        print(f"File received: {file.filename}")
        
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            for key in session_data:
                session_data[key] = None
 
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session_data['dataset_path'] = filepath
            
            print(f"File saved to: {filepath}")

            try:
                if filepath.endswith('.csv'):
                    print("Reading CSV file")
                    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
                    
                    for encoding in encodings_to_try:
                        try:
                            print(f"Trying encoding: {encoding}")
                            df = pd.read_csv(filepath, encoding=encoding)
                            print(f"Successfully read with encoding: {encoding}")
                            break
                        except UnicodeDecodeError:
                            print(f"Failed with encoding: {encoding}")
                            continue
                    else:
                        print("All standard encodings failed, trying with error handling")
                        df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
                        
                elif filepath.endswith(('.xls', '.xlsx')):
                    print("Reading Excel file")
                    df = pd.read_excel(filepath)
                else:
                    print(f"Unsupported file format: {filepath}")
                    return jsonify({'error': 'Unsupported file format'}), 400
                
                print(f"Data shape: {df.shape}")
                df_clean = df.copy()
                for col in df_clean.columns:
                    if pd.api.types.is_object_dtype(df_clean[col]):
                        print(f"Converting column to string: {col}")
                        df_clean[col] = df_clean[col].astype(str)
                    elif pd.isna(df_clean[col]).any():
                        print(f"Column contains NaN values: {col}")
                        df_clean[col] = df_clean[col].fillna('null').astype(str)
                
                preview = df_clean.head(5).to_dict('records')
                columns = df_clean.columns.tolist()
                try:
                    json.dumps(preview)
                    print("Preview successfully serialized")
                except Exception as json_error:
                    print(f"JSON serialization error: {str(json_error)}")
                    preview = [
                        {col: str(row[col]) for col in df_clean.columns}
                        for _, row in df_clean.head(5).iterrows()
                    ]
                
                return jsonify({
                    'success': True,
                    'preview': preview,
                    'columns': columns,
                    'rows': len(df)
                })
            except Exception as e:
                print(f"Error reading file: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        else:
            print(f"File type not allowed: {file.filename}")
            return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("Analyze endpoint called")
        data = request.json
        print(f"Request data: {data}")
        
        user_description = data.get('description', '')
        task_type = data.get('task_type')
        is_labeled = data.get('is_labeled', False)
        target_column = data.get('target_column', '')
        
        print(f"Task type: {task_type}")
        print(f"Is labeled: {is_labeled}")
        print(f"Target column: {target_column}")
        
        if not session_data['dataset_path']:
            print("No dataset uploaded")
            return jsonify({'error': 'No dataset uploaded'}), 400
        
        if is_labeled and not target_column:
            print("Target column missing for labeled data")
            return jsonify({'error': 'Target column must be specified for labeled data'}), 400
        
        print(f"Dataset path: {session_data['dataset_path']}")
        
        session_data['task_type'] = task_type
        session_data['is_labeled'] = is_labeled
        session_data['target_column'] = target_column
        session_data['dataset_description'] = user_description
        print("Analyzing dataset...")
        try:
            dataset_info = analyze_dataset(session_data['dataset_path'], target_column if is_labeled else None)
            print("Dataset analysis completed")
        except Exception as analyze_error:
            print(f"Error analyzing dataset: {str(analyze_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error analyzing dataset: {str(analyze_error)}'}), 400
        print("Getting LLM description...")
        try:
            llm_description = get_llm_description(
                dataset_info,
                user_description,
                task_type,
                is_labeled,
                target_column
            )
            print("LLM description completed")
        except Exception as llm_error:
            print(f"Error getting LLM description: {str(llm_error)}")
            import traceback
            traceback.print_exc()
            llm_description = f"Dataset summary: This dataset contains {dataset_info['num_rows']} rows and {dataset_info['num_columns']} columns. It appears to be a {task_type} task."
            print("Using fallback description")
        
        session_data['llm_description'] = llm_description
        
        print("Selecting models...")
        try:
            selected_models = select_models(
                llm_description,
                dataset_info,
                task_type,
                is_labeled
            )
            print(f"Selected {len(selected_models)} models")
        except Exception as model_error:
            print(f"Error selecting models: {str(model_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error selecting models: {str(model_error)}'}), 400
        
        session_data['selected_models'] = selected_models
        
        print("Preparing response")
        try:
            for col_name, col_info in dataset_info['columns'].items():
                if col_info['type'] == 'numeric':
                    stats = col_info['stats']
                    for stat_key, stat_value in stats.items():
                        if isinstance(stat_value, (np.float32, np.float64)):
                            stats[stat_key] = float(stat_value)
                        elif isinstance(stat_value, (np.int32, np.int64)):
                            stats[stat_key] = int(stat_value)
            
            if dataset_info.get('target_info'):
                target_info = dataset_info['target_info']
                if 'distribution' in target_info:
                    new_dist = {}
                    for k, v in target_info['distribution'].items():
                        if isinstance(k, (np.int32, np.int64, np.float32, np.float64)):
                            k = str(k)
                        if isinstance(v, (np.int32, np.int64, np.float32, np.float64)):
                            v = int(v)
                        new_dist[k] = v
                    target_info['distribution'] = new_dist
                
                if 'classes' in target_info:
                    target_info['classes'] = [str(c) if isinstance(c, (np.int32, np.int64, np.float32, np.float64)) else c for c in target_info['classes']]
                
                for key, value in target_info.items():
                    if isinstance(value, (np.int32, np.int64)):
                        target_info[key] = int(value)
                    elif isinstance(value, (np.float32, np.float64)):
                        target_info[key] = float(value)
        
            if dataset_info.get('correlation'):
                for col1, col_values in dataset_info['correlation'].items():
                    for col2, value in col_values.items():
                        if isinstance(value, (np.float32, np.float64)):
                            dataset_info['correlation'][col1][col2] = float(value)
            for model in selected_models:
                if 'hyperparameters' in model:
                    for param, values in model['hyperparameters'].items():
                        if isinstance(values, np.ndarray):
                            model['hyperparameters'][param] = values.tolist()
            
            print("Successfully prepared response")
            
            response_data = {
                'success': True,
                'dataset_info': dataset_info,
                'llm_description': llm_description,
                'selected_models': selected_models
            }
            
            try:
                json.dumps(response_data)
                print("Response data successfully serialized")
            except TypeError as json_error:
                print(f"JSON serialization error: {str(json_error)}")
                response_data = {
                    'success': True,
                    'llm_description': llm_description,
                    'selected_models': selected_models
                }
                
            return jsonify(response_data)
            
        except Exception as serialization_error:
            print(f"Error serializing response: {str(serialization_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': True,
                'llm_description': llm_description,
                'error_details': 'Could not serialize full dataset info, but analysis completed successfully.'
            })
    except Exception as e:
        print(f"Overall error in analyze endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        print("Train endpoint called")
        
        if not session_data['dataset_path'] or not session_data['selected_models']:
            print("Missing dataset or models")
            return jsonify({'error': 'Dataset and models must be selected first'}), 400
        
        print(f"Dataset path: {session_data['dataset_path']}")
        print(f"Task type: {session_data['task_type']}")
        print(f"Is labeled: {session_data['is_labeled']}")
        print(f"Target column: {session_data['target_column']}")
        print(f"Selected models: {[m['name'] for m in session_data['selected_models']]}")

        try:
            if session_data['dataset_path'].endswith('.csv'):
                encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
                
                for encoding in encodings_to_try:
                    try:
                        print(f"Trying encoding: {encoding}")
                        df = pd.read_csv(session_data['dataset_path'], encoding=encoding)
                        print(f"Successfully read with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        print(f"Failed with encoding: {encoding}")
                        continue
                else:
                    print("All standard encodings failed, trying with error handling")
                    df = pd.read_csv(session_data['dataset_path'], encoding='latin1', on_bad_lines='skip')
            elif session_data['dataset_path'].endswith(('.xls', '.xlsx')):
                df = pd.read_excel(session_data['dataset_path'])
            else:
                print(f"Unsupported file format: {session_data['dataset_path']}")
                return jsonify({'error': 'Unsupported file format'}), 400
                
            print(f"Dataset loaded with shape: {df.shape}")
        except Exception as file_error:
            print(f"Error reading dataset: {str(file_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error reading dataset: {str(file_error)}'}), 500
        print("Starting model training...")
        try:
            trained_models, evaluation_results = train_models(
                session_data['dataset_path'],
                session_data['selected_models'],
                session_data['task_type'],
                session_data['is_labeled'],
                session_data['target_column'] if session_data['is_labeled'] else None
            )
            
            print(f"Training completed with {len(trained_models)} models")
            session_data['trained_models'] = trained_models
            session_data['evaluation_results'] = evaluation_results
        except Exception as training_error:
            print(f"Error during model training: {str(training_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error during model training: {str(training_error)}'}), 500
        print("Starting model evaluation...")
        try:
            full_evaluation = evaluate_models(
                session_data['dataset_path'],
                trained_models,
                session_data['task_type'],
                session_data['target_column'] if session_data['is_labeled'] else None
            )
            print("Evaluation completed")
        except Exception as eval_error:
            print(f"Error during model evaluation: {str(eval_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error during model evaluation: {str(eval_error)}'}), 500
        print("Preparing response...")
        try:
            from utils import convert_dict_types_for_json
            json_safe_evaluation = convert_dict_types_for_json(full_evaluation)
            try:
                json.dumps(json_safe_evaluation)
                print("Response successfully serialized")
            except TypeError as json_error:
                print(f"JSON serialization error: {str(json_error)}")
                json_safe_evaluation = {
                    'models': {
                        model_name: {
                            'metrics': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                                      for k, v in model_info.get('metrics', {}).items() 
                                      if k != 'confusion_matrix'}
                        }
                        for model_name, model_info in full_evaluation.get('models', {}).items()
                    }
                }
            
            return jsonify({
                'success': True,
                'evaluation_results': json_safe_evaluation
            })
        except Exception as response_error:
            print(f"Error preparing response: {str(response_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Training completed but error preparing results: {str(response_error)}'}), 500
    except Exception as e:
        print(f"Overall error in train endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = data.get('input_data', {})
        
        if not session_data['trained_models']:
            return jsonify({'error': 'No trained models available'}), 400
        
        predictions = predict_with_models(
            session_data['trained_models'],
            input_data,
            session_data['task_type']
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/download_model/<model_name>', methods=['GET'])
def download_model(model_name):
    try:
        if not session_data['trained_models'] or model_name not in session_data['trained_models']:
            return jsonify({'error': 'Model not found'}), 404
    
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_name}.joblib")
        joblib.dump(session_data['trained_models'][model_name], model_path)
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f"{model_name}.joblib",
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return jsonify({'error': f'Error downloading model: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)