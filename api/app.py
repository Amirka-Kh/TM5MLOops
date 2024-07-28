import json

from flask import Flask, request, jsonify, abort, make_response
import mlflow
import mlflow.pyfunc
import os
import pandas as pd
from omegaconf import DictConfig, OmegaConf


# Set up paths and model loading
BASE_PATH = os.path.expandvars("$PYTHONPATH")
model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

cfg = OmegaConf.load('./api/model_dir/registered_model_meta')
client = mlflow.MlflowClient()
model_uri = client.get_model_version_download_uri(cfg.model_name, cfg.model_version)
model_info = mlflow.models.get_model_info(model_uri)
signature_dict = model_info._signature_dict
signature_inputs = json.loads(signature_dict['inputs'])

input_types = {input_spec['name']: input_spec['type'] for input_spec in signature_inputs}
int_keys = {k for k, v in input_types.items() if v in ['long', 'integer']}
float_keys = {k for k, v in input_types.items() if v == 'double'}

expected_schema = {}
for k, v in input_types.items():
    if v == 'long':
        expected_schema[k] = 'int64'
    elif v == 'integer':
        expected_schema[k] = 'int32'
    elif v == 'double':
        expected_schema[k] = 'float64'
    else:
        expected_schema[k] = 'object'


app = Flask(__name__)


@app.route("/info", methods=["GET"])
def info():
    # Fetch model metadata
    response = make_response(str(model.metadata), 200)
    response.content_type = "text/plain"
    return response


@app.route("/", methods=["GET"])
def home():
    msg = """
    Welcome to our ML service to predict Customer satisfaction\n\n

    This API has two main endpoints:\n
    1. /info: to get info about the deployed model.\n
    2. /predict: to send predict requests to our deployed model.\n
    """
    response = make_response(msg, 200)
    response.content_type = "text/plain"
    return response


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the incoming JSON request
        data = request.get_json()

        if not data or 'inputs' not in data:
            abort(400, description="Invalid request format. Must include 'inputs' key.")

        # Extract feature values from the JSON
        inputs = data['inputs']

        def convert_value(value, key):
            """Convert a single value based on the key."""
            if key in int_keys:
                try:
                    return int(value)
                except ValueError:
                    return value  # Return as is if conversion fails
            elif key in float_keys:
                try:
                    return float(value)
                except ValueError:
                    return value  # Return as is if conversion fails
            return value

        # Corrected dictionary comprehension with key argument
        i_inputs = {k: convert_value(v, k) for k, v in inputs.items()}

        # Convert inputs to the correct format for the model (e.g., a DataFrame)
        input_data = pd.DataFrame(i_inputs, index=[0])

        # Convert columns to the expected types
        for column, dtype in expected_schema.items():
            if column in input_data.columns:
                input_data[column] = input_data[column].astype(dtype)

        # Predict using the model
        prediction = model.predict(input_data)

        # Prepare the response
        response = jsonify({'prediction': prediction.tolist()})
        response.status_code = 200
        return response

    except Exception as e:
        # Handle unexpected errors
        print(f"Error during prediction: {e}")
        abort(500, description="Internal server error during prediction.")


# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
