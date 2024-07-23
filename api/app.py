from flask import Flask, request, jsonify, abort, make_response
import mlflow
import mlflow.pyfunc
import os
import pandas as pd

# Specify which keys should be converted to int or float
int_keys = {
    'property_type', 'cancellation_policy', 'cleaning_fee', 'host_has_profile_pic',
    'host_identity_verified', 'instant_bookable', 'dryer', 'essentials', 'friendly',
    'heating', 'smoke', 'tv', 'apartment', 'bed', 'bedroom', 'room', 'first_review_year',
    'first_review_month', 'last_review_year', 'last_review_month', 'host_since_year',
    'host_since_month', 'room_type_Private room', 'room_type_Shared room', 'bed_type_Couch',
    'bed_type_Futon', 'bed_type_Pull-out Sofa', 'bed_type_Real Bed', 'city_Chicago',
    'city_DC', 'city_LA', 'city_NYC', 'city_SF'
}

float_keys = {
    'accommodates', 'bathrooms', 'host_response_rate', 'latitude', 'longitude',
    'number_of_reviews', 'review_scores_rating', 'bedrooms', 'beds', 'zipcode_freq',
    'neighbourhood_freq', 'detector', 'restaurants', 'walk', 'first_review_day',
    'last_review_day', 'host_since_day', 'room_type_Private room', 'room_type_Shared room',
    'bed_type_Couch', 'bed_type_Futon', 'bed_type_Pull-out Sofa', 'bed_type_Real Bed',
    'city_Chicago', 'city_DC', 'city_LA', 'city_NYC', 'city_SF'
}

# Define expected schema types for each column
expected_schema = {
    'property_type': 'int64',
    'room_type': 'object',
    'accommodates': 'float64',
    'bathrooms': 'float64',
    'bed_type': 'object',
    'cancellation_policy': 'int64',
    'cleaning_fee': 'int64',
    'city': 'object',
    'host_has_profile_pic': 'int64',
    'host_identity_verified': 'int64',
    'host_response_rate': 'float64',
    'instant_bookable': 'int64',
    'latitude': 'float64',
    'longitude': 'float64',
    'name': 'object',
    'number_of_reviews': 'float64',
    'review_scores_rating': 'float64',
    'thumbnail_url': 'object',
    'bedrooms': 'float64',
    'beds': 'float64',
    'zipcode_freq': 'float64',
    'neighbourhood_freq': 'float64',
    'detector': 'float64',
    'dryer': 'float64',
    'essentials': 'float64',
    'friendly': 'float64',
    'heating': 'float64',
    'smoke': 'float64',
    'tv': 'float64',
    'apartment': 'float64',
    'bed': 'float64',
    'bedroom': 'float64',
    'private': 'float64',
    'restaurants': 'float64',
    'room': 'float64',
    'walk': 'float64',
    'first_review_year': 'int32',
    'first_review_month': 'int64',
    'first_review_day': 'float64',
    'last_review_year': 'int32',
    'last_review_month': 'int64',
    'last_review_day': 'float64',
    'host_since_year': 'int32',
    'host_since_month': 'int64',
    'host_since_day': 'float64',
    'room_type_Private room': 'float64',
    'room_type_Shared room': 'float64',
    'bed_type_Couch': 'float64',
    'bed_type_Futon': 'float64',
    'bed_type_Pull-out Sofa': 'float64',
    'bed_type_Real Bed': 'float64',
    'city_Chicago': 'float64',
    'city_DC': 'float64',
    'city_LA': 'float64',
    'city_NYC': 'float64',
    'city_SF': 'float64'
}

# Set up paths and model loading
BASE_PATH = os.path.expandvars("$PYTHONPATH")
model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

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
