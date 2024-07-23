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

        i_inputs = {k: convert_value(v) for k, v in inputs.items()}

        # Convert inputs to the correct format for the model (e.g., a DataFrame)
        input_data = pd.DataFrame(i_inputs, index=[0])

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
