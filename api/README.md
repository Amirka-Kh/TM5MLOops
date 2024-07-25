# Model Serving

## Flask Application

---

To serve your model via Flask app go through these steps: 
1. Activate your virtual environment (which does have mlflow installation) 
    ```
    source {your_mlflow_env_name}/bin/activate 
    
    # Or create new one by 
    python -m venv {your_virtual_env_name}
    
    # Install necessary dependencies 
    python -m pip install -r mlflow.requirements.txt
    
    # Activate your new environment
    ```

2. Ensure you have `api/model_dir` directory. If you don't have it, run 
`mlflow models serve --model-uri models:/random_forest@champion --env-manager local -h localhost -p 5151`
This will deploy a local inference server which runs a Flask app and allows to make predictions.
However, we want to run our own flask application, thus, stop process. The `api` folder should have `model_dir` generated.

3. Start Flask application: `python api/app.py`
4. To test Flask application run:
```azure
curl -X POST http://localhost:5001/predict -H 'Content-Type: application/json' \
--data '{"inputs": {"property_type": "1", "room_type": "Entire home/apt", "accommodates": "4", "bathrooms": "2", "bed_type": "Real Bed", "cancellation_policy": "1", "cleaning_fee": "75", "city": "Chicago", "host_has_profile_pic": "1", "host_identity_verified": "1", "host_response_rate": "85", "instant_bookable": "1", "latitude": "41.8781", "longitude": "-87.6298", "name": "Spacious House", "number_of_reviews": "40", "review_scores_rating": "92", "thumbnail_url": "", "bedrooms": "2", "beds": "3", "zipcode_freq": "0.6", "neighbourhood_freq": "0.5", "detector": "1", "dryer": "1", "essentials": "1", "friendly": "1", "heating": "1", "smoke": "1", "tv": "1", "apartment": "1", "bed": "0", "bedroom": "1", "private": "2", "restaurants": "1", "room": "1", "walk": "0", "first_review_year": "1", "first_review_month": "2018", "first_review_day": "7", "last_review_year": "20", "last_review_month": "2021", "last_review_day": "4", "host_since_year": "15", "host_since_month": "2017", "host_since_day": "3", "room_type_Private room": "10", "room_type_Shared room": "1", "bed_type_Couch": "0", "bed_type_Futon": "0", "bed_type_Pull-out Sofa": "0", "bed_type_Real Bed": "1", "city_Chicago": "1", "city_DC": "0", "city_LA": "1", "city_NYC": "1", "city_SF": "1"}}'
```
4. Interact with model via gradio UI: `python api/gradio_app.py`

## Docker

---

To serve model via docker you can do two things:
1. You can actually build a docker image in MLflow without creating a Dockerfile with a single command line as follows:
`mlflow models build-docker -m <model-uri> --env-manager local -n <image_name>`
2. Build the docker image from Dockerfile in `api` folder: `docker build -t my_ml_service .`

After having an image - run it: `docker run --rm -p 5152:8080 my_ml_service`
