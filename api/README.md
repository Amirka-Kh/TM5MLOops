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
4. Interact with model via gradio UI: `python api/gradio_app.py`

## Docker

---

To serve model via docker you can do two things:
1. You can actually build a docker image in MLflow without creating a Dockerfile with a single command line as follows:
`mlflow models build-docker -m <model-uri> --env-manager local -n <image_name>`
2. Build the docker image from Dockerfile in `api` folder: `docker build -t my_ml_service .`

After having an image - run it: `docker run --rm -p 5152:8080 my_ml_service`
