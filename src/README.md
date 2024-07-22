# Prepare MLFlow

Before running experiments, we need to sample data, then preprocess it,
and store to feature store. After these steps mlflow will be ready for runs.
Here I will provide detailed steps how to prepare everything before running
mlflow:
1. Install necessary requirements. Activate your virtual environment (which does
have airflow installation) `source {your_virtual_env_name}/bin/activate` or create
new one by `python -m venv {your_virtual_env_name}`. After you activate virtual environment
install necessary dependencies `python -m pip install -r mlflow.requirements.txt` 
2. After preparing the environment you should add such parameters to `.bashrc`:
    ```
    # REPLACE <project-folder-path> with your project folder path
    cd <project-folder-path>
    echo "export ZENML_CONFIG_PATH=$PWD/services/zenml" >> ~/.bashrc
    echo "export PYTHONPATH=$PWD" >> ~/.bashrc
    
    # Run the file
    source ~/.bashrc
    ```
3. After we set environment, we need to prepare data. Run `python src/data.py` it will sample new
data and version it (write new sample version number to `data_version`).
4. 