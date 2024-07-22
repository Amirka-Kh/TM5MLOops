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
data and version it (write new sample version number to `data_version`). Then run `python src/version_data.py`
it will push new changes to dvc store and put tags in git for new data version.
4. Run zenml pipeline `python src/prepare_data.py`, it will transform data and put it to feature store.
We will use zenml api then to fetch processed data.
5. Run steps 3-4 one more time to have two different data features in datastore.
We will need them in mlflow experiment.
6. After that you should have two `features_target` artifacts you can check them by
`zenml artifact version list --name features_target`. Check if `tag` number is similar to one in
`config/data_version.yaml`, also if there are tags lower than number in data_version.yaml.
```
┠──────────────────────────────┼─────────────────┼─────────┼──────────────────────────────┼──────────────┼──────────────────────────────┼───────────────────────────────┼──────────┨
┃ ad968501-eaf2-417c-865a-7613 │ features_target │ 4       │ /mnt/c/Users/amira/PycharmPr │ DataArtifact │ module='zenml.materializers. │ module='pandas.core.frame'    │ ['1.18'] ┃
┃           f4bc245e           │                 │         │ ojects/MLOps/services/zenml/ │              │ pandas_materializer'         │ attribute='DataFrame'         │          ┃
┃                              │                 │         │ local_stores/c028da59-655d-4 │              │ attribute='PandasMaterialize │ type=<SourceType.DISTRIBUTION │          ┃
┃                              │                 │         │ 70f-9e3c-d943d6fe3ad6/custom │              │ r'                           │ _PACKAGE:                     │          ┃
┃                              │                 │         │ _artifacts/features_target/0 │              │ type=<SourceType.INTERNAL:   │ 'distribution_package'>       │          ┃
┃                              │                 │         │ f11a5b2-161e-4299-b025-a971d │              │ 'internal'>                  │ package_name='pandas'         │          ┃
┃                              │                 │         │ 67375e9                      │              │                              │ version='2.2.2'               │          ┃
┠──────────────────────────────┼─────────────────┼─────────┼──────────────────────────────┼──────────────┼──────────────────────────────┼───────────────────────────────┼──────────┨
┃ ea4625b3-f493-4e73-b73b-0955 │ features_target │ 5       │ /mnt/c/Users/amira/PycharmPr │ DataArtifact │ module='zenml.materializers. │ module='pandas.core.frame'    │ ['1.19'] ┃
┃           68bf6b0b           │                 │         │ ojects/MLOps/services/zenml/ │              │ pandas_materializer'         │ attribute='DataFrame'         │          ┃
┃                              │                 │         │ local_stores/c028da59-655d-4 │              │ attribute='PandasMaterialize │ type=<SourceType.DISTRIBUTION │          ┃
┃                              │                 │         │ 70f-9e3c-d943d6fe3ad6/custom │              │ r'                           │ _PACKAGE:                     │          ┃
┃                              │                 │         │ _artifacts/features_target/f │              │ type=<SourceType.INTERNAL:   │ 'distribution_package'>       │          ┃
┃                              │                 │         │ 1a69fb3-e581-4e15-b661-eed66 │              │ 'internal'>                  │ package_name='pandas'         │          ┃
┃                              │                 │         │ f4eaec7                      │              │                              │ version='2.2.2'               │          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━┷━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┛
```
7. Run mlflow `mlflow run . --env-manager=local`
