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


# Flask

1. To start Flask application first run: `python api/app.py`
2. To test Flask application run:
```azure
curl -X POST http://localhost:5001/predict -H 'Content-Type: application/json' \
--data '{"inputs": {"property_type": "1", "room_type": "Entire home/apt", "accommodates": "4", "bathrooms": "2", "bed_type": "Real Bed", "cancellation_policy": "1", "cleaning_fee": "75", "city": "Chicago", "host_has_profile_pic": "1", "host_identity_verified": "1", "host_response_rate": "85", "instant_bookable": "1", "latitude": "41.8781", "longitude": "-87.6298", "name": "Spacious House", "number_of_reviews": "40", "review_scores_rating": "92", "thumbnail_url": "", "bedrooms": "2", "beds": "3", "zipcode_freq": "0.6", "neighbourhood_freq": "0.5", "detector": "1", "dryer": "1", "essentials": "1", "friendly": "1", "heating": "1", "smoke": "1", "tv": "1", "apartment": "1", "bed": "0", "bedroom": "1", "private": "2", "restaurants": "1", "room": "1", "walk": "0", "first_review_year": "1", "first_review_month": "2018", "first_review_day": "7", "last_review_year": "20", "last_review_month": "2021", "last_review_day": "4", "host_since_year": "15", "host_since_month": "2017", "host_since_day": "3", "room_type_Private room": "10", "room_type_Shared room": "1", "bed_type_Couch": "0", "bed_type_Futon": "0", "bed_type_Pull-out Sofa": "0", "bed_type_Real Bed": "1", "city_Chicago": "1", "city_DC": "0", "city_LA": "1", "city_NYC": "1", "city_SF": "1"}}'
```
