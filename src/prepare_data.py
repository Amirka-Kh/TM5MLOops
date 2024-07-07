import zenml
from zenml.pipelines import pipeline
from zenml.steps import step

from src.read_datastore import read_datastore
from src.sample_data import preprocess_data, validate_features
import pandas as pd


@step
def extract_data_step() -> tuple:
    data, version = read_datastore()
    return data, version


@step
def transform_data_step(data: pd.DataFrame, version: str) -> tuple:
    processed_data, version = preprocess_data(data, version)
    return processed_data, version


@step
def validate_data_step(data: pd.DataFrame, version: str) -> tuple:
    validate_features(data, version)


@step(enable_cache=False)
def load(data: pd.DataFrame, version: str) -> tuple:
    load_features(data, version)


@pipeline
def data_prepare_pipeline(extract_data, transform_data, validate_data):
    data, version = extract_data()
    data, version = transform_data(data, version)
    validate_data(data, version)
    load(data, version)


def load_features(df: pd.DataFrame, version):
    # version is your custom version (set it to tags)
    zenml.save_artifact(data=df, name="features_target", tags=[version])

    from zenml.client import Client
    client = Client()

    # Retrieve list of artifacts
    list_of_art = client.list_artifact_versions(name="features_target", tag=version, sort_by="version").items

    # Descending order
    list_of_art.reverse()

    # Retrieve latest version of the artifact
    df = list_of_art[0].load()

    # Check output
    print(df.head())


if __name__=="__main__":
    run = data_prepare_pipeline()
