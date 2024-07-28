import pandas as pd
from typing import Tuple, Any
from typing_extensions import Annotated

import zenml
from zenml import step, pipeline, ArtifactConfig
from zenml.client import Client

from src.sample_data import read_datastore, preprocess_data, validate_features


@step(enable_cache=False)
def extract_data() -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="extracted_data", tags=["data_preparation"])],
    Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])]
]:
    data, version = read_datastore()
    return data, version


@step(enable_cache=False)
def transform_data(data: pd.DataFrame) -> Annotated[pd.DataFrame, ArtifactConfig(name="input_features", tags=["data_preparation"])]:
    processed_data = preprocess_data(data)
    return processed_data


@step(enable_cache=False)
def validate_data(data: pd.DataFrame, version: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="valid_input_features", tags=["data_preparation"])]:
    result = validate_features(data, version)
    return result


@step(enable_cache=False)
def load_data(df: pd.DataFrame, version: str) -> None:
    # version is your custom version (set it to tags)
    zenml.save_artifact(data=df, name="features_target", tags=[version])

    client = Client()

    # Retrieve list of artifacts
    list_of_art = client.list_artifact_versions(name="features_target", tag=version, sort_by="version").items

    # Descending order
    list_of_art.reverse()

    # Retrieve latest version of the artifact
    df = list_of_art[0].load()

    # Check output
    print(df.head())


@pipeline()
def data_prepare_pipeline():
    data, version = extract_data()
    data = transform_data(data)
    data = validate_data(data, version)
    load_data(data, version)


if __name__ == "__main__":
    run = data_prepare_pipeline()
