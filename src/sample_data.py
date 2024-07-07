from typing import Tuple, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import dvc.api

import os
import datetime

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from great_expectations.data_context import FileDataContext


"""
Phase 1: Business and data understanding
"""


def increment_version(version: str) -> str:
    major, minor = map(int, version.split('.'))
    minor += 1
    return f"{major}.{minor}"


@hydra.main(version_base="1.2", config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig) -> None:
    start = datetime.datetime.now()
    print(OmegaConf.to_yaml(cfg))

    # Read data from URL
    data = pd.read_csv(cfg.data.url)

    # Take seed for random sampling
    version = cfg.data.version
    major, minor = map(int, version.split('.'))

    # Sample the data
    sample = data.sample(frac=cfg.data.sample_size, random_state=int(minor))

    # Create the output directory if it doesn't exist
    os.makedirs("../data/samples", exist_ok=True)

    # Save the sample data to CSV
    sample.to_csv(f"../data/samples/{cfg.data.dataset_name}", index=False)

    # Update the version in the Hydra configuration
    new_version = increment_version(version)
    new_message = f"Add data version {new_version}"
    OmegaConf.update(cfg, 'data.version', new_version)
    OmegaConf.update(cfg, 'data.message', new_message)

    end = datetime.datetime.now()
    print(f"Data sampling was successful")
    print('time: ', end-start)


def validate_initial_data():
    # Create a FileDataContext
    context = FileDataContext(project_root_dir = "../services")

    # Connect a data source
    ds = context.sources.add_or_update_pandas(name="pandas_datasource")
    da1 = ds.add_csv_asset(name="airbnb", filepath_or_buffer="../data/samples/sample.csv")

    # Create a batch
    batch_request = da1.build_batch_request()
    batches = da1.get_batch_list_from_batch_request(batch_request)

    # Define expectations
    context.add_or_update_expectation_suite("initial_data_validation")
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="initial_data_validation"
    )
    ex1 = validator.expect_column_values_to_not_be_null(
        column="city",
        meta={"dimension": "Completeness"}
    )
    ex4 = validator.expect_column_values_to_be_between(
        column='review_scores_rating',
        min_value=0.0,
        max_value=101.0,
        mostly=.9,
        meta={
            "dimension": 'Consistency'
        }
    )

    validator.expect_column_values_to_not_be_null('id')
    validator.expect_column_values_to_not_be_null('log_price')
    validator.expect_column_values_to_be_between('log_price', 0, 10)
    validator.expect_column_values_to_not_be_null('accommodates')
    validator.expect_column_values_to_be_between('accommodates', 1, 16)
    validator.expect_column_values_to_not_be_null('bathrooms')
    validator.expect_column_values_to_be_between('bathrooms', 0, 10)
    validator.expect_column_values_to_not_be_null('bedrooms')
    validator.expect_column_values_to_be_between('bedrooms', 0, 10)
    validator.expect_column_values_to_not_be_null('beds')
    validator.expect_column_values_to_be_between('beds', 0, 20)
    validator.expect_column_values_to_not_be_null('review_scores_rating')
    validator.expect_column_values_to_be_between('review_scores_rating', 0, 100)

    # Run the validation
    validation_result = validator.validate()
    assert ex1['success']
    assert ex4['success']

    validator.save_expectation_suite(discard_failed_expectations=True)

    checkpoint = context.add_or_update_checkpoint(
        name="initial_data_validation_checkpoint",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "initial_data_validation"
            }
        ]
    )
    checkpoint_result = checkpoint.run()
    print(checkpoint_result.success)

    context.build_data_docs()
    context.open_data_docs()


"""
Phase 2: Data preparation/engineering
"""


@hydra.main(version_base="1.2", config_path="../configs", config_name="main")
def read_datastore(cfg: DictConfig):
    # Define location in datastore
    url = dvc.api.get_url(
        path=os.path.join("data/samples/sample.csv"),
        repo=os.path.join(cfg.data.repo),
        rev=str(cfg.data.version),
        remote=cfg.data.remote
    )

    # Define dataframe
    df = pd.read_csv(url)

    # Send dataframe and version
    return df, str(cfg.data.version)


def preprocess_data(data: pd.DataFrame, version: str) -> tuple[DataFrame, Any]:
    # Transform data
    # Handling missing values
    data['bathrooms'].fillna(data['bathrooms'].median(), inplace=True)
    data['bedrooms'].fillna(data['bedrooms'].median(), inplace=True)
    data['beds'].fillna(data['beds'].median(), inplace=True)
    data['host_has_profile_pic'].fillna('inplace', inplace=True)
    data['host_identity_verified'].fillna('False', inplace=True)
    data['host_response_rate'].fillna(0, inplace=True)
    data['neighbourhood'].fillna('Unknown', inplace=True)
    data['review_scores_rating'].fillna(data['review_scores_rating'].median(), inplace=True)
    data['thumbnail_url'].fillna('No URL', inplace=True)
    data['zipcode'].fillna('Unknown', inplace=True)

    # Fill missing date fields with a specific date (e.g., '1970-01-01') and convert to datetime
    data['first_review'].fillna('1970-01-01', inplace=True)
    data['last_review'].fillna('1970-01-01', inplace=True)
    data['host_since'].fillna('1970-01-01', inplace=True)

    # One-Hot Encoding for categorical features
    # categorical_features = ['property_type', 'room_type', 'bed_type', 'city']
    categorical_features = ['room_type', 'bed_type', 'city']

    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(data[categorical_features])
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(categorical_features))

    # Label Encoding for cancellation_policy
    for i in ['property_type', 'cancellation_policy']:
        label_encoder = LabelEncoder()
        data[i] = label_encoder.fit_transform(data[i])

    # Binary Encoding for boolean features
    binary_features = ['instant_bookable', 'host_has_profile_pic', 'host_identity_verified', 'cleaning_fee']
    for feature in binary_features:
        if feature in data.columns:
            data[feature] = data[feature].replace({'t': True, 'f': False, 'true': True, 'false': False}).astype(
                bool).astype(int)

    # Standardization for numerical features
    numerical_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'number_of_reviews']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    data['host_response_rate'] = scaler.fit_transform(data[['host_response_rate']])

    # Frequency Encoding for zipcode and neighbourhood
    for feature in ['zipcode', 'neighbourhood']:
        if feature in data.columns:
            freq_encoding = data.groupby(feature).size() / len(data)
            data[feature + '_freq'] = data[feature].map(freq_encoding)
            data.drop(columns=[feature], inplace=True)

    # TF-IDF Vectorization for text features
    text_features = ['amenities', 'description']
    for feature in text_features:
        if feature in data.columns:
            tfidf = TfidfVectorizer(max_features=8, stop_words='english', max_df=0.95,
                                    min_df=2)  # Adjust max_features as needed
            tfidf_matrix = tfidf.fit_transform(data[feature].fillna(''))
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out(), index=data.index)
            data = pd.concat([data, tfidf_df], axis=1)
            data.drop(columns=[feature], inplace=True)

    # Extracting year, month, day from date features and applying label encoding for month
    date_features = ['first_review', 'last_review', 'host_since']
    label_encoder = LabelEncoder()

    for feature in date_features:
        if feature in data.columns:
            data[feature + '_year'] = pd.to_datetime(data[feature]).dt.year
            data[feature + '_month'] = label_encoder.fit_transform(pd.to_datetime(data[feature]).dt.month)
            data[feature + '_day'] = pd.to_datetime(data[feature]).dt.day
            data.drop(columns=[feature], inplace=True)

    # Standardization for day columns if needed
    for feature in date_features:
        day_feature = feature + '_day'
        if day_feature in data.columns:
            data[day_feature] = scaler.fit_transform(data[[day_feature]])

    # Concatenate the encoded categorical features back to the DataFrame
    data_cleaned = pd.concat([data.reset_index(drop=True), onehot_encoded_df.reset_index(drop=True)], axis=1)

    # Return processed dataframe
    return data_cleaned, version


def validate_features(data: pd.DataFrame, version):
    # Create a FileDataContext
    context = FileDataContext(project_root_dir="../services")

    # Connect a data source
    ds = context.sources.add_or_update_pandas(name="features_datasource")
    da1 = ds.add_dataframe_asset(name="airbnb_features")

    # Create a batch request
    batch_request = da1.build_batch_request(dataframe=data)

    # Define a new expectation suite
    suite_name = "feature_expectations"
    context.add_or_update_expectation_suite(suite_name)

    # Add expectations to the suite
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # Completeness
    ex1 = validator.expect_column_values_to_not_be_null(column="id")
    assert ex1['success']

    # Uniqueness
    ex2 = validator.expect_column_values_to_be_unique(column="id")
    assert ex2['success']

    # Validity of log_price
    ex3 = validator.expect_column_values_to_be_between(column="log_price", min_value=0, max_value=10)
    assert ex3['success']

    # Validity of accommodates
    ex4 = validator.expect_column_values_to_be_between(column="accommodates", min_value=1, max_value=16)
    assert ex4['success']

    # Completeness of bathrooms
    ex5 = validator.expect_column_values_to_not_be_null(column="bathrooms")
    assert ex5['success']

    # Validity of bathrooms
    ex6 = validator.expect_column_values_to_be_between(column="bathrooms", min_value=0, max_value=10)
    assert ex6['success']

    # Completeness of bedrooms
    ex7 = validator.expect_column_values_to_not_be_null(column="bedrooms")
    assert ex7['success']

    # Validity of bedrooms
    ex8 = validator.expect_column_values_to_be_between(column="bedrooms", min_value=0, max_value=10)
    assert ex8['success']

    # Completeness of beds
    ex9 = validator.expect_column_values_to_not_be_null(column="beds")
    assert ex9['success']

    # Validity of beds
    ex10 = validator.expect_column_values_to_be_between(column="beds", min_value=0, max_value=20)
    assert ex10['success']

    # Completeness of review_scores_rating
    ex11 = validator.expect_column_values_to_not_be_null(column="review_scores_rating")
    assert ex11['success']

    # Validity of review_scores_rating
    ex12 = validator.expect_column_values_to_be_between(column="review_scores_rating", min_value=0, max_value=100)
    assert ex12['success']

    # Completeness of first_review
    ex13 = validator.expect_column_values_to_not_be_null(column="first_review_year")
    assert ex13['success']

    # Completeness of last_review
    ex14 = validator.expect_column_values_to_not_be_null(column="last_review_year")
    assert ex14['success']

    # Completeness of host_since
    ex15 = validator.expect_column_values_to_not_be_null(column="host_since_year")
    assert ex15['success']

    validator.save_expectation_suite(discard_failed_expectations=True)

    context.add_or_update_checkpoint(
        name="features_validation_checkpoint",
        config_version=version,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": suite_name
            }
        ]
    )

    # Run the checkpoint
    results = context.run_checkpoint(checkpoint_name='features_validation_checkpoint')
    return results['success']


if __name__ == "__main__":
    # data, version = read_datastore()
    # data, version = preprocess_data(data, version)
    # print(validate_features(data, version))
    sample_data()