import pandas as pd
from great_expectations.data_context import FileDataContext


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
    # assert ex5['success']

    # Validity of bathrooms
    ex6 = validator.expect_column_values_to_be_between(column="bathrooms", min_value=0, max_value=10)
    assert ex6['success']

    # Completeness of bedrooms
    ex7 = validator.expect_column_values_to_not_be_null(column="bedrooms")
    # assert ex7['success']

    # Validity of bedrooms
    ex8 = validator.expect_column_values_to_be_between(column="bedrooms", min_value=0, max_value=10)
    assert ex8['success']

    # Completeness of beds
    ex9 = validator.expect_column_values_to_not_be_null(column="beds")
    # assert ex9['success']

    # Validity of beds
    ex10 = validator.expect_column_values_to_be_between(column="beds", min_value=0, max_value=20)
    assert ex10['success']

    # Completeness of review_scores_rating
    ex11 = validator.expect_column_values_to_not_be_null(column="review_scores_rating")
    # assert ex11['success']

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
    validate_initial_data()
