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


if __name__ == "__main__":
    validate_initial_data()
