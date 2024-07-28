from sample_data import read_datastore, preprocess_data
from model import retrieve_model_with_alias
import giskard
from hydra import compose, initialize
import mlflow
import json


initialize(version_base="1.2", config_path="../configs")
cfg = compose(config_name="main")

df, version = read_datastore()
TARGET_COLUMN = 'log_price'
dataset_name = 'validation_dataset'

giskard_dataset = giskard.Dataset(
    df=df,
    target=TARGET_COLUMN,
    name=dataset_name,
)

model_name = cfg.model.best_model_name
model_alias = cfg.model.best_model_alias
model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias = model_alias)

client = mlflow.MlflowClient()
mv = client.get_model_version_by_alias(name = model_name, alias=model_alias)
model_version = mv.version

model_uri = client.get_model_version_download_uri(model_name, model_version)
model_info = mlflow.models.get_model_info(model_uri)
signature_dict = model_info._signature_dict


def parse_signature(signature_dict):
    inputs = json.loads(signature_dict['inputs'])
    input_dict = {entry['name']: entry['type'] for entry in inputs}
    return input_dict


def convert_dtype(dtype_str):
    if dtype_str == 'long':
        return 'int64'
    elif dtype_str == 'integer':
        return 'int32'
    elif dtype_str == 'double':
        return 'float64'
    elif dtype_str == 'string':
        return 'object'
    else:
        return dtype_str  # Default case


def add_missing_columns(data, missing_columns, input_signature):
    for column in missing_columns:
        if column not in data.columns:
            if input_signature[column] != 'object':
                data[column] = 0
            else:
                data['test_indication'].fillna(df['column'].mode()[0], inplace=True)
    return data


def transform(input_data, input_signature):
    missing_columns = [col for col in input_signature if col not in input_data.columns]
    input_data = add_missing_columns(input_data, missing_columns, input_signature)

    for column, dtype in input_signature.items():
        if column in input_data.columns:
            # Fill NaN values based on the target dtype
            if 'int' in convert_dtype(dtype):
                input_data[column].fillna(0, inplace=True)
            elif 'float' in convert_dtype(dtype):
                input_data[column].fillna(0.0, inplace=True)
            elif 'object' in convert_dtype(dtype):
                input_data[column].fillna('', inplace=True)
            # Convert the column to the appropriate dtype
            input_data[column] = input_data[column].astype(convert_dtype(dtype))

    return input_data


def predict(raw_df):
    new_data = preprocess_data(data=raw_df)
    input_signature = parse_signature(signature_dict)
    X = transform(new_data, input_signature)
    return model.predict(X)

predictions = predict(df[df.columns].head())
print(predictions)

giskard_model = giskard.Model(
  model=predict,
  model_type = "regression",
  name=model_name,
)

scan_results = giskard.scan(giskard_model, giskard_dataset)

# Save the results in `html` file
scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
scan_results.to_html(scan_results_path)


suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
test_suite = scan_results.generate_test_suite(suite_name)

test1 = giskard.testing.test_f1(model = giskard_model,
                                dataset = giskard_dataset,
                                threshold=cfg.model.r2_threshold)
test_suite.add_test(test1)

test_results = test_suite.run()
if (test_results.passed):
    print("Passed model validation!")
else:
    print("Model has vulnerabilities!")
