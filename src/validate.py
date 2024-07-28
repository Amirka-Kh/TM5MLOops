from model import load_features
from sample_data import read_datastore
from transform_data import transform_data # custom module
from model import retrieve_model_with_alias # custom module
from utils import init_hydra # custom module
import giskard
import hydra
import mlflow


cfg = init_hydra()

df, version = read_datastore()

TARGET_COLUMN = df['log_price'][0]

CATEGORICAL_COLUMNS = df.select_dtypes(include=['object']).columns

# dataset_name = cfg.data.dataset_name
dataset_name = 'validation_dataset'

# Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
giskard_dataset = giskard.Dataset(
    df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
    target=TARGET_COLUMN,  # Ground truth variable
    name=dataset_name, # Optional: Give a name to your dataset
    cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
)

model_name = cfg.model.best_model_name

# You can sweep over challenger aliases using Hydra
model_alias = cfg.model.best_model_alias

model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias = model_alias)

client = mlflow.MlflowClient()

mv = client.get_model_version_by_alias(name = model_name, alias=model_alias)

model_version = mv.version

transformer_version = cfg.data_transformer_version

def predict(raw_df):
    X = transform_data(
        df = raw_df,
        version = version,
        cfg = cfg,
        return_df = False,
        only_transform = True,
        transformer_version = transformer_version,
        only_X = True
      )

    return model.predict(X)

predictions = predict(df[df.columns].head())
print(predictions)