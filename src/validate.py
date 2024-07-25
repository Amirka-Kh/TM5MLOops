from model import load_features
from transform_data import transform_data # custom module
from model import retrieve_model_with_alias # custom module
from utils import init_hydra # custom module
import giskard
import hydra
import mlflow


cfg = init_hydra()

version  = cfg.test_data_version

X, y = load_features(name="features_target", version=version)

# Specify categorical columns and target column
TARGET_COLUMN = cfg.data.target_cols[0]

categorical_features = X_train.select_dtypes(include=['object']).columns

dataset_name = cfg.data.dataset_name


# Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
giskard_dataset = giskard.Dataset(
    df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
    target=TARGET_COLUMN,  # Ground truth variable
    name=dataset_name, # Optional: Give a name to your dataset
    cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
)