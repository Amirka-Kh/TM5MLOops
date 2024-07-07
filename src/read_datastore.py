import pandas as pd
import hydra
from omegaconf import DictConfig
import os
import dvc.api


@hydra.main(version_base="1.2", config_path="../configs", config_name="main")
def read_datastore(cfg: DictConfig):
    # Define data location in datastore
    url = dvc.api.get_url(
        path=os.path.join("data/sample/sample.csv"),
        repo=os.path.join(cfg.data.repo),
        rev=str(cfg.data.version),
        remote=cfg.data.remote
    )

    # Define dataframe
    df = pd.read_csv(url)

    # Send dataframe and version
    return df, str(cfg.data.version)


if __name__ == "__main__":
    read_datastore()
