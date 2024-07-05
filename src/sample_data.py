import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime


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


def increment_version(version: str) -> str:
    major, minor = map(int, version.split('.'))
    minor += 1
    return f"{major}.{minor}"


if __name__ == "__main__":
    sample_data()
