import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    start = datetime.datetime.now()
    # Read data from URL
    data = pd.read_csv(cfg.data.url)

    # Sample the data
    sample = data.sample(frac=cfg.data.sample_size)

    # Create the output directory if it doesn't exist
    os.makedirs("data/samples", exist_ok=True)

    # Save the sample data to CSV
    sample.to_csv(f"data/samples/{cfg.data.dataset_name}", index=False)

    end = datetime.datetime.now()

    print(end - start)

if __name__ == "__main__":
    sample_data()
