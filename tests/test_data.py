import os

import pytest
import pandas as pd
from src.sample_data import sample_data
from omegaconf import OmegaConf


@pytest.fixture(autouse=True, scope="session")
def prepare_data():
    # Prepare the data
    cfg = OmegaConf.create({
        "data": {
            "url": "https://storage.yandexcloud.net/sledobot.ru/Airbnb_Data.csv",
            "sample_size": 0.2,
            "dataset_name": "sample.csv"
        }
    })

    sample_data(cfg)
    return cfg


def test_sample_data(prepare_data):
    # Check if the sample file exists
    assert os.path.exists("data/samples/sample.csv")

    # Check if the sample file is not empty
    sample = pd.read_csv("data/samples/sample.csv")
    assert not sample.empty


def test_sample_size(prepare_data):
    # Check if the sample size is correct
    original_data = pd.read_csv(prepare_data.data.url)
    sample = pd.read_csv("data/samples/sample.csv")
    assert len(sample) == int(0.2 * len(original_data))


if __name__ == "__main__":
    pytest.main()
