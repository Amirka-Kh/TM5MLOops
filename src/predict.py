import json
import requests
import hydra
from model import load_features


@hydra.main(config_path="../configs", config_name="main", version_base=None) # type: ignore
def predict(cfg = None):

    X, y = load_features(name = "features_target", version = cfg.version, random_state=cfg.random_state)

    example = X.iloc[0,:]
    example_target = y[0]

    example = json.dumps(
    { "inputs": example.to_dict() }
    )

    payload = example

    response = requests.post(
        url=f"http://{cfg.hostname}:{cfg.port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(response.json())
    print("encoded target labels: ", example_target)
    # print("target labels: ", list(cfg.data.labels)[example_target])


if __name__=="__main__":
    predict()
