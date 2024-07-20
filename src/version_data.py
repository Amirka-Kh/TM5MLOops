import subprocess
import hydra
from omegaconf import DictConfig
import os


@hydra.main(version_base="1.2", config_path="../configs", config_name="main")
def version_data(cfg: DictConfig):
    version = cfg.version
    message = cfg.message

    # Stage and commit changes using DVC
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", message])

    # Tag the commit with the version number
    subprocess.run(["git", "tag", "-a", version, "-m", message])

    # Push the commit and the tag to the remote repository
    # subprocess.run(["git", "push"])
    # subprocess.run(["git", "push", "--tags"])

    # Push the data using DVC
    subprocess.run(["dvc", "push"])

    print(f"Data versioned successfully with version {version}.")


if __name__ == "__main__":
    version_data()
