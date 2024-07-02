import subprocess
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="main")
def version_data(cfg: DictConfig):
    version = cfg.version
    message = cfg.message if "message" in cfg else "Versioning data"

    # Stage and commit changes using DVC
    subprocess.run(["dvc", "add", "../data/samples/sample.csv"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"Add version {version} of the data"], check=True)

    # Tag the commit with the version number
    subprocess.run(["git", "tag", "-a", version, "-m", message], check=True)

    # Push the commit and the tag to the remote repository
    subprocess.run(["git", "push"], check=True)
    subprocess.run(["git", "push", "--tags"], check=True)

    # Push the data using DVC
    subprocess.run(["dvc", "push"], check=True)

    print(f"Data versioned successfully with version {version}.")


if __name__ == "__main__":
    version_data()
