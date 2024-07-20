# Scripts

We have the following scripts for now:
* `install_requirements.sh` should activate virtual environment and install dependencies
* `test_data.sh` should run code in folder `src` (data sampling and validating)
* `test_data.ps1` script for PowerShell, it needs adjustments since now it save the output of `dvc` 
command in `scripts` folder, which is not intended behaviour 