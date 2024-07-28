# Scripts

We have the following scripts for now:
* `install_requirements.sh` should activate virtual environment and install dependencies
* `test_data.sh` should run code in folder `src` (data sampling and validating)
* `test_data.ps1` script for PowerShell, it needs adjustments since now it save the output of `dvc` 
command in `scripts` folder, which is not intended behaviour

To test your prediction service run `./scripts/predict_samples.sh`. The output which I got after script run:
```
Running mlflow with version=1.19 and random_state=14...
{'predictions': [5.190058050507108]}
encoded target labels:  4.8283137373023015
2024/07/23 14:19:20 INFO mlflow.projects: === Run (ID 'f5917ef2519542d984751de6d8efa914') succeeded ===

Running mlflow with version=1.19 and random_state=42...
{'predictions': [4.576810891460253]}
encoded target labels:  4.8283137373023015
2024/07/23 14:20:56 INFO mlflow.projects: === Run (ID 'f2dce6ff4fef4be589f5411b65eb344e') succeeded ===

Running mlflow with version=1.19 and random_state=99...
{'predictions': [4.464805367661466]}
encoded target labels:  4.8283137373023015
2024/07/23 14:22:22 INFO mlflow.projects: === Run (ID 'b506af4daeb444b88294f9209c12362b') succeeded ===

Running mlflow with version=1.18 and random_state=14...
{'predictions': [5.190058050507108]}
encoded target labels:  4.8283137373023015
2024/07/23 14:23:50 INFO mlflow.projects: === Run (ID '3f3e31c48aba4cb3ac5f4f037b999d58') succeeded ===

Running mlflow with version=1.18 and random_state=42...
{'predictions': [4.576810891460253]}
encoded target labels:  4.8283137373023015
2024/07/23 14:25:16 INFO mlflow.projects: === Run (ID 'd116e60b11d9476e96b959c7b4542487') succeeded ===

Running mlflow with version=1.18 and random_state=99...
{'predictions': [4.464805367661466]}
encoded target labels:  4.8283137373023015
2024/07/23 14:26:45 INFO mlflow.projects: === Run (ID '6cc9dfe1cde54d87b671b69cf5c28d2d') succeeded ===

Running mlflow with version=1.7 and random_state=14...
{'predictions': [5.190058050507108]}
encoded target labels:  4.8283137373023015
2024/07/23 14:28:11 INFO mlflow.projects: === Run (ID '7845f04cabc5457c9d80307ddb8fcd79') succeeded ===

Running mlflow with version=1.7 and random_state=42...
{'predictions': [4.576810891460253]}
encoded target labels:  4.8283137373023015
2024/07/23 14:29:39 INFO mlflow.projects: === Run (ID '9761b90dcb74447cad6fc0bac8eabbbb') succeeded ===

Running mlflow with version=1.7 and random_state=99...
{'predictions': [4.464805367661466]}
encoded target labels:  4.8283137373023015
2024/07/23 14:31:05 INFO mlflow.projects: === Run (ID 'a9558cce666f468e91ffca2b94f021c9') succeeded ===
```
