import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.exceptions

# Prepare data for ML model
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model hyperparameters
params = {
    "hidden_layer_sizes": (100, 50),
    "max_iter": 500, # Use hydra for configuration management
    "verbose": True,
}

# Build and train the MLP model
mlp = MLPRegressor(**params)
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Calculate metrics -- Evaluate the model
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
print(mse, accuracy, precision, recall, f1)

experiment_name = "MLflow experiment 01"
run_name = "run 01"
try:
    # Create a new MLflow Experiment
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException as e:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

print(experiment_id)

with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:

    # Log the hyperparameters
    mlflow.log_params(params=params)

    # Log the performance metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1": f1
    })

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic MLPRegressor model for airbnb data")

        # Infer the model signature
    signature = infer_signature(X_test, y_test)

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=mlp,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_test,
        registered_model_name="MLPR_model_01",
        pyfunc_predict_fn="predict_proba"
    )

    sk_pyfunc = mlflow.sklearn.load_model(model_uri=model_info.model_uri)

    predictions = sk_pyfunc.predict(X_test)
    print(predictions)

    eval_data = pd.DataFrame(y_test)
    eval_data.columns = ["label"]
    eval_data["predictions"] = predictions

    results = mlflow.evaluate(
        data=eval_data,
        model_type="classifier",
        targets="label",
        predictions="predictions",
        evaluators=["default"]
    )

# # Plot the loss curve
# plt.figure(figsize=(10, 6))
# plt.plot(mlp.loss_curve_)
# plt.title('MLP Loss Curve')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()