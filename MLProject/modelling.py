import mlflow

accuracy = 0.90
loss = 0.10

# gunakan active run dari mlflow run
mlflow.log_param("model_type", "CNN")
mlflow.log_param("epochs", 1)

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("loss", loss)

print("Training selesai")
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")
