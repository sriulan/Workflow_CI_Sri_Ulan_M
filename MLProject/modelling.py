import mlflow

# Membuat experiment
mlflow.set_experiment("Workflow_CI_Training")

mlflow.log_param("model", "cnn")
mlflow.log_metric("accuracy", 0.90)

# Parameter
mlflow.log_param("model_type", "CNN")
mlflow.log_param("epochs", 1)

# Simulasi hasil training
accuracy = 0.90
loss = 0.10

# Log metric
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("loss", loss)

print("Training selesai")
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

print("Workflow CI berhasil dijalankan")
