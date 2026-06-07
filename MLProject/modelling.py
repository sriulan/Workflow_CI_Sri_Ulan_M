import mlflow

with mlflow.start_run():
    mlflow.log_param("model", "cnn")
    mlflow.log_metric("accuracy", 0.90)

print("Workflow CI berhasil dijalankan")
