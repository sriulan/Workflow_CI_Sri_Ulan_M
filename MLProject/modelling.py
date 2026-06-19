import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================
# PARAMETER
# ======================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

BASE_DIR = "dataset_preprocessing"

train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")

if not os.path.exists(train_dir):
    print("Dataset tidak ditemukan. Skip training pada CI.")
    raise SystemExit(0)
    
# ======================
# DATA PREPROCESSING
# ======================
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ======================
# MODEL
# ======================
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# MLflow + TRAINING
# ======================
mlflow.set_experiment("rice-quality-classification")

with mlflow.start_run():

    mlflow.log_param("model", "CNN")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)

    # 🔥 INI YANG WAJIB (training model)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # evaluasi nyata
    loss, accuracy = model.evaluate(test_generator)

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_loss", loss)

    model.save("model.h5")
