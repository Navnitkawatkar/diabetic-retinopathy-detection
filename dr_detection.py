"""
Diabetic Retinopathy Detection via Retinal Fundus Images
=========================================================
Minor Project Submission
Technology: Python, TensorFlow/Keras, OpenCV
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 20
NUM_CLASSES   = 5
LEARNING_RATE = 1e-4


DATASET_PATH  = "dataset/"
MODEL_PATH    = "dr_model.h5"

CLASS_NAMES = [
    "No DR (Grade 0)",
    "Mild DR (Grade 1)",
    "Moderate DR (Grade 2)",
    "Severe DR (Grade 3)",
    "Proliferative DR (Grade 4)"
]


# ─────────────────────────────────────────────
# STEP 1: IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(img_path, target_size=IMG_SIZE):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ben Graham preprocessing
    img = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), target_size / 30), -4,
        128
    )

    # Circular crop
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask,
               (img.shape[1] // 2, img.shape[0] // 2),
               int(min(img.shape[:2]) * 0.45),
               1, -1)
    img = img * mask[:, :, np.newaxis]

    img = cv2.resize(img, (target_size, target_size))
    img = img.astype(np.float32) / 255.0
    return img


# ─────────────────────────────────────────────
# STEP 2: DATA GENERATORS
# ─────────────────────────────────────────────
def build_data_generators(dataset_path):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )

    val_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_data = train_gen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42
    )

    val_data = val_gen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42
    )

    return train_data, val_data


# ─────────────────────────────────────────────
# STEP 3: MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES):
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


# ─────────────────────────────────────────────
# STEP 4: TRAINING
# ─────────────────────────────────────────────
def train_model(model, train_data, val_data):
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc",
                                  save_best_only=True)
    ]

    print("─── Phase 1: Training head (frozen backbone) ───")
    history1 = model.fit(train_data, validation_data=val_data,
                         epochs=10, callbacks=cb)

    # Unfreeze top 30 layers
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )

    print("─── Phase 2: Fine-tuning top layers ───")
    history2 = model.fit(train_data, validation_data=val_data,
                         epochs=EPOCHS, callbacks=cb)

    return history1, history2


# ─────────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────────
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved: training_history.png")


def plot_confusion_matrix(model, val_data):
    val_data.reset()
    y_true = val_data.classes
    y_pred = np.argmax(model.predict(val_data, verbose=0), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ─────────────────────────────────────────────
# STEP 6: PREDICT SINGLE IMAGE
# ─────────────────────────────────────────────
def predict_single_image(model, img_path):
    img = preprocess_image(img_path)
    img_batch = np.expand_dims(img, axis=0)
    probs = model.predict(img_batch, verbose=0)[0]
    grade = np.argmax(probs)
    confidence = probs[grade] * 100

    print(f"\n📷 Image: {img_path}")
    print(f"🔍 Predicted: {CLASS_NAMES[grade]}")
    print(f"📊 Confidence: {confidence:.1f}%")
    print("\nProbability distribution:")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar = "█" * int(prob * 30)
        print(f"  Grade {i}: {bar} {prob*100:.1f}%")

    return grade, confidence




# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Diabetic Retinopathy Detection")
    print("=" * 60)
    print(f"\nTensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU(s) available: {len(gpus)}")
    
    # Load saved model instead of retraining
    from tensorflow.keras.models import load_model
    model = load_model("dr_model.h5")
    print("✅ Model loaded from dr_model.h5!")

    # Predict on test image
    predict_single_image(model, "test_image.jpg")
