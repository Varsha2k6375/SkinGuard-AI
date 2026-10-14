import os
import tensorflow as tf

from dataset import load_datasets, split_dataset
from losses import bce_dice_loss
from metrics import (
    dice_coefficient,
    iou_score,
    precision,
    recall,
    pixel_accuracy
)
from config import MODEL_PATH


print("=" * 60)
print("U-Net Test Evaluation")
print("=" * 60)


# ==========================================
# Check saved model
# ==========================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: U-Net model not found!")
    print("Expected path:", MODEL_PATH)
    raise FileNotFoundError(MODEL_PATH)

print("\nSaved U-Net model found:")
print(MODEL_PATH)


# ==========================================
# Load Dataset
# ==========================================

print("\n" + "=" * 60)
print("Loading Dataset...")
print("=" * 60)

images, masks = load_datasets()

(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test
) = split_dataset(images, masks)

print("Training Images   :", len(X_train))
print("Validation Images :", len(X_valid))
print("Testing Images    :", len(X_test))


# ==========================================
# Load Existing U-Net
# ==========================================

print("\nLoading Existing U-Net Model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": pixel_accuracy
    }
)

print("U-Net model loaded successfully.")


# ==========================================
# Evaluate ONLY on Test Dataset
# ==========================================

print("\n" + "=" * 60)
print("Evaluating U-Net on Test Dataset...")
print("=" * 60)

results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


# ==========================================
# Display Results
# ==========================================

print("\n")
print("=" * 60)
print("U-NET TEST PERFORMANCE")
print("=" * 60)

for name, value in zip(model.metrics_names, results):
    print(f"{name:<25}: {value:.4f}")

print("=" * 60) 