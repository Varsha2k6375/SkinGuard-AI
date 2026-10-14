import os
import tensorflow as tf

from dataset import load_datasets, split_dataset
from unet_model import build_unet
from losses import bce_dice_loss
from metrics import (
    dice_coefficient,
    iou_score,
    precision,
    recall,
    pixel_accuracy
)
from config import (
    MODEL_PATH,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE
)

# ==========================================
# Create models folder
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

print("=" * 60)
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
# Build U-Net Model
# ==========================================

print("\nBuilding U-Net Model...")

model = build_unet()

# ==========================================
# Compile Model
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=bce_dice_loss,

    metrics=[
        dice_coefficient,
        iou_score,
        precision,
        recall,
        pixel_accuracy
    ]

)

# ==========================================
# Callbacks
# ==========================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath=MODEL_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1

)

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True,

    verbose=1

)

# ==========================================
# Train Model
# ==========================================

print("\nTraining Started...")

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_valid, y_valid),

    epochs=5,

    batch_size=BATCH_SIZE,

    callbacks=[checkpoint, early_stop],

    verbose=1

)

# ==========================================
# Evaluate Model
# ==========================================

print("\nEvaluating Model...")

results = model.evaluate(

    X_test,

    y_test,

    verbose=1

)

print("\n===============================")
print("Model Evaluation")
print("===============================")

for name, value in zip(model.metrics_names, results):

    print(f"{name} : {value:.4f}")

# ==========================================
# Save Model
# ==========================================

model.save(MODEL_PATH)

print("\n====================================")
print("U-Net Model Saved Successfully")
print("Saved Path :", MODEL_PATH)
print("====================================")