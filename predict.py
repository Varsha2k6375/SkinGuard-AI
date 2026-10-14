import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from classification.config import  (
    MODEL_PATH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)

# ======================================
# Load Trained U-Net Model
# ======================================

model = load_model(
    MODEL_PATH,
    compile=False
)

# ======================================
# Predict Segmentation
# ======================================

def predict_segmentation(image_path, output_folder):

    # Read Image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found.")

    # Keep Original Image
    original = image.copy()

    # Resize
    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # Convert RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Normalize
    image = image.astype(np.float32) / 255.0

    # Batch Dimension
    image = np.expand_dims(image, axis=0)

    # ==================================
    # U-Net Prediction
    # ==================================

    prediction = model.predict(
        image,
        verbose=0
    )

    # Binary Mask
    mask = (prediction[0] > 0.5).astype(np.uint8)

    mask = mask[:, :, 0] * 255

    # Resize Mask Back
    mask = cv2.resize(
        mask,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    # Create Colored Mask
    colored_mask = np.zeros_like(original)

    colored_mask[:, :, 1] = mask

    # Overlay
    segmented = cv2.addWeighted(
        original,
        0.7,
        colored_mask,
        0.3,
        0
    )

    # Save
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        output_folder,
        filename
    )

    cv2.imwrite(
        output_path,
        segmented
    )

    return output_path