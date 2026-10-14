import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

from segmentation.losses import bce_dice_loss
from segmentation.metrics import (
    dice_coefficient,
    iou_score,
    precision,
    recall,
    pixel_accuracy
)

# Load U-Net model only once
model = load_model(
    "models/unet_model.keras",
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": pixel_accuracy
    }
)

def segment_image(image_path, output_folder):

    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0

    input_image = np.expand_dims(image, axis=0)

    prediction = model.predict(input_image, verbose=0)

    mask = (prediction[0] > 0.5).astype(np.uint8) * 255

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)

    cv2.imwrite(output_path, mask)

    return output_path