import os
import tensorflow as tf

from gradcam.heatmap import create_heatmap


# ==========================================
# Load Classification Model
# ==========================================

MODEL_PATH = (
    "models/efficientnet_model.keras"
)


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ==========================================
# Generate Grad-CAM Image
# ==========================================

def generate_gradcam_image(
    image_path,
    class_index
):

    # --------------------------------------
    # Output folder
    # --------------------------------------

    output_folder = (
        "static/uploads/gradcam"
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # --------------------------------------
    # Create heatmap
    # --------------------------------------

    output_path = create_heatmap(

        model=model,

        image_path=image_path,

        class_index=class_index,

        output_folder=output_folder

    )

    return output_path