import cv2
import os
import numpy as np
import bm3d


def preprocess_image(
        image_path,
        output_folder
):

    # -----------------------------
    # Check Image Path
    # -----------------------------

    if not os.path.exists(image_path):

        raise ValueError(
            "Image path not found : "
            + image_path
        )

    # -----------------------------
    # Read Image
    # -----------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "Unable to read image : "
            + image_path
        )

    # -----------------------------
    # Resize
    # -----------------------------

    resized = cv2.resize(

        image,

        (224, 224),

        interpolation=cv2.INTER_AREA

    )

    # -----------------------------
    # Convert BGR → RGB
    # -----------------------------

    rgb_image = cv2.cvtColor(

        resized,

        cv2.COLOR_BGR2RGB

    )

    # -----------------------------
    # Convert to Float [0, 1]
    # BM3D requires normalized
    # floating-point image
    # -----------------------------

    image_float = (

        rgb_image.astype(
            np.float32
        ) / 255.0

    )

    # -----------------------------
    # BM3D DENOISING
    # -----------------------------

    denoised_float = bm3d.bm3d(

        image_float,

        sigma_psd=0.05

    )

    # -----------------------------
    # Keep pixel values valid
    # -----------------------------

    denoised_float = np.clip(

        denoised_float,

        0.0,

        1.0

    )

    # -----------------------------
    # Convert back to 8-bit
    # -----------------------------

    denoised = (

        denoised_float * 255.0

    ).astype(

        np.uint8

    )

    # -----------------------------
    # RGB → BGR
    # for OpenCV processing
    # -----------------------------

    denoised_bgr = cv2.cvtColor(

        denoised,

        cv2.COLOR_RGB2BGR

    )

    # -----------------------------
    # CLAHE
    # Contrast Enhancement
    # -----------------------------

    lab = cv2.cvtColor(

        denoised_bgr,

        cv2.COLOR_BGR2LAB

    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8, 8)

    )

    enhanced_l = clahe.apply(l)

    enhanced = cv2.merge(

        (
            enhanced_l,
            a,
            b
        )

    )

    enhanced = cv2.cvtColor(

        enhanced,

        cv2.COLOR_LAB2BGR

    )

    # -----------------------------
    # Normalization
    # -----------------------------

    normalized = (

        enhanced.astype(
            np.float32
        ) / 255.0

    )

    # -----------------------------
    # Convert to 8-bit
    # for image saving
    # -----------------------------

    output_image = (

        normalized * 255.0

    ).astype(

        np.uint8

    )

    # -----------------------------
    # Save Image
    # -----------------------------

    os.makedirs(

        output_folder,

        exist_ok=True

    )

    output_path = os.path.join(

        output_folder,

        "processed_image.jpg"

    )

    cv2.imwrite(

        output_path,

        output_image

    )

    return output_path