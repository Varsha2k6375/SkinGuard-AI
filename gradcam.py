import os
import cv2
import bm3d
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

BM3D_SIGMA = 0.05

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

HEATMAP_THRESHOLD = 0.35

MIN_REGION_RATIO = 0.003

OVERLAY_ALPHA = 0.45


# ============================================================
# LOAD IMAGE + SAME PREPROCESSING AS CLASSIFICATION
# ============================================================

def load_image(image_path):

    print("\n===================================")
    print("LOADING IMAGE FOR GRAD-CAM")
    print("===================================")

    if not image_path:

        raise ValueError(
            "Grad-CAM Error: Image path is empty."
        )

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Grad-CAM Error: Image not found: {image_path}"
        )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            "Grad-CAM Error: Unable to read image."
        )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_image = image_rgb.copy()

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        image_rgb,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # RGB -> float [0,1]
    #
    # BM3D requires normalized values.
    # --------------------------------------------------------

    resized_float = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # BM3D
    # --------------------------------------------------------

    print(
        "Applying BM3D..."
    )

    denoised = bm3d.bm3d(
        resized_float,
        sigma_psd=BM3D_SIGMA
    )

    denoised = np.clip(
        denoised,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Convert back to 0-255
    # --------------------------------------------------------

    denoised_uint8 = (
        denoised * 255.0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    print(
        "Applying CLAHE..."
    )

    lab = cv2.cvtColor(
        denoised_uint8,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        [
            l_channel,
            a_channel,
            b_channel
        ]
    )

    processed = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # DO NOT divide by 255 after this point.
    #
    # EfficientNetB3 model expects 0-255 input and contains
    # its own input rescaling.
    #
    # ========================================================

    processed = processed.astype(
        np.float32
    )

    processed = np.expand_dims(
        processed,
        axis=0
    )

    print(
        "Original Image Shape :",
        original_image.shape
    )

    print(
        "Model Image Shape    :",
        processed.shape
    )

    print(
        "Pixel Minimum        :",
        float(np.min(processed))
    )

    print(
        "Pixel Maximum        :",
        float(np.max(processed))
    )

    return (
        original_image,
        processed
    )


# ============================================================
# FIND EFFICIENTNET BASE MODEL
# ============================================================

def find_base_model(model):

    print("\n===================================")
    print("SEARCHING EFFICIENTNET BASE MODEL")
    print("===================================")

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            if (
                "efficientnet"
                in layer.name.lower()
            ):

                print(
                    "Base Model Found :",
                    layer.name
                )

                return layer

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            print(
                "Base Model Found :",
                layer.name
            )

            return layer

    raise ValueError(
        "Grad-CAM Error: "
        "EfficientNet base model not found."
    )


# ============================================================
# FIND TARGET CONVOLUTIONAL LAYER
# ============================================================

def find_target_layer(base_model):

    print("\n===================================")
    print("SEARCHING GRAD-CAM TARGET LAYER")
    print("===================================")

    preferred_layers = [

        "top_activation",

        "top_bn",

        "block7a_project_bn",

        "block7a_project_conv"

    ]

    for layer_name in preferred_layers:

        try:

            layer = base_model.get_layer(
                layer_name
            )

            if len(
                layer.output.shape
            ) == 4:

                print(
                    "Target Layer :",
                    layer.name
                )

                print(
                    "Feature Shape:",
                    layer.output.shape
                )

                return layer

        except Exception:

            continue

    # --------------------------------------------------------
    # Search backwards
    # --------------------------------------------------------

    for layer in reversed(
        base_model.layers
    ):

        try:

            output_shape = (
                layer.output.shape
            )

            if len(
                output_shape
            ) == 4:

                if (
                    output_shape[1] is not None
                    and
                    output_shape[2] is not None
                ):

                    print(
                        "Target Layer :",
                        layer.name
                    )

                    print(
                        "Feature Shape:",
                        output_shape
                    )

                    return layer

        except Exception:

            continue

    raise ValueError(
        "Grad-CAM Error: "
        "No convolutional target layer found."
    )


# ============================================================
# FIND CLASSIFIER LAYERS
# ============================================================

def find_classifier_layers(
    model,
    base_model
):

    base_index = None

    for index, layer in enumerate(
        model.layers
    ):

        if layer is base_model:

            base_index = index

            break

    if base_index is None:

        raise ValueError(
            "Grad-CAM Error: "
            "EfficientNet base model index not found."
        )

    classifier_layers = (
        model.layers[
            base_index + 1:
        ]
    )

    print("\n===================================")
    print("CLASSIFIER LAYERS")
    print("===================================")

    for layer in classifier_layers:

        print(
            "-",
            layer.name
        )

    return classifier_layers


# ============================================================
# BUILD GRAD-CAM MODEL
# ============================================================

def build_gradcam_model(model):

    base_model = find_base_model(
        model
    )

    target_layer = find_target_layer(
        base_model
    )

    feature_model = tf.keras.models.Model(

        inputs=base_model.input,

        outputs=[
            target_layer.output,
            base_model.output
        ],

        name="gradcam_feature_model"

    )

    classifier_layers = (
        find_classifier_layers(
            model,
            base_model
        )
    )

    return (
        feature_model,
        classifier_layers,
        target_layer
    )


# ============================================================
# GENERATE RAW GRAD-CAM
# ============================================================

def generate_raw_gradcam(
    model,
    image_array,
    class_index
):

    print("\n===================================")
    print("GENERATING GRAD-CAM")
    print("===================================")

    if model is None:

        raise ValueError(
            "Grad-CAM Error: Model is None."
        )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    if len(
        image_tensor.shape
    ) == 3:

        image_tensor = tf.expand_dims(
            image_tensor,
            axis=0
        )

    (
        feature_model,
        classifier_layers,
        target_layer
    ) = build_gradcam_model(
        model
    )

    print(
        "Using Target Layer :",
        target_layer.name
    )

    print(
        "Requested Class     :",
        class_index
    )

    # ========================================================
    # GRADIENT TAPE
    # ========================================================

    with tf.GradientTape() as tape:

        (
            conv_outputs,
            base_output
        ) = feature_model(
            image_tensor,
            training=False
        )

        x = base_output

        # ----------------------------------------------------
        # Pass through classifier
        # ----------------------------------------------------

        for layer in classifier_layers:

            if isinstance(
                layer,
                tf.keras.layers.InputLayer
            ):

                continue

            try:

                x = layer(
                    x,
                    training=False
                )

            except TypeError:

                x = layer(x)

        predictions = x

        print(
            "Prediction Shape :",
            predictions.shape
        )

        class_score = predictions[
            :,
            int(class_index)
        ]

    # ========================================================
    # GRADIENT
    # ========================================================

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Grad-CAM Error: "
            "Gradients are None."
        )

    print(
        "Gradient Shape    :",
        gradients.shape
    )

    print(
        "Feature Map Shape :",
        conv_outputs.shape
    )

    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    # ========================================================
    # WEIGHT FEATURE MAPS
    # ========================================================

    heatmap = tf.reduce_sum(

        conv_outputs *
        pooled_gradients,

        axis=-1

    )

    # ========================================================
    # RELU
    # ========================================================

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    maximum = tf.reduce_max(
        heatmap
    )

    maximum = float(
        maximum.numpy()
    )

    if maximum > 0:

        heatmap = (
            heatmap /
            maximum
        )

    else:

        heatmap = tf.zeros_like(
            heatmap
        )

    heatmap = heatmap.numpy()

    print(
        "Raw Heatmap Shape :",
        heatmap.shape
    )

    print(
        "Heatmap Min       :",
        float(np.min(heatmap))
    )

    print(
        "Heatmap Max       :",
        float(np.max(heatmap))
    )

    return heatmap


# ============================================================
# LOAD SEGMENTATION MASK
# ============================================================

def load_segmentation_mask(
    mask_path,
    target_shape
):

    if not mask_path:

        return None

    if not os.path.exists(
        mask_path
    ):

        print(
            "Segmentation mask not found."
        )

        return None

    mask = cv2.imread(

        mask_path,

        cv2.IMREAD_GRAYSCALE

    )

    if mask is None:

        print(
            "Unable to read segmentation mask."
        )

        return None

    height, width = target_shape

    mask = cv2.resize(

        mask,

        (
            width,
            height
        ),

        interpolation=cv2.INTER_NEAREST

    )

    _, mask = cv2.threshold(

        mask,

        127,

        255,

        cv2.THRESH_BINARY

    )

    mask = (
        mask.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # Morphological closing
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel

    )

    return mask


# ============================================================
# APPLY SEGMENTATION MASK
# ============================================================

def restrict_heatmap_to_lesion(
    heatmap,
    segmentation_mask
):

    if segmentation_mask is None:

        print(
            "No segmentation mask supplied."
        )

        return heatmap

    print(
        "Applying segmentation mask "
        "to Grad-CAM..."
    )

    mask = cv2.resize(

        segmentation_mask,

        (
            heatmap.shape[1],
            heatmap.shape[0]
        ),

        interpolation=cv2.INTER_LINEAR

    )

    # --------------------------------------------------------
    # Restrict activation to lesion
    # --------------------------------------------------------

    heatmap = (
        heatmap *
        mask
    )

    maximum = np.max(
        heatmap
    )

    if maximum > 0:

        heatmap = (
            heatmap /
            maximum
        )

    return heatmap


# ============================================================
# REMOVE WEAK ACTIVATION
# ============================================================

def clean_heatmap(
    heatmap,
    threshold=0.20
):

    cleaned = heatmap.copy()

    cleaned[
        cleaned < threshold
    ] = 0

    cleaned = cv2.GaussianBlur(

        cleaned,

        (5, 5),

        0

    )

    maximum = np.max(
        cleaned
    )

    if maximum > 0:

        cleaned = (
            cleaned /
            maximum
        )

    return cleaned


# ============================================================
# RESIZE HEATMAP
# ============================================================

def resize_heatmap(
    heatmap,
    width,
    height
):

    return cv2.resize(

        heatmap,

        (
            width,
            height
        ),

        interpolation=cv2.INTER_CUBIC

    )


# ============================================================
# CREATE COLORED HEATMAP
# ============================================================

def create_colored_heatmap(
    heatmap
):

    heatmap_uint8 = np.uint8(

        np.clip(
            heatmap,
            0,
            1
        ) * 255

    )

    colored = cv2.applyColorMap(

        heatmap_uint8,

        cv2.COLORMAP_JET

    )

    colored = cv2.cvtColor(

        colored,

        cv2.COLOR_BGR2RGB

    )

    return colored


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=OVERLAY_ALPHA
):

    height = original_image.shape[0]

    width = original_image.shape[1]

    heatmap_resized = resize_heatmap(

        heatmap,

        width,

        height

    )

    colored_heatmap = (
        create_colored_heatmap(
            heatmap_resized
        )
    )

    original_uint8 = (
        original_image.astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Only strong activation is displayed
    # --------------------------------------------------------

    activation_mask = (
        heatmap_resized >=
        HEATMAP_THRESHOLD
    ).astype(
        np.float32
    )

    activation_mask = np.expand_dims(
        activation_mask,
        axis=-1
    )

    colored_heatmap = (
        colored_heatmap.astype(
            np.float32
        )
    )

    original_float = (
        original_uint8.astype(
            np.float32
        )
    )

    blended = (

        original_float *
        (
            1 -
            alpha *
            activation_mask
        )

        +

        colored_heatmap *
        (
            alpha *
            activation_mask
        )

    )

    blended = np.clip(
        blended,
        0,
        255
    ).astype(
        np.uint8
    )

    return blended


# ============================================================
# CREATE IMPORTANT REGION
# ============================================================

def create_important_region(
    original_image,
    heatmap,
    threshold=HEATMAP_THRESHOLD
):

    height = original_image.shape[0]

    width = original_image.shape[1]

    heatmap_resized = resize_heatmap(

        heatmap,

        width,

        height

    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    binary = (

        heatmap_resized >=
        threshold

    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (9, 9),
        np.uint8
    )

    binary = cv2.morphologyEx(

        binary,

        cv2.MORPH_CLOSE,

        kernel

    )

    binary = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        kernel

    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    result = original_image.copy()

    if not contours:

        return result

    image_area = (
        width *
        height
    )

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area >= (
            image_area *
            MIN_REGION_RATIO
        ):

            valid_contours.append(
                contour
            )

    if not valid_contours:

        return result

    largest_contour = max(

        valid_contours,

        key=cv2.contourArea

    )

    # --------------------------------------------------------
    # Highlight important region
    # --------------------------------------------------------

    overlay = result.copy()

    cv2.drawContours(

        overlay,

        [largest_contour],

        -1,

        (255, 40, 40),

        thickness=cv2.FILLED

    )

    result = cv2.addWeighted(

        result,

        0.75,

        overlay,

        0.25,

        0

    )

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    cv2.drawContours(

        result,

        [largest_contour],

        -1,

        (255, 0, 0),

        3

    )

    return result


# ============================================================
# COMPLETE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_path,
    class_index,
    segmentation_mask_path=None
):

    print("\n")
    print("=" * 60)
    print("COMPLETE GRAD-CAM ANALYSIS")
    print("=" * 60)

    print(
        "Image       :",
        image_path
    )

    print(
        "Class Index :",
        class_index
    )

    # --------------------------------------------------------
    # Load + preprocess
    # --------------------------------------------------------

    (
        original_image,
        image_array
    ) = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Generate raw Grad-CAM
    # --------------------------------------------------------

    heatmap = generate_raw_gradcam(

        model,

        image_array,

        class_index

    )

    # --------------------------------------------------------
    # Apply segmentation mask
    # --------------------------------------------------------

    if segmentation_mask_path:

        segmentation_mask = (
            load_segmentation_mask(

                segmentation_mask_path,

                heatmap.shape

            )
        )

        heatmap = (
            restrict_heatmap_to_lesion(

                heatmap,

                segmentation_mask

            )
        )

    else:

        segmentation_mask = None

    # --------------------------------------------------------
    # Clean weak activation
    # --------------------------------------------------------

    heatmap = clean_heatmap(

        heatmap,

        threshold=0.15

    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_gradcam_overlay(

        original_image,

        heatmap

    )

    # --------------------------------------------------------
    # Important region
    # --------------------------------------------------------

    important_region = (
        create_important_region(

            original_image,

            heatmap

        )
    )

    print("\n")
    print("=" * 60)
    print("GRAD-CAM COMPLETED")
    print("=" * 60)

    return (

        overlay,

        important_region,

        heatmap

    )


# ============================================================
# SAVE GRAD-CAM RESULTS
# ============================================================

def save_gradcam_results(
    overlay,
    important_region,
    heatmap,
    output_directory,
    image_name="gradcam"
):

    os.makedirs(

        output_directory,

        exist_ok=True

    )

    # --------------------------------------------------------
    # Grad-CAM overlay
    # --------------------------------------------------------

    overlay_path = os.path.join(

        output_directory,

        image_name +
        "_gradcam.jpg"

    )

    overlay_bgr = cv2.cvtColor(

        overlay,

        cv2.COLOR_RGB2BGR

    )

    cv2.imwrite(

        overlay_path,

        overlay_bgr

    )

    # --------------------------------------------------------
    # Important region
    # --------------------------------------------------------

    region_path = os.path.join(

        output_directory,

        image_name +
        "_important_region.jpg"

    )

    region_bgr = cv2.cvtColor(

        important_region,

        cv2.COLOR_RGB2BGR

    )

    cv2.imwrite(

        region_path,

        region_bgr

    )

    # --------------------------------------------------------
    # Raw heatmap
    # --------------------------------------------------------

    heatmap_path = os.path.join(

        output_directory,

        image_name +
        "_heatmap.jpg"

    )

    heatmap_color = (
        create_colored_heatmap(
            heatmap
        )
    )

    heatmap_bgr = cv2.cvtColor(

        heatmap_color,

        cv2.COLOR_RGB2BGR

    )

    cv2.imwrite(

        heatmap_path,

        heatmap_bgr

    )

    print("\n===================================")
    print("GRAD-CAM FILES SAVED")
    print("===================================")

    print(
        "Grad-CAM      :",
        overlay_path
    )

    print(
        "Important     :",
        region_path
    )

    print(
        "Heatmap       :",
        heatmap_path
    )

    return {

        "gradcam": overlay_path,

        "important_region": region_path,

        "heatmap": heatmap_path

    }