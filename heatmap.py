import tensorflow as tf
import numpy as np
import cv2
import os


# ============================================================
# FIND EFFICIENTNET BASE MODEL
# ============================================================

def find_base_model(model):

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            print(
                "EfficientNet Base Model Found:",
                layer.name
            )

            return layer

    raise ValueError(
        "Grad-CAM Error: EfficientNet base model not found."
    )


# ============================================================
# FIND TARGET LAYER
# ============================================================

def find_target_layer(base_model):

    # --------------------------------------------------------
    # Try top_activation
    # --------------------------------------------------------

    try:

        target_layer = base_model.get_layer(
            "top_activation"
        )

        print(
            "Grad-CAM Target Layer:",
            target_layer.name
        )

        return target_layer

    except Exception:

        pass


    # --------------------------------------------------------
    # Search backwards
    # --------------------------------------------------------

    for layer in reversed(
        base_model.layers
    ):

        try:

            output_shape = layer.output.shape

            if (
                len(output_shape) == 4
                and output_shape[1] is not None
                and output_shape[2] is not None
            ):

                print(
                    "Grad-CAM Target Layer:",
                    layer.name
                )

                return layer

        except Exception:

            continue


    raise ValueError(
        "Grad-CAM Error: "
        "No suitable convolutional layer found."
    )


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image_path):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            "Image not found: "
            + image_path
        )


    image = cv2.imread(
        image_path
    )


    if image is None:

        raise ValueError(
            "Unable to read image."
        )


    # OpenCV BGR -> RGB

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    original = image_rgb.copy()


    # --------------------------------------------------------
    # Resize to model input
    # --------------------------------------------------------

    resized = cv2.resize(
        image_rgb,
        (224, 224)
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    img_array = (
        resized.astype(
            np.float32
        ) / 255.0
    )


    return (
        original,
        img_array
    )


# ============================================================
# FIND CLASSIFIER OUTPUT
# ============================================================

def build_gradcam_model(model):

    base_model = find_base_model(
        model
    )

    target_layer = find_target_layer(
        base_model
    )


    # --------------------------------------------------------
    # Create feature model
    # --------------------------------------------------------

    feature_model = tf.keras.models.Model(

        inputs=base_model.input,

        outputs=[
            target_layer.output,
            base_model.output
        ],

        name="gradcam_feature_model"

    )


    # --------------------------------------------------------
    # Find base model index
    # --------------------------------------------------------

    base_index = None

    for i, layer in enumerate(
        model.layers
    ):

        if layer is base_model:

            base_index = i

            break


    if base_index is None:

        raise ValueError(
            "Grad-CAM Error: "
            "Unable to locate EfficientNet base model."
        )


    classifier_layers = model.layers[
        base_index + 1:
    ]


    print(
        "Classifier Layers:"
    )


    for layer in classifier_layers:

        print(
            " -",
            layer.name
        )


    return (
        feature_model,
        classifier_layers
    )


# ============================================================
# GENERATE RAW GRAD-CAM
# ============================================================

def generate_raw_gradcam(
    model,
    img_array,
    class_index
):

    print("\n===================================")
    print("GENERATING RAW GRAD-CAM")
    print("===================================")


    if model is None:

        raise ValueError(
            "Grad-CAM Error: Model is None."
        )


    img_array = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32
    )


    if len(
        img_array.shape
    ) == 3:

        img_array = tf.expand_dims(
            img_array,
            axis=0
        )


    if len(
        img_array.shape
    ) != 4:

        raise ValueError(
            "Invalid image shape: "
            + str(img_array.shape)
        )


    print(
        "Grad-CAM Input Shape:",
        img_array.shape
    )


    (
        feature_model,
        classifier_layers
    ) = build_gradcam_model(
        model
    )


    # ========================================================
    # GRADIENT TAPE
    # ========================================================

    with tf.GradientTape() as tape:

        (
            conv_outputs,
            base_output
        ) = feature_model(

            img_array,

            training=False

        )


        # ----------------------------------------------------
        # PASS THROUGH CLASSIFIER
        # ----------------------------------------------------

        x = base_output


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

                x = layer(
                    x
                )


        predictions = x


        print(
            "Prediction Shape:",
            predictions.shape
        )


        # ----------------------------------------------------
        # Select requested class
        # ----------------------------------------------------

        class_channel = predictions[
            :,
            int(class_index)
        ]


    # ========================================================
    # CALCULATE GRADIENT
    # ========================================================

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )


    if grads is None:

        raise ValueError(
            "Grad-CAM Error: Gradients are None."
        )


    print(
        "Gradient Shape:",
        grads.shape
    )

    print(
        "Feature Map Shape:",
        conv_outputs.shape
    )


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )


    # ========================================================
    # REMOVE BATCH
    # ========================================================

    conv_outputs = conv_outputs[0]

    pooled_grads = pooled_grads[0]


    # ========================================================
    # WEIGHT FEATURE MAPS
    # ========================================================

    heatmap = tf.reduce_sum(

        conv_outputs *
        pooled_grads,

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

    max_value = tf.reduce_max(
        heatmap
    )


    max_value = float(
        max_value.numpy()
    )


    if max_value > 0:

        heatmap = (
            heatmap /
            max_value
        )

    else:

        heatmap = tf.zeros_like(
            heatmap
        )


    heatmap = heatmap.numpy()


    print(
        "Raw Grad-CAM Heatmap:",
        heatmap.shape
    )


    return heatmap


# ============================================================
# CREATE HEATMAP OVERLAY
# ============================================================

def create_heatmap_overlay(
    original_image,
    heatmap
):

    height, width = (
        original_image.shape[:2]
    )


    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap_resized = cv2.resize(

        heatmap,

        (
            width,
            height
        )

    )


    # --------------------------------------------------------
    # Convert 0-1 -> 0-255
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 *
        heatmap_resized
    )


    # --------------------------------------------------------
    # Apply JET
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(

        heatmap_uint8,

        cv2.COLORMAP_JET

    )


    heatmap_color = cv2.cvtColor(

        heatmap_color,

        cv2.COLOR_BGR2RGB

    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(

        original_image,
        0.45,

        heatmap_color,
        0.55,

        0

    )


    return overlay


# ============================================================
# IMPORTANT REGION
# ============================================================

def create_important_region(
    original_image,
    heatmap
):

    height, width = (
        original_image.shape[:2]
    )


    heatmap_resized = cv2.resize(

        heatmap,

        (
            width,
            height
        )

    )


    # --------------------------------------------------------
    # Threshold important region
    # --------------------------------------------------------

    threshold = 0.55


    mask = (
        heatmap_resized >= threshold
    ).astype(
        np.uint8
    ) * 255


    # --------------------------------------------------------
    # Smooth mask
    # --------------------------------------------------------

    kernel = np.ones(
        (7, 7),
        np.uint8
    )


    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel

    )


    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    result = original_image.copy()


    # --------------------------------------------------------
    # Draw important regions
    # --------------------------------------------------------

    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area < 100:

            continue


        cv2.drawContours(

            result,

            [contour],

            -1,

            (255, 0, 0),

            3

        )


    return result


# ============================================================
# COMPLETE GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    model,
    image_path,
    class_index
):

    print("\n===================================")
    print("COMPLETE GRAD-CAM")
    print("===================================")


    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    (
        original_image,
        img_array
    ) = prepare_image(
        image_path
    )


    # --------------------------------------------------------
    # Raw heatmap
    # --------------------------------------------------------

    heatmap = generate_raw_gradcam(

        model,

        img_array,

        class_index

    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_heatmap_overlay(

        original_image,

        heatmap

    )


    # --------------------------------------------------------
    # Important region
    # --------------------------------------------------------

    important_region = create_important_region(

        original_image,

        heatmap

    )


    print(
        "Grad-CAM Generated Successfully"
    )


    print(
        "===================================\n"
    )


    return (
        overlay,
        important_region,
        heatmap
    )


# ============================================================
# COLOR LEGEND
# ============================================================

def create_color_legend(
    output_path
):

    height = 500
    width = 80


    gradient = np.linspace(
        255,
        0,
        height
    ).astype(
        np.uint8
    )


    gradient = np.repeat(
        gradient[:, np.newaxis],
        width,
        axis=1
    )


    legend = cv2.applyColorMap(

        gradient,

        cv2.COLORMAP_JET

    )


    cv2.imwrite(

        output_path,

        legend

    )


    print(
        "Grad-CAM Legend Created:",
        output_path
    )