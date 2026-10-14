import tensorflow as tf
from tensorflow.keras import backend as K


# =====================================
# Dice Coefficient
# =====================================

def dice_coefficient(y_true, y_pred):

    smooth = 1.0

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)

    return (

        (2.0 * intersection + smooth)

        /

        (K.sum(y_true) + K.sum(y_pred) + smooth)

    )


# =====================================
# IoU (Intersection over Union)
# =====================================

def iou_score(y_true, y_pred):

    smooth = 1.0

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)

    union = (

        K.sum(y_true)

        +

        K.sum(y_pred)

        -

        intersection

    )

    return (

        intersection + smooth

    ) / (

        union + smooth

    )


# =====================================
# Precision
# =====================================

def precision(y_true, y_pred):

    true_positive = K.sum(
        K.round(
            K.clip(
                y_true * y_pred,
                0,
                1
            )
        )
    )

    predicted_positive = K.sum(
        K.round(
            K.clip(
                y_pred,
                0,
                1
            )
        )
    )

    return true_positive / (predicted_positive + K.epsilon())


# =====================================
# Recall
# =====================================

def recall(y_true, y_pred):

    true_positive = K.sum(
        K.round(
            K.clip(
                y_true * y_pred,
                0,
                1
            )
        )
    )

    possible_positive = K.sum(
        K.round(
            K.clip(
                y_true,
                0,
                1
            )
        )
    )

    return true_positive / (possible_positive + K.epsilon())


# =====================================
# Pixel Accuracy
# =====================================

def pixel_accuracy(y_true, y_pred):

    y_pred = K.round(y_pred)

    correct_pixels = K.sum(
        K.cast(
            K.equal(y_true, y_pred),
            "float32"
        )
    )

    total_pixels = K.cast(
        K.prod(K.shape(y_true)),
        "float32"
    )

    return correct_pixels / total_pixels