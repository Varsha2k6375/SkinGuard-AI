import tensorflow as tf
from tensorflow.keras import backend as K


# =====================================
# Dice Coefficient
# =====================================

def dice_coefficient(y_true, y_pred):

    smooth = 1.0

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (

        (2.0 * intersection + smooth)

        /

        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    )


# =====================================
# Dice Loss
# =====================================

def dice_loss(y_true, y_pred):

    return 1 - dice_coefficient(y_true, y_pred)


# =====================================
# Binary Cross Entropy Loss
# =====================================

def binary_crossentropy_loss(y_true, y_pred):

    return tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )


# =====================================
# Combined BCE + Dice Loss
# =====================================

def bce_dice_loss(y_true, y_pred):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    dice = dice_loss(
        y_true,
        y_pred
    )

    return bce + dice