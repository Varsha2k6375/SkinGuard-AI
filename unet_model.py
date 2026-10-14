import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate,
    Dropout
)
from tensorflow.keras.models import Model


# =====================================
# Convolution Block
# =====================================

def conv_block(inputs, filters):

    x = Conv2D(filters, (3, 3), padding="same", activation="relu")(inputs)
    x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)

    return x


# =====================================
# Encoder Block
# =====================================

def encoder_block(inputs, filters):

    x = conv_block(inputs, filters)

    p = MaxPooling2D((2, 2))(x)

    p = Dropout(0.2)(p)

    return x, p


# =====================================
# Decoder Block
# =====================================

def decoder_block(inputs, skip_features, filters):

    x = Conv2DTranspose(
        filters,
        (2, 2),
        strides=2,
        padding="same"
    )(inputs)

    x = Concatenate()([x, skip_features])

    x = conv_block(x, filters)

    return x


# =====================================
# U-Net Model
# =====================================

def build_unet(input_shape=(256, 256, 3)):

    inputs = Input(input_shape)

    # Encoder

    s1, p1 = encoder_block(inputs, 64)

    s2, p2 = encoder_block(p1, 128)

    s3, p3 = encoder_block(p2, 256)

    s4, p4 = encoder_block(p3, 512)

    # Bottleneck

    b1 = conv_block(p4, 1024)

    # Decoder

    d1 = decoder_block(b1, s4, 512)

    d2 = decoder_block(d1, s3, 256)

    d3 = decoder_block(d2, s2, 128)

    d4 = decoder_block(d3, s1, 64)

    # Output Layer

    outputs = Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="sigmoid"
    )(d4)

    model = Model(inputs, outputs, name="U-Net")

    return model


# =====================================
# Test Model
# =====================================

if __name__ == "__main__":

    model = build_unet()

    model.summary()