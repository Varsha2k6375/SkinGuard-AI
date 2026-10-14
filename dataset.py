import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    IMAGE_PATH,
    MASK_PATH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    RANDOM_SEED
)


def load_datasets():

    images = []
    masks = []

    image_files = sorted(os.listdir(IMAGE_PATH))

    for file in image_files:

        image_file = os.path.join(IMAGE_PATH, file)

        mask_name = os.path.splitext(file)[0] + "_segmentation.png"
        mask_file = os.path.join(MASK_PATH, mask_name)

        # Skip if mask does not exist
        if not os.path.exists(mask_file):
            continue

        # Read image
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        # Read mask
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(
            mask,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Expand mask dimension
        mask = np.expand_dims(mask, axis=-1)

        images.append(image)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    return images, masks


def split_dataset(images, masks):

    X_train, X_temp, y_train, y_temp = train_test_split(
        images,
        masks,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )


if __name__ == "__main__":

    images, masks = load_datasets()

    print("=" * 50)
    print("U-Net Dataset Loaded Successfully")
    print("=" * 50)

    print("Total Images :", len(images))
    print("Total Masks  :", len(masks))

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    ) = split_dataset(images, masks)

    print()

    print("Training Images   :", len(X_train))
    print("Validation Images :", len(X_valid))
    print("Testing Images    :", len(X_test))