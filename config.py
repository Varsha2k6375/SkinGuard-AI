import os

# ===============================
# Dataset Paths
# ===============================

DATASET_PATH = "datasets/isic_segmentation"

IMAGE_PATH = "datasets/isic_segmentation/images"

MASK_PATH = "datasets/isic_segmentation/masks"

MODEL_PATH = "models/unet_model.keras"
# ===============================
# Image Settings
# ===============================

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNELS = 3

# ===============================
# Training Settings
# ===============================

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001

# ===============================
# Dataset Split
# ===============================

TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.10
TEST_SPLIT = 0.10

# ===============================
# Model Save Path
# ===============================

MODEL_PATH = "models/unet_model.keras"

# ===============================
# Random Seed
# ===============================

RANDOM_SEED = 42