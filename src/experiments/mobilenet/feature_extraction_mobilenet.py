
#import libraries
import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Dataset folder
DATASET_DIR = "dog_breed_balanced"

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load pretrained model MobileNetV2
print("Loading MobileNetV2...")

model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

print("Model loaded!")

features = []
labels = []
filenames = []

# Get breed folders
breed_folders = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

# Process each breed
for breed in breed_folders:
    breed_path = os.path.join(DATASET_DIR, breed)

    print(f"\nProcessing {breed}...")

    image_batch = []
    batch_labels = []
    batch_filenames = []

    image_files = sorted(os.listdir(breed_path))

    for filename in image_files:
        image_path = os.path.join(breed_path, filename)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        image = preprocess_input(image.astype(np.float32))

        image_batch.append(image)
        batch_labels.append(breed)
        batch_filenames.append(filename)

        # Process full batches
        if len(image_batch) == BATCH_SIZE:
            batch_features = model.predict(
                np.array(image_batch),
                verbose=0
            )

            features.extend(batch_features)
            labels.extend(batch_labels)
            filenames.extend(batch_filenames)

            image_batch = []
            batch_labels = []
            batch_filenames = []

    # Process remaining images
    if image_batch:
        batch_features = model.predict(
            np.array(image_batch),
            verbose=0
        )

        features.extend(batch_features)
        labels.extend(batch_labels)
        filenames.extend(batch_filenames)

    print(f"Finished {breed}")

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)
filenames = np.array(filenames)

# Save extracted features
np.savez(
    "dog_features_mobilenet.npz",
    X=X,
    y=y,
    filenames=filenames
)

print("\nFeature extraction complete!")
print("Images processed:", len(X))
print("Feature dimensions:", X.shape)
print("Breeds:", np.unique(y))
print("Saved to dog_features_mobilenet.npz")
