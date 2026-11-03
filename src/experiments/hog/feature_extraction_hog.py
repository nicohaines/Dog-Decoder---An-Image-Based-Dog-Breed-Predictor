from pathlib import Path

import cv2
import numpy as np # Numerical operations and array handling
from skimage.feature import hog # Histogram of Oriented Gradients for texture and shape analysis

DATA_DIR = Path("dog_breed_balanced") # Directory containing the dog breed images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"} # Supported image file extensions


def extract_features(image_path):
    """Extract color, grayscale, and HOG features from a dog image."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (128, 128))

    # Original color histogram: 512 features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    color_hist = cv2.normalize(color_hist, color_hist).flatten()

    # Original grayscale intensity histogram: 32 features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_hist = cv2.calcHist(
        [gray], [0], None, [32], [0, 256]
    )
    texture_hist = cv2.normalize(texture_hist, texture_hist).flatten()

    # New HOG features: local edge directions and shapes
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    # Keep the original 544 features first, then append HOG
    return np.concatenate([color_hist, texture_hist, hog_features])


def main():
    feature_vectors = []
    labels = []

    for breed_folder in sorted(DATA_DIR.iterdir()):
        if not breed_folder.is_dir():
            continue

        for image_path in sorted(breed_folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                features = extract_features(image_path)
                feature_vectors.append(features)
                labels.append(breed_folder.name)
            except ValueError as error:
                print(error)

    X = np.asarray(feature_vectors, dtype=np.float32)
    y = np.asarray(labels)

    print("Images processed:", len(X))
    print("Features per image:", X.shape[1] if X.ndim == 2 else 0)
    print("Breed categories:", len(set(y)))

    np.savez_compressed(
        "dog_features_hog.npz",
        X=X,
        y=y
    )
    print("Saved features to dog_features_hog.npz")


if __name__ == "__main__":
    main()
