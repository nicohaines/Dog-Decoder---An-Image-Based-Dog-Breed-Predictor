
import os
import hashlib
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

DATASET_DIR = "dog_breed_balanced"

# Store image information
images = []

# Find all images
for breed in sorted(os.listdir(DATASET_DIR)):
    breed_path = Path(DATASET_DIR) / breed

    if not breed_path.is_dir():
        continue

    for file in sorted(breed_path.iterdir()):
        if file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        try:
            with Image.open(file) as img:
                images.append({
                    "path": str(file),
                    "breed": breed,
                    "hash": imagehash.phash(img)
                })
        except Exception as e:
            print(f"Could not process {file}: {e}")

print("Images checked:", len(images))

# 1. Check exact duplicates using file hashes
exact_groups = defaultdict(list)

for item in images:
    with open(item["path"], "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    exact_groups[file_hash].append(item["path"])

exact_duplicates = [
    group for group in exact_groups.values()
    if len(group) > 1
]

print("\nEXACT DUPLICATES")
print("Duplicate groups:", len(exact_duplicates))

for group in exact_duplicates:
    print(group)

# 2. Check near duplicates using perceptual hashes
near_duplicates = []

for i in range(len(images)):
    for j in range(i + 1, len(images)):
        distance = images[i]["hash"] - images[j]["hash"]

        if distance <= 5:
            near_duplicates.append((
                images[i]["path"],
                images[j]["path"],
                images[i]["breed"],
                images[j]["breed"],
                distance
            ))

print("\nNEAR DUPLICATES")
print("Pairs found:", len(near_duplicates))

for first, second, breed1, breed2, distance in near_duplicates:
    print(f"\nDistance: {distance}")
    print(f"Image 1: {first} ({breed1})")
    print(f"Image 2: {second} ({breed2})")

print("\nDuplicate check complete!")
