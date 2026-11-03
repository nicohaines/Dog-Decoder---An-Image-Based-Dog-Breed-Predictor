
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load MobileNetV2 features
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dog_features_mobilenet.npz"

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]

print("Dataset shape:", X.shape)
print("Number of images:", len(y))

# Same stratified 80/20 split as previous experiments
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Five-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

k_values = [1, 3, 5, 7, 9]

best_score = 0
best_k = None

print("\nMobileNetV2 + KNN Results")
print("-" * 50)

# Test different K values
for k in k_values:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    mean_score = scores.mean()
    std_score = scores.std()

    print(
        f"k={k}: CV accuracy = "
        f"{mean_score * 100:.2f}% "
        f"(+/- {std_score * 100:.2f}%)"
    )

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

# Evaluate the best model on the held-out test set
print("\nBest K:", best_k)
print(f"Best CV accuracy: {best_score * 100:.2f}%")

best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, predictions)

print("\nFinal Test Accuracy:", f"{test_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
