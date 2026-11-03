
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

data = np.load("dog_features_hog.npz")
X = data["X"]
y = data["y"]

# The first 544 features are the original histograms.
X_hist = X[:, :544]

# The remaining features are HOG.
X_hog = X[:, 544:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Use the same training/test indices for all feature experiments.
train_indices, test_indices = train_test_split(
    np.arange(len(y)),
    test_size=0.20,
    random_state=42,
    stratify=y
)

feature_sets = {
    "Histograms only": X_hist,
    "HOG only": X_hog,
    "Combined features": X
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for feature_name, features in feature_sets.items():
    X_tr = features[train_indices]
    X_te = features[test_indices]

    for k in [1, 3, 5, 7, 9]:
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k)
        )

        scores = cross_val_score(
            model, X_tr, y[train_indices],
            cv=cv, scoring="accuracy"
        )

        results.append((scores.mean(), feature_name, k))
        print(
            f"{feature_name:20s} k={k}: "
            f"CV accuracy={scores.mean():.2%} "
            f"(+/- {scores.std():.2%})"
        )

# Choose settings using training cross-validation only.
best_score, best_features, best_k = max(results, key=lambda r: r[0])

print("\nBest cross-validation result:")
print(f"Features: {best_features}")
print(f"k: {best_k}")
print(f"Mean CV accuracy: {best_score:.2%}")

# Evaluate the selected configuration on the held-out test set.
chosen_X = feature_sets[best_features]
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=best_k)
)
model.fit(chosen_X[train_indices], y[train_indices])
predictions = model.predict(chosen_X[test_indices])

print(f"\nFinal test accuracy: {accuracy_score(y[test_indices], predictions):.2%}")
