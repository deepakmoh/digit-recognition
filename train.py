"""Download MNIST, augment it, train a neural network, and save model.joblib."""
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

from recognizer import MODEL_PATH

rng = np.random.default_rng(0)


def augment(images: np.ndarray) -> np.ndarray:
    """Small random rotations, scalings and shifts, so the model copes with
    the variety of real mouse/touch handwriting."""
    out = np.empty_like(images)
    for i, im in enumerate(images):
        img = Image.fromarray((im * 255).astype(np.uint8))
        angle = rng.uniform(-12, 12)
        scale = rng.uniform(0.85, 1.1)
        dx, dy = rng.uniform(-2, 2, size=2)
        img = img.rotate(angle, resample=Image.BILINEAR, translate=(dx, dy))
        if abs(scale - 1) > 0.01:
            size = max(1, int(round(28 * scale)))
            scaled = img.resize((size, size), Image.BILINEAR)
            canvas = Image.new("L", (28, 28))
            canvas.paste(scaled, ((28 - size) // 2, (28 - size) // 2))
            img = canvas
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
    return out


def main():
    t0 = time.time()
    print("Loading MNIST (first run downloads ~15 MB)...")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                        data_home="data", return_X_y=True)
    X = (X / 255.0).astype(np.float32).reshape(-1, 28, 28)
    y = y.astype(int)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    print("Augmenting training set...")
    X_aug = augment(X_train)
    X_fit = np.concatenate([X_train, X_aug]).reshape(-1, 784)
    y_fit = np.concatenate([y_train, y_train])

    print(f"Training on {len(X_fit):,} images...")
    model = MLPClassifier(hidden_layer_sizes=(512, 256), alpha=1e-4, batch_size=256,
                          learning_rate_init=1e-3, max_iter=30, early_stopping=True,
                          n_iter_no_change=4, random_state=0, verbose=True)
    model.fit(X_fit, y_fit)

    pred = model.predict(X_test.reshape(-1, 784))
    print(f"\nTest accuracy on 10,000 unseen MNIST digits: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, digits=3))
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Saved {MODEL_PATH} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
