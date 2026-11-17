"""Turn a drawn image into MNIST-style 28x28 digits and classify them.

Pipeline:
  1. Grayscale, make ink bright on a dark background.
  2. Split the drawing into separate digits (connected components, with
     strokes that overlap horizontally merged, so a two-stroke "4" or "5"
     stays one digit).
  3. Normalise each digit the way MNIST was built: fit into a 20x20 box,
     place it in a 28x28 frame, and centre it by centre of mass.
  4. Classify each 28x28 image with the trained model.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageFilter

MODEL_PATH = Path(__file__).with_name("model.joblib")
INK_THRESHOLD = 40  # 0-255; pixels brighter than this count as ink


def to_ink_array(img: Image.Image) -> np.ndarray:
    """Return a float array in [0, 255] with ink bright and background dark."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    # Most pixels are background; if the background is light, invert.
    if np.median(arr) > 127:
        arr = 255.0 - arr
    return arr


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (top, bottom, left, right) of 8-connected ink blobs."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        top, bottom, left, right = y, y, x, x
        q = deque([(y, x)])
        seen[y, x] = True
        while q:
            cy, cx = q.popleft()
            top, bottom = min(top, cy), max(bottom, cy)
            left, right = min(left, cx), max(right, cx)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        boxes.append((top, bottom, left, right))
    return boxes


def segment(arr: np.ndarray, scale: int = 4) -> list[tuple[int, int, int, int]]:
    """Find digit bounding boxes in full-resolution coordinates, left to right."""
    h, w = arr.shape
    # Work on a downsampled mask so the flood fill stays fast.
    small = arr[: h - h % scale, : w - w % scale]
    small = small.reshape(h // scale, scale, w // scale, scale).max(axis=(1, 3))
    boxes = _components(small > INK_THRESHOLD)
    if not boxes:
        return []

    # Drop specks much smaller than the biggest stroke.
    biggest = max((b - t + 1) * (r - l + 1) for t, b, l, r in boxes)
    boxes = [bx for bx in boxes if (bx[1] - bx[0] + 1) * (bx[3] - bx[2] + 1) >= biggest * 0.02]

    # Merge strokes whose horizontal extents overlap (parts of one digit).
    boxes.sort(key=lambda b: b[2])
    merged = [list(boxes[0])]
    for t, b, l, r in boxes[1:]:
        m = merged[-1]
        overlap = min(m[3], r) - max(m[2], l)
        narrower = min(m[3] - m[2], r - l) + 1
        if overlap > 0.4 * narrower:
            m[0], m[1] = min(m[0], t), max(m[1], b)
            m[2], m[3] = min(m[2], l), max(m[3], r)
        else:
            merged.append([t, b, l, r])

    return [
        (t * scale, min(h, (b + 1) * scale), l * scale, min(w, (r + 1) * scale))
        for t, b, l, r in merged
    ]


def normalise(crop: np.ndarray) -> np.ndarray:
    """Crop of one digit -> 28x28 float array in [0, 1], MNIST style."""
    ys, xs = np.nonzero(crop > INK_THRESHOLD)
    crop = crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    # Pad thin digits (like "1") to a square first so they aren't stretched.
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    square[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = crop

    img = Image.fromarray(square.astype(np.uint8))
    img = img.filter(ImageFilter.MaxFilter(_odd(side // 40)))  # thicken a little
    img = img.resize((20, 20), Image.LANCZOS)
    digit = np.asarray(img, dtype=np.float32)

    out = np.zeros((28, 28), dtype=np.float32)
    out[4:24, 4:24] = digit
    # Shift so the centre of mass sits at the centre of the frame.
    total = out.sum()
    if total > 0:
        yy, xx = np.indices(out.shape)
        cy, cx = (yy * out).sum() / total, (xx * out).sum() / total
        out = np.roll(out, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))
    return np.clip(out / 255.0, 0.0, 1.0)


def _odd(n: int) -> int:
    n = max(1, n)
    return n if n % 2 else n + 1


class Recognizer:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found. Run `python train.py` first.")
        self.model = joblib.load(model_path)

    def predict_digits(self, digits: np.ndarray) -> np.ndarray:
        """digits: (n, 28, 28) in [0, 1] -> (n, 10) probabilities."""
        return self.model.predict_proba(digits.reshape(len(digits), -1))

    def recognize(self, img: Image.Image) -> dict:
        arr = to_ink_array(img)
        boxes = segment(arr)
        if not boxes:
            return {"number": "", "digits": []}
        digits = np.stack([normalise(arr[t:b, l:r]) for t, b, l, r in boxes])
        probs = self.predict_digits(digits)
        results = []
        for (t, b, l, r), p, d in zip(boxes, probs, digits):
            order = np.argsort(p)[::-1]
            results.append({
                "digit": int(order[0]),
                "confidence": float(p[order[0]]),
                "alternatives": [{"digit": int(k), "p": float(p[k])} for k in order[1:3]],
                "box": [int(l), int(t), int(r - l), int(b - t)],
                "preview": (d * 255).astype(np.uint8).tolist(),
            })
        return {"number": "".join(str(r["digit"]) for r in results), "digits": results}
