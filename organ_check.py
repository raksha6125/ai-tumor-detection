import cv2
import numpy as np

def is_abdominal_ct(image_path):
    """
    A LIGHTWEIGHT and FLEXIBLE CT scan checker.
    Rejects photos and colourful images, but allows any CT/MRI-like images.
    """

    img = cv2.imread(image_path)
    if img is None:
        return False

    # Resize small
    small = cv2.resize(img, (128, 128))

    # -----------------------------
    # RULE 1: Check if image is mostly grayscale
    # -----------------------------
    b, g, r = cv2.split(small)
    color_diff = np.mean(np.abs(b - g)) + np.mean(np.abs(g - r))

    # If colors differ too much, it's not a CT scan
    if color_diff > 40:   # earlier was too strict
        return False

    # -----------------------------
    # RULE 2: Check brightness distribution
    # -----------------------------
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # CT usually has a wide grayscale spread
    std_intensity = np.std(gray)

    if std_intensity < 20:   # very dark or very flat image → reject
        return False

    # -----------------------------
    # RULE 3: Basic shape check (allow almost anything)
    # -----------------------------
    h, w = img.shape[:2]
    ratio = max(h, w) / min(h, w)
    if ratio > 3:    # portrait photos (selfies) often fail here
        return False

    # If all checks passed → accept
    return True
