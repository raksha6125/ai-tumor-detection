import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(image_path):
    """
    Reads an uploaded image and prepares it for U-Net model prediction.
    Ensures it has 3 channels (RGB) and is normalized.
    """
    # Read image as color (3 channels)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Resize to match model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    # Expand batch dimension for model prediction
    img = np.expand_dims(img, axis=0)  # shape becomes (1,128,128,3)

    return img
