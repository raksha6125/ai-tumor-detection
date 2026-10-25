import os
import cv2
import numpy as np
import random

# Create dataset folders
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)

def random_shape(mask, img, image_size):
    """Draw random tumor-like shapes on both image and mask."""
    shape_type = random.choice(["circle", "ellipse", "irregular"])
    x, y = random.randint(30, image_size[0]-30), random.randint(30, image_size[1]-30)

    if shape_type == "circle":
        r = random.randint(10, 25)
        cv2.circle(img, (x, y), r, (255), -1)
        cv2.circle(mask, (x, y), r, (255), -1)

    elif shape_type == "ellipse":
        ax, ay = random.randint(10, 25), random.randint(10, 25)
        angle = random.randint(0, 180)
        cv2.ellipse(img, (x, y), (ax, ay), angle, 0, 360, (255), -1)
        cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, (255), -1)

    else:  # irregular blob
        pts = np.array([
            [x + random.randint(-15, 15), y + random.randint(-15, 15)]
            for _ in range(random.randint(5, 10))
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (255))
        cv2.fillPoly(mask, [pts], (255))

def generate_sample(image_size=(128, 128)):
    """Generate a synthetic CT-like pancreas image and its tumor mask."""
    img = np.zeros(image_size, dtype=np.uint8)
    mask = np.zeros(image_size, dtype=np.uint8)

    # Simulate organ texture with soft blobs
    for _ in range(40):
        x, y = random.randint(0, image_size[0]), random.randint(0, image_size[1])
        r = random.randint(3, 12)
        color = random.randint(30, 180)
        cv2.circle(img, (x, y), r, (color), -1)

    # Apply Gaussian blur to make it look like a scan
    img = cv2.GaussianBlur(img, (9, 9), 0)

    # Draw 1–3 tumor-like shapes
    for _ in range(random.randint(1, 3)):
        random_shape(mask, img, image_size)

    # Slightly brighten tumor area
    img = cv2.add(img, mask // 4)

    # Normalize and clip to valid range
    img = np.clip(img, 0, 255)

    return img, mask

# Generate 100 samples
for i in range(1, 101):
    img, mask = generate_sample()
    cv2.imwrite(f"dataset/images/sample_{i:03}.png", img)
    cv2.imwrite(f"dataset/masks/sample_{i:03}.png", mask)

print("✅ 100 synthetic CT-like images and masks created in 'dataset/' folders!")
