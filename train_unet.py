import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os, cv2, glob, json

# ---------------- CONFIG ----------------
IMG_SIZE = 128
DATASET_PATH = "dataset/"  # Must contain 'images/' and 'masks/' subfolders
MODEL_PATH = "model/unet_model.h5"
METRICS_PATH = "model/metrics.json"

os.makedirs("model", exist_ok=True)

# ---------------- DATA LOADER ----------------
def load_data():
    images, masks = [], []
    for img_path in glob.glob(os.path.join(DATASET_PATH, "images", "*.png")):
        mask_path = os.path.join(DATASET_PATH, "masks", os.path.basename(img_path))
        if not os.path.exists(mask_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0
        images.append(img)
        masks.append(np.expand_dims(mask, axis=-1))
    return np.array(images), np.array(masks)

# ---------------- METRICS ----------------
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

def iou_metric(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

# ---------------- MODEL ----------------
def build_unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(b)

    # Decoder
    u3 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(b)
    u3 = layers.concatenate([u3, c3])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    u2 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)

    u1 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = layers.Conv2D(16, 3, activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(16, 3, activation='relu', padding='same')(c6)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', dice_coefficient, iou_metric])
    return model

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("📂 Loading dataset...")
    X, y = load_data()
    print(f"✅ Loaded {len(X)} images")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    print("🧠 Building model...")
    model = build_unet()
    model.summary()

    print("🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=8
    )

    print("💾 Saving model...")
    model.save(MODEL_PATH)

    # ---------------- METRICS CALCULATION ----------------
    print("📊 Evaluating model on validation data...")
    results = model.evaluate(X_val, y_val, verbose=0)
    val_loss, val_acc, val_dice, val_iou = results

    metrics_data = {
        "loss": float(val_loss),
        "accuracy": float(val_acc * 100),
        "dice_score": float(val_dice),
        "iou_score": float(val_iou)
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("\n✅ Training complete!")
    print(f"📈 Dice Coefficient: {val_dice:.4f}")
    print(f"📈 IoU Score: {val_iou:.4f}")
    print(f"📈 Pixel Accuracy: {val_acc * 100:.2f}%")
    print(f"💾 Model saved to {MODEL_PATH}")
    print(f"💾 Metrics saved to {METRICS_PATH}")
