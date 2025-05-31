import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import os

# ----- CONFIG -----
IMAGE_SIZE = (224, 224)
MODEL_PATH = "model/banknote_classifier_mobilenetv2.h5"

# ----- LOAD MODEL -----
model = load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input})

# ----- LOAD CLASS NAMES (same order as used in training) -----
# You can hardcode class names if known, e.g.:
# class_names = ['KRW_1000', 'USD_1', ...]
# OR dynamically load them (if dataset structure is available):
# Use a dummy dataset just to get the class names
dummy_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "testset",  # Must point to a directory with subfolders named after classes
    image_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False
)
class_names = dummy_ds.class_names

# ----- LOAD & PREPROCESS SINGLE IMAGE -----
test_img = tf.keras.preprocessing.image_dataset_from_directory(
    "real_img",
    labels="inferred",
    image_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False,  # Important for correct label-order alignment
    label_mode="categorical"
)

test_img = test_img.prefetch(tf.data.AUTOTUNE)

# ----- PREDICT -----
for img, labels in test_img:
    predictions = model.predict(img)

sorted_indices = sorted(range(len(predictions[0])), key=lambda i: predictions[0][i], reverse=True)

if (predictions[0][sorted_indices[0]] < 0.6):
    print()
    print("No banknote detected with sufficient confidence.")
    print(f"Predictions below threshold: {predictions[0][sorted_indices[0]]:.4f}")
    print()
else:
    print()
    print(" Predicted class  | Confidence")
    print("------------------|-----------")
    for i in sorted_indices[0:5]:  # Top 5 predictions
        class_name = class_names[i]
        confidence = predictions[0][i]
        print(f"{class_name:17} | {confidence:.4f}")
    print()
