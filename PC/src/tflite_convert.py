import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Softmax
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "model/banknote_classifier_mobilenetv2.h5"

# Load model
model = load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input})

# Call model with dummy input shape to build it (if needed)
# This ensures model.input and model.output are defined
_ = model(tf.keras.Input(shape=(224, 224, 3)))  # Adjust input shape if needed

# Append softmax if not already there
if not isinstance(model.layers[-1], tf.keras.layers.Softmax):
    print("Appending Softmax layer before conversion.")
    output = Softmax()(model(model.input))
    model = Model(inputs=model.input, outputs=output)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open("model/banknote_model_with_softmax.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted and saved as banknote_model_with_softmax.tflite")
