import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model/banknote_model_with_softmax.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess your single image
img_path = "real_img/test_img/image.jpg"
img = Image.open(img_path).resize((224, 224))

# Convert to numpy array
input_data = np.array(img, dtype=np.float32)

# IMPORTANT: Since your model has preprocess_input embedded, 
# do NOT normalize here. Just keep raw pixel values in [0,255].

# Add batch dimension
input_data = np.expand_dims(input_data, axis=0)

# Set tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output and predicted class
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

print(f"Predicted class: {predicted_class}")
