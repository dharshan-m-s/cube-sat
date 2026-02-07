import tensorflow as tf
import numpy as np
import cv2
import json
import config

# Load model
model = tf.keras.models.load_model(config.MODEL_NAME)

# Load class map
with open("classes.json") as f:
    class_map = json.load(f)

inv_map = {v:k for k,v in class_map.items()}

def predict(path):

    img = cv2.imread(path)

    if img is None:
        print("Image not found")
        return

    img = cv2.resize(img, config.IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]

    print("\nProbabilities:")
    for i,p in enumerate(pred):
        print(inv_map[i], ":", round(float(p),3))

    idx = np.argmax(pred)

    print("\nFINAL RESULT →",
          inv_map[idx],
          "Confidence:",
          round(float(pred[idx]),3))

# CHANGE THIS TO YOUR IMAGE
predict("test_image.jpg")
