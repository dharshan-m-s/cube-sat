import tensorflow as tf
import config

print("TensorFlow version:", tf.__version__)

# ---------- GPU SETUP ----------

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("GPU detected:", gpus)

    # Mixed precision for RTX 4050
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
else:
    print("No GPU - running on CPU")

# ---------- DATA LOADING ----------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    config.DATASET_PATH,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    config.DATASET_PATH,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ---------- MODEL ----------

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)

# float32 output needed for mixed precision
out = Dense(4, activation='softmax', dtype='float32')(x)

model = Model(base.input, out)

model.compile(
    optimizer=Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------- TRAIN ----------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=config.EPOCHS
)

model.save(config.MODEL_NAME)

print("Model saved as", config.MODEL_NAME)

# Save class mapping
import json
with open("classes.json","w") as f:
    json.dump(train_data.class_indices, f)
