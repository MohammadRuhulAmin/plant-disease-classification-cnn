import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
CHANNELS = 3
n_classes = 3

# --- Data Loading ---
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/mnt/c/development/Thesis/PotatoDiseaseClassification-CNN/DataSets/PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names

# --- Dataset Partitioning Function ---
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# --- Performance Tuning ---
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- Preprocessing & Augmentation Layers ---
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# --- Building Parallel Depth-CNN using Functional API ---
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
inputs = layers.Input(shape=input_shape)

# Apply preprocessing
x = resize_and_rescale(inputs)
x = data_augmentation(x)

# Parallel Branch 1: 3x3 Kernel
branch_3x3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
branch_3x3 = layers.MaxPooling2D((2, 2))(branch_3x3)

# Parallel Branch 2: 5x5 Kernel
branch_5x5 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(x)
branch_5x5 = layers.MaxPooling2D((2, 2))(branch_5x5)

# Parallel Branch 3: 7x7 Kernel
branch_7x7 = layers.Conv2D(32, (7, 7), padding='same', activation='relu')(x)
branch_7x7 = layers.MaxPooling2D((2, 2))(branch_7x7)

# Concatenate all parallel branches
merged = layers.Concatenate()([branch_3x3, branch_5x5, branch_7x7])

# Deep Feature Extraction
x = layers.Conv2D(64, (3, 3), activation='relu')(merged)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Fully Connected Layers
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)

# Create Model
model = models.Model(inputs=inputs, outputs=outputs)

# --- Model Summary & Compilation ---
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# --- Training ---
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

# --- Evaluation & Visualization ---
scores = model.evaluate(test_ds)

# Accuracy & Loss Plots
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

# --- Saving Model ---
model_version = "Parallel_CNN_v1"
model.save(f"./models/{model_version}.keras")