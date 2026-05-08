import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

# 1. Global constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
CHANNELS = 3
n_classes = 3

# 2. Data Loading (Based on  original code)
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names

# Data partition function
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

# Performance Optimization
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# 3. Custom Layers for TR-SE-NET

def squeeze_and_excitation_block(input_tensor, ratio=16):
    """SE Block: Channel-based attention mechanism"""
    init = input_tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    return layers.Multiply()([init, se])

def transformer_block(x, num_heads=4, key_dim=64):
    """Transformer Block: Global context understanding"""
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attention_output = layers.Dropout(0.1)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed Forward Network
    ffn = layers.Dense(x.shape[-1], activation="relu")(x)
    ffn = layers.Dropout(0.1)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x

# 4. Model Creation (TR-SE-NET Architecture)

inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

# Pre-processing and Augmentation
x = layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(inputs)
x = layers.Rescaling(1.0/255)(x)
x = layers.RandomFlip("horizontal_and_vertical")(x)
x = layers.RandomRotation(0.2)(x)

# Stage 1: Initial Feature Extraction (CNN Part)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = squeeze_and_excitation_block(x) 
x = layers.MaxPooling2D((2, 2))(x)

# Stage 2: Transformer Block Integration
# Reducing image dimensions so that the transformer can process it efficiently
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = squeeze_and_excitation_block(x)
x = layers.MaxPooling2D((2, 2))(x)


shape = x.shape
x_reshaped = layers.Reshape((-1, shape[-1]))(x) 
x_transformed = transformer_block(x_reshaped)
x = layers.Reshape((shape[1], shape[2], shape[3]))(x_transformed)

# Stage 3: Final Classification
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Model Summary
model.summary()

# 5. Model compile and Training
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

# 6. Evaluation and Plotting (Based on your code)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 7. Model Saving
model.save("./models/plant_tr_se_net_v1.keras")