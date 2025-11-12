import numpy as np
import tensorflow as tf

# --- config ---
img_size = 28
batch_size = 128
lr = 1e-4 # try 1e-4 ~ 3e-3

# --- preprocess: add channel -> resize -> RGB -> [0,1] ---
def preprocess(img, label):
    img = tf.expand_dims(img, -1)                     # (28,28,1)
    img = tf.image.resize(img, (img_size, img_size))  # (H,W,1)
    img = tf.image.grayscale_to_rgb(img)              # (H,W,3)
    img = tf.cast(img, tf.float32) / 255.0            # [0,1]
    return img, label

# --- load data ---
(data_train, labels_train), (data_test, labels_test) = tf.keras.datasets.fashion_mnist.load_data()

# --- build datasets (map -> batch -> prefetch) ---
ds_train = (tf.data.Dataset.from_tensor_slices((data_train, labels_train))
            .shuffle(10000, reshuffle_each_iteration=True)
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

ds_val = (tf.data.Dataset.from_tensor_slices((data_test, labels_test))
          .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))

# --- model ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_size, img_size, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
])

opt = tf.keras.optimizers.Adam(learning_rate=lr)  
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ds_train, validation_data=ds_val, epochs=50)

model.save("fmnist_model.keras")               # full model
# Or just weights (Keras 3 requires .weights.h5 suffix)
model.save_weights("fmnist_model.weights.h5")

### Load & Infer
# Load full model (compile not needed for inference)
# model = tf.keras.models.load_model("fmnist_model.keras", compile=False)

# # Example: single image (28×28 grayscale) -> preprocess and predict
# def preprocess_image(path):
#     img = tf.io.read_file(path)
#     img = tf.io.decode_png(img, channels=1)
#     img = tf.image.resize(img, (28, 28))
#     img = tf.cast(img, tf.float32) / 255.0
#     return tf.expand_dims(img, 0)  # [1, 28, 28, 1]

# x = preprocess_image("any_image.png")
# probs = model.predict(x)                 # shape [1, 10]
# pred = int(tf.argmax(probs, axis=-1))    # class index
# print("pred:", pred, "probs:", probs[0])
