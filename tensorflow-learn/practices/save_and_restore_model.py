#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

#%%
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


#%%
def buid_model():
    model = keras.models.Sequential(
        [
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    return model

model = buid_model()
model.summary()

#%%
checkpoint_path = "tensorflow-learn/practices/models/model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only = True,
    verbose=1
)
model = buid_model()
model.fit(
    train_images, train_labels, epochs=10,
    validation_data=(test_images, test_labels), callbacks=[cp_callback]
)


#%% [markdown]
# create a new, untrained model
model = buid_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#%%
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#%%
checkpoint_path = "tensorflow-learn/practices/models/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=5
)
model = buid_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(
    train_images, train_labels,
    epochs=50, callbacks=[cp_callback],
    validation_data=(test_images, test_labels),
    verbose=0
)


#%%
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

#%%
model = buid_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#%%
model.save_weights("tensorflow-learn/tutorial/checkpoints/my_checkpoint")
model = buid_model()
model.load_weights("tensorflow-learn/tutorial/checkpoints/my_checkpoint")
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#%%
model = buid_model()
model.fit(
    train_images, train_labels, epochs=5
)
model.save("tensorflow-learn/tutorial/checkpoints/my_whole_model.h5")

#%%
new_model = keras.models.load_model("tensorflow-learn/tutorial/checkpoints/my_whole_model.h5")
new_model.summary()

#%%
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy:{:5.2f}%".format(100*acc))

#%%
