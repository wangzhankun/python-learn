#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#%%
dataset_path = keras.utils.get_file("E:/aboutme/STUDY/python-learn/tensorflow-learn/practices/auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

#%%
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

#%% [markdown]
# ## clean the data
# the dataset contains a few unknown values
dataset.isna().sum()


#%% [markdown]
# to keep this initial tutorial simple drop those rows
dataset = dataset.dropna()

#%% [markdown]
# The 'Origin' column is really categorical, not numeric.
# So convert that to a one-hot

#%%
origin = dataset.pop('Origin')


#%%

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

#%%
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#%% [markdown]
# ## Inspect the data
# Have a quick look at the joint distribution of a few pairs of columns from the trainning data
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

#%% [markdown]
# Also look at the overall statistic
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

#%%
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


#%%
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


#%%
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#%% [markdown]
# ## The model
def buid_model():
    model = keras.Sequential(
        [
            layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ]
    )
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    return model


#%%
model = buid_model()

#%% [markdown]
# ## Inspect the model
# Use the ```summary()``` method to print a simple description of the model

#%%
model.summary()

#%%
# try out the model
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

#%%
EPOCHES = 1000
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHES, validation_split=0.2,
    verbose=0
)

#%%
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.head()

#%%
hist.tail()

#%%
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(
        hist['epoch'], hist['val_mean_absolute_error'],
        label='Val Error'
    )
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()

plot_history(history)

#%%
model = buid_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
    normed_train_data, train_labels,
    epochs=1000, validation_split=0.2, verbose=0,
    callbacks=[early_stop]
)
plot_history(history)

#%%
loss, mae, mse, acc = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

#%% [markdown]
# ## Make Prediction
test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

#%%
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')

#%%
