#%%
import tensorflow as tf 

# setup a linear classifier
classifier = tf.estimator.LinearClassifier()

# train the model on some example data
classifier.train(input_fn=train_input_fn,step=2000)

#use it to predict
predictions = classifier.predict(input_fn = predict_input_fn)