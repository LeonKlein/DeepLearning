

import numpy as np
import tensorflow as tf

from model import cnn_model_fn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_steps=500, keep_checkpoint_max=200)
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir='checkpoints', config=checkpointing_config)


predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": mnist.test.images},
    num_epochs=1,
    shuffle=False)


out = mnist_classifier.predict(input_fn=predict_input_fn)
predictions = [gen["classes"] for gen in out]

accuracy = predictions - np.argmax(mnist.test.labels, axis=1)

pred = 1 - np.count_nonzero(accuracy) / len(mnist.test.labels)

print("Accuracy: ", pred)


### load self made picture


picture = np.load("picture.npy").reshape(28 * 28)
picture = picture[np.newaxis, :]


predict_input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": picture},
    num_epochs=1,
    shuffle=False)


out2 = mnist_classifier.predict(input_fn=predict_input_fn2)
prediction = [gen["probabilities"] for gen in out2]
print(prediction)
