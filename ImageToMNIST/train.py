import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from model import cnn_model_fn


checkpointing_config = tf.estimator.RunConfig(save_checkpoints_steps=500, keep_checkpoint_max=1)
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='checkpoints', config=checkpointing_config)

epochs = 2


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": mnist.train.images},
    y=mnist.train.labels,
    batch_size=32,
    num_epochs=None,
    shuffle=True)

train_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": mnist.train.images},
    y=mnist.train.labels,
    batch_size=100,
    num_epochs=1,
    shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": mnist.test.images},
    y=mnist.test.labels,
    batch_size=100,
    num_epochs=1,
    shuffle=False
)


losses = []
accuracy = []

for _ in range(epochs):
    mnist_classifier.train(input_fn=train_input_fn, steps=500)
    train_results = mnist_classifier.evaluate(input_fn=train_eval_input_fn)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    print(train_results)
    losses.append(train_results['loss'])
    accuracy.append(eval_results['loss'])

print("losses", losses)
print("validation loss", accuracy)


plt.plot(np.arange(len(losses)) + 1, losses, 'bo', label='Training loss')
plt.plot(np.arange(len(losses)) + 1, accuracy, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend()
plt.figure()
plt.show()

