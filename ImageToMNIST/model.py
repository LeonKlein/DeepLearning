import numpy as np
import tensorflow as tf


dense_units = 150
filters1 = 64
filters2 = 88
filters3 = 128
filters4 = 64


def cnn_model_fn(features, labels, mode):
    """Model for mnist classification, using the estimator api.
    """

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=filters1,
                             kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=filters2,
                             kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=filters3, kernel_size=[
                             3, 3], padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=filters4, kernel_size=[
                             3, 3], padding="same", activation=tf.nn.relu)

    flatten = tf.reshape(conv4, [-1, 7 * 7 * filters4])
    dense = tf.layers.dense(
        inputs=flatten, units=dense_units, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        print("done")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(
        labels=labels, predictions=tf.nn.softmax(logits))
    predictions['loss'] = loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["probabilities"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
