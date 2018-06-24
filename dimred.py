import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.decomposition
from sklearn.cluster import KMeans
import os
import shutil
from itertools import permutations

out = '/tmp/dimred'
if os.path.exists(out):
    shutil.rmtree(out, ignore_errors=True)



with np.load('dimredux-challenge-01-data.npz') as fh:
    data = fh["data_x"]
    validation_x = fh["validation_x"]
    validation_y = fh["validation_y"]

test_data = np.copy(data)
print(data.shape)

tau = 8

data_plus_tau = data[tau:]
data = data[:-tau]

# make mean free
data_plus_tau -= np.mean(data_plus_tau, axis=0)
data -= np.mean(data, axis=0)
test_data -= np.mean(test_data, axis=0)
validation_x -= np.mean(validation_x, axis=0)

# data whitening
PCA = sklearn.decomposition.PCA(whiten=True)

data = PCA.fit_transform(data)
data_plus_tau = PCA.fit_transform(data_plus_tau)
validation_x = PCA.fit_transform(validation_x)
test_data = PCA.fit_transform(test_data)

print(np.mean(data, axis=0))
print(data[:,0].shape)
print(np.mean(test_data, axis=0))
print(test_data.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], zdir='z', s=1)
plt.show()

# encoder

dense_units1 = 200
dense_units2 = 100
# decoder

dense_units3 = 100
dense_units4 = 200


def autoenc_model_fn(features, labels, mode):


    dense1 = tf.layers.dense(
        inputs=features['x'], units=dense_units1, activation=tf.nn.leaky_relu)

    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(
        inputs=dropout1, units=dense_units2, activation=tf.nn.leaky_relu)

    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # one dimensional representaion
    onedim = tf.layers.dense(
        inputs=dropout2, units=1, activation=tf.nn.leaky_relu)
    
    # Decoder

    dense3 = tf.layers.dense(
        inputs=onedim, units=dense_units3, activation=tf.nn.leaky_relu)
    dropout3 = tf.layers.dropout(inputs=dense3, rate=0.5,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    
    dense4 = tf.layers.dense(
        inputs=dropout3, units=dense_units4, activation=tf.nn.leaky_relu)
    dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    


    output_layer = tf.layers.dense(
        inputs=dropout4, units=3, activation=tf.nn.leaky_relu)


    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=onedim)
    
    loss = tf.losses.mean_squared_error(
        labels=labels, predictions=output_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=output_layer,
            loss=loss,
            train_op=train_op)
    
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=output_layer)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops,
        predictions=output_layer)

def accuracy(pred, val):
    """
    Tries all possible permutations of the states and returns accuracy compared
    to the validation.
    """
    acc = []
    for perm in permutations([0,1,2,3]):
        val_perm = np.zeros(len(val))
        a,b,c,d = perm
        val_perm[val==0], val_perm[val==1], val_perm[val==2], val_perm[val==3] = a,b,c,d
        diff = pred - val_perm
        acc.append(1 - np.count_nonzero(diff) / len(diff))   
    return max(acc)
    
def cluster(inp):
    """Returns the labels of the clustered states"""
    return KMeans(n_clusters=4).fit(inp).labels_

config = tf.estimator.RunConfig(
    save_checkpoints_steps=500, keep_checkpoint_max=200)
autoenc = tf.estimator.Estimator(model_fn=autoenc_model_fn, model_dir=out, config=config)


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": data},
    y=data_plus_tau,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

train_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": data},
    y=data_plus_tau,
    batch_size=100,
    num_epochs=1,
    shuffle=True)


validate_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": validation_x},
    num_epochs=1,
    shuffle=False)


predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    num_epochs=1,
    shuffle=False)

epochs = 250
losses = []
val_loss = []

for _ in range(epochs):
    autoenc.train(input_fn=train_input_fn, steps=500)
    #train
    train_results = autoenc.evaluate(input_fn=train_eval_input_fn)
    #validate
    eval_result = []
    predictions_validation = autoenc.predict(input_fn=validate_input_fn)
    for point in predictions_validation:
        eval_result.append(point)
    eval_result_clustered = cluster(np.array(eval_result))
    losses.append(train_results['loss'])
    val_loss.append(accuracy(eval_result_clustered, validation_y))
    # tensorboard
    tf.summary.scalar('epoch', _)
    tf.summary.scalar('training_loss', losses)
    tf.summary.scalar('validation_loss', val_loss)


print("losses", losses)
print("accuracy", val_loss)

# print training and validation error
plt.plot(np.arange(len(losses)) + 1, losses, 'bo', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)

plt.figure()
plt.plot(np.arange(len(losses)) + 1, val_loss, 'r', label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend()
plt.figure()

plt.show()

# one dimensional porjection
projection = []
predictions = autoenc.predict(input_fn=predict_input_fn)
for point in predictions:
    projection.append(point)

# save the prediction

projection = np.array(projection)
labels = cluster(projection)

np.save('onedim_data', projection)

np.save('prediction', labels)



plt.plot(projection, 'b.', markersize=1)
plt.figure()
plt.plot(labels[:300])
plt.show() 


# Validation


projection_val = []
predictions_validation = autoenc.predict(input_fn=validate_input_fn)
for point in predictions_validation:
    projection_val.append(point)

projection_val_clustered = cluster(np.array(projection_val))




print("Accuracy", accuracy(projection_val_clustered, validation_y))



