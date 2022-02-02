import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pickle
import os
import time
from . import Configurable


# class BaseCustomModel(tf.keras.Model, Configurable):

@tf.function
def train_step(model,
               optimizer,
               loss,
               metric,
               train_ds):
    # pb_i = Progbar(len(list(train_ds)), stateful_metrics='acc')

    for x_batch_train, y_batch_train in train_ds:  # tqdm
        with tf.GradientTape() as tape:
            y_preds = model(x_batch_train,
                            training=True)
            cost = loss(y_batch_train, y_preds)

        grads = tape.gradient(cost, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        metric.update_state(y_batch_train, y_preds)
        # pb_i.add(1, values=[metric.result().numpy()])


# Function to run the validation step.
@tf.function
def val_step(model, val_ds, val_acc_metric):
    for x, y in val_ds:
        y_preds = model.call(x, training=False)
        val_acc_metric.update_state(y, y_preds)


def training_loop(model,
                  x_train, y_train,
                  x_val, y_val,
                  batch_size,
                  num_epochs, optimizer,
                  path_directory,
                  saving_step=10,
                  loss=tf.keras.losses.Huber(),
                  metric=tf.keras.metric.MeanSquaredError(),
                  function=np.min):

    train_acc_results, val_acc_results = ([np.inf] for i in range(2))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(8)

    for epoch in range(num_epochs + 1):

        x_train, y_train = shuffle(x_train, y_train)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

        train_step(model,
                   optimizer,
                   loss,
                   metric,
                   train_ds)

        # End epoch
        metric_result = metric.result().numpy()
        train_acc_results.append(metric_result)
        metric.reset_states()

        if epoch % saving_step == 0:
            val_step(model, val_ds, metric)
            val_metric_result = metric.result().numpy()

            print(
                "Epoch {0}: Train metric {1}, Val metric {2}".format(
                    str(epoch),
                    str(metric_result),
                    str(round(val_metric_result, 4)),
                ))

            if val_metric_result < function(val_acc_results):
                model.save_weights(os.path.join(path_directory, 'model'))

            val_acc_results.append(val_metric_result)
            metric.reset_states()

    # History of the training
    losses = dict(train_loss_results=metric,
                  val_acc_results=val_metric_result
                  )
    with open(os.path.join(path_directory, 'history.pickle'), 'wb') as d:
        pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)
