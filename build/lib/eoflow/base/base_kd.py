from sklearn.utils import shuffle
import numpy as np
import pickle
import os
import tensorflow as tf


from .base_custom_training import BaseModelCustomTraining


class BaseModelKD(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)

    @staticmethod
    def _init_dataset_training(x_s, y_s, x_t, y_t, batch_size):
        x_s = x_s.astype('float32')
        x_t = x_t.astype('float32')
        y_s = y_s.astype('float32')
        y_t = y_t.astype('float32')
        return tf.data.Dataset.from_tensor_slices((x_s, y_s, x_t, y_t)).batch(batch_size)

    @tf.function
    def trainstep_teacher(self,
                          teacher_model,
                          train_ds):

        for Xs, ys, _, _ in train_ds:
            with tf.GradientTape() as gradients_task:
                ys_pred, _ = teacher_model.call(Xs, training=True)
                ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                cost_teacher = teacher_model.loss(ys, ys_pred)

            grads_task = gradients_task.gradient(cost_teacher, teacher_model.trainable_variables)
            cost_teacher += sum(teacher_model.losses)

            # Update weights
            teacher_model.optimizer.apply_gradients(zip(grads_task, teacher_model.trainable_variables))
            teacher_model.loss_metric.update_state(cost_teacher)

    @tf.function
    def trainstep_student(self,
                          teacher_model,
                          train_ds,
                          lambda_,
                          gamma_,
                          temperature,
                          consis_loss,
                          criterion):#tf.keras.losses.CategoricalCrossentropy(from_logits=True)):

        for Xs, ys, Xt, yt in train_ds:
            with tf.GradientTape() as gradients_task:
                ys_pred, fmap = teacher_model.call(Xs, training=False)
                fmap_ = tf.nn.softmax(fmap/temperature, axis=-1)
                #print(tf.reduce_max(fmap_))
                ys_pred = tf.reshape(ys_pred, tf.shape(ys))

                yt_pred, fmap_student = self.call(Xt, training=True)
                fmap_student_ = tf.nn.softmax(fmap_student/temperature, axis=-1)
                #print(tf.reduce_max(fmap_student_))

                yt_pred = tf.reshape(yt_pred, tf.shape(yt))

                cost_student = self.loss(yt, yt_pred) + \
                               lambda_ * criterion(fmap_, fmap_student_) + \
                               gamma_ * consis_loss((ys_pred + 1e-6), (yt_pred + 1e-6))

            grads_disc = gradients_task.gradient(cost_student, self.trainable_variables)

            cost_student += sum(self.losses)
            self.optimizer.apply_gradients(zip(grads_disc, self.trainable_variables))
            self.loss_metric.update_state(cost_student)
            self.metric.update_state(yt, yt_pred)

    @tf.function
    def valstep_kd(self, val_ds):
        for Xt, yt in val_ds:
            y_pred, _ = self.call(Xt, training=False)
            y_pred = tf.reshape(y_pred, tf.shape(yt))
            cost = self.loss(yt, y_pred)
            cost = tf.reduce_mean(cost)

            self.loss_metric.update_state(cost)
            self.metric.update_state(yt, y_pred)


    def fit_kd(self,
               src_dataset,
               trgt_dataset,
               val_dataset,
               test_dataset,
               teacher_model,
               batch_size,
               num_epochs,
               model_directory,
               save_steps=10,
               patience=30,
               lamda_ = 1,
               gamma_ = 1,
               temperature = 1.0,
               reduce_lr=False,
               pretrain_student_path = None,
               pretrain_teacher_path = None,
               consis_loss=tf.keras.losses.KLDivergence(),
               criterion=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
               function=np.min):

        #print('sALUT')
        train_loss, val_loss, val_acc, test_loss, test_acc, \
        disc_loss, task_loss = ([np.inf] if function == np.min else [-np.inf] for i in range(7))

        x_s, y_s = src_dataset
        x_t, y_t = trgt_dataset
        x_test, y_test = test_dataset
        x_v, y_v = val_dataset

        val_ds = tf.data.Dataset.from_tensor_slices((x_v.astype('float32'), y_v.astype('float32'))).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test.astype('float32'), y_test.astype('float32'))).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)
        wait = 0

        _ = self(tf.zeros(list(x_t.shape)))

        if pretrain_student_path is not None:
            self.load_weights(os.path.join(pretrain_student_path, 'model'))

        _ = teacher_model(tf.zeros(list(x_s.shape)))
        if pretrain_teacher_path is not None:
            teacher_model.load_weights(os.path.join(pretrain_teacher_path, 'model'))

        for epoch in range(num_epochs + 1):
            x_s, y_s, x_t, y_t = shuffle(x_s, y_s, x_t, y_t)
            if patience and epoch >= patience and self.config.finetuning:
                for i in range(len(self.layers[0].layers)):
                    self.layers[0].layers[i].trainable = True

            train_ds = self._init_dataset_training(x_s, y_s, x_t, y_t, batch_size)

            self.trainstep_teacher(teacher_model, train_ds)
            task_loss_epoch = teacher_model.loss_metric.result().numpy()
            train_loss.append(task_loss_epoch)
            teacher_model.loss_metric.reset_states()

            self.trainstep_student(teacher_model, train_ds, lamda_, gamma_, temperature, consis_loss=consis_loss, criterion=criterion)
            enc_loss_epoch = self.loss_metric.result().numpy()
            train_acc_result = self.metric.result().numpy()
            self.loss_metric.reset_states()
            self.metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.valstep_kd(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                self.valstep_kd(test_ds)
                test_loss_epoch = self.loss_metric.result().numpy()
                test_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: Task Acc {1}, Train loss {2}, Val loss {3}, Val acc {4}, Test loss {5}, Test acc {6}".format(
                        str(epoch),
                        str(train_acc_result), str(enc_loss_epoch),
                        str(round(val_loss_epoch, 4)), str(round(val_acc_result, 4)),
                        str(round(test_loss_epoch, 4)), str(round(test_acc_result, 4))
                    ))

                if (function is np.min and val_loss_epoch < function(val_loss)
                        or function is np.max and val_loss_epoch > function(val_loss)):
                    # wait = 0
                    print('Best score seen so far ' + str(val_loss_epoch))
                    self.save_weights(os.path.join(model_directory, 'best_model'))
                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)

        self.save_weights(os.path.join(model_directory, 'last_model'))

        # History of the training
        losses = dict(train_loss_results=train_loss[1:],
                      val_loss_results=val_acc[1:],
                      disc_loss_results=disc_loss[1:],
                      task_loss_results=task_loss[1:]
                      )

        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)

