import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..utils.tf_utils import plot_to_image


class CustomReduceLRoP:
    """ Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This callback monitors a
        quantity and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.
        Example:
        ```python
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                        patience=5, min_lr=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```
    Arguments:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced. new_lr = lr *
            factor
        patience: number of epochs with no improvement after which learning rate
            will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
            quantity monitored has stopped decreasing; in `max` mode it will be
            reduced when the quantity monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred from the name of the
            monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
            significant changes.
        cooldown: number of epochs to wait before resuming normal operation after
            lr has been reduced.
        min_lr: lower bound on the learning rate.
        reduce_exp: reducing the learning rate exponentially
    """

    def __init__(self,
                 ## Custom modification:  Deprecated due to focusing on validation loss
                 # monitor='val_loss',
                 factor=0.1,
                 patience=30,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=10e-6,
                 sign_number=4,
                 ## Custom modification: Passing optimizer as arguement
                 optim_lr=None,
                 ## Custom modification:  Exponentially reducing learning
                 reduce_lin=False,
                 **kwargs):

        ## Custom modification: Optimizer Error Handling
        if tf.is_tensor(optim_lr) == False:
            raise ValueError('Need optimizer !')
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        ## Custom modification: Passing optimizer as arguement
        self.optim_lr = optim_lr

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.sign_number = sign_number

        ## Custom modification: Exponentially reducing learning
        self.reduce_lin = reduce_lin
        self.reduce_lr = True

        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            print('Learning Rate Plateau Reducing mode %s is unknown, '
                  'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if self.mode in ['min', 'auto']:
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, loss, logs=None):

        logs = logs or {}

        logs['lr'] = float(self.optim_lr.numpy())

        current = float(loss)

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:

                ## Custom modification: Optimizer Learning Rate
                # old_lr = float(K.get_value(self.model.optimizer.lr))
                old_lr = float(self.optim_lr.numpy())
                if old_lr > self.min_lr and self.reduce_lr == True:
                    ## Custom modification: Linear learning Rate
                    if self.reduce_lin == True:
                        new_lr = old_lr - self.factor
                        ## Custom modification: Error Handling when learning rate is below zero
                        if new_lr <= 0:
                            print('Learning Rate is below zero: {}, '
                                  'fallback to minimal learning rate: {}. '
                                  'Stop reducing learning rate during training.'.format(new_lr, self.min_lr))
                            self.reduce_lr = False
                    else:
                        new_lr = old_lr * self.factor

                    new_lr = max(new_lr, self.min_lr)

                    ## Custom modification: Optimizer Learning Rate
                    # K.set_value(self.model.optimizer.lr, new_lr)
                    self.optim_lr.assign(new_lr)

                    if self.verbose > 0:
                        print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                              'rate to %s.' % (epoch + 1, float(new_lr)))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0



class VisualizationCallback(tf.keras.callbacks.Callback):
    """ Keras Callback for saving prediction visualizations to TensorBoard. """

    def __init__(self, val_images, log_dir, time_index=0, rgb_indices=[2, 1, 0]):
        """
        :param val_images: Images to run predictions on. Tuple of (images, labels).
        :type val_images: (np.array, np.array)
        :param log_dir: Directory where the TensorBoard logs are written.
        :type log_dir: str
        :param time_index: Time index to use, when multiple time slices are available, defaults to 0
        :type time_index: int, optional
        :param rgb_indices: Indices for R, G and B bands in the input image, defaults to [0,1,2]
        :type rgb_indices: list, optional
        """
        super().__init__()

        self.val_images = val_images
        self.time_index = time_index
        self.rgb_indices = rgb_indices

        self.file_writer = tf.summary.create_file_writer(log_dir)

    @staticmethod
    def plot_predictions(input_image, labels, predictions, n_classes):
        # TODO: fix figsize (too wide?)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        scaled_image = np.clip(input_image*2.5, 0., 1.)
        ax1.imshow(scaled_image)
        ax1.title.set_text('Input image')

        cnorm = mpl.colors.NoNorm()
        cmap = plt.cm.get_cmap('Set3', n_classes)

        ax2.imshow(labels, cmap=cmap, norm=cnorm)
        ax2.title.set_text('Labeled classes')

        img = ax3.imshow(predictions, cmap=cmap, norm=cnorm)
        ax3.title.set_text('Predicted classes')

        plt.colorbar(img, ax=[ax1, ax2, ax3], shrink=0.8, ticks=list(range(n_classes)))

        return fig

    def prediction_summaries(self, step):
        images, labels = self.val_images
        preds_raw = self.model.predict(images)

        pred_shape = tf.shape(preds_raw)

        # If temporal data only use time_index slice
        if images.ndim == 5:
            images = images[:, self.time_index, :, :, :]

        # Crop images and labels to output size
        labels = tf.image.resize_with_crop_or_pad(labels, pred_shape[1], pred_shape[2])
        images = tf.image.resize_with_crop_or_pad(images, pred_shape[1], pred_shape[2])

        # Take RGB values
        images = images.numpy()[..., self.rgb_indices]

        num_classes = labels.shape[-1]

        # Get class ids
        preds_raw = np.argmax(preds_raw, axis=-1)
        labels = np.argmax(labels, axis=-1)

        vis_images = []
        for image_i, labels_i, pred_i in zip(images, labels, preds_raw):
            # Plot predictions and convert to image
            fig = self.plot_predictions(image_i, labels_i, pred_i, num_classes)
            img = plot_to_image(fig)

            vis_images.append(img)

        n_images = len(vis_images)
        vis_images = tf.concat(vis_images, axis=0)

        with self.file_writer.as_default():
            tf.summary.image('predictions', vis_images, step=step, max_outputs=n_images)

    def on_epoch_end(self, epoch, logs=None):
        self.prediction_summaries(epoch)