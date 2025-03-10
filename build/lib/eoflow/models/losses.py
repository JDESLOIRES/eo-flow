import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction

# import tensorflow_probability as tfp
# from keras import backend as K

import numpy as np


class LCC(Loss):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
    Furthermore, the constant log term is discarded.
    """

    def __init__(self, reduction=Reduction.AUTO, name="LCC"):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, ground_truth, predictions):
        def _convert_float(x):
            return tf.cast(x, tf.float32)

        predictions = tf.cast(predictions, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)

        pred_mean, pred_var = tf.nn.moments(predictions, (0,))
        gt_mean, gt_var = tf.nn.moments(ground_truth, (0,))

        mean_cent_prod = tf.reduce_mean(
            (predictions - pred_mean) * (ground_truth - gt_mean)
        )

        return 1.0 - (2.0 * _convert_float(mean_cent_prod)) / (
            _convert_float(pred_var)
            + _convert_float(gt_var)
            + tf.square(_convert_float(pred_mean) - _convert_float(gt_mean))
        )


def CCC(y_true, y_pred):
    """Lin's Concordance correlation coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient

    The concordance correlation coefficient is the correlation between two variables that fall on the 45 degree line through the origin.

    It is a product of
    - precision (Pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)

    Interpretation:
    - `rho_c =  1` : perfect agreement
    - `rho_c =  0` : no agreement
    - `rho_c = -1` : perfect disagreement

    Args:
    - y_true: ground truth
    - y_pred: predicted values

    Returns:
    - concordance correlation coefficient (float)
    """

    import keras.backend as K

    # covariance between y_true and y_pred
    N = K.int_shape(y_pred)[-1]
    s_xy = (
        1.0
        / (N - 1.0 + K.epsilon())
        * K.sum((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred)))
    )
    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)
    # variances
    s_x_sq = K.var(y_true)
    s_y_sq = K.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)

    return ccc


def CCC_numpy(y_true, y_pred):
    """Reference numpy implementation of Lin's Concordance correlation coefficient"""

    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0, 1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)

    return ccc


class NegLL(Loss):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
    Furthermore, the constant log term is discarded.
    """

    def __init__(self, reduction=Reduction.AUTO, name="NegLL"):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_obs, y_pred, sigma=0.5):
        """
        This function expects the log(var) to guarantee a positive variance with var = exp(log(var)).
        :param prediction: Predicted mean values
        :param log_variance: Predicted log(variance)
        :param target: Ground truth labels
        :return: gaussian negative log likelihood
        """
        # add a small constant to the variance for numeric stability
        dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
        return K.sum(-dist.log_prob(y_obs))


class GaussianNLL(Loss):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
    Furthermore, the constant log term is discarded.
    """

    def __init__(self, reduction=Reduction.AUTO, name="GaussianNLL"):
        super().__init__(reduction=reduction, name=name)
        self.eps = 1e-8

    def __call__(self, prediction, log_variance, target):
        """
        This function expects the log(var) to guarantee a positive variance with var = exp(log(var)).
        :param prediction: Predicted mean values
        :param log_variance: Predicted log(variance)
        :param target: Ground truth labels
        :return: gaussian negative log likelihood
        """
        # add a small constant to the variance for numeric stability
        variance = tf.math.exp(log_variance)  # + self.eps
        variance = tf.clip_by_value(
            t=variance,
            clip_value_min=tf.constant(1e-6),
            clip_value_max=tf.constant(10.0),
        )
        return 0.5 / variance * (prediction - target) ** 2 + 0.5 * tf.math.log(variance)


class LaplacianNLL(Loss):
    """
    Laplacian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
    Furthermore, the constant log term is discarded.
    """

    def __init__(self, reduction=Reduction.AUTO, name="LaplacianNLL"):
        super().__init__(reduction=reduction, name=name)
        # self.eps = 1e-8

    def __call__(self, prediction, log_variance, target):
        """
        This function expects the log(var) to guarantee a positive variance with var = exp(log(var)).
        :param prediction: Predicted mean values
        :param log_variance: Predicted log(variance)
        :param target: Ground truth labels
        :return: gaussian negative log likelihood
        """
        # add a small constant to the variance for numeric stability
        variance = tf.math.exp(log_variance)
        variance = tf.clip_by_value(
            t=variance,
            clip_value_min=tf.constant(1e-6),
            clip_value_max=tf.constant(10.0),
        )
        return 1 / variance * tf.math.abs(prediction - target) + log_variance


class RMAPE(Loss):
    """
    Laplacian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
    Furthermore, the constant log term is discarded.
    """

    def __init__(self, reduction=Reduction.AUTO, name="RMAPE"):
        super().__init__(reduction=reduction, name=name)
        self.eps = 1e-4

    def call(self, y_true, y_pred):
        """ """
        return tf.math.sqrt(
            tf.reduce_mean(
                tf.math.square((y_true - y_pred + self.eps) / (y_true + self.eps))
            )
        )


class RMSE(Loss):
    """
    Laplacian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
    Furthermore, the constant log term is discarded.
    """

    def __init__(self, reduction=Reduction.AUTO, name="RMSE"):
        super().__init__(reduction=reduction, name=name)
        self.eps = 1e-4

    def call(self, y_true, y_pred):
        """ """
        y_pred = tf.cast(y_pred, "float32")
        y_true = tf.cast(y_true, "float32")
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

        return tf.cast(rmse, "float32")


def cropped_loss(loss_fn):
    """Wraps loss function. Crops the labels to match the logits size."""

    def _loss_fn(labels, logits):
        logits_shape = tf.shape(logits)
        labels_crop = tf.image.resize_with_crop_or_pad(
            labels, logits_shape[1], logits_shape[2]
        )

        return loss_fn(labels_crop, logits)

    return _loss_fn


class PearsonR(Loss):
    def __init__(self, reduction=Reduction.AUTO, name="Person"):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
        my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        xm, ym = y_true - mx, y_pred - my
        t1_norm = tf.nn.l2_normalize(xm, axis=1)
        t2_norm = tf.nn.l2_normalize(ym, axis=1)
        cosine = tf.keras.losses.CosineSimilarity(axis=1)
        return cosine(t1_norm, t2_norm)


class CosineSim(Loss):
    def __init__(self, reduction=Reduction.AUTO, name="Cosine"):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        cosine = tf.keras.losses.CosineSimilarity(axis=1)
        return cosine(y_true, y_pred)


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.nn.l2_normalize(xm, axis=1)
    t2_norm = tf.nn.l2_normalize(ym, axis=1)
    return tf.losses.cosine_distance(t1_norm, t2_norm, axis=1)


class CategoricalCrossEntropy(Loss):
    """Wrapper class for cross-entropy with class weights"""

    def __init__(
        self,
        from_logits=True,
        class_weights=None,
        reduction=Reduction.AUTO,
        name="FocalLoss",
    ):
        """Categorical cross-entropy.

        :param from_logits: Whether predictions are logits or softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be applied to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'FocalLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.from_logits = from_logits
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        # Calculate Cross Entropy
        loss = -y_true * tf.math.log(y_pred)

        # Multiply cross-entropy with class-wise weights
        if self.class_weights is not None:
            loss = tf.multiply(loss, self.class_weights)

        # Sum over classes
        loss = tf.reduce_sum(loss, axis=-1)

        return loss


class CategoricalFocalLoss(Loss):
    """Categorical version of focal loss.

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        Keras implementation: https://github.com/umbertogriffo/focal-loss-keras
    """

    def __init__(
        self,
        gamma=2.0,
        alpha=0.25,
        from_logits=True,
        class_weights=None,
        reduction=Reduction.AUTO,
        name="FocalLoss",
    ):
        """Categorical version of focal loss.

        :param gamma: gamma value, defaults to 2.
        :type gamma: float
        :param alpha: alpha value, defaults to .25
        :type alpha: float
        :param from_logits: Whether predictions are logits or softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be applied to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'FocalLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.class_weights = class_weights

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = self.alpha * tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

        # Multiply focal loss with class-wise weights
        if self.class_weights is not None:
            loss = tf.multiply(cross_entropy, self.class_weights)

        # Sum over classes
        loss = tf.reduce_sum(loss, axis=-1)

        return loss


class JaccardDistanceLoss(Loss):
    """Implementation of the Jaccard distance, or Intersection over Union IoU loss.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    Implementation taken from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """

    def __init__(
        self,
        smooth=1,
        from_logits=True,
        class_weights=None,
        reduction=Reduction.AUTO,
        name="JaccardLoss",
    ):
        """Jaccard distance loss.

        :param smooth: Smoothing factor. Default is 1.
        :type smooth: int
        :param from_logits: Whether predictions are logits or softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be applied to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'JaccardLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.smooth = smooth
        self.from_logits = from_logits

        self.class_weights = class_weights

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))

        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))

        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)

        loss = (1 - jac) * self.smooth

        if self.class_weights is not None:
            loss = tf.multiply(loss, self.class_weights)

        loss = tf.reduce_sum(loss, axis=-1)

        return loss


class TanimotoDistanceLoss(Loss):
    """Implementation of the Tanimoto distance, which is modified version of the Jaccard distance.

    Tanimoto = (|X & Y|)/ (|X|^2+ |Y|^2 - |X & Y|)
            = sum(|A*B|)/(sum(|A|^2)+sum(|B|^2)-sum(|A*B|))

    Implementation taken from
    https://github.com/feevos/resuneta/blob/145be5519ee4bec9a8cce9e887808b8df011f520/nn/loss/loss.py#L7
    """

    def __init__(
        self,
        smooth=1.0e-5,
        from_logits=True,
        class_weights=None,
        reduction=Reduction.AUTO,
        normalise=False,
        name="TanimotoLoss",
    ):
        """Tanimoto distance loss.

        :param smooth: Smoothing factor. Default is 1.0e-5.
        :type smooth: float
        :param from_logits: Whether predictions are logits or softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be applied to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: Reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param normalise: Whether to normalise loss by number of positive samples in class, defaults to `False`
        :type normalise: bool
        :param name: Name of the loss, defaults to 'TanimotoLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.smooth = smooth
        self.from_logits = from_logits
        self.normalise = normalise

        self.class_weights = class_weights

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        n_classes = y_true.shape[-1]

        volume = (
            tf.reduce_mean(tf.reduce_sum(y_true, axis=(1, 2)), axis=0)
            if self.normalise
            else tf.ones(n_classes, dtype=tf.float32)
        )

        weights = tf.math.reciprocal(tf.math.square(volume))
        new_weights = tf.where(tf.math.is_inf(weights), tf.zeros_like(weights), weights)
        weights = tf.where(
            tf.math.is_inf(weights),
            tf.ones_like(weights) * tf.reduce_max(new_weights),
            weights,
        )

        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))

        sum_ = tf.reduce_sum(y_true * y_true + y_pred * y_pred, axis=(1, 2))

        num_ = tf.multiply(intersection, weights) + self.smooth

        den_ = tf.multiply(sum_ - intersection, weights) + self.smooth

        tanimoto = num_ / den_

        loss = 1 - tanimoto

        if self.class_weights is not None:
            loss = tf.multiply(loss, self.class_weights)

        loss = tf.reduce_sum(loss, axis=-1)

        return loss
