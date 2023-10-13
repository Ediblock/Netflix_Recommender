import tensorflow as tf
import numpy as np

class CustomMSE(tf.keras.losses.Loss):
    """Class calculated mse error from the ground true values and predictions. It accepts also masking argument which
    when gives at masks output loss before summation on each row within batch. 1 - full loss value, 0 - no loss value"""
    def __init__(self, name = "CustomMSE"):
        super().__init__(name=name)

    def __call__(self, y_true, y_pred, mask_loss=None):
        if isinstance(mask_loss, np.ndarray):
            mask_loss = mask_loss.tolist()
        if isinstance(mask_loss, tf.Tensor):
            mask_loss = mask_loss.numpy().tolist()
        if mask_loss is None:
            squared_list = tf.math.squared_difference(y_true, y_pred)
            number_of_elements = squared_list.get_shape()[1]
            summed_list = tf.math.reduce_sum(squared_list, axis=1)
            return tf.math.reduce_sum(summed_list)/number_of_elements
        else:
            if not isinstance(mask_loss, tf.Tensor):
                mask_loss = tf.constant(mask_loss)
            squared_list = tf.math.squared_difference(y_true, y_pred)
            #Counting number of ones in mask_loss, this is the number of elements which we should divide by
            number_of_elements = mask_loss.numpy().sum()
            masked_list = tf.math.multiply(squared_list, mask_loss)
            summed_list = tf.math.reduce_sum(masked_list, axis=1)
            return tf.math.reduce_sum(summed_list)/number_of_elements

    def call(self, y_true, y_pred):
        return self.__call__(y_true, y_pred, mask_loss=None)