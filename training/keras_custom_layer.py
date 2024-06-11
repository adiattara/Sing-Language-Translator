import tensorflow as tf
from tqdm.notebook import tqdm; tqdm.pandas();
from PIL import Image, ImageEnhance; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
from matplotlib import animation, rc; rc('animation', html='jshtml')
import plotly.io as pio
pio.templates.default = "simple_white"
import math
import pandas as pd

class GatherLayer(tf.keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super(GatherLayer, self).__init__(**kwargs)
        self.indices = indices

    def call(self, inputs):
        return tf.gather(inputs, tf.cast(self.indices, dtype=tf.int32), axis=1)


# Find main hand and extract it
class ChooseHand(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        hand_inputs_final = []
        for s in range(inputs.shape[0]):
            left_hand_inputs = inputs[s, 1404:1467]
            right_hand_inputs = inputs[s, 1566:1629]
            hand_inputs = tf.cond(tf.equal(tf.reduce_sum(tf.abs(left_hand_inputs)), 0),
                                  lambda: right_hand_inputs,
                                  lambda: left_hand_inputs)
            hand_inputs_final.append(hand_inputs)
        stacked_hand_inputs = tf.stack(hand_inputs_final, axis=-1)
        transposed_hand_inputs = tf.transpose(stacked_hand_inputs, perm=[1, 0])
        return transposed_hand_inputs


# Calculate distance between non-consecutive joints in main hand
class HandKineticLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, tensor):
        end_ldmrk = [4 * 3, 8 * 3, 12 * 3, 16 * 3, 19 * 3]
        for index in range(0, 60, 3):
            start = (index + 3) if index in end_ldmrk else (index + 6)

            dist_list = []
            for x in range(start, 63, 3):
                dx = tensor[:, index] - tensor[:, x]
                dy = tensor[:, index + 1] - tensor[:, x + 1]
                dist = tf.sqrt(tf.square(dx) + tf.square(dy))
                dist_list.append(dist)

            dist_tensor = tf.stack(dist_list, axis=-1)
            tensor = tf.concat([tensor, dist_tensor], axis=-1)
        return tensor


# Calculate distance between non-consecutive joints in lips
class FaceKineticLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, tensor):
        for index in range(0, 36, 2):
            start = (index + 4)

            dist_list = []
            for x in range(start, 40, 2):
                dx = tensor[:, index] - tensor[:, x]
                dy = tensor[:, index + 1] - tensor[:, x + 1]
                dist = tf.sqrt(tf.square(dx) + tf.square(dy))
                dist_list.append(dist)

            dist_tensor = tf.stack(dist_list, axis=-1)
            tensor = tf.concat([tensor, dist_tensor], axis=-1)
        return tensor


# Calculate angle between non-consecutive joints in main hand
class AngularLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, tensor):
        for index in range(0, 60, 3):
            start = (index + 3)
            angle_list = []
            for j in range(start, 63, 3):
                vector = tensor[:, j:j + 2] - tensor[:, index:index + 2]
                norms = tf.norm(vector, axis=1)
                x_tilt = (tf.clip_by_value(vector[:, 0] / norms, -1.0, 1.0))
                y_tilt = (tf.clip_by_value(vector[:, 1] / norms, -1.0, 1.0))
                angle = tf.stack([x_tilt, y_tilt], axis=1)
                angle_list.append(angle)
            angle_tensor = tf.concat(angle_list, axis=-1)
            tensor = tf.concat([tensor, angle_tensor], axis=-1)
            tensor = tf.cast(tensor, tf.float32)
        return tensor