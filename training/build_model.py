import tensorflow as tf
from tqdm.notebook import tqdm; tqdm.pandas();
from PIL import Image, ImageEnhance; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
from matplotlib import animation, rc; rc('animation', html='jshtml')
import plotly.io as pio
pio.templates.default = "simple_white"
from keras_custom_layer import *

def DenseLayer(inputs, nn, dropout):
    x = tf.keras.layers.Dense(nn)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x
def build_model1():
    batch_size = 64

    inputs = tf.keras.Input(batch_shape=(batch_size,) + (3258,), dtype=tf.float32)
    hand_inputs = ChooseHand()(inputs)
    hand_inputs = HandKineticLayer()(hand_inputs)
    hand_inputs = AngularLayer()(hand_inputs)
    hand_coord = hand_inputs[:, :42]
    hand_features = hand_inputs[:, 63:]
    hand_inputs = tf.keras.layers.Concatenate(axis=1)([hand_coord, hand_features])

    lower_lip_indices = [78 * 3, 191 * 3, 80 * 3, 81 * 3, 82 * 3, 13 * 3, 312 * 3, 311 * 3, 310 * 3, 415 * 3,
                         95 * 3, 88 * 3, 178 * 3, 87 * 3, 14 * 3, 317 * 3, 402 * 3, 318 * 3, 324 * 3, 308 * 3,
                         78 * 3 + 1, 191 * 3 + 1, 80 * 3 + 1, 81 * 3 + 1, 82 * 3 + 1, 13 * 3 + 1, 312 * 3 + 1,
                         311 * 3 + 1, 310 * 3 + 1, 415 * 3 + 1,
                         95 * 3 + 1, 88 * 3 + 1, 178 * 3 + 1, 87 * 3 + 1, 14 * 3 + 1, 317 * 3 + 1, 402 * 3 + 1,
                         318 * 3 + 1, 324 * 3 + 1, 308 * 3 + 1]

    upper_lip_indices = [61 * 3, 185 * 3, 40 * 3, 39 * 3, 37 * 3, 0, 267 * 3, 269 * 3, 270 * 3, 409 * 3,
                         291 * 3, 146 * 3, 91 * 3, 181 * 3, 84 * 3, 17 * 3, 314 * 3, 405 * 3, 321 * 3, 375 * 3,
                         61 * 3 + 1, 185 * 3 + 1, 40 * 3 + 1, 39 * 3 + 1, 37 * 3 + 1, 0 + 1, 267 * 3 + 1, 269 * 3 + 1,
                         270 * 3 + 1, 409 * 3 + 1,
                         291 * 3 + 1, 146 * 3 + 1, 91 * 3 + 1, 181 * 3 + 1, 84 * 3 + 1, 17 * 3 + 1, 314 * 3 + 1,
                         405 * 3 + 1, 321 * 3 + 1, 375 * 3 + 1]

    lower_lip_inputs = GatherLayer(lower_lip_indices)(inputs)
    lower_lip_inputs = FaceKineticLayer()(lower_lip_inputs)

    upper_lip_inputs = GatherLayer(upper_lip_indices)(inputs)
    upper_lip_inputs = FaceKineticLayer()(upper_lip_inputs)

    upper_body_indices = [16 * 3, 14 * 3, 12 * 3, 11 * 3, 13 * 3, 15 * 3,
                          16 * 3 + 1, 14 * 3 + 1, 12 * 3 + 1, 11 * 3 + 1, 13 * 3 + 1, 15 * 3 + 1]
    pose_inputs = GatherLayer(upper_body_indices)(inputs)

    all_inputs = tf.keras.layers.Concatenate(axis=1)([hand_inputs, lower_lip_inputs, upper_lip_inputs, pose_inputs])

    vector = DenseLayer(all_inputs, 1024 // 2, 0.2)
    vector = DenseLayer(vector, 1024 // 4, 0.6)

    vector = tf.keras.layers.Flatten()(vector)
    output = tf.keras.layers.Dense(250, activation="softmax")(vector)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(tf.keras.optimizers.Adam(0.000333), "sparse_categorical_crossentropy", metrics=["acc"])
    return model