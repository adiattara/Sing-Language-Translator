import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import datetime
import numpy as np
import os
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.express as px
from PIL import Image, ImageEnhance; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
from matplotlib import animation, rc; rc('animation', html='jshtml')
import json
from sklearn.model_selection import train_test_split
import plotly.io as pio
pio.templates.default = "simple_white"
import math
import pandas as pd
from sklearn.preprocessing import normalize
from keras_custom_layer import *
from build_model import *


class PrepInputs(tf.keras.layers.Layer):
    def __init__(self, face_idx_range=(0, 468), lh_idx_range=(468, 489),
                 pose_idx_range=(489, 522), rh_idx_range=(522, 543)):
        super(PrepInputs, self).__init__()
        self.idx_ranges = [face_idx_range, lh_idx_range, pose_idx_range, rh_idx_range]
        self.flat_feat_lens = [3 * (_range[1] - _range[0]) for _range in self.idx_ranges]

    def call(self, x_in):
        x_in = tf.reshape(x_in, (-1, x_in.shape[1], x_in.shape[2] * 3))  # (batch_size, n_frames, n_features*3)
        xs = [x_in[:, :, _range[0] * 3:_range[1] * 3] for _range in self.idx_ranges]
        xs = [tf.reshape(_x, (-1, flat_feat_len)) for _x, flat_feat_len in zip(xs, self.flat_feat_lens)]

        xs[1:] = [
            tf.boolean_mask(_x, tf.reduce_all(tf.logical_not(tf.math.is_nan(_x)), axis=1), axis=0)
            for _x in xs[1:]
        ]

        x_means = [tf.math.reduce_mean(_x, axis=0) for _x in xs]
        x_stds = [tf.math.reduce_std(_x, axis=0) for _x in xs]

        x_out = tf.concat([*x_means, *x_stds], axis=0)
        x_out = tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))
        return tf.expand_dims(x_out, axis=0)

def seed_it_all(seed=42):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_json_file(json_path):
    with open(json_path, 'r') as f:
        sign_map = json.load(f)
    return sign_map


class CFG:
    data_dir = "./"
    sequence_length = 12
    rows_per_frame = 543


ROWS_PER_FRAME = 543


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


if __name__ =='__main__':
    sign_map = load_json_file(CFG.data_dir + 'sign_to_prediction_index_map.json')
    train_data = pd.read_csv(CFG.data_dir + 'train.csv')

    s2p_map = {k.lower(): v for k, v in load_json_file(CFG.data_dir + "sign_to_prediction_index_map.json").items()}
    p2s_map = {v: k for k, v in load_json_file(CFG.data_dir + "sign_to_prediction_index_map.json").items()}
    encoder = lambda x: s2p_map.get(x.lower())
    decoder = lambda x: p2s_map.get(x)


    train_x = np.load("../preprocessed_data/feature_data.npy").astype(np.float32)
    train_y = np.load("../preprocessed_data/feature_labels.npy").astype(np.uint8)
    N_TOTAL = train_x.shape[0]
    VAL_PCT = 0.1
    N_VAL   = int(N_TOTAL*VAL_PCT)
    N_TRAIN = N_TOTAL-N_VAL

    random_idxs = random.sample(range(N_TOTAL), N_TOTAL)
    train_idxs, val_idxs = np.array(random_idxs[:N_TRAIN]), np.array(random_idxs[N_TRAIN:])

    val_x, val_y = train_x[val_idxs], train_y[val_idxs]
    train_x, train_y = train_x[train_idxs], train_y[train_idxs]

    model1 = build_model1()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'file.keras',  # Sauvegarder le meilleur modèle sous le nom "file.keras"
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.8, verbose=1)
    ]

    # Charger les données d'entraînement et de validation
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.batch(batch_size=64, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_dataset = val_dataset.batch(batch_size=64, drop_remainder=True)

    # Entraîner le modèle en utilisant les callbacks
    model1.fit(train_dataset, epochs=100, validation_data=val_dataset, batch_size=64, callbacks=callbacks)

    saved_model_dir='../model_asl'
    tf.saved_model.save(model1, saved_model_dir)