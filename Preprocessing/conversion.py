import multiprocessing as mp
import numpy as np
import os
from tqdm.notebook import tqdm; tqdm.pandas();
from PIL import Image, ImageEnhance; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
from matplotlib import animation, rc; rc('animation', html='jshtml')
import torch
import json
import plotly.io as pio
import pandas as pd
from FeatureGen import *

ROWS_PER_FRAME = 543
LANDMARK_FILES_DIR = "../train_landmark_files"
TRAIN_FILE = "./train.csv"
label_map = json.load(open("./sign_to_prediction_index_map.json", "r"))
pio.templates.default = "simple_white"

feature_converter = FeatureGen()
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def convert_row(row):
    x = load_relevant_data_subset(os.path.join("../", row[1].path))
    x = feature_converter(torch.tensor(x)).cpu().numpy()
    return x, row[1].label


def convert_and_save_data():
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)
    npdata = np.zeros((df.shape[0], 3258))
    nplabels = np.zeros(df.shape[0])
    with mp.Pool() as pool:
        results = pool.imap(convert_row, df.iterrows(), chunksize=250)
        for i, (x, y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata[i, :] = x
            nplabels[i] = y

    np.save("feature_data.npy", npdata)
    np.save("feature_labels.npy", nplabels)