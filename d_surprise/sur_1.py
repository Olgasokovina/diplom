from surprise import  KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline,SVD,SVDpp,NMF,BaselineOnly,CoClustering
from surprise import Dataset
from surprise.accuracy import rmse, mae, fcp
from surprise import Reader

import pandas as pd
import pickle
import optuna


# # # Производим десериализацию и извлекаем из файла формата pkl
with open('train_time.pkl', 'rb') as pkl_file:
    train = pickle.load(pkl_file)

with open('test_time.pkl', 'rb') as pkl_file:
    test = pickle.load(pkl_file)