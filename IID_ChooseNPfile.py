import os
import pandas as pd
import numpy as np
import argparse
import time
import datetime
from colorama import Fore, Back, Style, init
# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)

### Choose Load np array
def ChooseLoadNpArray(filepath,split_file, Choose_method):

    if split_file == 'total_train':
        print("Training with total_train")
        if (Choose_method == 'normal'):
            # 20240520 EdgeIIoT after do labelencode and minmax chi_square45 75 25分
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_train_AfterFeatureSelect44_20240520.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_AfterFeatureSelect44_20240520.npy", allow_pickle=True)    
        
        client_str = "BaseLine"
        print(Choose_method)
    elif split_file == 'client1_train':
        if (Choose_method == 'normal'):
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2017\ALLday\Dirichlet\20250205
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250205\\x_Dirichlet_client1_20250205.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250205\\y_Dirichlet_client1_20250205.npy", allow_pickle=True)

        print("train_half1 x_train 的形狀:", x_train.shape)
        print("train_half1 y_train 的形狀:", y_train.shape)
        client_str = "client1"
        print("使用 train_half1 進行訓練")
    elif split_file == 'client2_train':
        if (Choose_method == 'normal'):
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2017\ALLday\Dirichlet\20250205
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250205\\x_Dirichlet_client2_20250205.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250205\\y_Dirichlet_client2_20250205.npy", allow_pickle=True)

        print("train_half2 x_train 的形狀:", x_train.shape)
        print("train_half2 y_train 的形狀:", y_train.shape)
        client_str = "client2"
        print("使用 train_half2 進行訓練")

    elif split_file == 'client3_train':
        if (Choose_method == 'normal'):
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2017\ALLday\Dirichlet\20250205
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250205\\x_Dirichlet_client2_20250205.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250205\\y_Dirichlet_client2_20250205.npy", allow_pickle=True)
        print("train_half3 x_train 的形狀:", x_train.shape)
        print("train_half3 y_train 的形狀:", y_train.shape)
        client_str = "client3"
        print("使用 train_half3 進行訓練")

    print("use file", split_file)
    return x_train, y_train,client_str