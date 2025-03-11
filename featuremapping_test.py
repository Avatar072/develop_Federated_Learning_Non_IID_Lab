import warnings
import os
import seaborn as sns
import time
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import copy
from collections import OrderedDict
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.metrics import confusion_matrix
from mytoolfunction import generatefolder,ParseCommandLineArgs,SaveDataframeTonpArray
from mytoolfunction import ChooseUseModel, getStartorEndtime,EvaluateVariation
from IID_ChooseNPfile import CICIDS2017_IID_ChooseLoadNpArray, CICIDS2018_IID_ChooseLoadNpArray, ChooseLoad_class_names
from collections import Counter
from Add_ALL_LayerToCount import DoCountModelWeightSum,evaluateWeightDifferences
from Add_ALL_LayerToCount import Calculate_Weight_Diffs_Distance_OR_Absolute
from colorama import Fore, Back, Style, init
import configparser
from sklearn.preprocessing import MinMaxScaler

filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
current_time = time.strftime("%Hh%Mm%Ss", time.localtime())
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
generatefolder(filepath + "\\", "dataset_AfterProcessed")
generatefolder(filepath + "\\dataset_AfterProcessed\\", "CICIDS2017")
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\featureMapping\\", today)
generatefolder(filepath + f"\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\featureMapping\\{today}\\", current_time)

# # 初始化 colorama（Windows 系統中必須）
# # 初始化 ConfigParser
# config = configparser.ConfigParser()
# # 讀取 ini 文件
# config.read('./config.ini', encoding='utf-8')
# # 獲取 Datasets 節點下的值
# choose_dataset = config.get('Datasets', 'choose_dataset')
# # 獲取 Setting_Adversarial_Attack 節點下的值
# set_attack = config.getboolean ('Setting_Adversarial_Attack', 'set_attack')
# # 獲取 Round 節點下的值
# # 使用 getint 來取得整數類型的值
# start_attack_round = config.getint('Round', 'start_attack_round')
# end_attack_round = config.getint('Round', 'end_attack_round')  
# save_model_round = config.getint('Round', 'save_model_round')
# # 顯示讀取的配置
# print(Fore.YELLOW+Style.BRIGHT+f"choose_dataset: {choose_dataset}")
# print(Fore.YELLOW+Style.BRIGHT+f"set_attack: {set_attack}")
# print(Fore.YELLOW+Style.BRIGHT+f"start_attack_round: {start_attack_round}")
# print(Fore.YELLOW+Style.BRIGHT+f"end_attack_round: {end_attack_round}")
# print(Fore.YELLOW+Style.BRIGHT+f"save_model_round: {save_model_round}")

# # 獲取 Count 節點下的值
# labelCount = config.getint('Count', 'labelCount')
# print(f"Count: {labelCount}")

# start_IDS = time.time()
# # #############################################################################
# # 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# # #############################################################################

# warnings.filterwarnings("ignore", category=UserWarning)
# #  Clear GPU Cache
# torch.cuda.empty_cache()
# # DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # 返回gpu数量；
# torch.cuda.device_count()
# # 返回gpu名字，设备索引默认从0开始；
# torch.cuda.get_device_name(0)
# # 返回当前设备索引；
# torch.cuda.current_device()
# print(f"DEVICE: {DEVICE}")
# #python client_IID.py --dataset_split client1_train --epochs 50 --method normal
# #python client_IID.py --dataset_split client2_train --epochs 50 --method normal
# #python client_IID.py --dataset_split client3_train --epochs 50 --method normal
# file, num_epochs,Choose_method = ParseCommandLineArgs(["dataset_split", "epochs", "method"])
# print(f"Dataset: {file}")
# print(f"Number of epochs: {num_epochs}")
# print(f"Choose_method: {Choose_method}")

# # 初始化變數為None或空列表
x_train =  np.array([]) # 預設初始化為一個空陣列
y_train =  np.array([])  # 預設初始化為一個空陣列

# x_test = np.array([])  # 預設初始化為一個空陣列
# y_test = np.array([])  # 預設初始化為一個空陣列

# client_str = ""
# # 預設初始化 class_names
# class_names_global, class_names_local, labels_to_calculate = None, None, None

# try:
#     # CICIDS2017
#     if choose_dataset == "CICIDS2017":
#         print(Fore.YELLOW+Style.BRIGHT+f"use dataset: {choose_dataset}")
#         x_train, y_train, x_test, y_test, client_str = CICIDS2017_IID_ChooseLoadNpArray(filepath, file, Choose_method)
#         class_names_local, labels_to_calculate = ChooseLoad_class_names("CICIDS2017")
        
#     # CICIDS2018
#     if choose_dataset == "CICIDS2018":
#         print(Fore.YELLOW+Style.BRIGHT+f"use dataset: {choose_dataset}")
#         x_train, y_train, x_test, y_test, client_str = CICIDS2018_IID_ChooseLoadNpArray(filepath, file, Choose_method)
#         class_names_local, labels_to_calculate = ChooseLoad_class_names("CICIDS2018")
# except Exception as e:
#     print(f"An error occurred: {e}")
# finally:
#     # 確保資料加載成功
#     if y_train is None or len(y_train) == 0:
#             raise ValueError("Failed to load y_train for "+f"{choose_dataset}")
#     else:
#             print("Execution finished.")

# counter = Counter(y_train)
# print(counter)

# input_dim = 36  # 輸入特徵的維度
output_dim = 123  # 輸出的維度

### 特徵映射 ###
# 定義一個簡單的非線性映射神經網絡
def FeatureMappingNetwork_instance(x_train):
    class FeatureMappingNetwork(nn.Module):
        def __init__(self):
            super(FeatureMappingNetwork, self).__init__()

            # 定義網絡層
            self.layers = nn.Sequential(
                nn.Linear(x_train.shape[1], 100),  # 第一層將輸入映射
                nn.ReLU(),  # 非線性激活函數
                nn.Linear(100, output_dim)  # 第二層將特徵映射到目標維度
            )

        def forward(self, x):
            return self.layers(x)  # 傳遞數據通過網絡層        
    
    return FeatureMappingNetwork()

def CheckLoadtrainFiledtype(x_train):
    if isinstance(x_train, np.ndarray):
        print("x_train is a NumPy array.")
    elif isinstance(x_train, torch.Tensor):
        print("x_train is a PyTorch tensor.")
    else:
        print("x_train is of unknown type.")

# 直接以npy做FeatureMapping 訓練效果差
def DoFeatureMapping(x_train):
    # 檢查 x_train 的類型，一開始載入時 x_train is a NumPy array.
    CheckLoadtrainFiledtype(x_train)
    # 創建feature mapping模型實例
    model = FeatureMappingNetwork_instance(x_train)
    # 經過網絡層映射後的特徵 轉換為 PyTorch 張量，並確保數據類型是 float32
    # 將 x_train 轉換為 PyTorch Tensor
    x_train = model(torch.tensor(x_train, dtype=torch.float32))
    print(x_train.shape[1])  # 這裡的 x_train 就是經過網絡層映射後的特徵
    # 檢查 x_train 的類型，經過網絡層映射後的特徵，這邊x_train is a PyTorch tensor.
    CheckLoadtrainFiledtype(x_train)
    #將PyTorch tensor轉為回去NumPy array
    # 使用 .detach() 移除梯度追蹤，再將其轉換為 NumPy 陣列
    # .detach()：這會返回一個新的張量，這個張量不會追蹤梯度。
    # .cpu()：如果張量在 GPU 上，先把它移動到 CPU。
    # .numpy()：將張量轉換為 NumPy 陣列。
    x_train_numpy = x_train.detach().cpu().numpy()

    return x_train_numpy


def DoFeatureMappingAndSavetoCSVfile(x_train,y_train,str_train_OR_test):

    # 創建模型實例
    model = FeatureMappingNetwork_instance(x_train)

    print(model)

    # 將 x_train 轉換為張量並進行映射
    x_train_mapped = model(torch.tensor(x_train, dtype=torch.float32))

    print(x_train_mapped.shape[1])  # 這裡的 x_train_mapped 就是經過網絡層映射後的特徵
    
    # x_train 是 NumPy 陣列，將其轉換為 PyTorch Tensor
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

    # 將 x_train 輸入到模型中，進行特徵映射
    mapped_features = model(x_train_tensor)


    # 將映射後的特徵轉換為 Pandas DataFrame
    mapped_df = pd.DataFrame(mapped_features.detach().numpy())

    # 將 y_train 轉換為 Pandas DataFrame
    y_train_df = pd.DataFrame(y_train, columns=['Label'])

    # 合併 x_train 映射後的特徵與 y_train
    # final_df = pd.concat([mapped_df, y_train_df], axis=1)
    
    # 全特徵都做minmax
    X = mapped_df
    X=X.values
    # scaler = preprocessing.StandardScaler() #資料標準化
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    scaler.fit(X)
    X=scaler.transform(X)
    # 將縮放後的值更新到 doScalerdataset 中
    mapped_df.iloc[:, :] = X
    # 將排除的列名和選中的特徵和 Label 合併為新的 DataFrame
    df = pd.concat([mapped_df,y_train_df], axis = 1)
    
    # 儲存為 CSV 檔案
    df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/featureMapping/{today}/{current_time}/{str_train_OR_test}_mapped_features_with_labels.csv", index=False)
    if str_train_OR_test == "train" :
        # 儲存為 npy 檔案
        SaveDataframeTonpArray(df, 
                                f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/featureMapping/{today}/{current_time}/", 
                                f"ALLDay","featrue_mapping_train")
    elif str_train_OR_test == "test" :
        # 儲存為 npy 檔案
        SaveDataframeTonpArray(df, 
                                f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/featureMapping/{today}/{current_time}/", 
                                f"ALLDay","featrue_mapping_test")


    print("CSV file has been saved.")