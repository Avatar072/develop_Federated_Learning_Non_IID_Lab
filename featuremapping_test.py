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
from mytoolfunction import generatefolder,ParseCommandLineArgs,SaveDataframeTonpArray,SaveDataToCsvfile
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

# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)
# # 初始化變數為None或空列表
x_train =  np.array([]) # 預設初始化為一個空陣列
y_train =  np.array([])  # 預設初始化為一個空陣列


# input_dim = 36  # 輸入特徵的維度
# output_dim = 123  # 輸出的維度
# 123 = CICIDS2017 & CICIDS2018 79feature + TONIOT 44 feature

### 特徵映射 ###
# 定義一個簡單的非線性映射神經網絡
def FeatureMappingNetwork_instance(x_train,output_dim):
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
def DoFeatureMapping_when_LoadingNpyfile(x_train):
    # 檢查 x_train 的類型，一開始載入時 x_train is a NumPy array.
    CheckLoadtrainFiledtype(x_train)
    # 創建feature mapping模型實例
    output_dim = 123  # 輸出的維度即要mapping的特徵數量
    model = FeatureMappingNetwork_instance(x_train,output_dim)
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
    output_dim = 123  # 輸出的維度即要mapping的特徵數量
    model = FeatureMappingNetwork_instance(x_train,output_dim)

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

def FeatureMappingAndMinMax(df,output_dim):
    # 除了Label外的特徵數量
    print(Fore.YELLOW+Style.BRIGHT+"Before feature Mapping feature count:",df.iloc[:,:-1].shape[1])
    x_train = df.iloc[:, :-1].values  # 所有特徵
    y_train = df.iloc[:, -1].values  # 標籤
    # 創建模型實例
    # 原本的維度即原特徵數量:x_train
    # 輸出的維度即要mapping的特徵數量:output_dim  
    model = FeatureMappingNetwork_instance(x_train,output_dim)
    # 將 x_train 轉換為 PyTorch 張量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

    # 進行特徵映射
    mapped_features = model(x_train_tensor)
    
    # 將映射後的特徵轉換為 Pandas DataFrame
    mapped_df = pd.DataFrame(mapped_features.detach().numpy())
    
    print(Fore.YELLOW+Style.BRIGHT+"After feature Mapping feature count:",mapped_df.iloc[:,:-1].shape[1])
    # 將 y_train 轉換為 Pandas DataFrame
    y_train_df = pd.DataFrame(y_train, columns=['Label'])

    # 合併 x_train 映射後的特徵與 y_train
    final_df = pd.concat([mapped_df, y_train_df], axis=1)

    # 對映射後的特徵進行MinMax標準化
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(final_df.iloc[:, :-1].values)
    final_df.iloc[:, :-1] = scaler.transform(final_df.iloc[:, :-1])

    return final_df

def CICIDS2017_DoFeatureMappingAfterReadCSV(choose_merge_days):
    generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\featureMapping\\", today)
    generatefolder(filepath + f"\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\featureMapping\\{today}\\", current_time)
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20250121\\Deleted79features\\10000筆資料\\ALLDay_Deleted79features_20250121.csv") 
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\CICIDS2017_AfterProcessed_DoLabelencode_ALLDay_10000.csv") 
    
    # 創建空的 DataFrame
    train_dataframes = pd.DataFrame()
    test_dataframes = pd.DataFrame()
    output_dim = 123  # 輸出的維度即要mapping的特徵數量
    # 進行FeatureMapping
    final_df = FeatureMappingAndMinMax(df,output_dim)
    # 儲存結果為CSV
    final_df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}/mapped_features_with_labels.csv", index=False)

    print("CSV file has been saved.")

    from CICIDS2017_Preprocess import DoBaselinesplit
    train_dataframes, test_dataframes= DoBaselinesplit(final_df,train_dataframes,test_dataframes)            
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}/encode_and_count_Deleted79features.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}", f"{choose_merge_days}_train_dataframes_featureMapping_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}", f"{choose_merge_days}_test_dataframes_featureMapping_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}", f"{choose_merge_days}_test_featureMapping",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}", f"{choose_merge_days}_train_featureMapping",today)

def CICIDS2018_DoFeatureMappingAfterReadCSV(choose_merge_days):
    generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\featureMapping\\", today)
    generatefolder(filepath + f"\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\featureMapping\\{today}\\", current_time)
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\sampled_data_max_10000_per_label.csv")
    # 創建空的 DataFrame
    train_dataframes = pd.DataFrame()
    test_dataframes = pd.DataFrame()
    output_dim = 123  # 輸出的維度即要mapping的特徵數量
    # 進行FeatureMapping
    final_df = FeatureMappingAndMinMax(df,output_dim)
    # 儲存結果為CSV
    final_df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/featureMapping/{today}/{current_time}/mapped_features_with_labels.csv", index=False)

    print("CSV file has been saved.")

    from CICIDS2018_Preprocess import DoBaselinesplit
    train_dataframes, test_dataframes= DoBaselinesplit(final_df,train_dataframes,test_dataframes)            
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/featureMapping/{today}/{current_time}/encode_and_count_featureMapping.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/featureMapping/{today}/{current_time}", f"{choose_merge_days}_train_dataframes_featureMapping_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/featureMapping/{today}/{current_time}", f"{choose_merge_days}_test_dataframes_featureMapping_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/featureMapping/{today}/{current_time}", f"{choose_merge_days}_test_featureMapping",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/featureMapping/{today}/{current_time}", f"{choose_merge_days}_train_featureMapping",today)

def TONIOT_DoFeatureMappingAfterReadCSV():
    generatefolder(filepath + "\\dataset_AfterProcessed\\TONIOT\\featureMapping\\", today)
    generatefolder(filepath + f"\\dataset_AfterProcessed\\TONIOT\\featureMapping\\{today}\\", current_time)
    df= pd.read_csv(f'./data/dataset_AfterProcessed/TONIOT/20250312/Train_Test_Network_AfterProcessed_updated_10000_ALLMinmax_and_Labelencode.csv')
    # 創建空的 DataFrame
    train_dataframes = pd.DataFrame()
    test_dataframes = pd.DataFrame()
    output_dim = 123  # 輸出的維度即要mapping的特徵數量
    # 進行FeatureMapping
    final_df = FeatureMappingAndMinMax(df,output_dim)
    # 儲存結果為CSV
    final_df.to_csv(f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}/mapped_features_with_labels.csv", index=False)

    print("CSV file has been saved.")

    from TONIOT_Preprocess import DoBaselinesplit
    train_dataframes, test_dataframes= DoBaselinesplit(final_df,train_dataframes,test_dataframes)            
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}/encode_and_count_featureMapping.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}", f"TONIOT_train_dataframes_featureMapping_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}", f"TONIOT_test_dataframes_featureMapping_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}", f"TONIOT_test_featureMapping",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}", f"TONIOT_train_featureMapping",today)


# 強制以npyfile進行特徵映射FeatureMapping
# DoFeatureMappingAndSavetoCSVfile(x_train,y_train, "train")
# DoFeatureMappingAndSavetoCSVfile(x_test,y_test, "test")
# print(Fore.RED +Back.BLUE+ Style.BRIGHT+str(x_train.shape[1]))  # 這裡的 x_train_mapped 就是經過網絡層映射後的特徵
# x_train = DoFeatureMapping(x_train)
# x_test = DoFeatureMapping(x_test)
# print(Fore.LIGHTYELLOW_EX +Back.BLUE+ Style.BRIGHT+str(x_train.shape[1]))  # 這裡的 x_train_mapped 就是經過網絡層映射後的特徵

# CICIDS2017_DoFeatureMappingAfterReadCSV("ALLday")

# CICIDS2018_DoFeatureMappingAfterReadCSV("csv_data")
TONIOT_DoFeatureMappingAfterReadCSV()