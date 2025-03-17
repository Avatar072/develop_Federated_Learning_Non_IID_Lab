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

def cropdatasetforStringtypeFeature(df,columns_to_exclude):
    # 除'dst_port', 'proto', 'ts', 'duration'外的特徵 for TONIOT
    # ['DestinationPort', 'Protocol', 'Timestamp', 'FlowDuration'] for CICIDS2017
    # ['DestinationPort', 'Protocol', 'Timestamp', 'FlowDuration'] for CICIDS2018
    dofeatureMappingdataset = df[[col for col in df.columns if col not in columns_to_exclude]]
    # 'dst_port', 'proto', 'ts', 'duration'
    undofeatureMappingdataset = df[[col for col in df.columns if col in columns_to_exclude]]
    print(dofeatureMappingdataset.info())
    print(undofeatureMappingdataset.info())

    return undofeatureMappingdataset, dofeatureMappingdataset

def Do_FeatureMapping_After_AlignmentStringTypeFeature(df, columns_to_exclude, output_dim):
    undofeatureMappingdataset,dofeatureMappingdataset = cropdatasetforStringtypeFeature(df,columns_to_exclude)
    output_dim = 116  # 輸出的維度即要mapping的特徵數量
    # CICIDS2017/2018 feature:79，79 = 75 + 4 string(要對齊的特徵)
    # TONIOT feature:41，41 = 37 + 4 string(要對齊的特徵)
    # 輸出的維度:79+41=120
    # 120-4string = 要mapping的特徵為116
    dofeatureMappingdataset = FeatureMappingAndMinMax(dofeatureMappingdataset,output_dim)
    # 將'DestinationPort', 'Protocol', 'Timestamp', 'FlowDuration'和mapping後的特徵和 Label 合並為新的 DataFrame
    final_df = pd.concat([undofeatureMappingdataset,dofeatureMappingdataset], axis = 1)
    return final_df

def CICIDS2017_DoFeatureMappingAfterReadCSV(choose_merge_days):
    generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\featureMapping\\", today)
    generatefolder(filepath + f"\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\featureMapping\\{today}\\", current_time)
    # 79 feature
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20250121\\Deleted79features\\10000筆資料\\ALLDay_Deleted79features_20250121.csv") 
    # 83 feature
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\CICIDS2017_AfterProcessed_DoLabelencode_ALLDay_10000.csv") 
    # 79 feature and Label merged 
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\featureMapping\\20250314\\rename_Label實驗\\Label_merge.csv") 

    # 創建空的 DataFrame
    train_dataframes = pd.DataFrame()
    test_dataframes = pd.DataFrame()
    output_dim = 123  # 不對齊直接進行FeatureMapping 輸出的維度即要mapping的特徵數量(CICIDS feature:79+TONIOT feature:44)
    # 不對齊直接進行FeatureMapping
    final_df = FeatureMappingAndMinMax(df,output_dim)
    # 對齊string特徵後再進行FeatureMapping
    # columns_to_exclude = ['DestinationPort', 'Protocol', 'Timestamp', 'FlowDuration']
    # final_df  = Do_FeatureMapping_After_AlignmentStringTypeFeature(df, columns_to_exclude, output_dim)
    # 儲存結果為CSV
    final_df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}/mapped_features_with_labels.csv", index=False)
    print("CSV file has been saved.")
    from CICIDS2017_Preprocess import DoBaselinesplitAfter_LabelMerge
    train_dataframes, test_dataframes= DoBaselinesplitAfter_LabelMerge(final_df,train_dataframes,test_dataframes)            
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
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\sampled_data_max_10000_per_label.csv")
    # 79 feature and Label merged 
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\featureMapping\\20250314\\rename_Label實驗\\Label_merge.csv")
    
    # 創建空的 DataFrame
    train_dataframes = pd.DataFrame()
    test_dataframes = pd.DataFrame()
    output_dim = 123  # 不對齊直接進行FeatureMapping 輸出的維度即要mapping的特徵數量(CICIDS feature:79+TONIOT feature:44)
    # 不對齊直接進行FeatureMapping
    final_df = FeatureMappingAndMinMax(df,output_dim)
    # 對齊string特徵後再進行FeatureMapping
    # columns_to_exclude = ['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration']
    # final_df  = Do_FeatureMapping_After_AlignmentStringTypeFeature(df, columns_to_exclude, output_dim)
    # 儲存結果為CSV
    final_df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/featureMapping/{today}/{current_time}/mapped_features_with_labels.csv", index=False)

    print("CSV file has been saved.")

    from CICIDS2018_Preprocess import DoBaselinesplitAfter_LabelMerge
    train_dataframes, test_dataframes= DoBaselinesplitAfter_LabelMerge(final_df,train_dataframes,test_dataframes)     
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
    
    # 44 feature版本
    # df= pd.read_csv(f'./data/dataset_AfterProcessed/TONIOT/20250312/Train_Test_Network_AfterProcessed_updated_10000_ALLMinmax_and_Labelencode.csv')
    
    # 留['dst_port', 'proto', 'ts', 'duration']41 feature版本
    # df= pd.read_csv(f'./data/dataset_AfterProcessed/TONIOT/DeleteFeature/TONIOT_Deletedfeatures_20250313_new.csv')
    
    # 44 feature and 把backdoor和ddos互相更換encode值
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\featureMapping\\20250314\\rename_Label實驗\\Label_merge.csv")


    # 創建空的 DataFrame
    train_dataframes = pd.DataFrame()
    test_dataframes = pd.DataFrame()
    output_dim = 123  # 不對齊直接進行FeatureMapping 輸出的維度即要mapping的特徵數量(CICIDS feature:79+TONIOT feature:44)
    # 不對齊直接進行FeatureMapping
    final_df = FeatureMappingAndMinMax(df,output_dim)
    # 對齊string特徵後再進行FeatureMapping
    # columns_to_exclude = ['dst_port', 'proto', 'ts', 'duration']
    # final_df  = Do_FeatureMapping_After_AlignmentStringTypeFeature(df, columns_to_exclude, output_dim)

    # 儲存結果為CSV
    final_df.to_csv(f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}/mapped_features_with_labels.csv", index=False)

    print("CSV file has been saved.")

    from TONIOT_Preprocess import DoBaselinesplit
    train_dataframes, test_dataframes= DoBaselinesplit(final_df,train_dataframes,test_dataframes,False)  
               
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}/encode_and_count_featureMapping.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        # label_counts = test_dataframes['type'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        # label_counts = train_dataframes['type'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}", f"TONIOT_train_dataframes_featureMapping_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}", f"TONIOT_test_dataframes_featureMapping_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}", f"TONIOT_test_featureMapping",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}", f"TONIOT_train_featureMapping",today)

# 用於刪除特定特徵跟設置新順序
def TONIOTdropStringtypeAfterReadCSV_():
    # 44 feature版本
    df= pd.read_csv(f'./data/dataset_AfterProcessed/TONIOT/20250312/Train_Test_Network_AfterProcessed_updated_10000_ALLMinmax_and_Labelencode.csv')
    generatefolder(filepath + f"\\dataset_AfterProcessed\\TONIOT\\DeleteFeature\\{today}\\", current_time)
    # 刪除特定特徵for feauture mapping 將特徵44刪至41個
    # df.drop(columns=['ts','src_ip','src_port', 'dst_ip', 'dst_port', 'proto'], inplace=True)
    df.drop(columns=['src_ip','src_port', 'dst_ip'], inplace=True)
    # 設置新順序
    new_column_order = ['dst_port', 'proto', 'ts', 'duration']
    df_new_order = df[new_column_order]  
    # 特徵列
    crop_dataset=df.iloc[:,:-1]
    columns_to_exclude = ['ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'duration']
    # 除columns_to_exclude外的特徵
    doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    
    # 將除columns_to_exclude外的特徵和設置新順序的特徵和 Label 合並為新的 DataFrame
    afterminmax_dataset = pd.concat([df_new_order,doScalerdataset,df['type']], axis = 1)
    SaveDataToCsvfile(afterminmax_dataset, f"./data/dataset_AfterProcessed/TONIOT/DeleteFeature", f"TONIOT_Deletedfeatures_{today}_new") 
    print(afterminmax_dataset.info())

    # 41 feature版本
    df= pd.read_csv(f'./data/dataset_AfterProcessed/TONIOT/DeleteFeature/TONIOT_Deletedfeatures_20250313_new.csv')
    generatefolder(filepath + f"\\dataset_AfterProcessed\\TONIOT\\DeleteFeature\\{today}\\", current_time)

    # 創建空的 DataFrame
    train_dataframes = pd.DataFrame()
    test_dataframes = pd.DataFrame()
    from TONIOT_Preprocess import DoBaselinesplit
    train_dataframes, test_dataframes= DoBaselinesplit(df,train_dataframes,test_dataframes)            
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/TONIOT/DeleteFeature/{today}/{current_time}/encode_and_count_featureMapping.csv", "a+") as file:
        label_counts = test_dataframes['type'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['type'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/DeleteFeature/{today}/{current_time}/", f"TONIOT_train_dataframes_DeleteFeature41_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/TONIOT/DeleteFeature/{today}/{current_time}/", f"TONIOT_test_dataframes_DeleteFeature41_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/TONIOT/DeleteFeature/{today}/{current_time}/", f"TONIOT_test_DeleteFeature41",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/DeleteFeature/{today}/{current_time}/", f"TONIOT_train_DeleteFeature41",today)

# 　把label 分為大類 並取上限隨機10000
def CICIDS2017_DorenameLabel(choose_merge_days):
    generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\featureMapping\\", today)
    generatefolder(filepath + f"\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\featureMapping\\{today}\\", current_time)
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20250121\\Deleted79features\\10000筆資料\\ALLDay_Deleted79features_20250121.csv") 
    # df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}/Label_rename.csv", index=False)
    from CICIDS2017_Preprocess import ReplaceMorethanTenthousandQuantity
    # 把WEB Attack算一類 ; Dos算一類
    df['Label'] = df['Label'].replace({4: 3,  5: 3,  6: 3, 9: 4, 12: 5, 13: 5, 14: 5, 7: 6, 11: 7, 10: 9})
    df = ReplaceMorethanTenthousandQuantity(df)
    # 檢查結果
    print(df['Label'].value_counts())
    df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/featureMapping/{today}/{current_time}/Label_merge.csv", index=False)

def CICIDS2018_DorenameLabel(choose_merge_days):
    generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\featureMapping\\", today)
    generatefolder(filepath + f"\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\featureMapping\\{today}\\", current_time)
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\sampled_data_max_10000_per_label.csv")
    from CICIDS2017_Preprocess import ReplaceMorethanTenthousandQuantity
    # 把WEB Attack算一類 ; Dos算一類; DDos算一類
    df['Label'] = df['Label'].replace({2: 5, 3: 5, 13: 5, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 6, 12: 4, 14: 7})
    df = ReplaceMorethanTenthousandQuantity(df)
    # 檢查結果
    print(df['Label'].value_counts())
    df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/featureMapping/{today}/{current_time}/Label_merge.csv", index=False)

def TONIOT_DorenameLabel():
    generatefolder(filepath + "\\dataset_AfterProcessed\\TONIOT\\featureMapping\\", today)
    generatefolder(filepath + f"\\dataset_AfterProcessed\\TONIOT\\featureMapping\\{today}\\", current_time)
    
    # 44 feature版本
    df= pd.read_csv(f'./data/dataset_AfterProcessed/TONIOT/20250312/Train_Test_Network_AfterProcessed_updated_10000_ALLMinmax_and_Labelencode.csv')
    # 查找原本 'type' 為 1 的行
    print(df[df['type'] == 1].head())

    # 查找原本 'type' 為 2 的行
    print(df[df['type'] == 2].head())
    # 把ddos跟backdoor互調位置
    df['type'] = df['type'].replace({2: 1, 1: 2})
    # from TONIOT_Preprocess import ReplaceMorethanTenthousandQuantity
    # 查找更換後 'type' 為 1 的行
    print(df[df['type'] == 1].head())

    # 查找更換後 'type' 為 2 的行
    print(df[df['type'] == 2].head())
    # df = ReplaceMorethanTenthousandQuantity(df)
    # 檢查結果
    print(df['type'].value_counts())
    df.to_csv(f"./data/dataset_AfterProcessed/TONIOT/featureMapping/{today}/{current_time}/Label_merge.csv", index=False)

    
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
# CICIDS2017_DorenameLabel("ALLday")
# CICIDS2018_DorenameLabel("csv_data")
# TONIOT_DorenameLabel()