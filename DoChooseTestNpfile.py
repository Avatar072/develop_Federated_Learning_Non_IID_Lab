import os
import pandas as pd
import numpy as np
import argparse
import time
import datetime
from colorama import Fore, Back, Style, init
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)

class CICIDS2019TestLoader:
    def __init__(self, filepath,Attack_method):
        self.filepath = filepath
        self.Attack_method = Attack_method

    def load_test_data(self, Choose_Attacktype, split_file):
        if split_file == 'test' and Choose_Attacktype == 'normal':
            # 使用不同日期或特徵選擇方式的檔案
            print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2019 test data with normal attack type")
            # 20240422 CICIDS2019 after PCA do labelencode and minmax
            # x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_CICIDS2019_01_12_test_20240422.npy", allow_pickle=True)
            # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_CICIDS2019_01_12_test_20240422.npy", allow_pickle=True)

            # 20240422 CICIDS2019 after PCA do labelencode and minmax chi-square 45
            # x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_CICIDS2019_AfterFeatureSelect44_20240422.npy", allow_pickle=True)
            # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_CICIDS2019_AfterFeatureSelect44_20240422.npy", allow_pickle=True)

            # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_20240502.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_20240502.npy", allow_pickle=True)
            # 20241102 CIC-IDS2019 after do labelencode and minmax 75 25分 do Person相關係數選擇
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_CICIDS2019_AfterPearsonFeatureSelect52_20241102.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_CICIDS2019_AfterPearsonFeatureSelect52_20241102.npy", allow_pickle=True)
            # 20241102 CIC-IDS2019 after do labelencode and minmax 75 25分 do PCA
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal AfterPCA")
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_AfterPCA77_20241102.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_AfterPCA77_20241102.npy", allow_pickle=True)
            # 20241102 CIC-IDS2019 after do labelencode and minmax 58 42分 二元分類
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal Afterbinary")
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12__binary__test_20241102.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12__binary__test_20241102.npy", allow_pickle=True)
            
            # 20241102 CIC-IDS2019 after do labelencode and minmax 75 25分 DO CandW attack
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal After Do CandW attack")
            # x_test = np.load(f"./Adversarial_Attack_Test/20241102_CandW_C1/x_DoCandW_test_20241102.npy")
            # y_test = np.load(f"./Adversarial_Attack_Test/20241102_CandW_C1/y_DoCandW_test_20241102.npy")

            # 20241102 CIC-IDS2019 after do labelencode and minmax 75 25分 DO CandW attack 0.5
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal After Do CandW attack")
            # x_test = np.load(f"./Adversarial_Attack_Test/20241102_CandW_C0.5/x_DoC_0.5andW_test_20241102.npy")
            # y_test = np.load(f"./Adversarial_Attack_Test/20241102_CandW_C0.5/y_DoC_0.5andW_test_20241102.npy")
            # 20241102 CIC-IDS2019 after do labelencode and minmax 75 25分 DO CandW attack 0.5
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal After Do feature squeezed")
            x_test = np.load(f"./Adversarial_Attack_Denfense/Processed_Advlist_C&W/x_DoCandW_test_squeezed.npy")
            y_test = np.load(f"./Adversarial_Attack_Denfense/Processed_Advlist_C&W/y_DoCandW_test_squeezed.npy")
            return x_test, y_test
        elif split_file == 'test' and Choose_Attacktype == 'Evasion_Attack':
            print("Using CICIDS2019 with Evasion_Attack")
            # if (self.Attack_method == 'JSMA'):
            # if (self.Attack_method == 'FGSM'):
            # if (self.Attack_method == 'PGD'):
            # if (self.Attack_method == 'CandW'):
            # 可在此加載相關的檔案
            return None, None
        elif split_file == 'test' and Choose_Attacktype == 'Poisoning_Attack':
            print("Using CICIDS2019 with Ponsion_Attack")
            # 可在此加載相關的檔案
            return None, None
        else:
            raise ValueError("Invalid Choose_Attacktype or split_file")
class TONIOTTestLoader:
    def __init__(self,filepath,Attack_method):
         self.filepath = filepath
         self.Attack_method = Attack_method
    
    def load_test_data(self,Choose_Attacktype,split_file):
        if split_file=='test' and Choose_Attacktype =='normal':
            print(Fore.GREEN+Style.BRIGHT+"Loading TONIOT test data with normal attack type")
            # 20240523 TONIoT after do labelencode and minmax  75 25分
            x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_test_ToN-IoT_20240523.npy", allow_pickle=True)
            y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_test_ToN-IoT_20240523.npy", allow_pickle=True)   

            # 20240523 TONIoT after do labelencode and minmax  75 25分 DOJSMA
            # x_test  = np.load(f"./Adversarial_Attack_Test/20240721_0.5_0.5/x_DoJSMA_test_20240721.npy")
            # y_test  = np.load(f"./Adversarial_Attack_Test/20240721_0.5_0.5/y_DoJSMA_test_20240721.npy")


            return x_test, y_test
        elif split_file == 'test' and Choose_Attacktype == 'Evasion_Attack':
            print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Evasion_Attack")
            # 可在此加載相關的檔案
            if (self.Attack_method == 'JSMA'):
                # 20241030 TONIoT after do labelencode and minmax  75 25分 JSMA
                x_test = np.load("D:\\develop_Federated_Learning_Non_IID_Lab\\Adversarial_Attack_Test\\20241030\\x_DoJSMA_test_theta_0.05_20241030.npy", allow_pickle=True)
                y_test = np.load("D:\\develop_Federated_Learning_Non_IID_Lab\\Adversarial_Attack_Test\\20241030\\y_DoJSMA_test_theta_0.05_20241030.npy", allow_pickle=True)   
            return x_test, y_test
            # if (self.Attack_method == 'FGSM'):
            # if (self.Attack_method == 'PGD'):
            # if (self.Attack_method == 'CandW'):
            
        elif split_file == 'test' and Choose_Attacktype == 'Poisoning_Attack':
            print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Poisoning_Attack")
            # 可在此加載相關的檔案
            return None, None
        else:
            raise ValueError("Invalid Choose_Attacktype or split_file")


'''
ChooseLoadTestNpArray 函數根據 choose_datasets 選擇適當的類別來加載測試數據。
每個類別的 load_test_data 方法包含不同的檔案選項和攻擊類型 (normal、Evasion_Attack、Poisoning_Attack)。
'''

def ChooseLoadTestNpArray(choose_datasets,split_file, filepath, Choose_Attacktype,Attack_method):
    if choose_datasets == "CICIDS2019":
        loader = CICIDS2019TestLoader(filepath,Attack_method)
    elif choose_datasets == "TONIOT":
        loader = TONIOTTestLoader(filepath,Attack_method)
    else:
        raise ValueError("Unknown dataset type")

    x_test, y_test = loader.load_test_data(Choose_Attacktype, split_file)
    print("use file", split_file)
    return x_test, y_test