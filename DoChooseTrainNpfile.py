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

class CICIDS2019BaseLine_TrainLoader:
    client_str = 'BaseLine'
    def __init__(self, filepath,Attack_method):
        self.filepath = filepath
        self.Attack_method = Attack_method

    def load_train_data(self, Choose_Attacktype, split_file):
        if split_file=='baseLine_train' and Choose_Attacktype =='normal':
            # 使用不同日期或特徵選擇方式的檔案
            print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal attack type")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_CICIDS2019_01_12_train_20240422.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_CICIDS2019_01_12_train_20240422.npy", allow_pickle=True)
            # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
            x_train = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_20240502.npy", allow_pickle=True)
            y_train = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_20240502.npy", allow_pickle=True)
            # 20241030 CIC-IDS2019 after do labelencode and minmax 75 25分 do GDA 高斯資料增強
            # x_train = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/x_CICIDS2019_train_augmented.npy", allow_pickle=True)
            # y_train = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/y_CICIDS2019_train_augmented.npy", allow_pickle=True)
            # 20241102 CIC-IDS2019 after do labelencode and minmax 75 25分 do Person相關係數選擇
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal AfterPearsonFeatureSelect52")
            # x_train = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_CICIDS2019_AfterPearsonFeatureSelect52_20241102.npy", allow_pickle=True)
            # y_train = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_CICIDS2019_AfterPearsonFeatureSelect52_20241102.npy", allow_pickle=True)
            # 20241102 CIC-IDS2019 after do labelencode and minmax 75 25分 do PCA
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal AfterPCA")
            # x_train = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_AfterPCA77_20241102.npy", allow_pickle=True)
            # y_train = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_AfterPCA77_20241102.npy", allow_pickle=True)
            
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal Afterbinary")
            # x_train = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12__binary__train_20241102.npy", allow_pickle=True)
            # y_train = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12__binary__train_20241102.npy", allow_pickle=True)
            
            return x_train, y_train,self.client_str
        elif split_file == 'baseLine_train' and Choose_Attacktype == 'Evasion_Attack':
            print("Using CICIDS2019 with Evasion_Attack")
            # if (Attack_method == 'JSMA'):
            # if (Attack_method == 'FGSM'):
            # if (Attack_method == 'PGD'):
            # if (Attack_method == 'CandW'):
            # 可在此加載相關的檔案
            return None, None
        elif split_file == 'baseLine_train' and Choose_Attacktype == 'Poisoning_Attack':
            print("Using CICIDS2019 with Ponsion_Attack")
            # 可在此加載相關的檔案
            return None, None
        else:
            raise ValueError("Invalid Choose_Attacktype or split_file")
# def ChooseLoadNpArray(filepath, choose_datasets,split_file, Choose_method):

class TONIOTTrainLoaderBase:
    def __init__(self, filepath,Attack_method):
        self.filepath = filepath
        self.Attack_method = Attack_method

    def load_normal_data(self, x_path, y_path):
        """通用的正常數據加載方法，子類中可以直接使用或覆寫"""
        x_train = np.load(x_path, allow_pickle=True)
        y_train = np.load(y_path, allow_pickle=True)
        return x_train, y_train

    def load_jsma_attack_data(self, x_path, y_path):
        """JSMA 攻擊數據加載方法，可在子類中調用"""
        x_train = np.load(x_path, allow_pickle=True)
        y_train = np.load(y_path, allow_pickle=True)
        return x_train, y_train

    # 其他攻擊方法的模版，也可以在子類中覆寫
    def load_fgsm_attack_data(self):
        print("FGSM attack data loading not implemented.")
        return None, None

    def load_pgd_attack_data(self):
        print("PGD attack data loading not implemented.")
        return None, None

    def load_candw_attack_data(self):
        print("C&W attack data loading not implemented.")
        return None, None

    def load_train_data(self, Choose_Attacktype, split_file, Attack_method=None):
        raise NotImplementedError("Each client loader must implement its own `load_train_data` method.")


class TONIOTBaseLine_TrainLoader(TONIOTTrainLoaderBase):
    client_str = 'BaseLine'
    def load_train_data(self,Choose_Attacktype,split_file):
        if split_file=='baseLine_train' and Choose_Attacktype =='normal':
                print(Fore.GREEN+Style.BRIGHT+"Loading TONIOT" +f"{split_file} with normal attack type")
                # 20240523 TONIoT after do labelencode and minmax  75 25分
                # 直接加載數據
                x_train = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_20240523.npy", allow_pickle=True)
                y_train = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_20240523.npy", allow_pickle=True)
                print(Fore.GREEN+Style.BRIGHT+"Debug: x_train shape:", x_train.shape if x_train is not None else None)
                print(Fore.GREEN+Style.BRIGHT+"Debug: y_train shape:", y_train.shape if y_train is not None else None)
                print(Fore.GREEN+Style.BRIGHT+"Debug: client_str:", self.client_str)
                # 確認三個值是否正確
                return x_train, y_train, self.client_str

        elif split_file == 'baseLine_train' and Choose_Attacktype == 'Evasion_Attack':
                print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Evasion_Attack")
                # 可在此加載相關的檔案
                if (self.Attack_method == 'JSMA'):
                    x_train = self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_baseLine_20240801.npy"
                    y_train = self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_baseLine_20240801.npy"
                    # x_train, y_train = self.load_jsma_attack_data(x_path, y_path)
                    return x_train, y_train,self.client_str
                else:
                    raise ValueError(Fore.RED+Back.RED+Style.BRIGHT+"Invalid Choose_Attacktype or split_file")
                # if (self.Attack_method == 'FGSM'):
                    #  return self.load_fgsm_attack_data()
                # if (self.Attack_method == 'PGD'):
                # if (self.Attack_method == 'CandW'):
                return None, None
        elif split_file == 'baseLine_train' and Choose_Attacktype == 'Poisoning_Attack':
                print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Poisoning_Attack")
                # 可在此加載相關的檔案
                return None, None
        else:
            raise ValueError("Invalid Choose_Attacktype or split_file")


class TONIOTClient1_TrainLoader(TONIOTTrainLoaderBase):    
    client_str = "client1"
    def load_train_data(self,Choose_Attacktype,split_file):
        if split_file=='client1_train' and Choose_Attacktype =='normal':
                print(Fore.GREEN+Style.BRIGHT+"Loading TONIOT" +f"{split_file} with normal attack type")
                # 20240523 client1 use TONIoT after do labelencode and minmax  隨機劃分75 25分
                x_train = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_random_train_half1_20240523.npy", allow_pickle=True)
                y_train = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_random_train_half1_20240523.npy", allow_pickle=True)  
                # x_train, y_train = self.load_normal_data(x_path, y_path)
                return x_train, y_train,self.client_str
        elif split_file == 'client1_train' and Choose_Attacktype == 'Evasion_Attack':
                print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Evasion_Attack")
                # 可在此加載相關的檔案
                if (self.Attack_method == 'JSMA'):
                    x_train = self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half1_20240801.npy"
                    y_train = self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half1_20240801.npy"
                    # x_train, y_train = self.load_jsma_attack_data(x_path, y_path)
                    return x_train, y_train,self.client_str
                else:
                    raise ValueError(Fore.RED+Back.RED+Style.BRIGHT+"Invalid Choose_Attacktype or split_file")
                # if (self.Attack_method == 'FGSM'):
                    # return self.load_fgsm_attack_data()
                # if (self.Attack_method == 'PGD'):
                # if (self.Attack_method == 'CandW'):
                return None, None,self.client_str
        elif split_file == 'client1_train' and Choose_Attacktype == 'Poisoning_Attack':
                print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Poisoning_Attack")
                # 可在此加載相關的檔案
                return None, None,self.client_str
        else:
            raise ValueError("Invalid Choose_Attacktype or split_file")

class TONIOTClient2_TrainLoader(TONIOTTrainLoaderBase):
    client_str = "client2"
    def load_train_data(self,Choose_Attacktype,split_file,Attack_method=None):
        if split_file=='client2_train' and Choose_Attacktype =='normal':
                print(Fore.GREEN+Style.BRIGHT+"Loading TONIOT" +f"{split_file} with normal attack type")
                # 20240523 client2 use TONIoT after do labelencode and minmax  隨機劃分75 25分
                x_train = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_random_train_half2_20240523.npy", allow_pickle=True)
                y_train = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_random_train_half2_20240523.npy", allow_pickle=True)  
                return x_train, y_train,self.client_str
        elif split_file == 'client2_train' and Choose_Attacktype == 'Evasion_Attack':
                print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Evasion_Attack")
                # 可在此加載相關的檔案
                if (Attack_method == 'JSMA'):
                    x_train = self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half2_20240801.npy"
                    y_train = self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half2_20240801.npy"
                    # x_train, y_train = self.load_jsma_attack_data(x_path, y_path)
                    return x_train, y_train,self.client_str
                else:
                    raise ValueError(Fore.RED+Back.RED+Style.BRIGHT+"Invalid Choose_Attacktype or split_file")
                # if (Attack_method == 'FGSM'):
                    # return self.load_fgsm_attack_data()
                # if (Attack_method == 'PGD'):
                # if (Attack_method == 'CandW'):
                return None, None,self.client_str
        elif split_file == 'client2_train' and Choose_Attacktype == 'Poisoning_Attack':
                print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Poisoning_Attack")
                # 可在此加載相關的檔案
                return None, None,self.client_str
        else:
            raise ValueError("Invalid Choose_Attacktype or split_file")

class TONIOTClient3_TrainLoader(TONIOTTrainLoaderBase):
    client_str = "client3"
    def load_train_data(self,Choose_Attacktype,split_file):
        if split_file=='client3_train' and Choose_Attacktype =='normal':
                print(Fore.GREEN+Style.BRIGHT+"Loading TONIOT" +f"{split_file} with normal attack type")
                # 20240523 client3 use TONIoT after do labelencode and minmax  隨機劃分75 25分
                x_train  = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_random_train_half3_20240523.npy", allow_pickle=True)
                y_train  = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_random_train_half3_20240523.npy", allow_pickle=True)  
                return x_train, y_train,self.client_str
        elif split_file == 'client3_train' and Choose_Attacktype == 'Evasion_Attack':
                print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Evasion_Attack")
                # 調試輸出傳入的參數
                print(Fore.RED+Back.RED+Style.BRIGHT+"Debug: Choose_Attacktype =", Choose_Attacktype)
                print(Fore.RED+Back.RED+Style.BRIGHT+"Debug: split_file =", split_file)
                print(Fore.RED+Back.RED+Style.BRIGHT+"Debug: Attack_method =", self.Attack_method)
                # 可在此加載相關的檔案
                if (self.Attack_method == 'JSMA'):
                    print(Fore.GREEN+Style.BRIGHT+"Using TONIOT" +f"{split_file}"+f"{self.Attack_method}+with Evasion_Attack")
                    # 載入被JSMA攻擊的數據 theta=0.05
                    x_train = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half3_20240801.npy", allow_pickle=True)
                    y_train = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half3_20240801.npy", allow_pickle=True)
                    # x_train, y_train = self.load_jsma_attack_data(x_path, y_path)
                    return x_train, y_train,self.client_str
                else:
                    raise ValueError(Fore.RED+Back.RED+Style.BRIGHT+"Invalid Choose_Attacktype or split_file")
                # if (self.Attack_method == 'FGSM'):
                    # return self.load_fgsm_attack_data()
                # if (self.Attack_method == 'FGSM'):
                # if (self.Attack_method == 'PGD'):
                # if (self.Attack_method == 'CandW'):
                return None, None, self.client_str
        elif split_file == 'client3_train' and Choose_Attacktype == 'Poisoning_Attack':
                print(Fore.GREEN+Style.BRIGHT+"Using TONIOT with Poisoning_Attack")
                # 可在此加載相關的檔案
                return None, None, self.client_str
        else:
            raise ValueError("Invalid Choose_Attacktype or split_file")
'''
ChooseLoadTrainNpArray 函數根據 choose_datasets 選擇適當的類別來加載測試數據。
每個類別的 load_test_data 方法包含不同的檔案選項和攻擊類型 (normal、Evasion_Attack、Poisoning_Attack)。
'''

def ChooseLoadTrainNpArray(choose_datasets,split_file, filepath, Choose_Attacktype,Attack_method):
    if choose_datasets == "CICIDS2019":
        loader = CICIDS2019BaseLine_TrainLoader(filepath,Attack_method)
    elif choose_datasets == "TONIOT":
        if split_file =="baseLine_train": 
            loader = TONIOTBaseLine_TrainLoader(filepath,Attack_method)
        if split_file =="client1_train": 
            loader = TONIOTClient1_TrainLoader(filepath,Attack_method)
        if split_file =="client2_train": 
            loader = TONIOTClient2_TrainLoader(filepath,Attack_method)
        if split_file =="client3_train": 
            loader = TONIOTClient3_TrainLoader(filepath,Attack_method)
    else:
        raise ValueError("Unknown dataset type")

     # 調用 loader 的 load_train_data 方法來加載數據
    x_train, y_train, client_str = loader.load_train_data(Choose_Attacktype, split_file)
    print("use file", split_file)
    return x_train, y_train, client_str