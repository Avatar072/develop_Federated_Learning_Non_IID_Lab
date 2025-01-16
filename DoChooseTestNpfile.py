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
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal After Do labelencode and minmax")
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_20240502.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_20240502.npy", allow_pickle=True)
            # 20241119 CIC-IDS2019 after do labelencode and all featrue minmax 75 25分
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_dataframes_ALLMinmax_20241119.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_dataframes_ALLMinmax_20241119.npy", allow_pickle=True)
            
            # 20250113 CIC-IDS2019 after do labelencode do PCA and all featrue minmax 75 25分
            print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2019 after do labelencode do pca" +f"{split_file} with normal attack type")
            x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\x_01_12_test_AfterPCA79_20250113.npy", allow_pickle=True)
            y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\y_01_12_test_AfterPCA79_20250113.npy", allow_pickle=True)
            
            return x_test, y_test
        elif split_file == 'test' and Choose_Attacktype == 'Evasion_Attack':
            print("Using CICIDS2019 with Evasion_Attack")
            # if (self.Attack_method == 'JSMA'):
            if (self.Attack_method == 'FGSM'):
            # 20241116 CIC-IDS2019 after do labelencode and minmax 75 25分 DO FGSM
                print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal After Do FGSM attack")
                
                ################################# ALL feature minmax
                # FGSM = 0.05
                # x_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/x_test_CICIDS2019_adversarial_samples_eps0.05.npy")
                # y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/y_test_CICIDS2019_adversarial_labels_eps0.05.npy")
                # FGSM = 0.1
                # ALL feature minmax
                # x_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/x_test_CICIDS2019_adversarial_samples_eps0.1.npy")
                # y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/y_test_CICIDS2019_adversarial_labels_eps0.1.npy")
                # FGSM = 0.15
                # ALL feature minmax
                x_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/x_test_CICIDS2019_adversarial_samples_eps0.15.npy")
                y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/y_test_CICIDS2019_adversarial_labels_eps0.15.npy")
                # FGSM = 0.2
                # ALL feature minmax
                # x_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/x_test_CICIDS2019_adversarial_samples_eps0.2.npy")
                # y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/y_test_CICIDS2019_adversarial_labels_eps0.2.npy")
                # FGSM = 0.25
                # ALL feature minmax
                # x_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/x_test_CICIDS2019_adversarial_samples_eps0.25.npy")
                # y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/y_test_CICIDS2019_adversarial_labels_eps0.25.npy")
                # FGSM = 0.3
                # ALL feature minmax
                # x_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/x_test_CICIDS2019_adversarial_samples_eps0.3.npy")
                # y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/y_test_CICIDS2019_adversarial_labels_eps0.3.npy")
                # FGSM = 1.0
                # ALL feature minmax
                # x_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/x_test_CICIDS2019_adversarial_samples_eps1.0.npy")
                # y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/y_test_CICIDS2019_adversarial_labels_eps1.0.npy")
                
                # FGSM = 0.05-1.0 DO FS bit 8
                # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"{split_file} with normal After Do FGSM attack FGSM = 0.05-1.0 DO FS bit 8") 
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_1.0_FS_bit8/x_01_12_test_DoFS_adversarial_samples_eps0.05.npy")
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_1.0_FS_bit8/x_01_12_test_DoFS_adversarial_samples_eps0.1.npy")
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_1.0_FS_bit8/x_01_12_test_DoFS_adversarial_samples_eps0.15.npy")
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_1.0_FS_bit8/x_01_12_test_DoFS_adversarial_samples_eps0.2.npy")
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_1.0_FS_bit8/x_01_12_test_DoFS_adversarial_samples_eps0.25.npy")
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_1.0_FS_bit8/x_01_12_test_DoFS_adversarial_samples_eps0.3.npy")
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_1.0_FS_bit8/x_01_12_test_DoFS_adversarial_samples_eps1.0.npy")

                # FGSM = 0.05 DO FS bit 6
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_FS_bit6/x_01_12_test_DoFS_adversarial_samples_eps0.05.npy")
                # FGSM = 0.05 DO FS bit 4
                # x_test = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/FS/20241126/test_FGSM_eps0.05_FS_bit4/x_01_12_test_DoFS_adversarial_samples_eps0.05.npy")
                
                y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/FGSM_Attack/20241124/all_feature_minmax/y_test_CICIDS2019_adversarial_labels_eps0.05.npy")
                
            # if (self.Attack_method == 'PGD'):
            # if (self.Attack_method == 'CandW'):
            # 可在此加載相關的檔案
            return x_test, y_test
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
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_test_ToN-IoT_20240523.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_test_ToN-IoT_20240523.npy", allow_pickle=True)   
            # 20241229 TONIoT after do labelencode and ALLminmax  75 25分
            x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\x_ToN-IoT_test_dataframes_ALLMinmax_20241229.npy", allow_pickle=True)
            y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\TONIOT\\y_ToN-IoT_test_dataframes_ALLMinmax_20241229.npy", allow_pickle=True)   

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

class CICIDS2018TestLoader:
    def __init__(self, filepath,Attack_method):
        self.filepath = filepath
        self.Attack_method = Attack_method

    def load_test_data(self, Choose_Attacktype, split_file):
        if split_file == 'test' and Choose_Attacktype == 'normal':
            # 使用不同日期或特徵選擇方式的檔案
            print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2018 test data with normal attack type")
            # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f"{split_file} with normal After Do labelencode and minmax")
            # 20250106 CIC-IDS2018 after do labelencode and all featrue minmax 75 25分
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_test_20250106.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_20250106.npy", allow_pickle=True)
            # 20250113 CIC-IDS2018 after do labelencode and all featrue minmax 75 25分 do PCA
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f"{split_file} with normal After Do labelencode and minmax and PCA")
            x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_test_AfterPCA79_20250113.npy", allow_pickle=True)
            y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_AfterPCA79_20250113.npy", allow_pickle=True)
            return x_test, y_test
        elif split_file == 'test' and Choose_Attacktype == 'Evasion_Attack':
            print("Using CICIDS2018 with Evasion_Attack")
            # if (self.Attack_method == 'JSMA'):
            if (self.Attack_method == 'FGSM'):
            # 20241116 CIC-IDS2019 after do labelencode and minmax 75 25分 DO FGSM
                print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f"{split_file} with normal After Do FGSM attack")
                
            ################################# ALL feature minmax
            # if (self.Attack_method == 'PGD'):
            # if (self.Attack_method == 'CandW'):
            # 可在此加載相關的檔案
            return x_test, y_test
        elif split_file == 'test' and Choose_Attacktype == 'Poisoning_Attack':
            print("Using CICIDS2018 with Ponsion_Attack")
            # 可在此加載相關的檔案
            return None, None
        else:
            raise ValueError("Invalid Choose_Attacktype or split_file")
class CICIDS2017TestLoader:
    def __init__(self, filepath,Attack_method):
        self.filepath = filepath
        self.Attack_method = Attack_method

    def load_test_data(self, Choose_Attacktype, split_file):
        if split_file == 'test' and Choose_Attacktype == 'normal':
            # 使用不同日期或特徵選擇方式的檔案
            print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2017 test data with normal attack type")
            # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax")
            # 20250107 CIC-IDS2017 after do labelencode and all featrue minmax 75 25分
            # x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_test_20250107.npy", allow_pickle=True)
            # y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_20250107.npy", allow_pickle=True)
            # 20250113 CIC-IDS2017 after do labelencode and except str and PCA all featrue minmax 75 25分
            x_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_test_AfterPCA79_20250113.npy", allow_pickle=True)
            y_test = np.load(self.filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_AfterPCA79_20250113.npy", allow_pickle=True)
            return x_test, y_test
        elif split_file == 'test' and Choose_Attacktype == 'Evasion_Attack':
            print("Using CICIDS2017 with Evasion_Attack")
            # if (self.Attack_method == 'JSMA'):
            if (self.Attack_method == 'FGSM'):
            # 20241116 CIC-IDS2019 after do labelencode and minmax 75 25分 DO FGSM
                print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do FGSM attack")
                
            ################################# ALL feature minmax
            # if (self.Attack_method == 'PGD'):
            # if (self.Attack_method == 'CandW'):
            # 可在此加載相關的檔案
            return x_test, y_test
        elif split_file == 'test' and Choose_Attacktype == 'Poisoning_Attack':
            print("Using CICIDS2017 with Ponsion_Attack")
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
    elif choose_datasets == "CICIDS2018":
        loader = CICIDS2018TestLoader(filepath,Attack_method)
    elif choose_datasets == "CICIDS2017":
        loader = CICIDS2017TestLoader(filepath,Attack_method)
    elif choose_datasets == "TONIOT":
        loader = TONIOTTestLoader(filepath,Attack_method)
    else:
        raise ValueError("Unknown dataset type")

    x_test, y_test = loader.load_test_data(Choose_Attacktype, split_file)
    print("use file", split_file)
    return x_test, y_test