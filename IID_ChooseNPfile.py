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
def CICIDS2017_IID_ChooseLoadNpArray(filepath,split_file, Choose_method):

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
            # # 20250317使用是a=0.5 123 feature
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.5\\x_Dirichlet_client1_20250317.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.5\\y_Dirichlet_client1_20250317.npy", allow_pickle=True)
            # 20250319使用是a=0.5 79 feature Label merge
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and drop feature to 79 feature do Label group")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250319\\x_Dirichlet_client1_20250319.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250319\\y_Dirichlet_client1_20250319.npy", allow_pickle=True)
            # 20250320使用是 use 20250305 a=0.5 do featuremapping 123 feature and Label merge
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and use 2025025 a=0.5 do featuremapping 123 feature and Label merge")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250320\\x_Dirichlet_client1_feature_123_20250320.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250320\\y_Dirichlet_client1_feature_123_20250320.npy", allow_pickle=True)

            # # 20250317使用是a=0.1 123 feature
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.1\\x_Dirichlet_client1_20250317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.1\\y_Dirichlet_client1_20250317.npy", allow_pickle=True)

        print("train_half1 x_train 的形狀:", x_train.shape)
        print("train_half1 y_train 的形狀:", y_train.shape)
        client_str = "client1"
        print("使用 train_half1 進行訓練")
    elif split_file == 'client2_train':
        if (Choose_method == 'normal'):
            # 20250317使用是a=0.5 123 feature
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.5\\x_Dirichlet_client2_20250317.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.5\\y_Dirichlet_client2_20250317.npy", allow_pickle=True)

            # 20250319使用是a=0.5 79 feature Label merge
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and drop feature to 79 feature do Label group")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250319\\x_Dirichlet_client2_20250319.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250319\\y_Dirichlet_client2_20250319.npy", allow_pickle=True)
            # 20250320使用是 use 20250305 a=0.5 do featuremapping 123 feature and Label merge
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and use 2025025 a=0.5 do featuremapping 123 feature and Label merge")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250320\\x_Dirichlet_client2_feature_123_20250320.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250320\\y_Dirichlet_client2_feature_123_20250320.npy", allow_pickle=True)

            # 20250317使用是a=0.1 123 feature
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.1\\x_Dirichlet_client2_20250317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.1\\y_Dirichlet_client2_20250317.npy", allow_pickle=True)

        print("train_half2 x_train 的形狀:", x_train.shape)
        print("train_half2 y_train 的形狀:", y_train.shape)
        client_str = "client2"
        print("使用 train_half2 進行訓練")   
    # 20240121 CICIDS2017 after do labelencode and minmax  75 25分 drop feature to 79 feature
    # x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_test_Deleted79features_20250121.npy", allow_pickle=True)
    # # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_AfterDeleted79features_20250121_ChangeLabelencode.npy", allow_pickle=True)
    # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_Deleted79features_20250121.npy", allow_pickle=True)

    # 20250317 CIC-IDS2017 after do labelencode and except str and drop feature to 79 feature and all featrue minmax 75 25分
    # 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
    print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and drop feature to 79 feature do feature mapping to 123 feature")
    x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLday_test_featureMapping_20250317.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLday_test_featureMapping_20250317.npy", allow_pickle=True)
    # 20250319使用是a=0.5 79 feature Label merge
    # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and drop feature to 79 feature do Label group")
    # x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250319\\x_test_20250319.npy", allow_pickle=True)
    # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250319\\y_test_20250319.npy", allow_pickle=True)
    # 20250320使用是 use 20250305 a=0.5 do featuremapping 123 feature and Label merge
    # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and use 2025025 a=0.5 do featuremapping 123 feature and Label merge")
    # x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250320\\x_test_feature_123_20250320.npy", allow_pickle=True)
    # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\alpha_0.5\\20250320\\y_test_feature_123_20250320.npy", allow_pickle=True)

    print("use file", split_file)
    return x_train, y_train, x_test, y_test, client_str


def CICIDS2018_IID_ChooseLoadNpArray(filepath,split_file, Choose_method):

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
            # 20250306 CICIDS2018 使用a=0.5
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2018\csv_data\Dirichlet\20250306
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250306\\x_Dirichlet_client1_20250306.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250306\\y_Dirichlet_client1_20250306.npy", allow_pickle=True)

            # 20250329 CICIDS2018 使用a=0.1 after do labelencode 79 feature and all featrue minmax 75 25分
            # 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2018\csv_data\Dirichlet\20250329
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250329\\alpha_0.1\\x_Dirichlet_client1_20250329.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250329\\alpha_0.1\\y_Dirichlet_client1_20250329.npy", allow_pickle=True)


        print("train_half1 x_train 的形狀:", x_train.shape)
        print("train_half1 y_train 的形狀:", y_train.shape)
        client_str = "client1"
        print("使用 train_half1 進行訓練")
    elif split_file == 'client2_train':
        if (Choose_method == 'normal'):
            # 20250306 CICIDS2018 使用a=0.5
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2018\csv_data\Dirichlet\20250306
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250306\\x_Dirichlet_client2_20250306.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250306\\y_Dirichlet_client2_20250306.npy", allow_pickle=True)
            # 20250329 CICIDS2018 使用a=0.1 after do labelencode 79 feature and all featrue minmax 75 25分
            # 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2018\csv_data\Dirichlet\20250329
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250329\\alpha_0.1\\x_Dirichlet_client2_20250329.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250329\\alpha_0.1\\y_Dirichlet_client2_20250329.npy", allow_pickle=True)


        print("train_half2 x_train 的形狀:", x_train.shape)
        print("train_half2 y_train 的形狀:", y_train.shape)
        client_str = "client2"
        print("使用 train_half2 進行訓練")

    # elif split_file == 'client3_train':
    #     # if (Choose_method == 'normal'):
    #     #     # 20250306 CICIDS2018 使用a=0.5
    #     #     # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2018\csv_data\Dirichlet\20250306
    #     #     x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250306\\x_Dirichlet_client3_20250306.npy", allow_pickle=True)
    #     #     y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250306\\y_Dirichlet_client3_20250306.npy", allow_pickle=True)

    #     print("train_half3 x_train 的形狀:", x_train.shape)
    #     print("train_half3 y_train 的形狀:", y_train.shape)
    #     client_str = "client3"
    #     print("使用 train_half3 進行訓練")


    # x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_test_20250106.npy", allow_pickle=True)
    # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_20250106.npy", allow_pickle=True)
    # 0317 a=0.1
    x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_test_featureMapping_20250317.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_featureMapping_20250317.npy", allow_pickle=True)

    print("use file", split_file)
    return x_train, y_train, x_test, y_test, client_str


def ChooseLoad_class_names(str_choose_dataset):

    if str_choose_dataset == 'CICIDS2017':
        # 15個Label ver
        # class_names_local = {0: '0_BENIGN', 1: '1_Bot', 2: '2_DDoS', 3: '3_DoS GoldenEye', 4: '4_DoS Hulk', 5: '5_DoS Slowhttptest', 6: '6_DoS slowloris', 
        #                     7: '7_Infilteration', 8: '8_Web Attack', 9: '9_Heartbleed', 10: '10_PortScan', 11: '11_FTP-BruteForce', 12: '12_FTP-Patator', 
        #                     13: '13_SSH-Bruteforce', 14: '14_SSH-Patator'
        #                     }  

        # # 0317 a=0.5
        class_names_local = {0: 'Benign', 1: 'Bot', 2: 'DDoS', 3: 'DoS', 4: 'Infiltration', 5: 'Web Attack', 
                             6: 'FTP-Patator', 7: 'SSH-Patator', 8: 'Heartbleed', 9: 'PortScan'
                            }  
        labels_to_calculate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    
    elif str_choose_dataset == 'CICIDS2018':
        # 0329 a=0.1
        class_names_local = {0: 'Benign', 1: 'Bot', 2: 'DDoS', 3: 'DoS', 
                             4: 'Infiltration', 5: 'Web Attack',
                             6: 'FTP-BruteForce', 7: 'SSH-Bruteforce'
                            }  
        labels_to_calculate = [0, 1, 2, 3, 4, 5, 6, 7]


    elif str_choose_dataset == 'TONIOT':
        class_names_local = {0: 'Normal', 1: 'Backdoor', 2: 'DDoS', 3: 'DoS', 
                             4: 'Injection', 5: 'Mitm',6: 'Password', 
                             7: 'Ransomware',8: 'Scanning', 9: 'XSS'
                            }  
        labels_to_calculate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


    return class_names_local, labels_to_calculate