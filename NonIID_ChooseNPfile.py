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
def CICIDS2017_NonIID_ChooseLoadNpArray(filepath,split_file, Choose_method):

    if split_file == 'client1_train':
        if (Choose_method == 'normal'):
            
            # 使用total train 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017 total train" +f"{split_file} with normal After Do labelencode and minmax 79 feature do feature mapping to 123 feature")
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\Noniid\\CICIDS2017_AddedLabel_Noniid_featureMapping_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\Noniid\\CICIDS2017_AddedLabel_Noniid_featureMapping_y.npy", allow_pickle=True)


            # # 20250317使用是a=0.1 123 feature
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and a=0.1 123 feature do Label group")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.1\\Noniid\\client1_Dirichlet_Added_Noniid_Label_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.1\\Noniid\\client1_Dirichlet_Added_Noniid_Label_y.npy", allow_pickle=True)

            # # # 20250317使用是a=0.5 123 feature
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and a=0.5 123 feature do Label group")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.5\\Noniid\\client1_Dirichlet_Added_Noniid_Label_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.5\\Noniid\\client1_Dirichlet_Added_Noniid_Label_y.npy", allow_pickle=True)


        print("train_half1 x_train 的形狀:", x_train.shape)
        print("train_half1 y_train 的形狀:", y_train.shape)
        client_str = "client1"
        print("使用 train_half1 進行訓練")
    elif split_file == 'client2_train':
        if (Choose_method == 'normal'):
            # 20250317使用是a=0.1 123 feature
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and a=0.1 123 feature do Label group")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.1\\Noniid\\client2_Dirichlet_Added_Noniid_Label_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.1\\Noniid\\client2_Dirichlet_Added_Noniid_Label_y.npy", allow_pickle=True)

            # 20250317使用是a=0.5 123 feature
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and a=0.5 123 feature do Label group")
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.5\\Noniid\\client2_Dirichlet_Added_Noniid_Label_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250317\\alpha_0.5\\Noniid\\client2_Dirichlet_Added_Noniid_Label_y.npy", allow_pickle=True)

        print("train_half2 x_train 的形狀:", x_train.shape)
        print("train_half2 y_train 的形狀:", y_train.shape)
        client_str = "client2"
        print("使用 train_half2 進行訓練")   
    
    # 20250317 CIC-IDS2017 after do labelencode and except str and drop feature to 79 feature and all featrue minmax 75 25分
    # 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
    print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"{split_file} with normal After Do labelencode and minmax and drop feature to 79 feature do feature mapping to 123 feature")
    x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLday_test_featureMapping_20250317.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLday_test_featureMapping_20250317.npy", allow_pickle=True)

    print("use file", split_file)
    return x_train, y_train, x_test, y_test, client_str


def CICIDS2018_NonIID_ChooseLoadNpArray(filepath,split_file, Choose_method):

    if split_file == 'client1_train':
        if (Choose_method == 'normal'):
            # 20250329 CICIDS2018 使用a=0.5 after do labelencode 79 feature and all featrue minmax 75 25分
            # 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2018\csv_data\Dirichlet\20250329
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018 a=0.5" +f"{split_file} with normal After Do labelencode and minmax 79 feature do feature mapping to 123 feature")
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250329\\alpha_0.5\\Noniid\\client1_Dirichlet_Added_Noniid_Label_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250329\\alpha_0.5\\Noniid\\client1_Dirichlet_Added_Noniid_Label_y.npy", allow_pickle=True)

        print("train_half1 x_train 的形狀:", x_train.shape)
        print("train_half1 y_train 的形狀:", y_train.shape)
        client_str = "client1"
        print("使用 train_half1 進行訓練")
    elif split_file == 'client2_train':
        if (Choose_method == 'normal'):
            # 使用total train 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
            print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018 total train" +f"{split_file} with normal After Do labelencode and minmax 79 feature do feature mapping to 123 feature")
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\Noniid\\CICIDS2018_AddedLabel_Noniid_featureMapping_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\Noniid\\CICIDS2018_AddedLabel_Noniid_featureMapping_y.npy", allow_pickle=True)

            # 20250329 CICIDS2018 使用a=0.5 after do labelencode 79 feature and all featrue minmax 75 25分
            # 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2018\csv_data\Dirichlet\20250329
            # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018 a=0.5" +f"{split_file} with normal After Do labelencode and minmax 79 feature do feature mapping to 123 feature")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250329\\alpha_0.5\\Noniid\\client2_Dirichlet_Added_Noniid_Label_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\20250329\\alpha_0.5\\Noniid\\client2_Dirichlet_Added_Noniid_Label_y.npy", allow_pickle=True)

        print("train_half2 x_train 的形狀:", x_train.shape)
        print("train_half2 y_train 的形狀:", y_train.shape)
        client_str = "client2"
        print("使用 train_half2 進行訓練")

    x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_test_featureMapping_20250317.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_featureMapping_20250317_ChangeLabelencode.npy", allow_pickle=True)

    print("use file", split_file)
    return x_train, y_train, x_test, y_test, client_str

def TONIOT_NonIID_ChooseLoadNpArray(filepath,split_file, Choose_method):
    if split_file == 'client1_train':
        if (Choose_method == 'normal'):
            # 20250414 TONIOT 使用a=0.1 after do labelencode 44 feature and all featrue minmax 75 25分
            # 44 feature use Label meraged BaseLine data do feature mapping to 123 feature
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\TONIOT\Dirichlet\20250414
            print(Fore.BLUE+Style.BRIGHT+"Loading TONIOT a=0.1" +f"{split_file} with normal After Do labelencode and minmax 44 feature do feature mapping to 123 feature")
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Dirichlet\\20250414\\alpha_0.1\\Noniid\\client1_Dirichlet_Added_Noniid_Label_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Dirichlet\\20250414\\alpha_0.1\\Noniid\\client1_Dirichlet_Added_Noniid_Label_y.npy", allow_pickle=True)

        print("train_half1 x_train 的形狀:", x_train.shape)
        print("train_half1 y_train 的形狀:", y_train.shape)
        client_str = "client1"
        print("使用 train_half1 進行訓練")
    elif split_file == 'client2_train':
        if (Choose_method == 'normal'):
            print(Fore.BLUE+Style.BRIGHT+"Loading TONIOT total train" +f"{split_file} with normal After Do labelencode and minmax 79 feature do feature mapping to 123 feature")
            # 44 feature use Label meraged BaseLine data do feature mapping to 123 feature
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Npfile\\Noniid\\TONIIOT_AddedLabel_featureMapping_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Npfile\\Noniid\\TONIIOT_AddedLabel_featureMapping_y.npy", allow_pickle=True)

            # 20250414 TONIOT 使用a=0.1 after do labelencode 44 feature and all featrue minmax 75 25分
            # 44 feature use Label meraged BaseLine data do feature mapping to 123 feature
            # D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\TONIOT\Dirichlet\20250414
            # print(Fore.BLUE+Style.BRIGHT+"Loading TONIOT a=0.1" +f"{split_file} with normal After Do labelencode and minmax 44 feature do feature mapping to 123 feature")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Dirichlet\\20250414\\alpha_0.1\\Noniid\\client2_Dirichlet_Added_Noniid_Label_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Dirichlet\\20250414\\alpha_0.1\\Noniid\\client2_Dirichlet_Added_Noniid_Label_y.npy", allow_pickle=True)

        print("train_half2 x_train 的形狀:", x_train.shape)
        print("train_half2 y_train 的形狀:", y_train.shape)
        client_str = "client2"
        print("使用 train_half2 進行訓練")

    # 44 feature use Label meraged BaseLine data do feature mapping to 123 feature
    # 20250317 TONIoT after do labelencode and all featrue minmax 75 25分 44 feature do backdoor和ddos互相更換encode值 feature mapping to 123 feature
    # labels_to_calculate = [0, 2, 3, 12, 13, 14, 15, 16, 17, 18]
    # print(Fore.BLUE+Style.BRIGHT+"Loading TONIOT" +f"test with normal After Do labelencode and minmax")
    x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Npfile\\x_TONIOT_test_featureMapping_20250317.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Npfile\\y_TONIOT_test_featureMapping_20250317_ChangeLabelEncode_for_Noniid.npy", allow_pickle=True)


    print("use file", split_file)
    return x_train, y_train, x_test, y_test, client_str

def NonIID_ChooseLoad_class_names(str_choose_dataset):

    if str_choose_dataset == 'CICIDS2017':
        class_names_local = {0: 'Benign', 1: 'Bot', 2: 'DDoS', 3: 'DoS', 4: 'Infiltration', 5: 'Web Attack', 
                             6: 'FTP-Patator', 7: 'SSH-Patator', 8: 'Heartbleed', 9: 'PortScan'
                            }  
        # labels_to_calculate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    
    elif str_choose_dataset == 'CICIDS2018':
        class_names_local = {0: 'Benign', 1: 'Bot', 2: 'DDoS', 3: 'DoS', 
                             4: 'Infiltration', 5: 'Web Attack',
                             10: 'FTP-BruteForce', 11: 'SSH-Bruteforce'
                            }  
        # labels_to_calculate = [0, 1, 2, 3, 4, 5, 10, 11]


    elif str_choose_dataset == 'TONIOT':
        class_names_local = {0: 'Benign', 12: 'Backdoor', 2: 'DDoS', 3: 'DoS', 
                             13: 'Injection', 14: 'Mitm', 15: 'Password', 
                             16: 'Ransomware',17: 'Scanning', 18: 'XSS'
                            }  
        # labels_to_calculate = [0, 2, 3, 12, 13, 14, 15, 16, 17, 18]
    
    class_names_global = {
                                0: 'Benign', 
                                1: 'Bot', 
                                2: 'DDoS', 
                                3: 'DoS', 
                                4: 'Infiltration', 
                                5: 'Web Attack', 
                                6: 'FTP-Patator', 
                                7: 'SSH-Patator', 
                                8: 'Heartbleed', 
                                9: 'PortScan', 
                                10: 'FTP-BruteForce', 
                                11: 'SSH-Bruteforce', 
                                12: 'Backdoor', 
                                13: 'Injection', 
                                14: 'Mitm',
                                15: 'Password',
                                16: 'Ransomware',
                                17: 'Scanning',
                                18: 'XSS'
                                } 

    # return class_names_local, labels_to_calculate,  class_names_global
    return class_names_local,  class_names_global