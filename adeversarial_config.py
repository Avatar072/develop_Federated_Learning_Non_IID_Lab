import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from art.attacks.evasion import ProjectedGradientDescent,FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from art.defences.preprocessor import GaussianAugmentation
from art.estimators.classification import PyTorchClassifier
import os
import random
import time
import datetime
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.metrics import classification_report
from mytoolfunction import ChooseUseModel, getStartorEndtime
from mytoolfunction import generatefolder, SaveDataToCsvfile, SaveDataframeTonpArray
from IID_ChooseNPfile import CICIDS2017_IID_ChooseLoadNpArray,ChooseLoad_class_names,CICIDS2018_IID_ChooseLoadNpArray,TONIOT_IID_ChooseLoadNpArray
from colorama import Fore, Back, Style, init
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
    # 每層神經元512下所訓練出來的正常model
    # CICIDS2019 load model
    # string feature 未做minmax
    # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2019\\BaseLine_After_local_train_model_bk.pth'
    # all_feature_minmax    
    # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2019\\20241119\\all_feature_minmax_baseline\\BaseLine\\BaseLine_After_local_train_model_bk.pth'
    
def SettingAderversarialConfig(choose_dataset):
    if choose_dataset == "CICIDS2017":
        # CICIDS2017 load model
        # 設定 FGSM 攻擊 eps
        # epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
        epsilons = [0.01]
        # epsilons = [0.05]
        # epsilons = [0.5]
        # BaseLine的正常model
        # feature 79
        # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2017\\20250121\\79feature\\BaseLine_After_local_train_model_bk.pth'
        # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2017\\20250317_Label_merge後baseLine\\BaseLine_After_local_train_model_79_feature_Label_merge.pth'
        # feature 123
        model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2017\\20250317\\BaseLine_After_local_train_model_feature123.pth'
        # client1 的正常model
        # feature 123
        # model_path = 'E:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20250219\\Local_model_After_local_train_model_300_bk.pth'
        # feature 79
        # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20250317\\Local_model_After_local_train_300_bk.pth'
        # labelCount = 15
        labelCount = 10
        # class_names = {
        #                         # CICIDS2017
        #                         0: '0_BENIGN', 
        #                         1: '1_Bot', 
        #                         2: '2_DDoS', 
        #                         3: '3_DoS GoldenEye', 
        #                         4: '4_DoS Hulk', 
        #                         5: '5_DoS Slowhttptest', 
        #                         6: '6_DoS slowloris', 
        #                         7: '7_FTP-Patator', 
        #                         8: '8_Heartbleed', 
        #                         9: '9_Infiltration', 
        #                         10: '10_PortScan', 
        #                         11: '11_SSH-Patator', 
        #                         12: '12_Web Attack Brute Force', 
        #                         13: '13_Web Attack Sql Injection', 
        #                         14: '14_Web Attack XSS'
        #                         } 
        class_names = {
                        0: 'Benign', 
                        1: 'Bot', 
                        2: 'DDoS', 
                        3: 'DoS', 
                        4: 'Infiltration', 
                        5: 'Web Attack', 
                        6: 'FTP-Patator', 
                        7: 'SSH-Patator', 
                        8: 'Heartbleed', 
                        9: 'PortScan'
                        } 
        # 20250121 CIC-IDS2017 after do labelencode and except str and drop feature to 79 feature and all featrue minmax 75 25分
        # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f" with normal After Do labelencode and minmax and drop feature to 79 feature")
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_train_Deleted79features_20250121.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_Deleted79features_20250121.npy", allow_pickle=True)
        # x_train, y_train, client_str = ChooseLoadNpArray(filepath, 'client1_train', 'normal')
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250205\\x_Dirichlet_client1_20250205.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Dirichlet\\20250205\\y_Dirichlet_client1_20250205.npy", allow_pickle=True)
        # x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_test_Deleted79features_20250121.npy", allow_pickle=True)
        # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_Deleted79features_20250121.npy", allow_pickle=True)
        # CICIDS2017
        print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f" with normal After Do labelencode and minmax and drop feature to 79 feature do feature mapping to 123 feature")
        x_train, y_train, x_test, y_test, client_str = CICIDS2017_IID_ChooseLoadNpArray(filepath,'client1_train', 'normal')
        class_names, labels_to_calculate = ChooseLoad_class_names("CICIDS2017")
    elif choose_dataset == "CICIDS2018":
        # CICIDS2018 load model
        # 設定 FGSM 攻擊 eps
        # epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
        # epsilons = [0.01, 0.5, 1.0]
        epsilons = [0.01]
        # epsilons = [0.05]
        # epsilons = [0.5]
        # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2018\\20250106\\only_minmax\\BaseLine_After_local_train_model_bk.pth'
        # D:\develop_Federated_Learning_Non_IID_Lab\single_AnalyseReportFolder\CICIDS2018\20250106\only_minmax\BaseLine\normal
        labelCount = 8
        # 20250121 CIC-IDS2018 after do labelencode and except str and drop feature to 79 feature and all featrue minmax 75 25分
        # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f" with normal After Do labelencode and minmax and drop feature to 79 feature")
        model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2018\\20250317\\BaseLine_After_local_train_model_feature123.pth'
        # 20250329 CIC-IDS2018 after do labelencode and 79 feature and all featrue minmax 75 25分 featuremapping 123
        print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f" with normal After Do labelencode and minmax and 79 featuremapping 123e")
        # CICIDS2018
        x_train, y_train, x_test, y_test, client_str = CICIDS2018_IID_ChooseLoadNpArray(filepath,'client1_train', 'normal')
        class_names, labels_to_calculate = ChooseLoad_class_names("CICIDS2018")    
    elif choose_dataset == 'TONIOT':
        # TON_IOT load model
        # string feature 未做minmax
        # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\TONIOT\\20241229\\BaseLine_After_local_train_model_StringNoMinmax_20241229.pth'
        # all_feature_minmax
        # 設定 FGSM 攻擊 eps
        # epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
        epsilons = [0.01]
        # epsilons = [0.05]
        # epsilons = [0.5]

        labelCount = 10
        print(Fore.BLUE+Style.BRIGHT+"Loading TONIOT" +f" with normal After Do labelencode and minmax and 44 featuremapping 123 feature")
        model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\TONIOT\\20250317\\BaseLine_After_local_train_model_feature123.pth'

        
        # 20250317 TONIOT after do labelencode and 79 feature and all featrue minmax 75 25分 featuremapping 123
        x_train, y_train, x_test, y_test, client_str = TONIOT_IID_ChooseLoadNpArray(filepath,'client1_train', 'normal')
        class_names, labels_to_calculate = ChooseLoad_class_names("TONIOT")    
   

    return epsilons, model_path, labelCount, class_names, labels_to_calculate, x_train, y_train, x_test, y_test
