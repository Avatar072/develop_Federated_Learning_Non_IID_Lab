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
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F

# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)

today = datetime.date.today()
today = today.strftime("%Y%m%d")

### 生成列名列表
column_names = ["principal_Component" + str(i) for i in range(1, 79)] + ["Label"]

### 獲取開始or結束時間
def getStartorEndtime(Start_or_End_Str,Start_or_End,fliepath):
    timestamp = time.time()# 用於獲取當前時間的時間戳，返回一個浮點數

    # 將時間戳轉換為日期時間物件
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    
    # 格式化日期时间為 "yyyy-mm-dd hh:mm:ss"
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")

    print(f"{Start_or_End}_time: {formatted_time}")
    # 開啟檔案並寫入開始時間
    with open(f"{fliepath}/StartTime_and_Endtime.csv", "a+") as file:
        file.write(f"{Start_or_End_Str}:{str(formatted_time)}\n")

    return timestamp, formatted_time
### 計算花費時間
def CalculateTime(end_IDS, start_IDS):
    #  end_IDS = time.time()
     execution_time = end_IDS - start_IDS
     print(f"Code execution time: {execution_time} seconds")
    
### 檢查資料夾是否存在 回傳True表示沒存在
def CheckFolderExists (folder_name):
    if not os.path.exists(folder_name):
        return True
    else:
        return False
    
### 檢查檔案是否存在
def CheckFileExists (file):
   if os.path.isfile(file):
    print(f"{file} 是一個存在的檔案。")
    return True
   else:
    print(f"{file} 不是一個檔案或不存在。")
    return False
    
### Save data to csv
def SaveDataToCsvfile(df, folder_name, filename):
    # 抓取當前工作目錄名稱
    current_directory = os.getcwd()
    print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+"當前工作目錄", Fore.YELLOW+Style.BRIGHT+current_directory)
    # folder_name = filename + "_folder"
    print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+"資料夾名稱"+folder_name)
    folder_name = generatefolder(current_directory + "\\",folder_name)
    csv_filename = os.path.join(current_directory, 
                                folder_name, filename + ".csv")
    print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+"存檔位置跟檔名"+csv_filename)
    df.to_csv(csv_filename, index=False)

### 建立一個資料夾
def generatefolder(fliepath, folder_name):
    if fliepath is None:
        fliepath = os.getcwd()
        print("當前工作目錄", fliepath) 
    
    if folder_name is None:
        folder_name = "my_AnalyseReportfolder"

    file_not_exists  = CheckFolderExists(fliepath +folder_name)
    print("file_not_exists:",file_not_exists)
    # 使用os.path.exists()檢文件夹是否存在
    if file_not_exists:
        # 如果文件夹不存在，就创建它
        os.makedirs(fliepath + folder_name)
        print(f"資料夾 '{fliepath +folder_name}' 創建。")
    else:
        print(f"資料夾 '{fliepath +folder_name}' 已存在，不需再創建。")
        
    return folder_name
### 合併DataFrame成csv
def mergeDataFrameAndSaveToCsv(trainingtype,x_train,y_train, filename, weaklabel=0, epochs=500):
    # 创建两个DataFrame分别包含x_train和y_train
    df_x_train = pd.DataFrame(x_train)                     # 特徵數據
    df_y_train = pd.DataFrame(y_train, columns=['Label'])  # 標籤數據

    # 使用concat函数将它们合并 axis=1 按列合併（水平合併）
    generateNewdata = pd.concat([df_x_train, df_y_train], axis=1)

    # 保存合并后的DataFrame为CSV文件
    if trainingtype == "GAN":
        generateNewdata.columns = column_names
        SaveDataToCsvfile(generateNewdata, f"{trainingtype}_data_{filename}", f"{trainingtype}_data_generate_weaklabel_{weaklabel}_epochs_{epochs}")
    elif trainingtype == "SMOTE":
        SaveDataToCsvfile(generateNewdata, f"{filename}_epochs_{epochs}", f"{trainingtype}_data_generate_weaklabel_{weaklabel}_epochs_{epochs}")
    else:
        SaveDataToCsvfile(generateNewdata, f"{trainingtype}",f"{filename}")

def ParseCommandLineArgs(commands):
    
    # e.g
    # python BaseLine.py -h
    # python BaseLine.py --dataset train_half1
    # python BaseLine.py --dataset train_half2
    # python BaseLine.py --epochs 100
    # python BaseLine.py --dataset train_half1 --epochs 100
    # python DoGAN.py --dataset train_half1 --epochs 10 --weaklabel 8
    # default='train_half1'
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Federated Learning Client')

    # 添加一个参数来选择数据集
    # parser.add_argument('--dataset_split', type=str, choices=['total_train','train_half1', 'train_half2', 'train_half3'], default='total_train',
    #                     help='Choose the dataset for training (total_train or train_half1 or train_half2 or train_half3)')
    parser.add_argument('--dataset_split', type=str, choices=['baseLine_train','client1_train', 'client2_train', 'client3_train'], default='total_train',
                        help='Choose the dataset for training (total_train or train_half1 or train_half2 or train_half3)')
    # 添加一个参数来设置训练的轮数
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

     # 添加一个参数来设置训练的轮数
    parser.add_argument('--weaklabel', type=int, default=8, help='encode of weak label')

    # add load method
    parser.add_argument('--method', type=str, choices=['normal','SMOTE', 'GAN','Evasion_Attack','Poisoning_Attack'], default='normal',
                        help='Choose the process method for training (normal or SMOTE or GAN)')
    
    # add choose dataset
    parser.add_argument('--Load_dataset', type=str, choices=['CICIDS2017','CICIDS2018', 'CICIDS2019', 'TONIOT'], 
                        default='CICIDS2017',
                        help='Load Choose the dataset')
    # 解析命令行参数
    args = parser.parse_args()

    # 根据输入的命令列表来确定返回的参数
    if 'Load_dataset' in commands and 'dataset_split' in commands and 'epochs' in commands and 'method' in commands:
        return args.Load_dataset,args.dataset_split,args.epochs, args.method

    if 'dataset_split' in commands and 'epochs' in commands and 'method' in commands:
        return args.dataset_split, args.epochs, args.method
    elif 'dataset_split' in commands and 'epochs' in commands and 'weaklabel' in commands:
        return args.dataset_split, args.epochs, args.weaklabel
    elif 'dataset_split' in commands and 'epochs' in commands:
        return args.dataset_split, args.epochs
    elif 'dataset_split' in commands:
        return args.dataset_split
    elif 'dataset_split' in commands:
        return args.epochs

# 测试不同的命令
# print(ParseCommandLineArgs(['dataset']))
# print(ParseCommandLineArgs(['epochs']))
# print(ParseCommandLineArgs(['dataset', 'epochs']))
# print(ParseCommandLineArgs(['dataset', 'epochs', 'label']))

### Choose Load np array
def ChooseLoadNpArray(filepath,split_file, Choose_method):

    if split_file == 'total_train':
        print("Training with total_train")
        if (Choose_method == 'normal'):
            # x_train = np.load(filepath + "x_train_1.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_train_1.npy", allow_pickle=True)
            # 20231113 only do labelencode and minmax
            # x_train = np.load(filepath + "x_train_20231113.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_train_20231113.npy", allow_pickle=True)

            # 20240323 after Chi-square45
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_train_cicids2017_AfterFeatureSelect44_BaseLine_SplitIP_20240323.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_train_cicids2017_AfterFeatureSelect44_BaseLine_SplitIP_20240323.npy", allow_pickle=True)

            # 20240325 after PCA45
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_train_AfterPCA38_20240325.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_train_AfterPCA38_20240325.npy", allow_pickle=True)


            # # 20231220 after PCA do labelencode and minmax
            # x_train = np.load(filepath + "x_train_ToN-IoT_afterPCA_20231220.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_train_ToN-IoT_afterPCA_20231220.npy", allow_pickle=True)
            
            # # 20240502 CIC-IDS2017 after do labelencode and minmax  75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_train_20240502.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_train_20240502.npy", allow_pickle=True)    
            
            # # 20240502 CIC-IDS2017 after do labelencode and minmax chi_square45 75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLday_train_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLday_train_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)    
            
            # 20240422 CIC-IDS2019 after do labelencode and minmax 
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_CICIDS2019_01_12_train_20240422.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_CICIDS2019_01_12_train_20240422.npy", allow_pickle=True)
            # 20240422 CIC-IDS2019 after do labelencode and minmax chi-square 45
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_CICIDS2019_AfterFeatureSelect44_20240422.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_CICIDS2019_AfterFeatureSelect44_20240422.npy", allow_pickle=True)

            # if choose_datasets == "CICIDS2019":
                # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_CICIDS2019_01_12_train_20240422.npy", allow_pickle=True)
                # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_CICIDS2019_01_12_train_20240422.npy", allow_pickle=True)
                # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
                # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_20240502.npy", allow_pickle=True)
                # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_20240502.npy", allow_pickle=True)
                # 20241030 CIC-IDS2019 after do labelencode and minmax 75 25分 do GDA 高斯資料增強
                # x_train = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/x_CICIDS2019_train_augmented.npy", allow_pickle=True)
                # y_train = np.load(f"./Adversarial_Attack_Denfense/CICIDS2019/y_CICIDS2019_train_augmented.npy", allow_pickle=True)
            # 20240519 EdgeIIoT after do labelencode and minmax  75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_train_20240519.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_20240519.npy", allow_pickle=True)    
            
            # 20240520 EdgeIIoT after do labelencode and minmax chi_square45 75 25分
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_train_AfterFeatureSelect44_20240520.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_AfterFeatureSelect44_20240520.npy", allow_pickle=True)    
            # if choose_datasets == "TONIOT":
            # # 20240523 TONIoT after do labelencode and minmax  75 25分
            #     x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_20240523.npy", allow_pickle=True)
            #     y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_20240523.npy", allow_pickle=True)  

        elif (Choose_method == 'SMOTE'):
            # x_train = np.load(filepath + "x_total_train_SMOTE_ALL_Label.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_SMOTE_ALL_Label.npy", allow_pickle=True)
            # x_train = np.load(filepath + "x_total_train_SMOTE_ALL_Label14.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_SMOTE_ALL_Label14.npy", allow_pickle=True)
            # x_train = np.load(filepath + "x_train_half1_20231114.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_train_half1_20231114.npy", allow_pickle=True)
            # # # 20231207 borderLineSMOTE2 Lable8 k=5 and Label 9 k=5 Label 9 Label13 k=5 M = 10 4000
            # x_train = np.load(filepath + "x_total_train_BorederlineSMOTE_borderline-2_Label8_and_Label9_Label13_20231207.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_BorederlineSMOTE_borderline-2_Label8_and_Label9_Label13_20231207.npy", allow_pickle=True)
            # # # # 20231208 borderLineSMOTE1 Lable8 k=5 and Label 9 k=5 Label 9 Label13 k=5 M = 10 4000
            # x_train = np.load(filepath + "x_total_train_BorederlineSMOTE_borderline-1_Label8_and_Label9_Label13_20231208.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_BorederlineSMOTE_borderline-1_Label8_and_Label9_Label13_20231208.npy", allow_pickle=True)
            
            # # 20231220 after do labelencode and minmax
            # # # # # # 20231208 SMOTE Lable1 k=5 and Label3 k=5 Label6  k=5 10000
            # x_train = np.load(filepath + "x_total_train_SMOTE_Label1_and_Label3_Label6_20231225.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_SMOTE_Label1_and_Label3_Label6_20231225.npy", allow_pickle=True)
            # # 20231225 borderLineSMOTE1 Lable1 k=5 and Label 3 k=5 Label 6 k=5 M = 10 10000
            # x_train = np.load(filepath + "x_total_train_BorederlineSMOTE_borderline-1_Label1_and_Label3_Label6_20231225.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_BorederlineSMOTE_borderline-1_Label1_and_Label3_Label6_20231225.npy", allow_pickle=True)
            # # 20231225 borderLineSMOTE2 Lable1 k=5 and Label 3 k=5 Label 6 k=5 M = 10 10000
            # x_train = np.load(filepath + "x_total_train_BorederlineSMOTE_borderline-2_Label1_and_Label3_Label6_20231225.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_BorederlineSMOTE_borderline-2_Label1_and_Label3_Label6_20231225.npy", allow_pickle=True)
            
            # # # # # # 20231227 SMOTE Lable1 Label3 Label4 Label6  Label9 k=5
            x_train = np.load(filepath + "x_total_train_SMOTE_Label1_and_Label3_Label4_Label6_Label9_20231227.npy", allow_pickle=True)
            y_train = np.load(filepath + "y_total_train_SMOTE_Label1_and_Label3_Label4_Label6_Label9_20231227.npy", allow_pickle=True)

            #  # # # # # 20231227 BorederlineSMOTE_borderline-1 Lable1 Label3 Label4 Label6  Label9 k=5 M=10
            # x_train = np.load(filepath + "x_total_train_BorederlineSMOTE_borderline-1_Label1_and_Label3_Label4_Label6_Label9_20231227.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_BorederlineSMOTE_borderline-1_Label1_and_Label3_Label4_Label6_Label9_20231227.npy", allow_pickle=True)
            
             # # # # # 20231227 BorederlineSMOTE_borderline-2 Lable1 Label3 Label4 Label6  Label9 k=5 M=10
            # x_train = np.load(filepath + "x_total_train_BorederlineSMOTE_borderline-2_Label1_and_Label3_Label4_Label6_Label9_20231227.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_BorederlineSMOTE_borderline-2_Label1_and_Label3_Label4_Label6_Label9_20231227.npy", allow_pickle=True)
            
        elif (Choose_method == 'GAN'):
            # x_train = np.load(filepath + "x_GAN_data_total_train_weakpoint_14.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_GAN_data_total_train_weakpoint_14.npy", allow_pickle=True)
            x_train = np.load(filepath + "x_train_20231106_afterGAN_Label14.npy", allow_pickle=True)
            # 將複數的資料實部保留並轉換為浮點：
            x_train = x_train.real.astype(np.float64)
            y_train = np.load(filepath + "y_train_20231106_afterGAN_Label14.npy", allow_pickle=True)
        
        client_str = "BaseLine"
        print(Choose_method)

    elif split_file == 'client1_train':
        if (Choose_method == 'normal'):
            # # 20240110 non iid client1 use cicids2017 after chi-square
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\x_train_CICIDS2017_20240110.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\y_train_CICIDS2017_20240110.npy", allow_pickle=True)
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\x_train_CICIDS2017_addlossvalue_20240110.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\y_train_CICIDS2017_addlossvalue_20240110.npy", allow_pickle=True)
            # # 20240316 non iid client1 use cicids2017 Monday_and_Firday after chi-square
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_Monday_and_Firday_train_cicids2017_AfterFeatureSelect44_20240316.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_Monday_and_Firday_train_cicids2017_AfterFeatureSelect44_20240316.npy", allow_pickle=True)
            # # 20240314 non iid client1 use cicids2017 Monday_and_Firday after PCA
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_Monday_and_Firday_train_AfterPCA38_20240314.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_Monday_and_Firday_train_AfterPCA38_20240314.npy", allow_pickle=True)
            # # 20240315 non iid client1 use cicids2017 Monday_and_Firday after PCA
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_Monday_and_Firday_train_AfterPCA77_20240315.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_Monday_and_Firday_train_AfterPCA77_20240315.npy", allow_pickle=True)
            # # 20240317 non iid client1 use cicids2017 Monday_and_Firday after chi-square 45 add tonniot
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_Monday_and_Firday_train_dataframes_AfterFeatureSelect44_ADD_TONIOT_20240316.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_Monday_and_Firday_train_dataframes_AfterFeatureSelect44_ADD_TONIOT_20240316.npy", allow_pickle=True)
            # # 20240317 non iid client1 use cicids2017 Monday_and_Firday after chi-square 45 add tonniot remove all IP port
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_Monday_and_Firday_train_dataframes_AfterFeatureSelect44_ADD_TONIOT_rmove_ip_port_20240316.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_Monday_and_Firday_train_dataframes_AfterFeatureSelect44_ADD_TONIOT_rmove_ip_port_20240316.npy", allow_pickle=True)
            # # 20240317 non iid client1 use cicids2017 Monday_and_Firday  tonniot add cicids2017 39 feature then PCA 
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_Monday_and_Firday_train_AfterPCA77_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_Monday_and_Firday_train_AfterPCA77_20240317.npy", allow_pickle=True)
            # # 20240317 non iid client1 use cicids2017 Monday_and_Firday PCA 38
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_Monday_and_Firday_train_AfterPCA38_20240318.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_Monday_and_Firday_train_AfterPCA38_20240318.npy", allow_pickle=True)
             # # 20240319 non iid client1 use cicids2017 ALLday  chi-square_45
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_train_cicids2017_AfterFeatureSelect44_20240319.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_train_cicids2017_AfterFeatureSelect44_20240319.npy", allow_pickle=True)
            # 20240323 non iid client1 use cicids2017 ALLday  chi-square_45 change ip encode
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_train_dataframes_AfterFeatureSelect_Noniid_change_ip_20240323.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_train_dataframes_AfterFeatureSelect_Noniid_change_ip_20240323.npy", allow_pickle=True)
            # 20240428 non iid client1 use cicids2017 ALLday  chi-square_45
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_train_cicids2017_AfterFeatureSelect44_20240428.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_train_cicids2017_AfterFeatureSelect44_20240428.npy", allow_pickle=True)

            # 20240502 after do labelencode and minmax cicids2017 ALLDay  iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2017\\ALLday\\x_ALLDay_train_half1_20240502.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2017\\ALLday\\y_ALLDay_train_half1_20240502.npy", allow_pickle=True)

            # 20240503 after do labelencode and minmax cicids2019 01_12  iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2019\\01_12\\x_01_12_train_half1_20240503.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2019\\01_12\\y_01_12_train_half1_20240503.npy", allow_pickle=True)
            
            # 20240505 non iid after do labelencode and minmax chi-square_45 cicids2017 ALLday
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2017\\ALLday\\CICIDS2017_AddedLabel_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2017\\ALLday\\CICIDS2017_AddedLabel_y.npy", allow_pickle=True)

            # 20240507 after do labelencode and minmax EdgeIIoT iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_train_half1_20240507.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_half1_20240507.npy", allow_pickle=True)

            # 20240507 after do labelencode and minmax Kub iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\Kub\\x_Kub_train_half1_20240507.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\Kub\\y_Kub_train_half1_20240507.npy", allow_pickle=True)

            # 20240507 after do labelencode and minmax Wustl iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\Wustl\\x_Wustl_train_half1_20240507.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\Wustl\\y_Wustl_train_half1_20240507.npy", allow_pickle=True)

            # 20240523 non iid after do labelencode and minmax chi-square_45 cicids2017 ALLday
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\CICIDS2017_AddedLabel_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\CICIDS2017_AddedLabel_y.npy", allow_pickle=True)

            # 20240523 client1 use TONIoT after do labelencode and minmax  均勻劃分75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_train_half1_20240523.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_train_half1_20240523.npy", allow_pickle=True)  

            # 20240523 client1 use TONIoT after do labelencode and minmax  隨機劃分75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_random_train_half1_20240523.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_random_train_half1_20240523.npy", allow_pickle=True)  

            # 20250113 CIC-IDS2017 after do labelencode all featrue minmax 75 25分 do PCA Non-iid 
            # 20250121 CIC-IDS2017 after do labelencode and all featrue minmax 75 25分 do Do feature drop to 79 feature 
            # Non-iid
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\Noniid\\CICIDS2017_AddedLabel_Noniid_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\Noniid\\CICIDS2017_AddedLabel_Noniid_y.npy", allow_pickle=True)

        elif (Choose_method == 'SMOTE'):
            # # # # 20240317 Chi-square 45 SMOTE  K=5         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_train_half1_SMOTE_Monday_and_Firday_ALL_Label_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_train_half1_SMOTE_Monday_and_Firday_ALL_Label_20240317.npy", allow_pickle=True)
            # # # 20240317 Chi-square 45 BL-SMOTE1  K=5  M = 10         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_borderline-1_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_borderline-1_20240317.npy", allow_pickle=True)
            # # # 20240317 Chi-square 45 BL-SMOTE2  K=5  M = 10         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_borderline-2_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_borderline-2_20240317.npy", allow_pickle=True)
            # # # # 220240318 PCA 38 SMOTE  K=5         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_train_half1_SMOTE_Monday_and_Firday_ALL_Label_20240318.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_train_half1_SMOTE_Monday_and_Firday_ALL_Label_20240318.npy", allow_pickle=True)
            # # # 20240318 PCA 38 BL-SMOTE1  K=5  M = 10         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_borderline-1_20240318.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_borderline-1_20240318.npy", allow_pickle=True)
            # # # 20240318 PCA 38 BL-SMOTE2  K=5  M = 10         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\x_borderline-2_20240318.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\y_borderline-2_20240318.npy", allow_pickle=True)
            # # # # 20240319 Chi-square 45 SMOTE  K=5         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_train_half1_SMOTE_ALLday_ALL_Label_20240319.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_train_half1_SMOTE_ALLday_ALL_Label_20240319.npy", allow_pickle=True)
            # # # # 20240324 Chi-square 45 SMOTE  K=5     
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_train_half1_SMOTE_ALLday_ALL_Label_20240324.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_train_half1_SMOTE_ALLday_ALL_Label_20240324.npy", allow_pickle=True)
            # # # 20240324 Chi-square 45 BL-SMOTE1  K=5  M = 10         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_borderline-1_20240324.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_borderline-1_20240324.npy", allow_pickle=True)
            # # # 20240324 Chi-square 45 BL-SMOTE2  K=5  M = 10         
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_borderline-2_20240324.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_borderline-2_20240324.npy", allow_pickle=True)

            print("train_half1 SMOTE")

        elif (Choose_method == 'GAN'):
            # 20231114 after 百分百PCAonly do labelencode and minmax
            x_train = np.load(filepath + "x_train_half1_20231114.npy", allow_pickle=True)
            y_train = np.load(filepath + "y_train_half1_20231114.npy", allow_pickle=True)
        print("train_half1 x_train 的形狀:", x_train.shape)
        print("train_half1 y_train 的形狀:", y_train.shape)
        client_str = "client1"
        print("使用 train_half1 進行訓練")
    elif split_file == 'client2_train':
        if (Choose_method == 'normal'):
            # # 20240317 non iid client3 use TONIOT
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_20240317.npy", allow_pickle=True)
            # # 20240323 non iid client3 use TONIOT change ts change ip encode
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_TONIOT_train_change_ts_change_ip_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_TONIOT_train_change_ts_change_ip_20240317.npy", allow_pickle=True)
            # # 20240428 non iid client3 use TONIOT 
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_20240428.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_20240428.npy", allow_pickle=True)

            # 20240502 after do labelencode and minmax cicids2017 ALLDay  iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2017\\ALLday\\x_ALLDay_train_half2_20240502.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2017\\ALLday\\y_ALLDay_train_half2_20240502.npy", allow_pickle=True)

            # 20240503 after do labelencode and minmax cicids2019 01_12  iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2019\\01_12\\x_01_12_train_half2_20240503.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2019\\01_12\\y_01_12_train_half2_20240503.npy", allow_pickle=True)

            # 20240505 non iid after do labelencode and minmax TONIOT
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\TONIIOT_AddedLabel_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\\TONIOT\\TONIIOT_AddedLabel_y.npy", allow_pickle=True)

            # 20240507 after do labelencode and minmax EdgeIIoT iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_train_half2_20240507.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_half2_20240507.npy", allow_pickle=True)

            # 20240507 after do labelencode and minmax Kub iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\Kub\\x_Kub_train_half2_20240507.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\Kub\\y_Kub_train_half2_20240507.npy", allow_pickle=True)

            # 20240507 after do labelencode and minmax Wustl iid
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\Wustl\\x_Wustl_train_half2_20240507.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\Wustl\\y_Wustl_train_half2_20240507.npy", allow_pickle=True)

            # 20240523 non iid after do labelencode and minmax chi-square_45 TONIoT
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\TONIIOT_AddedLabel_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\TONIIOT_AddedLabel_y.npy", allow_pickle=True)

            # 20240523 client2 use TONIoT after do labelencode and minmax  均勻劃分75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_train_half2_20240523.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_train_half2_20240523.npy", allow_pickle=True)  

            # 20240523 client2 use TONIoT after do labelencode and minmax  隨機劃分75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_random_train_half2_20240523.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_random_train_half2_20240523.npy", allow_pickle=True)  

            # 20250113 CIC-IDS2018 after do labelencode and all featrue minmax 75 25分 do PCA Non-iid
            # 20250121 CIC-IDS2018 after do labelencode and all featrue minmax 75 25分 do Do feature drop to 79 feature 
            # Non-iid
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\Noniid\\CICIDS2018_AddedLabel_Noniid_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\Noniid\\CICIDS2018_AddedLabel_Noniid_y.npy", allow_pickle=True)
            
        elif (Choose_method == 'SMOTE'):
            # # # 20240317 Chi-square 45 SMOTE  K=5          
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_train_half2_SMOTE_Tuesday_and_Wednesday_and_Thursday_ALL_Label_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_train_half2_SMOTE_Tuesday_and_Wednesday_and_Thursday_ALL_Label_20240317.npy", allow_pickle=True)
            # # # 20240317 Chi-square 45 BL-SMOTE1 K=5 M = 10      
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_borderline-1_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_borderline-1_20240317.npy", allow_pickle=True)
            #  # # # 20240317 Chi-square 45 BL-SMOTE2 K=5 M = 10      
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_borderline-2_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_borderline-2_20240317.npy", allow_pickle=True)
            # # 20240318 PCA 38 SMOTE  K=5          
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_train_half2_SMOTE_Tuesday_and_Wednesday_and_Thursday_ALL_Label_20240318.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_train_half2_SMOTE_Tuesday_and_Wednesday_and_Thursday_ALL_Label_20240318.npy", allow_pickle=True)
            # # 20240318 PCA 38 BL-SMOTE1 K=5 M = 10      
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_borderline-1_20240318.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_borderline-1_20240318.npy", allow_pickle=True)
            # # 20240318 PCA 38 BL-SMOTE2 K=5 M = 10      
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_borderline-2_20240318.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_borderline-2_20240318.npy", allow_pickle=True)
          
            print("train_half2 SMOTE")
        elif (Choose_method == 'GAN'):
            # 20231114 after 百分百PCAonly do labelencode and minmax
            x_train = np.load(filepath + "x_train_half2_20231114.npy", allow_pickle=True)
            y_train = np.load(filepath + "y_train_half2_20231114.npy", allow_pickle=True)
        print("train_half2 x_train 的形狀:", x_train.shape)
        print("train_half2 y_train 的形狀:", y_train.shape)
        client_str = "client2"
        print("使用 train_half2 進行訓練")

    elif split_file == 'client3_train':
        if (Choose_method == 'normal'):
            # # 20240110 non iid client2 use TONIOT
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_20240110.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_20240110.npy", allow_pickle=True)
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_addlossvalue_20240110.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_addlossvalue_20240110.npy", allow_pickle=True)
            # # 20240316 non iid client1 use cicids2017 Tuesday_and_Wednesday_and_Thursday after chi-square
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_Tuesday_and_Wednesday_and_Thursday_train_cicids2017_AfterFeatureSelect44_20240316.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_Tuesday_and_Wednesday_and_Thursday_train_cicids2017_AfterFeatureSelect44_20240316.npy", allow_pickle=True)
            # # 20240314 non iid client1 use cicids2017 Tuesday_and_Wednesday_and_Thursday after PCA
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_Tuesday_and_Wednesday_and_Thursday_train_AfterPCA38_20240314.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_Tuesday_and_Wednesday_and_Thursday_train_AfterPCA38_20240314.npy", allow_pickle=True)       
            # # 20240315 non iid client1 use cicids2017 Tuesday_and_Wednesday_and_Thursday after PCA
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_Tuesday_and_Wednesday_and_Thursday_train_AfterPCA77_20240315.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_Tuesday_and_Wednesday_and_Thursday_train_AfterPCA77_20240315.npy", allow_pickle=True)       
            # # 20240317 non iid client2 use cicids2017 Tuesday_and_Wednesday_and_Thursday after chi-square 45 add tonniot
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_Tuesday_and_Wednesday_and_Thursday_train_dataframes_AfterFeatureSelect44_ADD_TONIOT_20240316.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_Tuesday_and_Wednesday_and_Thursday_train_dataframes_AfterFeatureSelect44_ADD_TONIOT_20240316.npy", allow_pickle=True)
            # # 20240317 non iid client2 use cicids2017 Tuesday_and_Wednesday_and_Thursday after chi-square 45 add tonniot remove all IP port
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_Tuesday_and_Wednesday_and_Thursday_train_dataframes_AfterFeatureSelect44_ADD_TONIOT_rmove_ip_port_20240316.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_Tuesday_and_Wednesday_and_Thursday_train_dataframes_AfterFeatureSelect44_ADD_TONIOT_rmove_ip_port_20240316.npy", allow_pickle=True)
            # # 20240317 non iid client2 use cicids2017 Tuesday_and_Wednesday_and_Thursday tonniot add cicids2017 39 feature then PCA 
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_Tuesday_and_Wednesday_and_Thursday_train_AfterPCA77_20240317.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_Tuesday_and_Wednesday_and_Thursday_train_AfterPCA77_20240317.npy", allow_pickle=True)
            # # 20240318 non iid client2 use cicids2017 Tuesday_and_Wednesday_and_Thursday PCA 38
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\x_Tuesday_and_Wednesday_and_Thursday_train_AfterPCA38_20240318.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\y_Tuesday_and_Wednesday_and_Thursday_train_AfterPCA38_20240318.npy", allow_pickle=True)
            
            # # 20240428 non iid client3 use cicids2019 chi45
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_CICIDS2019_AfterFeatureSelect44_20240428.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_CICIDS2019_AfterFeatureSelect44_20240428.npy", allow_pickle=True)
            
            # # 20240506 non iid client3 use cicids2019 chi45
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\CICIDS2019_AddedLabel_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\CICIDS2019_AddedLabel_y.npy", allow_pickle=True)
            
            # 20240523 non iid after do labelencode and minmax chi-square_45 EdgeIIoT
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AddedLabel_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AddedLabel_y.npy", allow_pickle=True)

            # 20240523 client3 use TONIoT after do labelencode and minmax  均勻劃分75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_train_half3_20240523.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_train_half3_20240523.npy", allow_pickle=True)  

            # # 20240523 client3 use TONIoT after do labelencode and minmax  均勻劃分75 25分 DoJSMA 0.0.5 0.02
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half3_20240725.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half3_20240725.npy", allow_pickle=True)  

            # 20240523 client3 use TONIoT after do labelencode and minmax  隨機劃分75 25分
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_random_train_half3_20240523.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_random_train_half3_20240523.npy", allow_pickle=True)  

            # # 20240523 client3 use TONIoT after do labelencode and minmax  隨機劃分75 25分 DoJSMA 0.0.5 0.02
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half3_20240801.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half3_20240801.npy", allow_pickle=True)  
            
            # 20250113 CIC-IDS2019 after do labelencode  and all featrue minmax 75 25分 DoPCA Non-iid
            # print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2019 after do labelencode do pca" +f"{split_file} with normal attack type")
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\Noniid\\CICIDS2019_AddedLabel_Noniid_x.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\Noniid\\CICIDS2019_AddedLabel_Noniid_y.npy", allow_pickle=True)
            
            # 20250121 01-12 and 03-11 merge ALLDay CIC-IDS2019 after do labelencode  and all featrue minmax 75 25分 Do feature drop to 79 feature 
            # Non-iid
            print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2019 after do labelencode do feature drop" +f"{split_file} with normal attack type")
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\Noniid\\CICIDS2019_AddedLabel_Noniid_x.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\Noniid\\CICIDS2019_AddedLabel_Noniid_y.npy", allow_pickle=True)
            

        elif (Choose_method == 'SMOTE'):
            # # 20240324 Chi-square 45 SMOTE  K=5          
            x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_half3_SMOTE_TONIOT_ALL_Label_20240324.npy", allow_pickle=True)
            y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_half3_SMOTE_TONIOT_ALL_Label_20240324.npy", allow_pickle=True)
            # # # 20240324 Chi-square 45 BL-SMOTE1 K=5 M = 10      
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_borderline-1_20240324.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_borderline-1_20240324.npy", allow_pickle=True)
            #  # # # 20240324 Chi-square 45 BL-SMOTE2 K=5 M = 10      
            # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_borderline-2_20240324.npy", allow_pickle=True)
            # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_borderline-2_20240324.npy", allow_pickle=True)
            print("train_half3 SMOTE")
        elif (Choose_method == 'GAN'):
            # 20231114 after 百分百PCAonly do labelencode and minmax
            x_train = np.load(filepath + "x_train_half2_20231114.npy", allow_pickle=True)
            y_train = np.load(filepath + "y_train_half2_20231114.npy", allow_pickle=True)
        print("train_half3 x_train 的形狀:", x_train.shape)
        print("train_half3 y_train 的形狀:", y_train.shape)
        client_str = "client3"
        print("使用 train_half3 進行訓練")


    print("use file", split_file)
    return x_train, y_train,client_str

# for do one hot
def ChooseTrainDatastes(filepath, my_command,Choose_method):
    # 加载选择的数据集
    
    if my_command == 'total_train':
        if (Choose_method == 'normal'):
            # print("Training with total_train")
            # train_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'train_dataframes_resplit.csv'))
            # x_train = np.array(train_dataframe.iloc[:, :-1])
            # y_train = np.array(train_dataframe.iloc[:, -1])
            dftrain = pd.read_csv(filepath + "data\\dataset_AfterProcessed\\20231113\\train_dataframes_20231113.csv")

            #特徵    
            x_columns = dftrain.columns.drop(dftrain.filter(like='Label_').columns)
            x_train = dftrain[x_columns].values.astype('float32')
            y_train = dftrain.filter(like='Label_').values.astype('float32')
            # 找到每一行中值為 1 的索引
            label_indices = np.argmax(y_train, axis=1)
            # 將 label_indices 賦值給 y_train，這樣 y_train 就包含了整數表示的標籤
            y_train = label_indices
           
        elif (Choose_method == 'SMOTE'):
            # x_train = np.load(filepath + "x_total_train_SMOTE_ALL_Label.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train_SMOTE_ALL_Label.npy", allow_pickle=True)
            x_train = np.load(filepath + "x_total_train_SMOTE_ALL_Label14.npy", allow_pickle=True)
            y_train = np.load(filepath + "y_total_train_SMOTE_ALL_Label14.npy", allow_pickle=True)
            
        elif (Choose_method == 'GAN'):
            # x_train = np.load(filepath + "x_total_train.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_total_train.npy", allow_pickle=True)
            # x_train = np.load(filepath + "x_GAN_data_total_train_weakpoint_14.npy", allow_pickle=True)
            # y_train = np.load(filepath + "y_GAN_data_total_train_weakpoint_14.npy", allow_pickle=True)
            x_train = np.load(filepath + "x_train_20231106_afterGAN_Label14.npy", allow_pickle=True)
            # 將複數的資料實部保留並轉換為浮點：
            x_train = x_train.real.astype(np.float64)
            y_train = np.load(filepath + "y_train_20231106_afterGAN_Label14.npy", allow_pickle=True)
        client_str = "BaseLine"

    elif my_command == 'train_half1':
        print("Training with train_half1")
        dftrain = pd.read_csv(filepath + "data\\dataset_AfterProcessed\\20231113\\train_half1_20231113.csv")

        #特徵    
        x_columns = dftrain.columns.drop(dftrain.filter(like='Label_').columns)
        x_train = dftrain[x_columns].values.astype('float32')
        y_train = dftrain.filter(like='Label_').values.astype('float32')
        # 找到每一行中值為 1 的索引
        label_indices = np.argmax(y_train, axis=1)
        # 將 label_indices 賦值給 y_train，這樣 y_train 就包含了整數表示的標籤
        y_train = label_indices
        client_str = "client1"
        
    elif my_command == 'train_half2':
        print("Training with train_half2")
        print("Training with train_half1")
        dftrain = pd.read_csv(filepath + "data\\dataset_AfterProcessed\\20231113\\train_half2_20231113.csv")
        #特徵    
        x_columns = dftrain.columns.drop(dftrain.filter(like='Label_').columns)
        x_train = dftrain[x_columns].values.astype('float32')
        y_train = dftrain.filter(like='Label_').values.astype('float32')
        # 找到每一行中值為 1 的索引
        label_indices = np.argmax(y_train, axis=1)
        # 將 label_indices 賦值給 y_train，這樣 y_train 就包含了整數表示的標籤
        y_train = label_indices
        client_str = "client2"
        
    # 返回所需的數據或其他變量
    return x_train, y_train, client_str


def ChooseTestDataSet(filepath):
    # test_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'test_dataframes.csv'))
    # x_test = np.array(test_dataframe.iloc[:, :-1])
    # y_test = np.array(test_dataframe.iloc[:, -1])
    dftest = pd.read_csv(filepath + "data\\dataset_AfterProcessed\\20231113\\test_dataframes_20231113.csv")
    #x_columns作用就是丟掉Label_開頭 也就是等於特徵    
    x_columns = dftest.columns.drop(dftest.filter(like='Label_').columns)
    x_test = dftest[x_columns].values.astype('float32')
    y_test = dftest.filter(like='Label_').values.astype('float32')
    # 找到每一行中值為 1 的索引
    label_indices = np.argmax(y_test, axis=1)
    # 將 label_indices 賦值給 y_train，這樣 y_train 就包含了整數表示的標籤
    y_test = label_indices
    
    return x_test, y_test

### sava dataframe to np array 
def SaveDataframeTonpArray(dataframe, filepath ,df_name, filename):
    #選擇了最后一列Lable之外的所有列，即選擇所有feature
    x = np.array(dataframe.iloc[:,:-1])
    y = np.array(dataframe.iloc[:,-1])

    #np.save
    np.save(f"{filepath}\\x_{df_name}_{filename}.npy", x)
    np.save(f"{filepath}\\y_{df_name}_{filename}.npy", y)

### find找到datasets中是string的行
def findStringCloumn(dataFrame):
        string_columns = dataFrame.select_dtypes(include=['object'])
        for column in string_columns.columns:
            print(f"{dataFrame} 中type為 'object' 的列: {column}")
            print(string_columns[column].value_counts())
            print("\n")

### check train_df_half1 and train_df_half2 dont have duplicate data
def CheckDuplicate(dataFrame1, dataFrame2):
    intersection = len(set(dataFrame1.index) & set(dataFrame2.index))
    print(f"{dataFrame1} 和 {dataFrame2} 的index交集数量:", intersection)
    print(f"{dataFrame1} 和 {dataFrame2}是否相同:", dataFrame1.equals(dataFrame2))
    
### print dataset information 
def printFeatureCountAndLabelCountInfo(dataFrame1, dataFrame2,label):
     # 計算feature數量
    num_features_dataFrame1 = dataFrame1.shape[1] - 1
    num_features_dataFrame2 = dataFrame2.shape[1] - 1 
     # 計算Label數量
    label_counts = dataFrame1[label].value_counts()
    label_counts2 = dataFrame2[label].value_counts()

    print(f"{str(dataFrame1)} 的feature:", num_features_dataFrame1)
    print(f"{str(dataFrame1)} 的label數:", len(label_counts))
    print(f"{str(dataFrame1)} 的除了最後一列Label列之外的所有列,即選擇feature數:\n", dataFrame1.iloc[:,:-1])
    findStringCloumn(dataFrame1)

    print(f"{str(dataFrame2)} 的feature:", num_features_dataFrame2)
    print(f"{str(dataFrame2)} 的label數:", len(label_counts2))
    print(f"{str(dataFrame2)} 的除了最後一列Label列之外的所有列,即選擇feature數:\n", dataFrame2.iloc[:,:-1])
    findStringCloumn(dataFrame2)

    CheckDuplicate(dataFrame1, dataFrame2)

### label encoding
# def label_Encoding(label):
#     label_encoder = preprocessing.LabelEncoder()
#     mergecompelete_dataset[label] = label_encoder.fit_transform(mergecompelete_dataset[label])
#     mergecompelete_dataset[label].unique()

def label_Encoding(label,df):
    label_encoder = preprocessing.LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])
    df[label].unique()
    print(f"{label}",df[label].unique())
    return df


# ##  清除CIC-IDS-2017 資料集中的dirty data，包含NaN、Infinity、包含空白或小于ASCII 32的字符
def clearDirtyData(df):
    # 檢查第一列featurea名稱是否包含空白或是小于ASCII 32的字元
    first_column = df.columns[0]
    is_dirty = first_column.isspace() or ord(first_column[0]) < 32

    # 將"inf"值替換為NaN
    df.replace("inf", np.nan, inplace=True)

    # 找到包含NaN、Infinity和"inf"值的行，並將其index添加到dropList
    nan_inf_rows = df[df.isin([np.nan, np.inf, -np.inf]).any(axis=1)].index.tolist()

    # 將第一列featurea名稱所在的index添加到dropList
    if is_dirty:
        nan_inf_rows.append(0)

    # 去重dropList中的index
    dropList = list(set(nan_inf_rows))

    # 刪除包含dirty data的行
    df_clean = df.drop(dropList)

    return df_clean

### for sorting the labeled data based on support
def sortingFunction(data):
    return data.shape[0]

# 使用分層劃分資料集 平均劃分資料集 
def splitdatasetbalancehalf(train_dataframes, label):
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for train_indices, test_indices in stratified_split.split(train_dataframes, train_dataframes[label]):
        df1 = train_dataframes.iloc[train_indices]
        df2 = train_dataframes.iloc[test_indices]
        label_counts = df1[label].value_counts()
        label_counts2 = df2[label].value_counts()
        print("train_half1\n",label_counts)
        print("train_half2\n",label_counts2)

    return df1,df2

def splitweakLabelbalance(weakLabel,original_dataset,size):
    # label_data = original_dataset[original_dataset['type'] == weakLabel]
    label_data = original_dataset[original_dataset['Label'] == weakLabel]
    # 使用train_test_split分別劃分取Label相等8、9、13、14的數據
    train_label, test_label = train_test_split(label_data, test_size=size, random_state=42)
    return train_label, test_label

def splitweakLabelbalance_afterOnehot(weak_label, original_dataset,size):

    weak_label_data = original_dataset[weak_label]
    # e.g:找到做完one hot Label_8 列中值等于1的行
    weak_label_data_equals_1_rows = original_dataset[weak_label_data == 1]
    weak_label_train, weak_label_test = train_test_split(weak_label_data_equals_1_rows, test_size=size, random_state=42)
    
    return weak_label_train, weak_label_test

# Choose Model
def ChooseUseModel(model_type, input, ouput):
    if model_type == "DNN":
        print("choose model is",model_type)
        class DNN(nn.Module):
            def __init__(self):
                super(DNN, self).__init__()
                self.layer1 = nn.Linear(input, 50)
                self.fc2 = nn.Linear(50, 50)
                self.fc3 = nn.Linear(50, 50)
                self.fc4 = nn.Linear(50, 50)
                self.fc5 = nn.Linear(50, 50)
                self.fc6 = nn.Linear(50, 50)
                self.fc7 = nn.Linear(50, 50)
                self.fc8 = nn.Linear(50, ouput)

            def forward(self, x):
                x = F.relu(self.layer1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                x = F.relu(self.fc6(x))
                x = F.relu(self.fc7(x))
                x = self.fc8(x)
                return x
        return DNN()  # 返回創建的model instance
    elif model_type == "ANN":
        print("choose model is",model_type)
        class ANN(nn.Module):
            def __init__(self):
                super(ANN,self).__init__()
                # input has two features and
                self.layer1 = nn.Linear(input,200)
                self.layer2 = nn.Linear(200,200)
                self.layer3 = nn.Linear(200,ouput)

            def forward(self,x):
                x = self.layer1(x)
                x = F.tanh(x)
                x = self.layer2(x)
                x = F.tanh(x)
                x = self.layer3(x)
                return x
        return ANN()  # 返回創建的model instance
    elif model_type == "MLP":
        print("choose model is",model_type)
        class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                 # 每層512神經元 for cicids2017
                self.layer1 = nn.Linear(input, 512)
                self.dropout1 = nn.Dropout(p=0.2)  # 第一層 Dropout
                self.fc2 = nn.Linear(512, 512)
                self.dropout2 = nn.Dropout(p=0.2)  # 第二層 Dropout
                self.fc3 = nn.Linear(512, 512)
                self.dropout3 = nn.Dropout(p=0.2)  # 第三層 Dropout
                self.fc4 = nn.Linear(512, 512)
                self.dropout4 = nn.Dropout(p=0.2)  # 第四層 Dropout
                self.layer5 = nn.Linear(512, ouput)
                # # 每層64神經元 for Toniot
                # self.layer1 = nn.Linear(input, 64)
                # self.fc2 = nn.Linear(64, 64)
                # self.fc3 = nn.Linear(64, 64)
                # self.fc4 = nn.Linear(64, 64)
                # self.layer5 = nn.Linear(64, ouput)
                # # 隱藏層分別配置了 40 、 30 和 15 個神經元 for CICIDS2019
                # self.layer1 = nn.Linear(input, 40)
                # self.fc2 = nn.Linear(40, 30)
                # self.fc3 = nn.Linear(30, 30)
                # self.fc4 = nn.Linear(30, 15)
                # self.layer5 = nn.Linear(15, ouput)

            def forward(self, x):
                # relu激活函数
                # 输出范围在 (0, max) 之间
                x = F.relu(self.layer1(x))
                x = self.dropout1(x)  # 第一層 Dropout
                x = F.relu(self.fc2(x))
                x = self.dropout2(x)  # 第二層 Dropout
                x = F.relu(self.fc3(x))
                x = self.dropout3(x)  # 第三層 Dropout
                x = F.relu(self.fc4(x))
                x = self.dropout4(x)  # 第四層 Dropout
                # 修改后的代码使用Sigmoid激活函数每層64神經元 for Toniot
                # 输出范围在 (0, 1) 之间
                # x = F.sigmoid(self.layer1(x))
                # x = F.sigmoid(self.fc2(x))
                # x = F.sigmoid(self.fc3(x))
                # x = F.sigmoid(self.fc4(x))
                # x = self.sigmoid(self.output(x))  # 使用 Sigmoid 作為輸出層激活函數
                x = self.layer5(x)
                return x
        return MLP()  # 返回創建的model instance

# 根據使用資料集載入使用的MLP模型
def Load_Model_BasedOnDataset(str_datasets, model_type, input, output, bool_dofeatureSelect=False):
    """
    ChooseUseModel(model_type, input, output)
    bool_dofeatureSelect=False 表示沒有特徵選擇

    CICIDS2017: input=77, output=15
    CICIDS2019: input=77, output=12
    TONIOT: input=45, output=10
    ########################################
    CIC系列input輸入是77是特徵扣掉
    'SourceIP', 'SourcePort', 
    'SourceIP', 'SourcePort',DestinationIP', 
    'DestinationPort', 'Timestamp', 'Label'
    """
    if not bool_dofeatureSelect:
        if str_datasets == "CICIDS2017":
            model = ChooseUseModel(model_type, input, output)
        elif str_datasets == "CICIDS2019":
            model = ChooseUseModel(model_type, input, output)
        elif str_datasets == "TONIOT":
            model = ChooseUseModel(model_type, input, output)

        print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+f"dofeatureSelect: {bool_dofeatureSelect}")
        print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+f"ChooseUseModel: {str_datasets}")
        print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+f"model_type: {model_type}")
        print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+f"input: {input}")
        print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+f"output: {output}")
    else:
        if str_datasets == "CICIDS2017":
            model = ChooseUseModel(model_type, 45, 15)
        elif str_datasets == "CICIDS2019":
            model = ChooseUseModel(model_type, 45, 12)
        elif str_datasets == "TONIOT":
            model = ChooseUseModel(model_type, 45, 10)

        print(Fore.RED +Back.WHITE+ Style.BRIGHT+f"dofeatureSelect: {bool_dofeatureSelect}")
        print(Fore.RED +Back.WHITE+ Style.BRIGHT+f"ChooseUseModel: {str_datasets}")
        print(Fore.RED +Back.WHITE+ Style.BRIGHT+f"model_type: {model_type}")
        print(Fore.RED +Back.WHITE+ Style.BRIGHT+f"input: {45}")
        print(Fore.RED +Back.WHITE+ Style.BRIGHT+f"output: {10}")

    return model

def ChooseDataSetNpFile(Str_ChooseDataset,filepath):
    if Str_ChooseDataset == "CICIDS2017":
        # 20240323 non iid client1 use cicids2017 ALLday  chi-square_45 change ip encode
        x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_train_dataframes_AfterFeatureSelect_Noniid_change_ip_20240323.npy", allow_pickle=True)
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_train_dataframes_AfterFeatureSelect_Noniid_change_ip_20240323.npy", allow_pickle=True)
        
        x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/x_ALLDay_test_dataframes_AfterFeatureSelect_Noniid_change_ip_20240323.npy")
        y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/y_ALLDay_test_dataframes_AfterFeatureSelect_Noniid_change_ip_20240323.npy")
    
    elif Str_ChooseDataset == "TONIOT":
        # # 20240323 non iid client2 use TONIOT change ts change ip encode
        x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_TONIOT_train_change_ts_change_ip_20240317.npy", allow_pickle=True)
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_TONIOT_train_change_ts_change_ip_20240317.npy", allow_pickle=True)
    
        x_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/x_TONIOT_test_change_ts_change_ip_20240317.npy")
        y_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/y_TONIOT_test_change_ts_change_ip_20240317.npy")

    elif Str_ChooseDataset == "CICIDS2019":

        x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_CICIDS2019_AfterFeatureSelect44_20240428.npy", allow_pickle=True)
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_CICIDS2019_AfterFeatureSelect44_20240428.npy", allow_pickle=True)

        x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/x_01_12_test_CICIDS2019_AfterFeatureSelect44_20240428.npy")
        y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/y_01_12_test_CICIDS2019_AfterFeatureSelect44_20240428.npy")

    elif Str_ChooseDataset == "EdgeIIoT":
        x_train = np.loadtxt(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_train.txt")
        y_train = np.loadtxt(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_train.txt")
            
        x_test = np.loadtxt(f"./data/dataset_AfterProcessed/EdgeIIoT/x_test.txt")
        y_test = np.loadtxt(f"./data/dataset_AfterProcessed/EdgeIIoT/y_test.txt")
    
    elif Str_ChooseDataset == "Kub":
        x_train = np.loadtxt(filepath + "\\dataset_AfterProcessed\\Kub\\x_train.txt")
        y_train = np.loadtxt(filepath + "\\dataset_AfterProcessed\\Kub\\y_train.txt")
            
        x_test = np.loadtxt(f"./data/dataset_AfterProcessed/Kub/x_test.txt")
        y_test = np.loadtxt(f"./data/dataset_AfterProcessed/Kub/y_test.txt")

    elif Str_ChooseDataset == "Wustl":
        x_train = np.loadtxt(filepath + "\\dataset_AfterProcessed\\Wustl\\x_train.txt")
        y_train = np.loadtxt(filepath + "\\dataset_AfterProcessed\\Wustl\\y_train.txt")
            
        x_test = np.loadtxt(f"./data/dataset_AfterProcessed/Wustl/x_test.txt")
        y_test = np.loadtxt(f"./data/dataset_AfterProcessed/Wustl/y_test.txt")
    
    return x_train, y_train,x_test,y_test

def DoReSplit(Str_ChooseDataset,df):
    if Str_ChooseDataset == "CICIDS2017":
        # BaseLine時
        # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
        # encode後對照如下
        # Heartbleed:8、
        # Infiltration:9、
        # Web Attack Sql Injection:13
            
    # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_Label = []
        List_test_Label = []
        for i in range(15):
            if i == 8 or i == 9 or i ==13:
                continue
            train_label_split, test_label_split = splitweakLabelbalance(i,df,0.2)
            List_train_Label.append(train_label_split)
            List_test_Label.append(test_label_split)         

        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)

        # Label encode mode  分別取出Label等於8、9、13的數據 對6633分
        train_label_Heartbleed, test_label_Heartbleed = splitweakLabelbalance(8,df,0.33)
        train_label_Infiltration, test_label_Infiltration = splitweakLabelbalance(9,df,0.33)
        train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = splitweakLabelbalance(13,df,0.33)

        # # 刪除Label相當於8、9、13的行
        test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
        train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
        # 合併Label8、9、13回去
        test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
        train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
        print("test",test_dataframes['Label'].value_counts())

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/encode_and_count_resplit.csv", "a+") as file:
            label_counts = test_dataframes['Label'].value_counts()
            print("test_dataframes\n", label_counts)
            file.write("test_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = train_dataframes['Label'].value_counts()
            print("train_dataframes\n", label_counts)
            file.write("train_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/{today}", f"Resplit_train_dataframes_{today}")
        SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/{today}", f"Resplit_test_dataframes_{today}")
        SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/{today}", f"Resplit_test",today)
        SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/{today}", f"Resplit_train",today)

    elif Str_ChooseDataset == "CICIDS2019":
        # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_Label = []
        List_test_Label = []
        for i in range(34):
            if i==0 or (i >= 23 and i <= 34):
                train_label_split, test_label_split = splitweakLabelbalance(i,df,0.2)
                List_train_Label.append(train_label_split)
                List_test_Label.append(test_label_split)         
        
        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)
        # encode後對照如下
        # WebDDoS:34
        # Label encode mode  分別取出Label等於12的數據 對6633分
        train_label_WebDDoS, test_label_WebDDoS = splitweakLabelbalance(34,df,0.33)
        # # 刪除Label相當於12的行
        test_dataframes = test_dataframes[~test_dataframes['Label'].isin([34])]
        train_dataframes = train_dataframes[~train_dataframes['Label'].isin([34])]
        # 合併Label12回去
        test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
        train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])            
        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/encode_and_count_resplit.csv", "a+") as file:
            label_counts = test_dataframes['Label'].value_counts()
            print("test_dataframes\n", label_counts)
            file.write("test_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = train_dataframes['Label'].value_counts()
            print("train_dataframes\n", label_counts)
            file.write("train_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/01_12/{today}", f"01_12_Resplit_train_dataframes_{today}")
        SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2019/01_12/{today}", f"01_12_Resplit_test_dataframes_{today}")
        SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/01_12/{today}", f"01_12_Resplit_test",today)
        SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/01_12/{today}", f"01_12_Resplit_train",today)

    elif Str_ChooseDataset == "EdgeIIoT":
        # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_Label = []
        List_test_Label = []
        for i in range(15):
            train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
            List_train_Label.append(train_label_split)
            List_test_Label.append(test_label_split)         

        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)

        print("test",test_dataframes['Label'].value_counts())

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/EdgeIIoT/encode_and_count_resplit.csv", "a+") as file:
            label_counts = test_dataframes['Label'].value_counts()
            print("test_dataframes\n", label_counts)
            file.write("test_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = train_dataframes['Label'].value_counts()
            print("train_dataframes\n", label_counts)
            file.write("train_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/EdgeIIoT/{today}", f"Resplit_train_dataframes_{today}")
        SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/EdgeIIoT/{today}", f"Resplit_test_dataframes_{today}")
        SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/EdgeIIoT/{today}", f"Resplit_test",today)
        SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/EdgeIIoT/{today}", f"Resplit_train",today)
    elif Str_ChooseDataset == "Kub":
        # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_Label = []
        List_test_Label = []
        for i in range(4):
            train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
            List_train_Label.append(train_label_split)
            List_test_Label.append(test_label_split)         

        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)

        print("test",test_dataframes['Label'].value_counts())

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/Kub/encode_and_count_resplit.csv", "a+") as file:
            label_counts = test_dataframes['Label'].value_counts()
            print("test_dataframes\n", label_counts)
            file.write("test_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = train_dataframes['Label'].value_counts()
            print("train_dataframes\n", label_counts)
            file.write("train_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/Kub/{today}", f"Resplit_train_dataframes_{today}")
        SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/Kub/{today}", f"Resplit_test_dataframes_{today}")
        SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/Kub/{today}", f"Resplit_test",today)
        SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/Kub/{today}", f"Resplit_train",today)
    elif Str_ChooseDataset == "Wustl":
        # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_Label = []
        List_test_Label = []
        for i in range(5):
            train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
            List_train_Label.append(train_label_split)
            List_test_Label.append(test_label_split)         

        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)

        print("test",test_dataframes['Label'].value_counts())

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/Kub/encode_and_count_resplit.csv", "a+") as file:
            label_counts = test_dataframes['Label'].value_counts()
            print("test_dataframes\n", label_counts)
            file.write("test_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = train_dataframes['Label'].value_counts()
            print("train_dataframes\n", label_counts)
            file.write("train_dataframes_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/Wustl/{today}", f"Resplit_train_dataframes_{today}")
        SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/Wustl/{today}", f"Resplit_test_dataframes_{today}")
        SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/Wustl/{today}", f"Resplit_test",today)
        SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/Wustl/{today}", f"Resplit_train",today)

def DoReStoreNpFileToCsv(x_train, y_train,x_test,y_test,Str_ChooseDataset):
     # 將特徵數據和標籤數據合併成一個 DataFrame
    columns_x = [f'feature_{i}' for i in range(x_train.shape[1])]
    df_train_x = pd.DataFrame(x_train,columns=columns_x)
    df_train_y = pd.DataFrame(y_train, columns=['Label'])

    df_test_x = pd.DataFrame(x_test,columns=columns_x)
    df_test_y = pd.DataFrame(y_test, columns=['Label'])
    


    if(CheckFileExists(f'./Restore_{Str_ChooseDataset}.csv')!=True):
        # 合併 x 和 y DataFrame
        df_train_combined = pd.concat([df_train_x, df_train_y], axis=1)
        df_test_combined = pd.concat([df_test_x,df_test_y ], axis=1)

        df_combined = pd.concat([df_train_combined, df_test_combined], axis=0)
        df_combined.to_csv(f'./Restore_{Str_ChooseDataset}.csv', index=False)
        df_combined = pd.read_csv(f'./Restore_{Str_ChooseDataset}.csv')
    else:
        df_combined = pd.read_csv(f'./Restore_{Str_ChooseDataset}.csv')
        
    return df_combined

### Replace the number of greater than 10,000
def ReplaceMorethanTenthousandQuantity(df,str_column_name):
    label_counts = df[str_column_name].value_counts()
    # 打印提取后的DataFrame
    print(label_counts)
    # 创建一个空的DataFrame来存储结果
    extracted_df = pd.DataFrame()

    # 获取所有不同的标签
    unique_labels = df[str_column_name].unique()

    # 遍历每个标签
    for label in unique_labels:
        # 选择特定标签的行
        label_df = df[df[str_column_name] == label]
    
        # 如果标签的数量超过1万，提取前1万行；否则提取所有行
        # if len(label_df) > 10000:
            # label_df = label_df.head(10000)
            
        # 如果標籤的數量超過1萬，隨機提取1萬行；否則提取所有行
        if len(label_df) > 10000:
            label_df = label_df.sample(n=10000, random_state=42)  # 使用指定的隨機種子(random_state)以保證可重現性
    
        # 将结果添加到提取的DataFrame中
        extracted_df = pd.concat([extracted_df, label_df])

    # 将更新后的DataFrame保存到文件
    # SaveDataToCsvfile(extracted_df, "./data/dataset_AfterProcessed","total_encoded_updated_10000")

    # 打印修改后的结果
    print(extracted_df[str_column_name].value_counts())
    return extracted_df


def ResotreTrainAndTestToCSVandReSplit(Str_ChooseDataset,filepath):
    x_train, y_train,x_test,y_test = ChooseDataSetNpFile(Str_ChooseDataset,filepath)
    df_combined = DoReStoreNpFileToCsv(x_train, y_train,x_test,y_test,Str_ChooseDataset)
    df_combined = ReplaceMorethanTenthousandQuantity(df_combined)
    DoReSplit(Str_ChooseDataset,df_combined)

def EvaluatePercent(Current_round_dis,Last_round_dis):

    # 檢查 Last_round_dis 是否為零，避免除以零
    if Last_round_dis == 0:
        # return float('inf')  # 這裡返回無限大，可以根據具體情況修改
         return 0 # 代表「沒有變化」
    Current_round_dis = float(Current_round_dis)
    Last_round_dis = float(Last_round_dis)
    percent_diff = abs(Current_round_dis-Last_round_dis)
    percent_diff = (percent_diff/Last_round_dis)*100
    return percent_diff

#針對Strig type做完label ecnode後補做minmax
def DominmaxforStringTypefeature(doScalerdataset):
    # 開始minmax
    X=doScalerdataset
    X=X.values
    # scaler = preprocessing.StandardScaler() #資料標準化
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    scaler.fit(X)
    X=scaler.transform(X)
    # 将缩放后的值更新到 doScalerdataset 中
    doScalerdataset.iloc[:, :] = X
    # 将排除的列名和选中的特征和 Label 合并为新的 DataFrame
    # afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,afterprocess_dataset['Label']], axis = 1)
    return doScalerdataset


## 將結合weakLabel Label8 的train_half1轉成np array
# gan_dataframe = pd.read_csv("D:\\Labtest20230911\\GAN_data_train_half1\\GAN_data_train_half1_ADD_weakLabel_8.csv")
# SaveDataframeTonpArray(gan_dataframe, "train_half1","weakpoint_8")
# gan_dataframe = pd.read_csv("D:\\Labtest20230911\\GAN_data_total_train\\GAN_data_total_train_ADD_weakLabel_14.csv")
# # # gan_dataframe = pd.read_csv("D:\\Labtest20230911\\GAN_data_train_half1\\GAN_data_train_half1_ADD_weakLabel_9.csv")
# SaveDataframeTonpArray(gan_dataframe, "GAN_data_total_train","weakpoint_14")
### DestinationPort拉出來到mytoolfunction.py單獨從做一次
# df = pd.read_csv("D:\\Labtest20230911\\data\\total_encoded_updated.csv")
# df['DestinationPort'] = df['DestinationPort'].astype(str)


# label_Encoding('DestinationPort',df)
# SaveDataToCsvfile(df, "./data", "total_encoded_updated_20231101")

############################################################# other
# # 命令行參數解析器
# parser = argparse.ArgumentParser(description='Federated Learning Client')

# # 添加一個參數來選擇數據集
# parser.add_argument('--dataset', type=str, choices=['train_half1', 'train_half2'], default='train_half1',
#                     help='選擇訓練數據集 (train_half1 或 train_half2)')

# args = parser.parse_args()

# # 根據命令行參數選擇數據集
# my_command = args.dataset
# # python BaseLine.py --dataset train_half1
# # python BaseLine.py --dataset train_half2

# # 載入選擇的數據集
# if my_command == 'train_half1':
#     # x_train = np.load(filepath + "x_train_half1.npy", allow_pickle=True)
#     # y_train = np.load(filepath + "y_train_half1.npy", allow_pickle=True)
#     x_train = np.load(filepath + "x_train_half1_weakpoint_8.npy", allow_pickle=True)
#     y_train = np.load(filepath + "y_train_half1_weakpoint_8.npy", allow_pickle=True)
#     client_str = "client1"
#     print("使用 train_half1 進行訓練")
# elif my_command == 'train_half2':
#     x_train = np.load(filepath + "x_train_half2.npy", allow_pickle=True)
#     y_train = np.load(filepath + "y_train_half2.npy", allow_pickle=True)
#     client_str = "client2"
#     print("使用 train_half2 進行訓練")

# test time function
# start_IDS = getStartorEndtime("start")
# # 暫停程式執行 5 秒
# time.sleep(10) #sleep 以秒為單位
# end_IDS = getStartorEndtime("end")
# CalculateTime(end_IDS, start_IDS)


### 添加图例
# # 绘制原始数据集
# plt.scatter(x_train[:, 0], x_train[:, 1], c='red', label='Original Data')

# # 绘制SMOTE采样后的数据集
# plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c='blue', marker='x', s=100, label='SMOTE Samples')


# plt.legend()
# plt.show()

# 找到SMOTE采样后的数据中Label 13的索引


# desired_sample_count = 500

# # 对Label14进行SMOTE
# sampling_strategy_label14 = {14: desired_sample_count}
# oversample_label14 = SMOTE(sampling_strategy=sampling_strategy_label14, k_neighbors=k_neighbors, random_state=42)
# X_resampled_label14, y_resampled_label14 = oversample_label14.fit_resample(x_train, y_train)

# # 对Label9进行SMOTE
# sampling_strategy_label9 = {9: desired_sample_count}
# oversample_label9 = SMOTE(sampling_strategy=sampling_strategy_label9, k_neighbors=k_neighbors, random_state=42)
# X_resampled_label9, y_resampled_label9 = oversample_label9.fit_resample(x_train, y_train)

# # 对Label13进行SMOTE
# sampling_strategy_label13 = {13: desired_sample_count}
# oversample_label13 = SMOTE(sampling_strategy=sampling_strategy_label13, k_neighbors=k_neighbors, random_state=42)
# X_resampled_label13, y_resampled_label13 = oversample_label13.fit_resample(x_train, y_train)

# # 对Label8进行SMOTE
# sampling_strategy_label8 = {8: desired_sample_count}
# oversample_label8 = SMOTE(sampling_strategy=sampling_strategy_label8, k_neighbors=k_neighbors, random_state=42)
# X_resampled_label8, y_resampled_label8 = oversample_label8.fit_resample(x_train, y_train)