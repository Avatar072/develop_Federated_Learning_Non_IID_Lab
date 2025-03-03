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
from mytoolfunction import generatefolder, ChooseLoadNpArray,ParseCommandLineArgs,ChooseTrainDatastes
from mytoolfunction import ChooseUseModel, getStartorEndtime,EvaluatePercent
from collections import Counter
from Add_ALL_LayerToCount import DoCountModelWeightSum,evaluateWeightDifferences
from Add_ALL_LayerToCount import Calculate_Weight_Diffs_Distance_OR_Absolute
from colorama import Fore, Back, Style, init
# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)
#CICIIDS2017 or Edge 62個特徵
# labelCount = 15
#TONIOT 44個特徵
# labelCount = 10
#CICIIDS2019
# labelCount = 13
#Wustl 41個特徵
# labelCount = 5
#Kub 36個特徵
# labelCount = 4
#CICIIDS2017、TONIOT、CICIIDS2019 聯集
# labelCount = 35

# CICIDS2017、CICIDS2018、CICIDS2019 聯集
labelCount = 27

# CICIIDS2017、TONIOT、EdgwIIOT 聯集
# labelCount = 31

filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
start_IDS = time.time()
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 返回gpu数量；
torch.cuda.device_count()
# 返回gpu名字，设备索引默认从0开始；
torch.cuda.get_device_name(0)
# 返回当前设备索引；
torch.cuda.current_device()
print(f"DEVICE: {DEVICE}")
#python client.py --dataset_split client1_train --epochs 50 --method normal
#python client.py --dataset_split client2_train --epochs 50 --method normal
#python client.py --dataset_split client3_train --epochs 50 --method normal
file, num_epochs,Choose_method = ParseCommandLineArgs(["dataset_split", "epochs", "method"])
print(f"Dataset: {file}")
print(f"Number of epochs: {num_epochs}")
print(f"Choose_method: {Choose_method}")
x_train, y_train, client_str = ChooseLoadNpArray(filepath, file, Choose_method)

counter = Counter(y_train)
y_train = y_train.astype(int)
print(counter)
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# generatefolder(filepath, "\\FL_AnalyseReportfolder")
generatefolder(f"./FL_AnalyseReportfolder/", today)
generatefolder(f"./FL_AnalyseReportfolder/{today}/", client_str)
generatefolder(f"./FL_AnalyseReportfolder/{today}/{client_str}/", Choose_method)
getStartorEndtime("starttime",start_IDS,f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}")


# # 20240316 after do labelencode and minmax
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y.npy", allow_pickle=True)
# # 20240317 after do labelencode and minmax add toniot
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_add_toniot.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_add_toniot.npy", allow_pickle=True)
# 20240317 after do labelencode and minmax add toniot remove all ip port
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_cicids2017_toniot_remove_ip_port.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_cicids2017_toniot_remove_ip_port.npy", allow_pickle=True)
# 20240317 after do labelencode and minmax tonniot add cicids2017 39 feature then PCA 
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_cicids2017_toniot_PCA.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_cicids2017_toniot_PCA.npy", allow_pickle=True)
# # 20240317 after do labelencode and minmax cicids2017 PCA 38 
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_cicids2017_toniot_PCA_38.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_cicids2017_toniot_PCA_38.npy", allow_pickle=True)
# 20240323 after do labelencode and minmax cicids2017 ALLDay and toniot  Chi_square_45
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_cicids2017_toniot_Chi_square_45.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_cicids2017_toniot_Chi_square_45.npy", allow_pickle=True)
# 20240319 after do labelencode and minmax cicids2017 ALLDay and toniot  Chi_square_45
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_cicids2017_toniot_Chi_square_45_change_ip_encode.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_cicids2017_toniot_Chi_square_45_change_ip_encode.npy", allow_pickle=True)
# 20240323 after do labelencode and minmax cicids2017 ALLDay and toniot  Chi_square_45 change ts change ip
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_cicids2017_toniot_Chi_square_45_change_ip.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_cicids2017_toniot_Chi_square_45_change_ip.npy", allow_pickle=True)

# 20240428 after do labelencode and minmax cicids2017 ALLDay and toniot  Chi_square_45
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_cicids2017_toniot_cicids2019_Chi_square_45.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_cicids2017_toniot_cicids2019_Chi_square_45.npy", allow_pickle=True)


# 20240502 after do labelencode and minmax cicids2017 ALLDay  iid
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2017\\ALLday\\x_ALLDay_test_20240502.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\\CICIDS2017\\ALLday\\y_ALLDay_test_20240502.npy", allow_pickle=True)

# 20240502 CIC-IDS2019 after do labelencode and minmax iid
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_20240502.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_20240502.npy", allow_pickle=True)

# 20240506 after do labelencode and minmax cicids2017 ALLDay and toniot  cicids2019 Chi_square_45
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_x_cicids2017_toniot_cicids2019_Chi_square_45_change_ip.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\merged_y_cicids2017_toniot_cicids2019_Chi_square_45_change_ip.npy", allow_pickle=True)

# 20240507 after do labelencode and minmax Edge iid
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\Edge\\x_Resplit_test_20240507.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\Edge\\y_Resplit_test_20240507.npy", allow_pickle=True)

# 20240507 after do labelencode and minmax Kub iid
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\Kub\\x_Resplit_test_20240507.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\Kub\\y_Resplit_test_20240507.npy", allow_pickle=True)

# # 20240507 after do labelencode and minmax Wustl iid
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\Wustl\\x_Resplit_test_20240507.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\Wustl\\y_Resplit_test_20240507.npy", allow_pickle=True)

# 20240523 after do labelencode and minmax cicids2017 ALLDay and toniot EdgwIIoT Chi_square_45
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_and_EdgeIIoT_test_combine\\merged_x_cicids2017_toniot_EdgeIIOT_Chi_square_45.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_and_EdgeIIoT_test_combine\\merged_y_cicids2017_toniot_EdgeIIOT_Chi_square_45.npy", allow_pickle=True)

# 20240523 TONIoT after do labelencode and minmax  75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_test_ToN-IoT_20240523.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_test_ToN-IoT_20240523.npy", allow_pickle=True)   


# 20240523 TONIoT after do labelencode and minmax  75 25分 DOJSMA attack for FL Client3
# x_test  = np.load(f"./Adversarial_Attack_Test/20240722_FL_cleint3_.0.5_0.02/x_DoJSMA_test_20240722.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20240722_FL_cleint3_.0.5_0.02/y_DoJSMA_test_20240722.npy")

# 20240523 TONIoT after do labelencode and minmax  75 25分
x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017_and_CICIDS2018_CICIDS2019_test\\merged_x_Non_IID_ALL_test.npy", allow_pickle=True)
y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017_and_CICIDS2018_CICIDS2019_test\\merged_y_Non_IID_ALL_test.npy", allow_pickle=True)   

counter = Counter(y_test)
print("test",counter)

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# 将测试数据移动到GPU上
x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)

print("Minimum label value:", min(y_train))
print("Maximum label value:", max(y_train))

# 定义训练和评估函数
def train(net, trainloader, epochs):
    print("train")
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    if client_str == "client3":  
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
    else:
        # 學長的參數
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

    for epoch in range(epochs):
        print("epoch",epoch)
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            output = net(images)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            ###訓練的過程    
        test_accuracy = test(net, testloader, start_IDS, client_str, "local_test",False)
        print(f"訓練週期 [{epoch+1}/{epochs}] - 測試準確度: {test_accuracy:.4f}")
        
    return test_accuracy

def test(net, testloader, start_time, client_str, str_globalOrlocal,bool_plot_confusion_matrix):
    print("test")
    correct = 0
    total = 0
    loss = 0  # 初始化损失值为0
    ave_loss = 0
    # with torch.no_grad():
    #     for images, labels in tqdm(testloader):
    #         output = net(images)
    #         _, predicted = torch.max(output.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # 迭代测试数据集
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
            # 使用神经网络模型进行前向传播
            outputs = net(images)
        
            # 计算损失
            loss += criterion(outputs, labels).item()
        
         # 计算预测的类别
            _, predicted = torch.max(outputs.data, 1)
        
            # 统计总样本数和正确分类的样本数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            # 计算滑动平均损失
            ave_loss = ave_loss * 0.9 + loss * 0.1

            # 将标签和预测结果转换为 NumPy 数组
            y_true = labels.data.cpu().numpy()
            y_pred = predicted.data.cpu().numpy()
        
            # 计算每个类别的召回率
            acc = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # print("correct:\n",correct)
            # print("total:\n",total)
            accuracy = correct / total
            #print("acc:\n",acc)
            # 将每个类别的召回率写入 "recall-baseline.csv" 文件
            # RecordRecall是用来存储每个类别的召回率（recall）值的元组
            # RecordAccuracy是用来存储其他一些数据的元组，包括整体的准确率（accuracy）
            #RecordRecall = []
            RecordRecall = ()
            RecordAccuracy = ()
            # labelCount = len(np.unique(y_train))# label數量要記得改
            print("labelCount:\n",labelCount)

            for i in range(labelCount):
                RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
                #RecordRecall.append(acc[str(i)]['recall'])    
            RecordAccuracy = RecordAccuracy + (accuracy, time.time() - start_time,)
          

            RecordRecall = str(RecordRecall)[1:-1]

            # 标志来跟踪是否已经添加了标题行
            header_written = False
            with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/recall-baseline_{client_str}_{str_globalOrlocal}.csv", "a+") as file:
                # file.write(str(RecordRecall))
                # file.writelines("\n")
                # 添加标题行
                #file.write("Label," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                # 写入Recall数据
                # file.write(f"{client_str}_Recall," + str(RecordRecall) + "\n")
                file.write(f"{RecordRecall}\n")
        
            # 将总体准确率和其他信息写入 "accuracy-baseline.csv" 文件
            with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}_{str_globalOrlocal}.csv", "a+") as file:
                # file.write(str(RecordAccuracy))
                # file.writelines("\n")
                # 添加标题行
                # file.write(f"{client_str}_Accuracy,Time\n")
                # 写入Accuracy数据
                file.write(f"{RecordAccuracy}\n")

            # 生成分类报告
            GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # 将字典转换为 DataFrame 并转置
            report_df = pd.DataFrame(GenrateReport).transpose()
            # 保存为 baseline_report 文件
            report_df.to_csv(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/baseline_report_{client_str}_{str_globalOrlocal}.csv",header=True)
    draw_confusion_matrix(y_true, y_pred, str_globalOrlocal, bool_plot_confusion_matrix)
    accuracy = correct / total
    print(f"測試準確度: {accuracy:.4f}")
    return accuracy

# 畫混淆矩陣
def draw_confusion_matrix(y_true, y_pred, str_globalOrlocal, bool_plot_confusion_matrix = False):
    #混淆矩陣
    if bool_plot_confusion_matrix:
        # df_cm的PD.DataFrame 接受三個參數：
        # arr：混淆矩陣的數據，這是一個二維陣列，其中包含了模型的預測和實際標籤之間的關係，以及它們在混淆矩陣中的計數。
        # class_names：類別標籤的清單，通常是一個包含每個類別名稱的字串清單。這將用作 Pandas 資料幀的行索引和列索引，以標識混淆矩陣中每個類別的位置。
        # class_names：同樣的類別標籤的清單，它作為列索引的標籤，這是可選的，如果不提供這個參數，將使用行索引的標籤作為列索引
        arr = confusion_matrix(y_true, y_pred)
        # class_names = [str(i) for i in range(labelCount)]
        #TONIoT
        # class_names = {
                        # 0: 'BENIGN', 
                        # 1: 'DDoS', 
                        # 0: 'normal', 
                        # 1: 'ddos',
                        # 2: 'backdoor', 
                        # 3: 'dos', 
                        # 4: 'injection', 
                        # 5: 'mitm', 
                        # 6: 'password', 
                        # 7: 'ransomware', 
                        # 8: 'scanning', 
                        # 9: 'xss', 
                        # 10: '10_PortScan', 
                        # 11: '11_SSH-Patator', 
                        # 12: '12_Web Attack Brute Force', 
                        # 13: '13_Web Attack Sql Injection', 
                        # 14: '14_Web Attack XSS'
                        # 15: '15_backdoor',
                        # 16: '16_dos',
                        # 17: '17_injection',
                        # 18: '18_mitm',
                        # 19: '19_password',
                        # 20: '20_ransomware',
                        # 21: '21_scanning',
                        # 22: '22_xss'
                        # } 
        # CICIDS2017 TONIOT CICIDS2019
        # class_names = {
        #                 0: '0_BENIGN', 
        #                 1: '1_Bot', 
        #                 2: '2_DDoS', 
        #                 3: '3_DoS GoldenEye', 
        #                 4: '4_DoS Hulk', 
        #                 5: '5_DoS Slowhttptest', 
        #                 6: '6_DoS slowloris', 
        #                 7: '7_FTP-Patator', 
        #                 8: '8_Heartbleed', 
        #                 9: '9_Infiltration', 
        #                 10: '10_PortScan', 
        #                 11: '11_SSH-Patator', 
        #                 12: '12_Web Attack Brute Force', 
        #                 13: '13_Web Attack Sql Injection', 
        #                 14: '14_Web Attack XSS',
        #                 15: '15_backdoor',
        #                 16: '16_dos',
        #                 17: '17_injection',
        #                 18: '18_mitm',
        #                 19: '19_password',
        #                 20: '20_ransomware',
        #                 21: '21_scanning',
        #                 22: '22_xss',
        #                 23: '23_DrDoS_DNS',
        #                 24: '24_DrDoS_LDAP',
        #                 25: '25_DrDoS_MSSQL',
        #                 26: '26_DrDoS_NTP', 
        #                 27: '27_DrDoS_NetBIOS',
        #                 28: '28_DrDoS_SNMP' ,
        #                 29: '29_DrDoS_SSDP',
        #                 30: '30_DrDoS_UDP',
        #                 31: '31_Syn',
        #                 32: '32_TFTP',
        #                 33: '33_UDPlag',
        #                 34: '34_WebDDoS' 
        #                 }
        # CICIDS2017 TONIOT EdgeIIOT 
        # class_names = {
        #                 0: '0_BENIGN', 
        #                 1: '1_Bot', 
        #                 2: '2_DDoS', 
        #                 3: '3_DoS GoldenEye', 
        #                 4: '4_DoS Hulk', 
        #                 5: '5_DoS Slowhttptest', 
        #                 6: '6_DoS slowloris', 
        #                 7: '7_FTP-Patator', 
        #                 8: '8_Heartbleed', 
        #                 9: '9_Infiltration', 
        #                 10: '10_PortScan', 
        #                 11: '11_SSH-Patator', 
        #                 12: '12_Web Attack Brute Force', 
        #                 13: '13_Web Attack Sql Injection', 
        #                 14: '14_Web Attack XSS',
        #                 15: '15_backdoor',
        #                 16: '16_dos',
        #                 17: '17_injection',
        #                 18: '18_mitm',
        #                 19: '19_password',
        #                 20: '20_ransomware',
        #                 21: '21_scanning',
        #                 22: '22_xss',
        #                 23: '23_DDoS_UDP',
        #                 24: '24_DDoS_ICMP',
        #                 25: '25_SQL_injection',
        #                 26: '26_Vulnerability_scanner', 
        #                 27: '27_DDoS_TCP',
        #                 28: '28_DDoS_HTTP' ,
        #                 29: '29_Uploading',
        #                 30: '30_Fingerprinting'
        #                 }      
        # CICIDS2017 CICIDS2018 CICIDS2019
        class_names = {
                        0: '0_BENIGN', 
                        1: '1_Bot', 
                        2: '2_DDoS', 
                        3: '3_DoS GoldenEye', 
                        4: '4_DoS Hulk', 
                        5: '5_DoS Slowhttptest', 
                        6: '6_DoS slowloris', 
                        7: '7_Infilteration', 
                        8: '8_Web Attack', 
                        9: '9_Heartbleed', 
                        10: '10_PortScan', 
                        11: '11_FTP-BruteForce', 
                        12: '12_FTP-Patator', 
                        13: '13_SSH-Bruteforce', 
                        14: '14_SSH-Patator',
                        15: '15_DrDoS_DNS',
                        16: '16_DrDoS_LDAP',
                        17: '17_DrDoS_MSSQL',
                        18: '18_DrDoS_NTP',
                        19: '19_DrDoS_NetBIOS',
                        20: '20_DrDoS_SNMP',
                        21: '21_DrDoS_SSDP',
                        22: '22_DrDoS_UDP',
                        23: '23_Syn',
                        24: '24_TFTP',
                        25: '25_UDPlag',
                        26: '26_WebDDoS'
                        }     
        # class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11','12','13','14','15','16','17','18','20','21']
        # class_names = ['0', '1', '2', '3']
        df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names)
        
        # 設置字體比例
        sns.set(font_scale=1.2)
        
        # 設置圖像大小和繪製熱圖
        plt.figure(figsize = (20,10))

        # 使用 heatmap 繪製混淆矩陣
        # annot=True 表示在單元格內顯示數值
        # fmt="d" 表示數值的格式為整數
        # cmap='BuGn' 設置顏色圖
        # annot_kws={"size": 13} 設置單元格內數值的字體大小
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn', annot_kws={"size": 15})
        
        # 設置標題和標籤
        plt.title(client_str +"_"+ Choose_method) # 圖片標題
        plt.xlabel("prediction",fontsize=15) # x 軸標籤
        plt.ylabel("Label (ground truth)", fontsize=18) # y 軸標籤
       
        # 設置 x 軸和 y 軸的字體大小和旋轉角度
        plt.xticks(rotation=0, fontsize=15) # x 軸刻度標籤不旋轉，字體大小為 15
        plt.yticks(rotation=0, fontsize=15) # y 軸刻度標籤不旋轉，字體大小為 15
        
        # 調整圖像間距
        # left, right, top, bottom 控制圖像在畫布上的邊距
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)
        # 保存圖像到指定路徑
        plt.savefig(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_{str_globalOrlocal}_confusion_matrix.png")
        plt.close('all')  # 清除圖形對象
        # plt.show()

# 創建用於訓練和測試的DataLoader
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=512, shuffle=True)  # 设置 shuffle 为 True
# test_data 的batch_size要設跟test_data(y_test)的筆數一樣 重要!!!
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
# #############################################################################
# 2. 使用 Flower 集成的code
# #############################################################################

# 定義Flower客戶端類
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.Record_List = [0] * 4  # 假設您需要至少 4 個元素 列表已經被初始化
        self.Current_total_Local_weight_sum = 0
        self.Current_total_FedAVG_weight_sum = 0
        self.Previous_Unattack_round_total_FedAVG_weight_sum = 0 #用於保存上一回合聚合後的未受攻擊汙染的權重
        self.Record_Previous_total_FedAVG_weight_sum = 0 #用於紀錄上一回合聚合後的的權重，權重可能已遭受到汙染
        self.Previous_Temp = 0
        self.global_round =0
        self.Local_train_accuracy = 0
        self.current_array = np.zeros(4)
        self.previous_array = np.zeros(4)
        self.client_id = str(client_str)
        self.original_trainloader = trainloader  # 保存原始訓練數據
        self.Reocrd_global_model_accuracy =0
        self.AllLayertotalSum_diff = 0
        self.Record_Local_Current_total_FedAVG_weight_sum = 0
        ####### dis
        self.Previous_Unattack_round_total_weight_diff_dis = 0 #用於保存上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以距離)
        self.Current_total_weight_diff_dis = 0 #當前回合全局模型與本地端模型間權重差異總和
        self.Previous_diff_dis_Temp = 0
        self.Record_Previous_total_weight_diff_dis = 0
        self.dis_percent_diff = 0
        self.Record_dis_percent_diff = 0
        self.Unattck_dis_percent_diff = 0
        self.LastRound_UnattackCounter = 0 # 用來計數最後一次的正常FedAvg後的模型
        self.bool_Unattack_Judage = True
        self.dis_threshold = 0
        self.Unattck_dis_threshold = 0
        ####### dis
        self.Previous_total_weight_diff_abs = 0 #用於保存上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以絕對值)
        self.Current_total_weight_diff_abs = 0 #當前每一回合全局模型與本地端模型間權重差異總和
        self.Current_total_weight_diff_dis_Norm = 0#當前每一回合全局模型與本地端模型間權重差異總和
        self.Previous_total_weight_diff_dis_Norm = 0 #用於保存上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以距離範數)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):#是知識載入
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
       
        #####################################################初始化一個模型#####################################################
        # 用於保存聚合後的未受攻擊汙染的全局模型
        # After_FedAVG_model_unattack_initial_model = ChooseUseModel("MLP", 44, labelCount)
        # 獲取初始化模型的 state_dict
        # After_FedAVG_model_unattack = After_FedAVG_model_unattack_initial_model.state_dict()
        #####################################################初始化一個模型#####################################################

        # 要記錄未受到攻擊聚合後的全局模型 先載入net這個state_dict當作初始化使用
        After_FedAVG_model_unattack = net
        # 紀錄全局回合次數
        self.global_round += 1
        print(f"Current global round: {self.global_round}")
        
        # 需紀錄上一回合FedAVG後的權重總和，這邊權重可能已遭受到攻擊而汙染
        self.Record_Previous_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
        # 需紀錄上一回合FedAVG後與Local train的模型間權重差異總和，這邊權重可能已遭受到攻擊而汙染
        self.Record_Previous_total_weight_diff_dis = self.Current_total_weight_diff_dis
       
        if(not self.bool_Unattack_Judage):
            # 需紀錄上一回合未操受攻擊FedAVG後與Local train的模型間權重差異總和，權重變化百分比
            # self.Record_dis_percent_diff = self.Unattck_dis_percent_diff
            # 測試
            self.Record_dis_percent_diff = self.dis_percent_diff
        else:
            # 需紀錄上一回合FedAVG後與Local train的模型間權重差異總和，權重變化百分比
            self.Record_dis_percent_diff = self.dis_percent_diff

        print("Last_round_After_FedAVG_may have been attacked", self.Record_Previous_total_FedAVG_weight_sum)
        print("Last_round_After_FedAVG_may have been attacked distance", self.Record_Previous_total_weight_diff_dis)
        print("Last_round_After_FedAVG_may have been attacked distance raise percent", self.Record_dis_percent_diff)

        #####################################################用accuracy保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################
        # if self.Reocrd_global_model_accuracy >= 0.8:
        #         self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
        #         # 保存上一回合未受到攻擊FedAVG後的正常權重總和 進行後續權重每層總和求差異計算
        #         self.Previous_Temp = self.Previous_Unattack_round_total_FedAVG_weight_sum
        #         print("Last_round_After_FedAVG", self.Previous_Unattack_round_total_FedAVG_weight_sum)

        #         # 保存上一回合未受到攻擊剛聚合完的全局模型
        #         torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack.pth")        
        # else:
        #         if self.global_round <=10:
        #              self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
        #         print("Previous_total_FedAVG_weight_sum", self.Previous_Unattack_round_total_FedAVG_weight_sum)
        #         print("Last_round_After_FedAVG_normal", self.Previous_Unattack_round_total_FedAVG_weight_sum)
        #####################################################用accuracy保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################

        # 更新客戶端模型參數為新的全局模型參數
        self.set_parameters(parameters)# 剛聚合完的權重 # 置新參數之前保存權重
        
        """
        global test 對每global round剛聚合完的gobal model進行測試 要在Local_train之前測試
        通常第1 round測出來會是0
        """
        # 保存模型剛聚合完的全局模型
        torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/gobal_model_Before_local_train_model_round_{self.global_round}.pth")
        After_FedAVG_model = torch.load(f'./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/gobal_model_Before_local_train_model_round_{self.global_round}.pth')
        # 測試剛聚合完的全局模型
        accuracy = test(net, testloader, start_IDS, client_str,f"global_test",True)
        self.Reocrd_global_model_accuracy = accuracy
        print("accuracy",accuracy)
        print("Reocrd_global_model_accuracy",self.Reocrd_global_model_accuracy)

        # 算聚合完的全局模型每層權重加總總和
        ######################################################Fedavg完的模型每層加總總和############################################# 
        weights_after_FedAVG = net.state_dict()
        # True 以絕對值加總 False 為直接加總
        # self.Current_total_FedAVG_weight_sum = DoCountModelWeightSum(weights_after_FedAVG,True,"After_FedAVG") 
        self.Current_total_FedAVG_weight_sum = DoCountModelWeightSum(weights_after_FedAVG,False,"After_FedAVG") 

        #####################################################保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################
        # if self.global_round <=10:
        #     # 初始回合直接給1當亂數初值
        #     if self.global_round == 1 :
        #         self.Current_total_FedAVG_weight_sum = 1
        #     print("After_FedAVG",self.Current_total_FedAVG_weight_sum)
        #     self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
        #     print("Previous_total_FedAVG_weight_sum", self.Previous_Unattack_round_total_FedAVG_weight_sum)
        # else:
        #     # weight_diff = float(self.Record_List[2])
        #     weight_diff = float(self.AllLayertotalSum_diff)
        #     print("**********************weight_diff**********************", weight_diff)
        #     # 權重差異超過當前本地權重總和的5%就要過濾掉                 
        #     # threshold = float(self.Record_List[0]) * 0.05
        #     threshold = float(self.Current_total_FedAVG_weight_sum) * 0.05
        #     print("**********************threshold**********************", threshold)
        #     print("**********************self.Previous_Unattack_round_total_FedAVG_weight_sum**********************", self.Previous_Unattack_round_total_FedAVG_weight_sum)
        #     if  weight_diff <= threshold:
        #             # 保存上一回合未受到攻擊FedAVG後的正常權重總和 進行後續權重每層總和求差異計算
        #             self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
        #             print("**********************Last_round_After_FedAVG_Unattack_round_total_FedAVG_weight_sum:**********************", self.Previous_Unattack_round_total_FedAVG_weight_sum)
        #             self.Previous_Temp = self.Previous_Unattack_round_total_FedAVG_weight_sum
        #             # 保存上一回合未受到攻擊剛聚合完的全局模型
        #             torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack.pth")
        #             print(f"Model saved to ./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack.pth with updated weights.")
        #     else:
        #             # 當條件不符合時，打印未更新的變數值並使用上一回合未受到攻擊的總和
        #             self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Previous_Temp
        #             print("**********************No update performed on Previous_Unattack_round_total_FedAVG_weight_sum.**********************")                
        #####################################################保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################

        #####################################################Fedavg完的模型每層加總總和############################################# 

        #  寫入Accuracy文件
        with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/accuracy-gobal_model_{client_str}.csv", "a+") as file:
            file.write(f"{accuracy}\n")
        
        
        #####################################################對抗式攻擊設定#################################################   
        ### 訓練中途加入JSMA Attack or FGSM attack
        # if (self.global_round >= 50 and self.global_round <= 100) and self.client_id == "client3":
        # if (self.global_round >= 50 and self.global_round <= 125) and self.client_id == "client3":
        # if (self.global_round >= 30 and self.global_round <= 80)  and self.client_id == "client2":
        # #     print(f"*********************{self.client_id}在第{self.global_round}回合開始使用被攻擊的數據*********************************************")
            
        # #     # 載入被FGSM攻擊的數據
        #     x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoFGSM_train_half2_20240827.npy", allow_pickle=True)
        #     y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoFGSM_train_half2_20240827.npy", allow_pickle=True)
            
        #     x_train_attacked = torch.from_numpy(x_train_attacked).type(torch.FloatTensor).to(DEVICE)
        #     y_train_attacked = torch.from_numpy(y_train_attacked).type(torch.LongTensor).to(DEVICE)
            
        #     train_data_attacked = TensorDataset(x_train_attacked, y_train_attacked)
        #     trainloader = DataLoader(train_data_attacked, batch_size=512, shuffle=True)
        # el
        if self.global_round >= 50  and self.client_id == "client3":
            print(f"*********************{self.client_id}在第{self.global_round}回合開始使用被攻擊的數據*********************************************")
            
            # 載入被JSMA攻擊的數據 theta=0.05
            # x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half3_20240801.npy", allow_pickle=True)
            # y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half3_20240801.npy", allow_pickle=True)
            
            # 載入被JSMA攻擊的數據 theta=0.1
            # x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half3_20240901_theta_0.1.npy", allow_pickle=True)
            # y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half3_20240901_theta_0.1.npy", allow_pickle=True)
            
            # 載入被JSMA攻擊的數據 theta=0.15
            # x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half3_theta_0.15_20240901.npy", allow_pickle=True)
            # y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half3_theta_0.15_20240901.npy", allow_pickle=True)
            
            # 載入被JSMA攻擊的數據 theta=0.2
            # x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half3_theta_0.2_20240901.npy", allow_pickle=True)
            # y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half3_theta_0.2_20240901.npy", allow_pickle=True)
            
            # 載入被JSMA攻擊的數據 theta=0.25
            # x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoJSMA_train_half3_theta_0.25_20240901.npy", allow_pickle=True)
            # y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoJSMA_train_half3_theta_0.25_20240901.npy", allow_pickle=True)
            
            # 載入被FGSM攻擊的數據
            # x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoFGSM_train_half3_20240826.npy", allow_pickle=True)
            # y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoFGSM_train_half3_20240826.npy", allow_pickle=True)
            
            # 載入被PGD攻擊的數據
            # x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_DoPGD_train_half3_20241017.npy", allow_pickle=True)
            # y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_DoPGD_train_half3_20241017.npy", allow_pickle=True)
            
            # 載入正常的數據
            # x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_dataframes_random_train_half3_20240523.npy", allow_pickle=True)
            # y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_dataframes_random_train_half3_20240523.npy", allow_pickle=True)  
            
            
            # 載入正常的non-iid數據
            print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2019 after do labelencode do pca" +f"cicids2019 with normal attack type")
            x_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\Noniid\\CICIDS2019_AddedLabel_Noniid_x.npy", allow_pickle=True)
            y_train_attacked = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\Noniid\\CICIDS2019_AddedLabel_Noniid_y.npy", allow_pickle=True)
            
            x_train_attacked = torch.from_numpy(x_train_attacked).type(torch.FloatTensor).to(DEVICE)
            y_train_attacked = torch.from_numpy(y_train_attacked).type(torch.LongTensor).to(DEVICE)
            
            train_data_attacked = TensorDataset(x_train_attacked, y_train_attacked)
            trainloader = DataLoader(train_data_attacked, batch_size=512, shuffle=True)
        else:
            print(f"*********************在第{self.global_round}回合結束攻擊*********************************************")
            trainloader = self.original_trainloader
        #####################################################對抗式攻擊設定#################################################  


        ####################################################歐基里德距離-模型每層差異求總和#################################################  
        # 先強制存第49round global當測試
        # if (self.global_round == 49):
        #     torch.save(net.state_dict(),f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack_49.pth")
      
        
        # 檢查 state_dict1 和 state_dict2 是否為 None

        if After_FedAVG_model is None:
            print("After_FedAVG_model is None")
        if After_FedAVG_model_unattack is None:
            print("After_FedAVG_model_unattack is None")
        # 上一回合未受到攻擊剛聚合完的全局模型 
        # 前10回合當前的聚合後的模型
        if self.global_round <=10: 
            try:
                After_FedAVG_model_unattack = After_FedAVG_model
            except Exception as e:
                print(f"Error loading After_FedAVG_model: {e}")
                After_FedAVG_model = None
        else:
            # After_FedAVG_model_unattack = After_FedAVG_model
            # 上一回合的距離跟這一回比距離突然大幅增大表示受到攻擊
            percent_threshold = 100 
            threshold = self.Record_dis_percent_diff
            print("*********bbbbbb*********Record_Previous_total_weight_diff_dis**********************", self.Record_Previous_total_weight_diff_dis)
            if threshold >= percent_threshold: #假設超過一百視為攻擊
                self.bool_Unattack_Judage = False #大於100那瞬間開始使用正常的模型
                self.Previous_diff_dis_Temp  = self.Previous_Unattack_round_total_weight_diff_dis 
                try:
                    After_FedAVG_model_unattack = torch.load(f'./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack_distance.pth')
                
                except Exception as e:
                    print(f"Error loading fedavg_unattack: {e}")
                    After_FedAVG_model_unattack = None
            
            # # 當條件不符合時，打印未更新的變數值並使用上一回合未受到攻擊的總和
            else:
                # 經測試在一進入第50回合就已受到攻擊影響 所以要在第50回合前 還沒進入else且bool_Unattack_Judage == False之前就存正常的global model存起來
                print(f"*********************在第{self.global_round}回合進入else*********************************************")
                
                # True表示正常未受到攻擊
                if self.bool_Unattack_Judage:
                    # 保存上一回合未受到攻擊FedAVG後的正常每層權重差異總和
                    self.Previous_diff_dis_Temp = self.Current_total_weight_diff_dis
                    # self.Previous_Unattack_round_total_weight_diff_dis = self.Previous_diff_dis_Temp   
                    self.Previous_Unattack_round_total_weight_diff_dis = self.Current_total_weight_diff_dis                    
                 
                    # 保存最後上一回合未受到攻擊剛聚合完的全局模型
                    After_FedAVG_model_unattack = net.state_dict()
                    torch.save(After_FedAVG_model_unattack,f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack_distance.pth")                
                
                # 先不用 if else 方便除錯
                # False表示受到攻擊
                if self.bool_Unattack_Judage == False:
                    # 要分析不加任何過濾機制時要加上註解
                    # 載入正常的global model
                    After_FedAVG_model_unattack = torch.load(f'./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack_distance.pth')
                   
                print("**********************No update performed on Previous_Unattack_round_total_weight_diff_dis.**********************")                

        ####################################################模型每層差異求總和#################################################  


        # 在本地訓練階段前保存模型
        weights_before_Localtrain = net.state_dict()
        torch.save(weights_before_Localtrain, 
                   f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/Local_model_before_local_train_{self.global_round}.pth")
        #####################################################本地訓練階段#################################################  
        # 本地訓練階段
        self.Local_train_accuracy = train(net, trainloader, epochs=num_epochs)
        #####################################################本地訓練階段#################################################  

        # 在本地訓練階段後保存模型
        weights_after_Localtrain = net.state_dict()
        torch.save(weights_after_Localtrain, f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/Local_model_After_local_train.pth")
        # if self.global_round >1: 
        # 載入本地訓練後的模型
        weights_after_Localtrain = torch.load(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/Local_model_After_local_train.pth")
        
        # if weights_after_Localtrain is None:
        #     print("weights_after_Localtrain is None")
        ######################################################Local train完的模型每層加總總和############################################# 
        print("Weights after local training:")
        # True 以絕對值加總 False 為直接加總
        # self.Current_total_Local_weight_sum = DoCountModelWeightSum(weights_after_Localtrain,
        #                                   True,
        #                                 self.client_id)    
        self.Current_total_Local_weight_sum = DoCountModelWeightSum(weights_after_Localtrain,False,self.client_id) 
        # 打印模型每層加總後權重總重總和
        print("self.Current_total_Local_weight_sum",self.Current_total_Local_weight_sum)
        print("self.Current_total_FedAVG_weight_sum",self.Current_total_FedAVG_weight_sum)
        # self.Record_List[0] = self.Current_total_Local_weight_sum
        self.Record_Local_Current_total_FedAVG_weight_sum = self.Current_total_Local_weight_sum
        ######################################################Local train完的模型每層加總總和############################################# 
        
        ##########################################計算兩個模型的每層權重差距 將每層權重差距值相加（以距離(distance)計算）##########################################
        # Calculate_Weight_Diffs_Distance_OR_Absolute True表示計算L2範數 False表示計算歐基里德距離

        #########################################################################以歐基里德距離#####################################################################
        # 計算兩個模型的每層權重差距 當前每一回合聚合後的全局模型與本地端模型間權重差異總和(以歐基里德距離)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/weight_diffs_dis_{client_str}.csv"
        weight_diffs_dis, self.Current_total_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                           After_FedAVG_model,
                                                                                                           diff_dis_csv_file_path,
                                                                                                          "distance",
                                                                                                           False)



        
        #載入上一回合聚合後的最後一次未受攻擊汙染的全局模型
        if(not self.bool_Unattack_Judage): #fasle表示受到攻擊
            After_FedAVG_model_unattack = torch.load(f'./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack_distance.pth')

            # if (self.global_round >= 50):
            # # 強值載入正常模型當試測
            #     After_FedAVG_model_unattack = torch.load(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/fedavg_unattack_49.pth")
      
            # 計算兩個模型的每層權重差距 上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以歐基里德距離)
            diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/weight_diffs_dis_{client_str}_unattack.csv"
            weight_diffs_dis, self.Previous_Unattack_round_total_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                                           After_FedAVG_model_unattack,
                                                                                                                           diff_dis_csv_file_path,
                                                                                                                          "distance",
                                                                                                                           False)
            self.dis_percent_diff = EvaluatePercent(self.Current_total_weight_diff_dis,
                                                    self.Previous_Unattack_round_total_weight_diff_dis)
            # 類似weight average算法計算閥值 當前回合距離佔20% 上一回合未受攻擊模型距離佔80%
            self.Unattck_dis_threshold = self.Current_total_weight_diff_dis*0.2 + self.Previous_Unattack_round_total_weight_diff_dis*0.8
        else:
            # 算每一回合權重距離變化的百分比  
            # 百分比變化=(當前可能受到攻擊的距離−上一回合聚合後的未受攻擊距離/上一回合聚合後的未受攻擊距離 )×100%      
            # self.dis_percent_diff = EvaluatePercent(self.Current_total_weight_diff_dis,
            #                                         self.Previous_Unattack_round_total_weight_diff_dis)

            self.dis_percent_diff = EvaluatePercent(self.Current_total_weight_diff_dis,
                                                    self.Record_Previous_total_weight_diff_dis)
            # 類似weight average算法計算閥值 當前回合距離佔20% 上一回合距離佔80%
            self.dis_threshold = self.Current_total_weight_diff_dis*0.2 + self.Record_Previous_total_weight_diff_dis*0.8

        # 計算兩個模型的每層權重差距 將每層權重差距值相加（以L2範數計算）
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/weight_diffs_dis_{client_str}_Norm.csv"
        
        weight_diffs_dis, self.Current_total_weight_diff_dis_Norm = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                    After_FedAVG_model,
                                                                                                    diff_dis_csv_file_path,
                                                                                                    "distance",
                                                                                                    True)
        # 計算兩個模型的每層權重差距 上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以以L2範數計算)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/weight_diffs_dis_{client_str}_unattack_Norm.csv"
        weight_diffs_dis, self.Previous_total_weight_diff_dis_Norm = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                    After_FedAVG_model_unattack,
                                                                                                    diff_dis_csv_file_path,
                                                                                                    "distance",
                                                                                                    True)

        # # 計算兩個模型的每層權重差距 將每層權重差距值相加（以絕對值(absolute)計算）
        # diff_abs_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/weight_diffs_abs_{client_str}.csv"   
        # weight_diffs_abs,self.Current_total_weight_diff_abs = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
        #                                                                                             After_FedAVG_model,
        #                                                                                             diff_abs_csv_file_path,
        #                                                                                             "absolute")
        
        # # 計算兩個模型的每層權重差距 上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以絕對值)
        # diff_abs_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/weight_diffs_abs_{client_str}_unattack.csv"   
        # weight_diffs_abs,self.Previous_total_weight_diff_abs = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
        #                                                                                             After_FedAVG_model_unattack,
        #                                                                                             diff_abs_csv_file_path,
        #                                                                                             "absolute")
        ######################################################模型每層差異求總和################################################ 


        ######################################################模型每層加總後求差異##############################################        
        # local train計算權重加總 - 實際每一回合FedAVG計算權重加總
        self.current_array[0],self.current_array[1],self.current_array[2],self.current_array[3] = evaluateWeightDifferences("Local-Current_FedAVG",
                                                                                                                                    self.Current_total_Local_weight_sum, 
                                                                                                                                    self.Current_total_FedAVG_weight_sum)
            
        
        # local train計算權重加總 - 上一回合FedAVG計算權重加總(最後一次未受到攻擊的回合)
        self.previous_array[0],self.previous_array[1],self.previous_array[2],self.previous_array[3] = evaluateWeightDifferences("Local-Current_FedAVG",
                                                                                                                                        self.Current_total_Local_weight_sum,     
                                                                                                                                        self.Previous_Unattack_round_total_FedAVG_weight_sum)
        self.AllLayertotalSum_diff = self.previous_array[0]
        
        # if self.global_round <=10:
        #     #前10回合計算兩個模型加總後差值使用local train計算權重加總 - 實際每一回合FedAVG計算權重加總
        #     self.AllLayertotalSum_diff = self.current_array[0]
        # else:
        #     # 權重差異超過當前本地權重總和的5%就要過濾掉
        #     if  weight_diff <= threshold:
        #         # 因weight_diff符合條件判斷小於等5%
        #         # 計算兩個模型加總後差值AllLayertotalSum_diff使用
        #         # local train計算權重加總 - 實際每一回合FedAVG計算權重加總                 
        #         self.AllLayertotalSum_diff = self.current_array[0]
        #     else:            
        #         # 權重差異超過當前本地權重總和的5%就要過濾掉
        #         # 計算兩個模型加總後差值AllLayertotalSum_diff使用
        #         # local train計算權重加總 - 上一回合FedAVG計算權重加總(最後一次未受到攻擊的回合)
        #         self.previous_array[0],self.previous_array[1],self.previous_array[2],self.previous_array[3] = evaluateWeightDifferences("Local-Current_FedAVG",
        #                                                                                                                                 self.Current_total_Local_weight_sum,     
        #                                                                                                                                 self.Previous_Unattack_round_total_FedAVG_weight_sum)
        #         self.AllLayertotalSum_diff = self.previous_array[0]
            

        # self.Record_List[2] = self.current_array[0]                                                                                                                         
   
        ######################################################模型每層加總後求差異##############################################    
        
        print("******************Current_total_Local_weight_sum**********************", self.Record_Local_Current_total_FedAVG_weight_sum)
        print("******************Previous_total_FedAVG_weight_sum**********************", self.Previous_Unattack_round_total_FedAVG_weight_sum)
        #step1上傳給權重，#step2在server做聚合，step3往下傳給server
        # return self.get_parameters(config={}), len(trainloader.dataset), {"accuracy": accuracy}
        return self.get_parameters(config={}), len(trainloader.dataset), {"global_round": self.global_round,
                                                                          "accuracy": accuracy,
                                                                          "Local_train_accuracy": self.Local_train_accuracy,
                                                                          "client_id": self.client_id,
                                                                          "Local_train_weight_sum":self.Current_total_Local_weight_sum,
                                                                          "Previous_round_FedAVG_weight_sum":self.Previous_Unattack_round_total_FedAVG_weight_sum,
                                                                          "Current_FedAVG_weight_sum":self.Current_total_FedAVG_weight_sum,
                                                                          "Local_train_weight_sum-Current_FedAVG weight_sum":float(self.current_array[0]),
                                                                          "Local_train_weight_sum-Previous_FedAVG weight_sum": float(self.previous_array[0]),
                                                                        #   "Current_total_weight_diff_abs": float(self.Current_total_weight_diff_abs),
                                                                        #   "Previous_total_weight_diff_abs": float(self.Previous_total_weight_diff_abs),
                                                                          "dis_percent_diff": float(self.dis_percent_diff), 
                                                                          "Current_total_weight_diff_dis": float(self.Current_total_weight_diff_dis),
                                                                          "Previous_total_weight_diff_dis": float(self.Previous_Unattack_round_total_weight_diff_dis)}

    def evaluate(self, parameters, config):
        # 当前 global round 数
        print(f"Evaluating global round: {self.global_round}")
        print("client_id",self.client_id)
        # local test
        # 這邊的測試結果會受到local train的影響
        # 保存模型
        torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/Local_model_After_local_train_model.pth")
        accuracy = test(net, testloader, start_IDS, client_str,f"local_test",True)
        # 寫入Accuracyg
        with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/accuracy-local_model_{client_str}.csv", "a+") as file:
            file.write(f"{accuracy}\n")

        self.Local_train_accuracy = accuracy
        self.set_parameters(parameters)#更新現有的知識#step4 更新model
        print(f"Client {self.client_id} returning metrics: {{accuracy: {accuracy}, client_id: {self.client_id}}}")
        
        with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/Local_train_weight_sum-FedAVG weight_sum_{client_str}.csv", "a+") as file:
                file.write(f"{self.Current_total_Local_weight_sum},"
                            f"{self.Current_total_FedAVG_weight_sum},"#當前每一回FedAVG後的權重
                            f"{self.Previous_Unattack_round_total_FedAVG_weight_sum},"#上一回未受到攻擊FedAVG後的權重
                            f"{self.Record_Previous_total_FedAVG_weight_sum},"#實際上一回FedAVG後的權重
                            f"{self.current_array[0]},"#當前每一回Local_train_weight_sum-Previous_FedAVG weight_sum
                            f"{self.previous_array[0]},"#上一回未受到攻擊Local_train_weight_sum-Previous_FedAVG weight_sum
                            f"{self.Current_total_weight_diff_abs},"#模型每層差異求總和（以絕對值計算）  
                            f"{self.Previous_total_weight_diff_abs},"#上一回未受到攻擊的全局模型與本地端模型每層差異總和（以絕對值計算）
                            f"{self.Current_total_weight_diff_dis},"#模型每層差異求總和（以距離計算）
                            f"{self.Record_Previous_total_weight_diff_dis},"#上一回的全局模型與本地端每層差異總和（以距離計算）
                            f"{self.dis_percent_diff},"#上一回的全局模型與本地端每層差異總和變化百分比（以距離計算）
                            f"{self.Previous_Unattack_round_total_weight_diff_dis},"#上一回未受到攻擊的全局模型與本地端每層差異總和（以距離計算）
                            f"{self.Unattck_dis_percent_diff},"#上一回未受到攻擊的全局模型與本地端每層差異總和變化百分比（以距離計算）
                            f"{self.dis_threshold},"#類似weight average算法計算閥值 當前回合距離佔20% 上一回合距離佔80%
                            f"{self.Unattck_dis_threshold},"#類似weight average算法計算閥值 當前回合距離佔20% 上一回合未受攻擊模型距離佔80%
                            f"{self.Current_total_weight_diff_dis_Norm},"#模型每層差異求總和（以距離範數計算）
                            f"{self.Previous_total_weight_diff_dis_Norm}\n")#上一回未受到攻擊的全局模型與本地端每層差異總和（以距離範數計算）

        
        self.Current_total_Local_weight_sum = float(self.Current_total_Local_weight_sum)  # 將字符串轉換為浮點數
        print("Local_weight_sum before multiplication:", self.Current_total_Local_weight_sum, "of type:", type(self.Current_total_Local_weight_sum))
        percentage_five = self.Current_total_Local_weight_sum * 0.05
        # # 保留小数点后两位
        percentage_five = round(percentage_five, 2)
        print("Local_train_weight_sum_percentage_five\n",percentage_five)
        return accuracy, len(testloader.dataset), {"global_round": self.global_round,
                                                   "accuracy": accuracy,
                                                   "Local_train_accuracy": self.Local_train_accuracy,
                                                   "client_id": self.client_id,
                                                   "Local_train_weight_sum":self.Current_total_Local_weight_sum,
                                                   "Previous_round_FedAVG_weight_sum":self.Previous_Unattack_round_total_FedAVG_weight_sum,
                                                   "Current_FedAVG_weight_sum":self.Current_total_FedAVG_weight_sum,
                                                   "Local_train_weight_sum-Current_FedAVG weight_sum":float(self.current_array[0]),
                                                    "Current_total_weight_diff_abs": float(self.Current_total_weight_diff_abs),
                                                    "Current_total_weight_diff_dis": float(self.Current_total_weight_diff_dis),
                                                    "Previous_total_weight_diff_abs": float(self.Previous_total_weight_diff_abs),
                                                    "Previous_total_weight_diff_dis": float(self.Previous_Unattack_round_total_weight_diff_dis)}

# 初始化神经网络模型
net = ChooseUseModel("MLP", x_train.shape[1], labelCount).to(DEVICE)

# 启动Flower客户端
fl.client.start_numpy_client(
    # server_address="127.0.0.1:53388",
    server_address="192.168.1.137:53388",
    client=FlowerClient(),
    
)

#紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime",end_IDS,f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}")