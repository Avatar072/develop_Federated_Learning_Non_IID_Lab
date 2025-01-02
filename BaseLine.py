import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")#https://blog.csdn.net/qq_43391414/article/details/120543028
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from mytoolfunction import generatefolder, ChooseLoadNpArray,ChooseTrainDatastes, ParseCommandLineArgs,ChooseTestDataSet
from mytoolfunction import ChooseUseModel, getStartorEndtime
from collections import Counter
from colorama import Fore, Back, Style, init
####################################################################################################
# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)
#CICIIDS2017 or Edge 62個特徵
# labelCount = 15
#CICIIDS2019
# labelCount = 13
#TONIoT
labelCount = 10
#Wustl 41個特徵
# labelCount = 5
#Kub 36個特徵
# labelCount = 4
#CICIIDS2017、TONIOT、CICIIDS2019 聯集
# labelCount = 35
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
start_IDS = time.time()
current_time = time.strftime("%Hh%Mm%Ss", time.localtime())
print(Fore.YELLOW+Style.BRIGHT+f"當前時間: {current_time}")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.__version__)

# python BaseLine.py --dataset train_half1 --epochs 100
# python BaseLine.py --dataset train_half2 --epochs 100
# python BaseLine.py --dataset total_train --epochs 500 --method normal
file, num_epochs,Choose_method = ParseCommandLineArgs(["dataset", "epochs", "method"])
print(f"Dataset: {file}")
print(f"Number of epochs: {num_epochs}")
print(f"Choose_method: {Choose_method}")
# ChooseLoadNpArray function  return x_train、y_train 和 client_str and Choose_method
x_train, y_train, client_str = ChooseLoadNpArray(filepath, file, Choose_method)
# x_train, y_train, client_str = ChooseTrainDatastes(filepath, file, Choose_method)   
print("特徵數",x_train.shape[1])
print(y_train)
# print(client_str)
counter = Counter(y_train)
print("train筆數",counter)
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在single_AnalyseReportFolder產生天日期的資料夾
# generatefolder(filepath, "\\single_AnalyseReportFolder")
generatefolder(f"./single_AnalyseReportFolder/", today)
generatefolder(f"./single_AnalyseReportFolder/{today}/{current_time}/", client_str)
generatefolder(f"./single_AnalyseReportFolder/{today}/{current_time}/{client_str}/", Choose_method)
getStartorEndtime("starttime",start_IDS,f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}")

# 20240324 after do chi-square
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_test_cicids2017_AfterFeatureSelect44_BaseLine_SpiltIP_20240323.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_test_cicids2017_AfterFeatureSelect44_BaseLine_SpiltIP_20240323.npy", allow_pickle=True)

# 20240325 after do PCA
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_test_AfterPCA38_20240325.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_test_AfterPCA38_20240325.npy", allow_pickle=True)

# 20240502 CIC-IDS2017 after do labelencode and minmax  75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLDay_test_20240502.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLDay_test_20240502.npy", allow_pickle=True)   

# # 20240502 CIC-IDS2017 after do labelencode and minmax chi_square45 75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)    
            
# 20240422 CICIDS2019 after PCA do labelencode and minmax
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_CICIDS2019_01_12_test_20240422.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_CICIDS2019_01_12_test_20240422.npy", allow_pickle=True)

# 20240422 CICIDS2019 after PCA do labelencode and minmax chi-square 45
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_CICIDS2019_AfterFeatureSelect44_20240422.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_CICIDS2019_AfterFeatureSelect44_20240422.npy", allow_pickle=True)

# 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_20240502.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_20240502.npy", allow_pickle=True)

# 20240519 EdgeIIoT after do labelencode and minmax  75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_test_20240519.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_test_20240519.npy", allow_pickle=True)    
            
# 20240520 EdgeIIoT after do labelencode and minmax chi_square45 75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_test_AfterFeatureSelect44_20240520.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_test_AfterFeatureSelect44_20240520.npy", allow_pickle=True)            

# 20240523 TONIoT after do labelencode and minmax  75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_test_ToN-IoT_20240523.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_test_ToN-IoT_20240523.npy", allow_pickle=True)   

# 20240523 TONIoT after do labelencode and minmax  75 25分 DOJSMA
# x_test  = np.load(f"./Adversarial_Attack_Test/20240721_0.5_0.5/x_DoJSMA_test_20240721.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20240721_0.5_0.5/y_DoJSMA_test_20240721.npy")

# 20241030 TONIoT after do labelencode and minmax  75 25分 JSMA
x_test = np.load("D:\\develop_Federated_Learning_Non_IID_Lab\\Adversarial_Attack_Test\\20241030\\x_DoJSMA_test_theta_0.05_20241030.npy", allow_pickle=True)
y_test = np.load("D:\\develop_Federated_Learning_Non_IID_Lab\\Adversarial_Attack_Test\\20241030\\y_DoJSMA_test_theta_0.05_20241030.npy", allow_pickle=True)   



counter = Counter(y_test)
print("test筆數",counter)

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)


labelCount=len(y_test.unique())
print("唯一值数量:", labelCount)

# 將測試數據移動到 GPU 上
x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)



# 定義訓練函數
def train(net, trainloader, epochs):
    print("訓練中")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)

    for epoch in range(epochs):
        print("epoch",epoch)
        net.train()# PyTorch 中的一個方法，模型切換為訓練模式
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            output = net(images)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        ###訓練的過程    
        test_accuracy = test(net, testloader, start_IDS, client_str,False)
        print(f"訓練週期 [{epoch+1}/{epochs}] - 測試準確度: {test_accuracy:.4f}")

# 定義測試函數
def test(net, testloader, start_time, client_str,plot_confusion_matrix):
    # print("測試中")
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    ave_loss = 0.0  # 初始化 ave_loss

    net.eval()  #PyTorch 中的一個方法，用於將神經網絡模型設置為測試模式
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 計算滑動平均損失
            ave_loss = ave_loss * 0.9 + loss * 0.1

            # 將標籤和預測結果轉換為 NumPy 陣列
            y_true = labels.data.cpu().numpy()
            y_pred = predicted.data.cpu().numpy()
        
            # 計算每個類別的召回率
            acc = classification_report(y_true, y_pred, digits=4, output_dict=True)
            accuracy = correct / total

            # 將每個類別的召回率寫入 "recall-baseline.csv" 檔案
            RecordRecall = ()
            RecordAccuracy = ()
           
            for i in range(labelCount):
                RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
                 
            RecordAccuracy = RecordAccuracy + (accuracy, time.time() - start_time,)
            RecordRecall = str(RecordRecall)[1:-1]

            # 標誌來跟踪是否已經添加了標題行
            header_written = False
            with open(f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/recall-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    # file.write("標籤," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                    header_written = True
                file.write(str(RecordRecall) + "\n")
        
            # 將總體準確度和其他信息寫入 "accuracy-baseline.csv" 檔案
            with open(f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    # file.write("標籤," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                    header_written = True
                file.write(f"精確度,時間\n")
                file.write(f"{accuracy},{time.time() - start_time}\n")

                # 生成分類報告
                GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
                report_df = pd.DataFrame(GenrateReport).transpose()
                report_df.to_csv(f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/baseline_report_{client_str}.csv",header=True)

    draw_confusion_matrix(y_true, y_pred,plot_confusion_matrix)
    accuracy = correct / total
    print(f"測試準確度: {accuracy:.4f}")
    return accuracy

# 畫混淆矩陣
def draw_confusion_matrix(y_true, y_pred, plot_confusion_matrix = False):
    #混淆矩陣
    if plot_confusion_matrix:
        # df_cm的PD.DataFrame 接受三個參數：
        # arr：混淆矩陣的數據，這是一個二維陣列，其中包含了模型的預測和實際標籤之間的關係，以及它們在混淆矩陣中的計數。
        # class_names：類別標籤的清單，通常是一個包含每個類別名稱的字串清單。這將用作 Pandas 資料幀的行索引和列索引，以標識混淆矩陣中每個類別的位置。
        # class_names：同樣的類別標籤的清單，它作為列索引的標籤，這是可選的，如果不提供這個參數，將使用行索引的標籤作為列索引
        arr = confusion_matrix(y_true, y_pred)
        #CICIDS2017
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
        #                 14: '14_Web Attack XSS'
        #                 # 15: '15_backdoor',
        #                 # 16: '16_dos',
        #                 # 17: '17_injection',
        #                 # 18: '18_mitm',
        #                 # 19: '19_password',
        #                 # 20: '20_ransomware',
        #                 # 21: '21_scanning',
        #                 # 22: '22_xss'
        #                 } 
        #TONIoT
        class_names = {
                        # 0: 'BENIGN', 
                        # 1: 'DDoS', 
                        0: 'normal', 
                        1: 'ddoS',
                        2: 'backdoor', 
                        3: 'dos', 
                        4: 'injection', 
                        5: 'mitm', 
                        6: 'password', 
                        7: 'ransomware', 
                        8: 'scanning', 
                        9: 'xss', 
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
                        } 
        # # CICIDS2019
        # class_names = {
        #                 0: '0_BENIGN', 
        #                 1: '1_DrDoS_DNS', 
        #                 2: '2_DrDoS_LDAP', 
        #                 3: '3_DrDoS_MSSQL',
        #                 4: '4_DrDoS_NTP', 
        #                 5: '5_DrDoS_NetBIOS', 
        #                 6: '6_DrDoS_SNMP', 
        #                 7: '7_DrDoS_SSDP', 
        #                 8: '8_DrDoS_UDP', 
        #                 9: '9_Syn', 
        #                 10: '10_TFTP', 
        #                 11: '11_UDPlag', 
        #                 12: '12_WebDDoS'
        #                 # 13: '13_Web Attack Sql Injection', 
        #                 # 14: '14_Web Attack XSS'
        #                 # 15: '15_backdoor',
        #                 # 16: '16_dos',
        #                 # 17: '17_injection',
        #                 # 18: '18_mitm',
        #                 # 19: '19_password',
        #                 # 20: '20_ransomware',
        #                 # 21: '21_scanning',
        #                 # 22: '22_xss'
        #                 } 
        # EdgeIIoT
        # class_names = {
        #                 0: 'BENIGN', 
        #                 1: 'DDoS_HTTP', 
        #                 2: 'DDoS_ICMP', 
        #                 3: 'DDoS_TCP',
        #                 4: 'DDoS_UDP', 
        #                 5: 'Fingerprinting', 
        #                 6: 'PortScan', 
        #                 7: 'SQL_injection', 
        #                 8: 'Uploading', 
        #                 9: 'Vulnerability_scanner', 
        #                 10: 'backdoor', 
        #                 11: 'mitm', 
        #                 12: 'password',
        #                 13: 'ransomware', 
        #                 14: 'xss'
        #                 } 
        # df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names)
        df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names.values())
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.title(client_str +"_"+ Choose_method)
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        # Rotate the x-axis labels (prediction categories)
        plt.xticks(rotation=30, ha='right',fontsize=12)
        plt.savefig(f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_confusion_matrix.png")
        plt.show()

# 創建用於訓練和測試的數據加載器
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=500, shuffle=True)
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

# 初始化神經網絡模型
# net = MLP().to(DEVICE)

net = ChooseUseModel("MLP", x_train.shape[1], labelCount).to(DEVICE)
# 訓練模型
train(net, trainloader, epochs=num_epochs)

#紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime",end_IDS,f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}")

# 評估模型
test_accuracy = test(net, testloader, start_IDS, client_str,True)
# 在训练或测试结束后，保存模型
torch.save(net.state_dict(), f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/BaseLine_After_local_train_model.pth")

print("測試數據量:\n", len(test_data))
print("訓練數據量:\n", len(train_data))
print(f"最終測試準確度: {test_accuracy:.4f}")