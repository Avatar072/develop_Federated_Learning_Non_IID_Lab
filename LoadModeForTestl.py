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
warnings.filterwarnings("ignore")  # https://blog.csdn.net/qq_43391414/article/details/120543028
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from mytoolfunction import generatefolder, ChooseLoadNpArray, ChooseTrainDatastes, ParseCommandLineArgs, ChooseTestDataSet
from mytoolfunction import ChooseUseModel, getStartorEndtime
from collections import Counter
####################################################################################################

# python LoadModeForTestl.py --dataset train_half1 --epochs 100
# python LoadModeForTestl.py --dataset train_half2 --epochs 100
# python LoadModeForTestl.py --dataset total_train --epochs 500 --method normal

labelCount = 10  # TONIoT

filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
start_IDS = time.time()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.__version__)

file, num_epochs, Choose_method = ParseCommandLineArgs(["dataset", "epochs", "method"])
print(f"Dataset: {file}")
print(f"Number of epochs: {num_epochs}")
print(f"Choose_method: {Choose_method}")

x_train, y_train, client_str = ChooseLoadNpArray(filepath, file, Choose_method)
print("特徵數", x_train.shape[1])
print(y_train)
counter = Counter(y_train)
print("train筆數", counter)
today = datetime.date.today().strftime("%Y%m%d")
generatefolder(f"./single_AnalyseReportFolder/", today)
generatefolder(f"./single_AnalyseReportFolder/{today}/", client_str)
generatefolder(f"./single_AnalyseReportFolder/{today}/{client_str}/", Choose_method)
getStartorEndtime("starttime", start_IDS, f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}")

# # 20240502 CIC-IDS2017 after do labelencode and minmax chi_square45 75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)    

# # 20240502 CIC-IDS2017 after do labelencode and minmax chi_square45 75 25分 Do JSMA
# x_test  = np.load(f"./Adversarial_Attack_Test/20240717/x_DoJSMA_test_20240718.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20240717/y_DoJSMA_test_20240718.npy")

# x_test  = np.load(f"./Adversarial_Attack_Test/20240718/theta_0.05_gamma_0.02/x_DoJSMA_test_20240718.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20240718/theta_0.05_gamma_0.02/y_DoJSMA_test_20240718.npy")

# 20240523 TONIoT after do labelencode and minmax  75 25分 DOJSMA attack
# x_test  = np.load(f"./Adversarial_Attack_Test/20240721_bk_0.5_0.5/x_DoJSMA_test_20240721.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20240721_bk_0.5_0.5/y_DoJSMA_test_20240721.npy")
# 每層神經元512下所訓練出來的model
# x_test  = np.load(f"./Adversarial_Attack_Test/20240721_0.05_0.02/x_DoJSMA_test_20240721.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20240721_0.05_0.02/y_DoJSMA_test_20240721.npy")

# 每層神經元64下所訓練出來的model
# x_test  = np.load(f"./Adversarial_Attack_Test/20240729_TONIOT_BaseLine_test_0.05_0.02/x_DoJSMA_test_20240729.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20240729_TONIOT_BaseLine_test_0.05_0.02/y_DoJSMA_test_20240729.npy")


# 20240523 TONIoT after do labelencode and minmax  75 25分
# x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_test_ToN-IoT_20240523.npy", allow_pickle=True)
# y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_test_ToN-IoT_20240523.npy", allow_pickle=True)   

# 20240523 TONIoT after do labelencode and minmax  75 25分 DOJSMA attack for FL Client3
# x_test  = np.load(f"./Adversarial_Attack_Test/20240722_FL_cleint3_.0.5_0.02/x_DoJSMA_test_20240722.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20240722_FL_cleint3_.0.5_0.02/y_DoJSMA_test_20240722.npy")

# 20241007 TONIoT after do labelencode and minmax  75 25分 DO PGD attack for TON_IOT test
x_test  = np.load(f"./Adversarial_Attack_Test/20241007_TONIOT_BaseLine_test_0.1_PGD/x_DoPGD_test_20241007.npy")
y_test  = np.load(f"./Adversarial_Attack_Test/20241007_TONIOT_BaseLine_test_0.1_PGD/y_DoPGD_test_20241007.npy")

# 20241013 TONIoT after do labelencode and minmax  75 25分 DO BIM attack for TON_IOT test
# x_test  = np.load(f"./Adversarial_Attack_Test/20241013_TONIOT_BaseLine_test_0.1_BIM/x_DoBIM_test_20241013.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20241013_TONIOT_BaseLine_test_0.1_BIM/y_DoBIM_test_20241013.npy")

# 20241015 TONIoT after do labelencode and minmax  75 25分 DO FGSM attack for TON_IOT test
# x_test  = np.load(f"./Adversarial_Attack_Test/20241015_TONIOT_BaseLine_test_0.1_FGSM/x_DoFGSM_test_20241015.npy")
# y_test  = np.load(f"./Adversarial_Attack_Test/20241015_TONIOT_BaseLine_test_0.1_FGSM/y_DoFGSM_test_20241015.npy")


counter = Counter(y_test)
print("test筆數", counter)

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

labelCount = len(y_test.unique())
print("唯一值数量:", labelCount)

x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)

# 定義測試函數
def test(net, testloader, start_time, client_str, plot_confusion_matrix):
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    ave_loss = 0.0

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            ave_loss = ave_loss * 0.9 + loss * 0.1

            y_true = labels.data.cpu().numpy()
            y_pred = predicted.data.cpu().numpy()
        
            acc = classification_report(y_true, y_pred, digits=4, output_dict=True)
            accuracy = correct / total

            RecordRecall = ()
            RecordAccuracy = ()
           
            for i in range(labelCount):
                RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
                 
            RecordAccuracy = RecordAccuracy + (accuracy, time.time() - start_time,)
            RecordRecall = str(RecordRecall)[1:-1]

            header_written = False
            with open(f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/recall-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    header_written = True
                file.write(str(RecordRecall) + "\n")
        
            with open(f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    header_written = True
                file.write(f"精確度,時間\n")
                file.write(f"{accuracy},{time.time() - start_time}\n")

                GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
                report_df = pd.DataFrame(GenrateReport).transpose()
                report_df.to_csv(f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/baseline_report_{client_str}.csv", header=True)

    draw_confusion_matrix(y_true, y_pred, plot_confusion_matrix)
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
        df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names)
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.title(client_str +"_"+ Choose_method)
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.savefig(f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_confusion_matrix.png")
        plt.show()

train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=500, shuffle=True)
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

net = ChooseUseModel("MLP", x_train.shape[1], labelCount).to(DEVICE)

# Load the saved model
# model_path = f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}/BaseLine_After_local_train_model.pth"
# 每層神經元512下所訓練出來的model
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\20240719_TONIOT_神經元512\\BaseLine\\normal\\BaseLine_After_local_train_model.pth'

# 每層神經元64下所訓練出來的model
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\20240729_TONIOT_BaseLine_神經元64\\BaseLine\\normal\\BaseLine_After_local_train_model.pth'
model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\BaseLine_After_local_train_model_1015_1st.pth'


# 每層神經元64下所訓練出來的model
# 測試正常_受到攻擊_每層神經元加總用絕對值_c1和c2lr=0.0001_c3_lr=0.001
# 使用c3的fedavg_Last_unattack_distance-測試結果效能異常
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20241005\\fedavg_Last_unattack_distance.pth'
# 使用c3的fedavg_unattack_distance-測試結果效能異常
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20241005\\fedavg_unattack_50.pth'
# gobal_model_Before_local_train_model_round_49測試結果效能異常
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20241005\\Local_model_before_local_train_150.pth'
# gobal_model_Before_local_train_model_round_50-測試結果效能異常
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20241005\\gobal_model_Before_local_train_model_round_50.pth'
# gobal_model_Before_local_train_model_round_51
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20241005\\gobal_model_Before_local_train_model_round_51.pth'


net.load_state_dict(torch.load(model_path))
print("Loaded model from", model_path)

# Test the loaded model
test_accuracy = test(net, testloader, start_IDS, client_str, True)

end_IDS = time.time()
getStartorEndtime("endtime", end_IDS, f"./single_AnalyseReportFolder/{today}/{client_str}/{Choose_method}")

print("測試數據量:\n", len(test_data))
print("訓練數據量:\n", len(train_data))
print(f"最終測試準確度: {test_accuracy:.4f}")
