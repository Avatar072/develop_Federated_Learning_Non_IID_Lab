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
from DoChooseTrainNpfile import ChooseLoadTrainNpArray
from DoChooseTestNpfile import ChooseLoadTestNpArray
from collections import Counter
from colorama import Fore, Back, Style, init
####################################################################################################
#CICIIDS2019
labelCount = 13
# 二元分類
# labelCount = 2
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
start_IDS = time.time()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.__version__)
torch.cuda.empty_cache()  # 清除 CUDA 快取

'''
執行範例
# python CICIDS2019_BaseLine.py --Load_dataset CICIDS2019 --dataset_split baseLine_train --epochs 500 --method normal
# python CICIDS2019_BaseLine.py --Load_dataset CICIDS2019 --dataset_split client1_train --epochs 500 --method normal
# python CICIDS2019_BaseLine.py --Load_dataset CICIDS2019 --dataset_split client2_train --epochs 500 --method normal
# python CICIDS2019_BaseLine.py --Load_dataset CICIDS2019 --dataset_split client3_train --epochs 500 --method normal
# --method normal表示未正常未受到攻擊且未做任何資料資強(GAN\SMOTE)
# --method Evasion_Attack表示受到逃避攻擊
'''
Load_dataset,split_file, num_epochs,Choose_method = ParseCommandLineArgs(["Load_dataset","dataset_split", "epochs", "method"])
print(Fore.YELLOW+Style.BRIGHT+f"Dataset: {Load_dataset}")
print(Fore.YELLOW+Style.BRIGHT+f"split: {split_file}")
print(Fore.YELLOW+Style.BRIGHT+f"Number of epochs: {num_epochs}")
print(Fore.YELLOW+Style.BRIGHT+f"Choose_method: {Choose_method}")
# 載入np file
# ChooseLoadNpArray function  return x_train、y_train 和 client_str and Choose_method
# 載入train
# x_train, y_train, client_str = ChooseLoadNpArray(filepath,Load_dataset,split_file,Choose_method)

# 載入train
# 正常
Choose_Attacktype = "normal"
# Choose_Attacktype = Choose_method
Attack_method = None
# Evasion_Attack
# Choose_Attacktype = "Evasion_Attack"
# Choose_Attacktype = Choose_method
# Attack_method = "JSMA"
x_train, y_train, client_str =ChooseLoadTrainNpArray(Load_dataset, split_file, filepath, Choose_Attacktype, Attack_method)

# 載入test
# 正常
# Evasion_Attack
test_Choose_Attacktype = "normal"
test_Attack_method = None
# Evasion_Attack
# test_Choose_Attacktype = "Evasion_Attack"
# test_Attack_method = "JSMA"
# test_Attack_method = "FGSM"
# test_Attack_method = "PGD"
# test_Attack_method = "CandW"
x_test,y_test = ChooseLoadTestNpArray('CICIDS2019','test', filepath, 'normal',test_Attack_method)
# 載入data frame(for one hot)
# x_train, y_train, client_str = ChooseTrainDatastes(filepath, file, Choose_method)   

# 在single_AnalyseReportFolder產生天日期的資料夾
today = datetime.date.today()
today = today.strftime("%Y%m%d")
current_time = time.strftime("%Hh%Mm%Ss", time.localtime())
print(Fore.YELLOW+Style.BRIGHT+f"當前時間: {current_time}")
generatefolder(f"./single_AnalyseReportFolder/CICIDS2019/", today)
generatefolder(f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/", client_str)
generatefolder(f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/{client_str}/", Choose_method)
getStartorEndtime("starttime",start_IDS,f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/{client_str}/{Choose_method}")

# 印出所選擇Npfile的資料
print("特徵數",x_train.shape[1])
print(y_train)
# print(client_str)
counter = Counter(y_train)
print(Fore.GREEN+Style.BRIGHT+"train筆數",counter)
counter = Counter(y_test)
print(Fore.GREEN+Style.BRIGHT+"test筆數",counter)

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)


print(Fore.WHITE + Back.RED+ Style.BRIGHT+f"Train labels range: {y_train.min().item()} - {y_train.max().item()}")
print(Fore.WHITE + Back.RED+ Style.BRIGHT+f"Test labels range: {y_test.min().item()} - {y_test.max().item()}")
print(Fore.WHITE + Back.RED+ Style.BRIGHT+f"Number of classes: {labelCount}")

labelCount=len(y_test.unique())
print(Fore.WHITE + Back.RED+ Style.BRIGHT+"唯一值数量:", labelCount)

# 將測試數據移動到 GPU 上
x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)



# 定義訓練函數
def train(net, trainloader, epochs):
    print("訓練中")
    #二元分類
    # criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    #多元分類
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)

    for epoch in range(epochs):
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+"epoch:",Fore.LIGHTBLUE_EX + Style.BRIGHT+str(epoch))
        net.train()# PyTorch 中的一個方法，模型切換為訓練模式
        for images, labels in tqdm(trainloader,desc="Training",colour="green",bar_format="{l_bar}{bar:10}{r_bar}",ascii="/*"):
            optimizer.zero_grad()
            output = net(images)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        ###訓練的過程    
        test_accuracy = test(net, testloader, start_IDS, client_str,False)
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+f"訓練週期 [{epoch+1}/{epochs}]")
        # print(f"- 測試準確度:"+ Fore.LIGHTGREEN_EX  + f"{test_accuracy:.4f}")
        print(Fore.RED +  f"------------------------------------------------\t")

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
            with open(f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/{client_str}/{Choose_method}/recall-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    # file.write("標籤," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                    header_written = True
                file.write(str(RecordRecall) + "\n")
        
            # 將總體準確度和其他信息寫入 "accuracy-baseline.csv" 檔案
            with open(f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    # file.write("標籤," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                    header_written = True
                file.write(f"精確度,時間\n")
                file.write(f"{accuracy},{time.time() - start_time}\n")

                # 生成分類報告
                GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
                report_df = pd.DataFrame(GenrateReport).transpose()
                report_df.to_csv(f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/{client_str}/{Choose_method}/baseline_report_{client_str}.csv",header=True)

    draw_confusion_matrix(y_true, y_pred,plot_confusion_matrix)
    accuracy = correct / total
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+f"測試準確度:"+Fore.LIGHTWHITE_EX+ f"{accuracy:.4f}"+
          "\t"+Fore.LIGHTYELLOW_EX + Style.BRIGHT+f"loss:"+Fore.LIGHTWHITE_EX + f"{ave_loss:.4f}")
    # print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+f"loss:"+Fore.LIGHTWHITE_EX + f"{ave_loss:.4f}")
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
        # # CICIDS2019
        class_names = {
                        #二元分類
                        # 0: '0_BENIGN', 
                        # 1: 'Attack', 
                        0: '0_BENIGN', 
                        1: '1_DrDoS_DNS', 
                        2: '2_DrDoS_LDAP', 
                        3: '3_DrDoS_MSSQL',
                        4: '4_DrDoS_NTP', 
                        5: '5_DrDoS_NetBIOS', 
                        6: '6_DrDoS_SNMP', 
                        7: '7_DrDoS_SSDP', 
                        8: '8_DrDoS_UDP', 
                        9: '9_Syn', 
                        10: '10_TFTP', 
                        11: '11_UDPlag', 
                        12: '12_WebDDoS'
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
        # df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names)
        df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names.values())
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        
        # 固定子圖參數
        plt.subplots_adjust(
            left=0.19,    # 左邊界
            bottom=0.167,  # 下邊界
            right=1.0,     # 右邊界
            top=0.88,      # 上邊界
            wspace=0.207,  # 子圖間的寬度間隔
            hspace=0.195   # 子圖間的高度間隔
        )
        
        plt.title(client_str +"_"+ Choose_method)
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        # Rotate the x-axis labels (prediction categories)
        plt.xticks(rotation=30, ha='right',fontsize=9)
        plt.savefig(f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_confusion_matrix.png")
        plt.show()

# 創建用於訓練和測試的數據加載器
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=500, shuffle=True)
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

# 初始化神經網絡模型
# net = MLP().to(DEVICE)

net = ChooseUseModel("MLP", x_train.shape[1], labelCount).to(DEVICE)
# # 訓練模型
train(net, trainloader, epochs=num_epochs)

#紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime",end_IDS,f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/{client_str}/{Choose_method}")

# 評估模型
test_accuracy = test(net, testloader, start_IDS, client_str,True)
# 在训练或测试结束后，保存模型
torch.save(net.state_dict(), f"./single_AnalyseReportFolder/CICIDS2019/{today}/{current_time}/{client_str}/{Choose_method}/BaseLine_After_local_train_model.pth")

print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+"測試數據量:\n", len(test_data))
print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+"訓練數據量:\n", len(train_data))
print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+f"最終測試準確度:"+
      Fore.WHITE + Back.RED+ f"{test_accuracy:.4f}")