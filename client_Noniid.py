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
from mytoolfunction import ChooseUseModel, getStartorEndtime,EvaluateVariation
from collections import Counter
from Add_ALL_LayerToCount import DoCountModelWeightSum,evaluateWeightDifferences
from Add_ALL_LayerToCount import Calculate_Weight_Diffs_Distance_OR_Absolute
from NonIID_ChooseNPfile import CICIDS2017_NonIID_ChooseLoadNpArray, CICIDS2018_NonIID_ChooseLoadNpArray,TONIOT_NonIID_ChooseLoadNpArray, NonIID_ChooseLoad_class_names
from Variance_Analysis import EvaluateVariance

from colorama import Fore, Back, Style, init
import configparser

# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)
# 初始化 ConfigParser
config = configparser.ConfigParser()
# 讀取 ini 文件
config.read('./config.ini', encoding='utf-8')
# 獲取 Datasets 節點下的值
choose_dataset = config.get('Datasets', 'choose_dataset')
# 獲取 Setting_Adversarial_Attack 節點下的值
set_attack = config.getboolean ('Setting_Adversarial_Attack', 'set_attack')
# 獲取 Round 節點下的值
# 使用 getint 來取得整數類型的值
start_attack_round = config.getint('Round', 'start_attack_round')
end_attack_round = config.getint('Round', 'end_attack_round')  
save_model_round = config.getint('Round', 'save_model_round')
# 顯示讀取的配置
print(Fore.YELLOW+Style.BRIGHT+f"choose_dataset: {choose_dataset}")
print(Fore.YELLOW+Style.BRIGHT+f"set_attack: {set_attack}")
print(Fore.YELLOW+Style.BRIGHT+f"start_attack_round: {start_attack_round}")
print(Fore.YELLOW+Style.BRIGHT+f"end_attack_round: {end_attack_round}")
print(Fore.YELLOW+Style.BRIGHT+f"save_model_round: {save_model_round}")

# CICIDS2017、CICIDS2018、TONIOT 聯集
labelCount = 19

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
#python client_Noniid.py --dataset_split client1_train --epochs 50 --method normal
#python client_Noniid.py --dataset_split client2_train --epochs 50 --method normal
#python client_Noniid.py --dataset_split client3_train --epochs 50 --method normal
file, num_epochs,Choose_method = ParseCommandLineArgs(["dataset_split", "epochs", "method"])
print(f"Dataset: {file}")
print(f"Number of epochs: {num_epochs}")
print(f"Choose_method: {Choose_method}")
# x_train, y_train, client_str = ChooseLoadNpArray(filepath, file, Choose_method)


# 初始化變數為None或空列表
x_train =  np.array([]) # 預設初始化為一個空陣列
y_train =  np.array([]) # 預設初始化為一個空陣列

x_test = np.array([])  # 預設初始化為一個空陣列
y_test = np.array([])  # 預設初始化為一個空陣列

client_str = ""
# 預設初始化 class_names
class_names_global, class_names_local, labels_to_calculate = None, None, None

labels_to_calculate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

try:
    # CICIDS2017
    if choose_dataset == "CICIDS2017":
        print(Fore.YELLOW+Style.BRIGHT+f"use dataset: {choose_dataset}")
        x_train, y_train, x_test, y_test, client_str = CICIDS2017_NonIID_ChooseLoadNpArray(filepath, file, Choose_method)
        class_names_local, class_names_global = NonIID_ChooseLoad_class_names("CICIDS2017")
        
    # CICIDS2018
    if choose_dataset == "CICIDS2018":
        print(Fore.YELLOW+Style.BRIGHT+f"use dataset: {choose_dataset}")
        x_train, y_train, x_test, y_test, client_str = CICIDS2018_NonIID_ChooseLoadNpArray(filepath, file, Choose_method)
        class_names_local, class_names_global = NonIID_ChooseLoad_class_names("CICIDS2018")

    # TONIOT
    if choose_dataset == "TONIOT":
        print(Fore.YELLOW+Style.BRIGHT+f"use dataset: {choose_dataset}")
        x_train, y_train, x_test, y_test, client_str = TONIOT_NonIID_ChooseLoadNpArray(filepath, file, Choose_method)
        class_names_local, class_names_global = NonIID_ChooseLoad_class_names("TONIOT")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # 確保資料加載成功
    if y_train is None or len(y_train) == 0:
            raise ValueError("Failed to load y_train for "+f"{choose_dataset}")
    else:
            print("Execution finished.")                  


counter = Counter(y_train)
y_train = y_train.astype(int)
print(Fore.GREEN+Style.BRIGHT+client_str+"\tlocal train筆數",counter) 
today = datetime.date.today()
today = today.strftime("%Y%m%d")
current_time = time.strftime("%Hh%Mm%Ss", time.localtime())
# generatefolder(filepath, "\\FL_AnalyseReportfolder")
generatefolder(f"./FL_AnalyseReportfolder/", today)
generatefolder(f"./FL_AnalyseReportfolder/{today}/{current_time}/", client_str)
generatefolder(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/", Choose_method)
getStartorEndtime("starttime",start_IDS,f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}")

# 20240317 CICIDS2017 和 CICIDS2018 和TONIOT after do labelencode and minmax  75 25分 and feature mapping to 123 feature
global_x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017_and_CICIDS2018_TONIOT_test\\merged_x_Non_IID_ALL_test.npy", allow_pickle=True)
global_y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017_and_CICIDS2018_TONIOT_test\\merged_y_Non_IID_ALL_test.npy", allow_pickle=True)   

counter = Counter(global_y_test)
print(Fore.GREEN+Style.BRIGHT+client_str+"\tglobal test筆數",counter)

# 先統一使用global的class name 方便繪圖
class_names_local = class_names_global

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

global_x_test = torch.from_numpy(global_x_test).type(torch.FloatTensor)
global_y_test = torch.from_numpy(global_y_test).type(torch.LongTensor)

# 将测试数据移动到GPU上
x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)

global_x_test = global_x_test.to(DEVICE)
global_y_test = global_y_test.to(DEVICE)

print("Minimum label value:", min(y_train))
print("Maximum label value:", max(y_train))

# 定義訓練和評估函數
def train(net, trainloader, epochs):
    record_final_epochs_round_counter = 0
    print("train")
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    if client_str == "client1":  
        # 調整測試 lr學習率 weight_decay為L2正規化的強度，這裡設為0.01
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0.01)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0.0001)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0.001)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.0)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.00008)




    else:
        # 學長的參數
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
        # 調整測試 lr學習率 weight_decay為L2正規化的強度，這裡設為0.01
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0.01)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0.0001)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)


    for epoch in range(epochs):
        print("epoch",epoch)
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            output = net(images)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()# 以train loss更新權重 反向傳播
            optimizer.step()
            ###訓練的過程    
        record_final_epochs_round_counter+=1#用於紀錄最後一次epochs的loss與Recall
        test_accuracy = test(net, local_testloader, start_IDS, client_str, "local_test",False,0,record_final_epochs_round_counter)
        print(f"訓練週期 [{epoch+1}/{epochs}] - 測試準確度: {test_accuracy:.4f}")
        # 存每次global round 初次的epochs的model
        # 用於觀測當前epochs內距離變化，初次的epochs的model與完成Local train的mode1之距離變化
        if epoch == 1:
            torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Initial_Local.pth")

    return test_accuracy

def test(net, testloader, start_time, client_str, str_globalOrlocal, bool_plot_confusion_matrix, int_record_round = 0,int_final_epochs_loss_and_Reacll_record_counter=0):
    print(Fore.GREEN+Style.BRIGHT+"進入test function")
    correct = 0
    total = 0
    loss = 0  # 初始化損失值為0
    ave_loss = 0
    # 迭代測試資料集
    with torch.no_grad(): #不讓模型更新把梯度關閉
        criterion = nn.CrossEntropyLoss()
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
            # 使用神經網路模型進行前向傳播
            outputs = net(images)
        
            # 計算損失
            loss += criterion(outputs, labels).item()
        
            # 計算預測的類別
            _, predicted = torch.max(outputs.data, 1)
        
            # 統計總樣本數和正確分類的樣本數
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            # 計算滑動平均損失
            ave_loss = ave_loss * 0.9 + loss * 0.1

            # 將標籤和預測結果轉換為 NumPy 數組
            y_true = labels.data.cpu().numpy()
            y_pred = predicted.data.cpu().numpy()
        
            # 计算每个类别的召回率
            acc = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # print("correct:\n",correct)
            # print("total:\n",total)
            # print("acc:\n",acc)
            # 將每個類別的召回率寫入 "recall-baseline.csv" 文件

            RecordRecall = ()
            RecordAccuracy = ()

            # local test只算各client各自持有的label的recall or 
            # global model用各client各自持有的local test測試的recall

            if str_globalOrlocal =="local_test" or str_globalOrlocal =="global_model_local_test":
                # for label in labels_to_calculate:#個別計算recall
                for label in range(labelCount):
                    label_str = str(label)  # 將標籤轉為字串
                    if label_str in acc:  # 檢查標籤是否存在於分類報告中
                        RecordRecall = RecordRecall + (acc[label_str]['recall'],)

            # global test算所有的Label
            elif str_globalOrlocal =="global_test":
                for i in range(labelCount):
                    RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
            
            
            RecordAccuracy = RecordAccuracy + (correct / total, time.time() - start_time,)
            RecordRecall = str(RecordRecall)[1:-1]

            # 標誌來追蹤是否已經添加了標題行
            header_written = False
            with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/loss-baseline_{client_str}_{str_globalOrlocal}.csv", "a+") as file:
                file.write(f"{ave_loss}\n")

            with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/recall-baseline_{client_str}_{str_globalOrlocal}.csv", "a+") as file:
                file.write(f"{RecordRecall}\n")
        
            # 將整體準確率和其他資訊寫入 "accuracy-baseline.csv" 文件
            with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}_{str_globalOrlocal}.csv", "a+") as file:
                file.write(f"{RecordAccuracy}\n")

            if int_final_epochs_loss_and_Reacll_record_counter==50: #50表示最後一個epochs時做紀錄
                with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/loss-baseline_{client_str}_local_final_epochs.csv", "a+") as file:
                    file.write(f"{ave_loss}\n")
            
                with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/recall-baseline_{client_str}_local_final_epochs.csv", "a+") as file:
                    file.write(f"{RecordRecall}\n")
        
            # 將整體準確率和其他資訊寫入 "accuracy-baseline.csv" 文件
            with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}_{str_globalOrlocal}.csv", "a+") as file:
                file.write(f"{RecordAccuracy}\n")
            # 產生分類報告
            GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # 將字典轉換為 DataFrame 並轉置
            report_df = pd.DataFrame(GenrateReport).transpose()
            # 將指定回合儲存為 baseline_report 文件 用於紀錄對抗式攻擊開始與結束
            if int_record_round != 0:
                report_df.to_csv(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/baseline_report_{client_str}_{str_globalOrlocal}_attack_round_{int_record_round}.csv",header=True)
                draw_confusion_matrix(y_true, y_pred, str_globalOrlocal, bool_plot_confusion_matrix,int_record_round)
            # 儲存為 baseline_report 文件
            report_df.to_csv(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/baseline_report_{client_str}_{str_globalOrlocal}.csv",header=True)
            # self.global_round
    draw_confusion_matrix(y_true, y_pred, str_globalOrlocal, bool_plot_confusion_matrix)
    accuracy = correct / total
    print(f"測試準確度: {accuracy:.4f}")
    return accuracy

# 畫混淆矩陣
def draw_confusion_matrix(y_true, y_pred, str_globalOrlocal, bool_plot_confusion_matrix = False, int_record_round = 0):

    # 若要更改全域變數的內容就需要這樣宣告，若只是單純讀值這樣就好
    global class_names_local
    global class_names_global

    #混淆矩陣
    if bool_plot_confusion_matrix:

        # arr：混淆矩陣的數據，這是一個二維陣列，其中包含了模型的預測和實際標籤之間的關係，以及它們在混淆矩陣中的計數。
        # class_names：類別標籤的清單，通常是一個包含每個類別名稱的字串清單。這將用作 Pandas 資料幀的行索引和列索引，
        # 以標識混淆矩陣中每個類別的位置。  同樣的類別標籤的清單，它作為列索引的標籤，這是可選的，
        # 如果不提供這個參數，將使用行索引的標籤作為列索引
        # 初始化宣告arr
        arr = confusion_matrix(y_true, y_pred, labels=labels_to_calculate)
        # 使用指定的標籤計算混淆矩陣
        if str_globalOrlocal == "local_test" or str_globalOrlocal == "global_model_local_test":
            arr = confusion_matrix(y_true, y_pred, labels=labels_to_calculate)
        elif str_globalOrlocal == "global_test":
            arr = confusion_matrix(y_true, y_pred)

        # 如果 class_names 未提供，根據 labels_to_calculate 動態生成
        if class_names_local is None:
            class_names_local = {label: f"Class_{label}" for label in labels_to_calculate}
        else:
            # 寫法
            # {key: value for key in iterable if condition}
            # 字典推導式，用於根據可迭代對象創建新的字典。
            # key：新字典中的鍵。
            # value：新字典中對應 key 的值。
            # iterable：可迭代對象（例如列表、集合等）。
            # if condition（可選）：條件過濾，只有滿足條件的條目才會包含在新字典中。
            # 過濾 class_names 僅保留 labels_to_calculate
            class_names_local = {label: class_names_local[label] for label in labels_to_calculate if label in class_names_local}
            class_names_global = {label: class_names_global[label] for label in labels_to_calculate if label in class_names_global}            

        # 初始化宣告df 
        df_cm = pd.DataFrame(arr, index=class_names_local.values(), columns=class_names_local.values())    
        # df_cm的PD.DataFrame 接受三個參數：
        if str_globalOrlocal == "local_test" or str_globalOrlocal == "global_model_local_test":
            df_cm = pd.DataFrame(arr, index=class_names_local.values(), columns=class_names_local.values())
        elif str_globalOrlocal == "global_test":
            df_cm = pd.DataFrame(arr, index=class_names_global.values(), columns=class_names_global.values())
        # 設置字體比例
        sns.set(font_scale=1.2)
        
        # 設置圖像大小和繪製熱圖
        plt.figure(figsize = (20,10))

        # 使用 heatmap 繪製混淆矩陣
        # annot=True 表示在單元格內顯示數值
        # fmt="d" 表示數值的格式為整數
        # cmap='BuGn' 設置顏色圖
        # annot_kws={"size": 13} 設置單元格內數值的字體大小
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn', annot_kws={"size": 10})
        
        # 固定子圖參數
        plt.subplots_adjust(
            left=0.19,    # 左邊界
            bottom=0.167,  # 下邊界
            right=1.0,     # 右邊界
            top=0.88,      # 上邊界
            wspace=0.207,  # 子圖間的寬度間隔
            hspace=0.195   # 子圖間的高度間隔
        )

        # 設置標題和標籤
        plt.title(client_str +"_"+ Choose_method) # 圖片標題
        plt.xlabel("prediction",fontsize=15) # x 軸標籤
        plt.ylabel("Label (ground truth)", fontsize=18) # y 軸標籤
       
        # 設置 x 軸和 y 軸的字體大小和旋轉角度
        # Rotate the x-axis labels (prediction categories)
        plt.xticks(rotation=30, ha='right',fontsize=12) 
        plt.yticks(rotation=0, fontsize=12) # y 軸刻度標籤不旋轉，字體大小為 15
        
        # 調整圖像間距
        # left, right, top, bottom 控制圖像在畫布上的邊距
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)
        # 保存圖像到指定路徑
        if int_record_round != 0:
            plt.savefig(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_{str_globalOrlocal}_attack_round_{int_record_round}_confusion_matrix.png")
        plt.savefig(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_{str_globalOrlocal}_confusion_matrix.png")
        plt.close('all')
        # plt.show()

# 創建用於訓練和測試的DataLoader
train_data = TensorDataset(x_train, y_train)
local_test_data = TensorDataset(x_test, y_test)
global_test_data = TensorDataset(global_x_test, global_y_test)
trainloader = DataLoader(train_data, batch_size=512, shuffle=True)  # 设置 shuffle 为 True
# test_data 的batch_size要設跟test_data(y_test)的筆數一樣 重要!!!
local_testloader = DataLoader(local_test_data, batch_size=len(local_test_data), shuffle=False)
global_testloader = DataLoader(global_test_data, batch_size=len(global_test_data), shuffle=False)
print(Fore.GREEN+Style.BRIGHT+"batch_size",len(local_test_data))
print(Fore.GREEN+Style.BRIGHT+"batch_size",len(global_test_data))
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
        self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis = 0 #用於保存上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以距離)
        self.Current_Global_vs_Local_total_weight_diff_dis = 0 #當前回合全局模型與本地端模型間權重差異總和
        self.Previous_and_Current_Local_model_weight_diff_dis = 0 #當前回合本地端模型與上一回合本地端模型權重差異總和
        self.Previous_and_Current_Global_model_weight_diff_dis = 0 #當前回合全局模型與上一回合全局模型權重差異總和
        self.Previous_diff_dis_Temp = 0
        self.Record_Previous_Global_vs_Local_total_weight_diff_dis = 0
        self.dis_variation = 0
        self.dis_variation_Global_Local = 0
        self.dis_variation_Previous_and_Current_Local_model = 0
        self.dis_variation_Previous_and_Current_Global_model = 0
        self.Record_dis_variation = 0
        self.Unattck_dis_variation = 0
        self.LastRound_UnattackCounter = 0 # 用來計數最後一次的正常FedAvg後的模型
        self.bool_Unattack_Judage = True
        self.dis_smooth_Global_Local = 0
        self.dis_smooth_Previous_and_Current_Local_model = 0
        self.Unattck_dis_smooth_Global_Local = 0
        self.Previous_and_Current_Local_model_weight_diff_dis_Unattack = 0
        self.Unattck_dis_smooth_Previous_and_Current_Local_model = 0
        self.Unattck_dis_variation_Global_Local = 0
        self.Unattck_dis_variation_Previous_and_Current_Local_model = 0
        self.Record_Previous_Local_weight_diff_dis = 0
        self.Record_Previous_Global_weight_diff_dis = 0
        self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis = 0
        self.Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis = 0
        self.UnAttack_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis = 0
        self.dis_smooth_Inital_Local = 0
        self.dis_variation_Inital_Local = 0
        self.dis_variance_Inital_Local_threshold = 0
        self.threshold_variance_List = []
        self.UnAttack_dis_variation_Inital_Local = 0
        self.threshold_List = []
        self.each_ten_round_sum = 0
        self.dis_threshold_Inital_Local = 0
        self.dis_each_ten_round_average = 0
        self.dis_each_ten_round_sum = 0
        self.Record_UnAttack_counter = 0

        ####### dis
        self.Previous_total_weight_diff_abs = 0 #用於保存上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以絕對值)
        self.Current_total_weight_diff_abs = 0 #當前每一回合全局模型與本地端模型間權重差異總和
        ####### Norm
        self.Norm_Current_Global_vs_Local_total_weight_diff_dis = 0#當前每一回合全局模型與本地端模型間權重差異總和
        self.Norm_Previous_Unattack_Global_vs_Local_total_weight_diff_dis = 0 #用於保存上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以距離範數)
        ####### To Save Unattack model
        self.Last_round_Unattack_After_FedAVG_model = net
        self.Last_round_Local_model_unattack = net

    def get_parameters(self, config):
        global net        
        # if (self.Record_UnAttack_counter != 0):
        if (not self.bool_Unattack_Judage):
            print(Fore.YELLOW+Back.RED+Style.BRIGHT+"發生攻擊進行權重抽換，上傳最後一次Local train後正常模型\t")
            model_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Unattack_AfterLocaltrain.pth"
            net.load_state_dict(torch.load(model_path))
            accuracy = test(net, local_testloader, start_IDS, client_str,f"local_test",True)
            print(Fore.YELLOW+Back.RED+Style.BRIGHT+"發生攻擊進行權重抽換，上傳最後一次Local train後正常模型之accuracy"+str({accuracy})+"\t")
            # 上傳最後一次未受攻擊的weight
            return [val.cpu().numpy() for _, val in net.state_dict().items()]
        else:
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):#是知識載入
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    ######################################################Fedavg完的模型每層加總總和############################################# 
    def Global_Model_each_layer_sum(self,weights_after_FedAVG):
        # 算聚合完的全局模型每層權重加總總和
        # True 以絕對值加總 False 為直接加總
        # self.Current_total_FedAVG_weight_sum = DoCountModelWeightSum(weights_after_FedAVG,True,"After_FedAVG") 
        self.Current_total_FedAVG_weight_sum = DoCountModelWeightSum(weights_after_FedAVG,False,"After_FedAVG") 
    
    ######################################################Local train完的模型每層加總總和############################################# 
    def Local_Model_each_layer_sum(self,weights_after_Localtrain):
        print("Weights after local training:")
        # True 以絕對值加總 False 為直接加總
        # self.Current_total_Local_weight_sum = DoCountModelWeightSum(weights_after_Localtrain,
        #                                   True,
        #                                 self.client_id)    
        self.Current_total_Local_weight_sum = DoCountModelWeightSum(weights_after_Localtrain,False,self.client_id) 
        # 打印模型每層加總後權重總重總和
        print(Fore.YELLOW+Style.BRIGHT+"self.Current_total_Local_weight_sum",self.Current_total_Local_weight_sum)
        print(Fore.YELLOW+Style.BRIGHT+"self.Current_total_FedAVG_weight_sum",self.Current_total_FedAVG_weight_sum)
        # self.Record_List[0] = self.Current_total_Local_weight_sum
        self.Record_Local_Current_total_FedAVG_weight_sum = self.Current_total_Local_weight_sum
        
    ######################################################模型每層加總後求差異##############################################        
    def Model_each_layer_sum_diff(self):
        # local train計算權重加總 - 實際每一回合FedAVG計算權重加總
        self.current_array[0],self.current_array[1],self.current_array[2],self.current_array[3] = evaluateWeightDifferences("Local-Current_FedAVG",
                                                                                                                                    self.Current_total_Local_weight_sum, 
                                                                                                                                    self.Current_total_FedAVG_weight_sum)
            
        
        # local train計算權重加總 - 上一回合FedAVG計算權重加總(最後一次未受到攻擊的回合)
        self.previous_array[0],self.previous_array[1],self.previous_array[2],self.previous_array[3] = evaluateWeightDifferences("Local-Last UnAttack_FedAVG",
                                                                                                                                        self.Current_total_Local_weight_sum,     
                                                                                                                                        self.Previous_Unattack_round_total_FedAVG_weight_sum)
        self.AllLayertotalSum_diff = self.previous_array[0]        
        print(Fore.GREEN+Style.BRIGHT+"Current_total_Local_weight_sum\t", self.Record_Local_Current_total_FedAVG_weight_sum)
        print(Fore.GREEN+Style.BRIGHT+"Previous_total_FedAVG_weight_sum\t", self.Previous_Unattack_round_total_FedAVG_weight_sum)

    
    #####################################################用accuracy保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################
    def By_Accuracy_Save_Previous_Unattack_After_FedAvg_Model_each_layer_sum_diff(self):
        if self.Reocrd_global_model_accuracy >= 0.8:
                    self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
                    # 保存上一回合未受到攻擊FedAVG後的正常權重總和 進行後續權重每層總和求差異計算
                    self.Previous_Temp = self.Previous_Unattack_round_total_FedAVG_weight_sum
                    print("Last_round_After_FedAVG", self.Previous_Unattack_round_total_FedAVG_weight_sum)

                    # 保存上一回合未受到攻擊剛聚合完的全局模型
                    torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/fedavg_unattack.pth")        
        else:
                            if self.global_round <=10:
                                self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
                            print("Previous_total_FedAVG_weight_sum", self.Previous_Unattack_round_total_FedAVG_weight_sum)
                            print("Last_round_After_FedAVG_normal", self.Previous_Unattack_round_total_FedAVG_weight_sum)         
    
    #####################################################保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################
    def Save_Previous_Unattack_After_FedAvg_Model_each_layer_sum_diff(self):
        if self.global_round <=10:
            # 初始回合直接給1當亂數初值
            if self.global_round == 1 :
                self.Current_total_FedAVG_weight_sum = 1
            print("After_FedAVG",self.Current_total_FedAVG_weight_sum)
            self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
            print("Previous_total_FedAVG_weight_sum", self.Previous_Unattack_round_total_FedAVG_weight_sum)
        else:
            # weight_diff = float(self.Record_List[2])
            weight_diff = float(self.AllLayertotalSum_diff)
            print("**********************weight_diff**********************", weight_diff)
            # 權重差異超過當前本地權重總和的5%就要過濾掉                 
            # threshold = float(self.Record_List[0]) * 0.05
            threshold = float(self.Current_total_FedAVG_weight_sum) * 0.05
            print("**********************threshold**********************", threshold)
            print("**********************self.Previous_Unattack_round_total_FedAVG_weight_sum**********************", self.Previous_Unattack_round_total_FedAVG_weight_sum)
            if  weight_diff <= threshold:
                    # 保存上一回合未受到攻擊FedAVG後的正常權重總和 進行後續權重每層總和求差異計算
                    self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
                    print("**********************Last_round_After_FedAVG_Unattack_round_total_FedAVG_weight_sum:**********************", self.Previous_Unattack_round_total_FedAVG_weight_sum)
                    self.Previous_Temp = self.Previous_Unattack_round_total_FedAVG_weight_sum
                    # 保存上一回合未受到攻擊剛聚合完的全局模型
                    torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/fedavg_unattack.pth")
                    print(f"Model saved to ./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/fedavg_unattack.pth with updated weights.")
            else:
                    # 當條件不符合時，打印未更新的變數值並使用上一回合未受到攻擊的總和
                    self.Previous_Unattack_round_total_FedAVG_weight_sum = self.Previous_Temp
                    print("**********************No update performed on Previous_Unattack_round_total_FedAVG_weight_sum.**********************")                

    
    def Record_Previous_Value(self):
        # 需紀錄上一回合FedAVG後的權重總和，這邊權重可能已遭受到攻擊而汙染
        self.Record_Previous_total_FedAVG_weight_sum = self.Current_total_FedAVG_weight_sum
        # 需紀錄上一回合FedAVG後與Local train的模型間權重差異總和，這邊權重可能已遭受到攻擊而汙染
        self.Record_Previous_Global_vs_Local_total_weight_diff_dis = self.Current_Global_vs_Local_total_weight_diff_dis
        # 需紀錄上一回合Local train後與Local train的模型間權重差異總和，這邊權重可能已遭受到攻擊而汙染
        self.Record_Previous_Local_weight_diff_dis = self.Previous_and_Current_Local_model_weight_diff_dis
        # 需紀錄上一回合FedAVG後與當前FedAVG後的模型間權重差異總和，這邊權重可能已遭受到攻擊而汙染
        self.Record_Previous_Global_weight_diff_dis = self.Previous_and_Current_Global_model_weight_diff_dis
        # 需紀錄每一回未Local train後與Local train的模型間權重差異總和，這邊權重可能已遭受到攻擊而汙染
        self.Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis = self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis
        print(Fore.GREEN+Style.BRIGHT+"Record_Previous_total_FedAVG_weight_sum", self.Record_Previous_total_FedAVG_weight_sum)
        print(Fore.GREEN+Style.BRIGHT+"Record_Previous_Global_vs_Local_total_weight_diff_dis", self.Record_Previous_Global_vs_Local_total_weight_diff_dis)
        print(Fore.GREEN+Style.BRIGHT+"Record_Previous_Local_weight_diff_dis", self.Record_Previous_Local_weight_diff_dis)
        print(Fore.GREEN+Style.BRIGHT+"Record_Previous_Global_weight_diff_dis", self.Record_Previous_Global_weight_diff_dis)
        print(Fore.GREEN+Style.BRIGHT+"Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis", self.Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis)

    ##########################################計算兩個模型的每層權重差距 將每層權重差距值相加（以歐基里德距離(distance)計算）##########################################
    def Euclidean_distance(self,weights_after_Localtrain,After_FedAVG_model,Previous_round_Local_weights,Previous_round_Global_weights):
        # Calculate_Weight_Diffs_Distance_OR_Absolute True表示計算L2範數 False表示計算歐基里德距離

        #########################################################################以歐基里德距離#####################################################################
        # 計算兩個模型的每層權重差距 當前每一回合聚合後的全局模型與本地端模型間權重差異總和(以歐基里德距離)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_Global_weight_diffs_dis_{client_str}.csv"
        weight_diffs_dis, self.Current_Global_vs_Local_total_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                           After_FedAVG_model,
                                                                                                           diff_dis_csv_file_path,
                                                                                                          "distance",
                                                                                                           False)
        # 計算兩個模型的每層權重差距 當前每一回合本地端模型與上一回合本地端模型間權重差異總和(以歐基里德距離)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_weight_diffs_dis_{client_str}.csv"
        weight_diffs_dis, self.Previous_and_Current_Local_model_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                           Previous_round_Local_weights,
                                                                                                           diff_dis_csv_file_path,
                                                                                                          "distance",
                                                                                                           False)

        # 計算兩個模型的每層權重差距 當前每一回合全局端模型與上一回合全局模型間權重差異總和(以歐基里德距離)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Global_weight_diffs_dis_{client_str}.csv"
        weight_diffs_dis, self.Previous_and_Current_Global_model_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(After_FedAVG_model,
                                                                                                           Previous_round_Global_weights,
                                                                                                           diff_dis_csv_file_path,
                                                                                                          "distance",
                                                                                                           False)
        
        # 類似weight average算法計算閥值 當前回合距離佔20% 上一回合距離佔80%
        self.dis_smooth_Global_Local = self.Current_Global_vs_Local_total_weight_diff_dis*0.2 + self.Record_Previous_Global_vs_Local_total_weight_diff_dis*0.8
        self.dis_smooth_Previous_and_Current_Local_model = self.Previous_and_Current_Local_model_weight_diff_dis*0.2 + self.Record_Previous_Local_weight_diff_dis*0.8

        print(Fore.BLUE+Style.BRIGHT+"Current_After_FedAVG_and_Current_Local_model_dis\t"+str(self.Current_Global_vs_Local_total_weight_diff_dis))
        print(Fore.BLUE+Style.BRIGHT+"Current_Local_model_and_Previous_Local_model_dis\t"+str(self.Previous_and_Current_Local_model_weight_diff_dis))
        
        # 算每一回合權重距離變化的百分比  
            # 百分比變化=(當前可能受到攻擊的距離−上一回合聚合後的未受攻擊距離/上一回合聚合後的未受攻擊距離 )×100%  
        self.dis_variation_Global_Local = EvaluateVariation(self.Current_Global_vs_Local_total_weight_diff_dis,
                                                                self.Record_Previous_Global_vs_Local_total_weight_diff_dis)
        self.dis_variation_Previous_and_Current_Local_model = EvaluateVariation(self.Previous_and_Current_Local_model_weight_diff_dis,
                                                                        self.Record_Previous_Local_weight_diff_dis)
    
        self.dis_variation_Previous_and_Current_Global_model = EvaluateVariation(self.Previous_and_Current_Global_model_weight_diff_dis,
                                                                        self.Record_Previous_Global_weight_diff_dis)
    ##########################################計算兩個模型的每層權重差距 將每層權重差距值相加（以歐基里德距離(distance)計算）##########################################
    def Unattack_Euclidean_distance(self,weights_after_Localtrain,Last_round_Unattack_After_FedAVG_model,Last_round_Local_model_unattack):
        # 計算兩個模型的每層權重差距 上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以歐基里德距離)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_Global_weight_diffs_dis_{client_str}_unattack.csv"
        weight_diffs_dis, self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                                     Last_round_Unattack_After_FedAVG_model,
                                                                                                                     diff_dis_csv_file_path,
                                                                                                                     "distance",
                                                                                                                     False)
                                                                                                                   
        # 計算兩個模型的每層權重差距 當前每一回合本地端模型與上一回合未受攻擊汙染本地端模型間權重差異總和(以歐基里德距離)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_weight_diffs_dis_{client_str}_unattack.csv"

        # 計算兩個模型的每層權重差距 上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以歐基里德距離)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_Global_weight_diffs_dis_{client_str}_unattack.csv"
        weight_diffs_dis, self.Previous_and_Current_Local_model_weight_diff_dis_Unattack = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                                     Last_round_Local_model_unattack,
                                                                                                                     diff_dis_csv_file_path,
                                                                                                                     "distance",
                                                                                                                     False)

        # 計算兩個模型的每層權重差距 上一回合聚合後的未受攻擊汙染的全局模型與本地端模型間權重差異總和(以L2範數計算)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/weight_diffs_dis_{client_str}_unattack_Norm.csv"
        weight_diffs_dis, self.Norm_Previous_Unattack_Global_vs_Local_total_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                    Last_round_Unattack_After_FedAVG_model,
                                                                                                    diff_dis_csv_file_path,
                                                                                                    "distance",
                                                                                                    True)
        print(Fore.CYAN+Style.BRIGHT+"Previous_Unattack_Global_vs_Local_total_weight_diff_dis\t"+str(self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis))
        print(Fore.CYAN+Style.BRIGHT+"Previous_Unattack_Local_vs_Current_Local_model_weight_diff_dis\t"+str(self.Previous_and_Current_Local_model_weight_diff_dis_Unattack))

        # 算每一回合權重距離變化的百分比  
            # 百分比變化=(當前可能受到攻擊的距離−上一回合聚合後的未受攻擊距離/上一回合聚合後的未受攻擊距離 )×100% 
        self.Unattck_dis_variation_Global_Local = EvaluateVariation(self.Current_Global_vs_Local_total_weight_diff_dis,
                                                            self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis)
        
        self.Unattck_dis_variation_Previous_and_Current_Local_model = EvaluateVariation(self.Previous_and_Current_Local_model_weight_diff_dis,
                                                            self.Previous_and_Current_Local_model_weight_diff_dis_Unattack)
        # 類似weight average算法計算閥值 當前回合距離佔20% 上一回合未受攻擊模型距離佔80%(FedAvg and Local)
        self.Unattck_dis_smooth_Global_Local = self.Current_Global_vs_Local_total_weight_diff_dis*0.2 + self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis*0.8
        # 類似weight average算法計算閥值 當前回合距離佔20% 上一回合未受攻擊模型距離佔80%(Local and Local)
        self.Unattck_dis_smooth_Previous_and_Current_Local_model = self.Previous_and_Current_Local_model_weight_diff_dis*0.2 + self.Previous_and_Current_Local_model_weight_diff_dis_Unattack*0.8

    ##########################################計算兩個模型的每層權重差距 將每層權重差距值相加（以歐基里德距離(distance)計算）##########################################
    def Initial_Local_weights_Euclidean_distance(self,weights_after_Localtrain,Initial_Local_weights):
        # 計算兩個模型的每層權重差距 每回合未訓練本地模型與本地訓練後本地模型間權重差異總和(以歐基里德距離)
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Initial_Local_and_Atfer_Local_trian_weight_diffs_dis_{client_str}.csv"
        weight_diffs_dis, self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(Initial_Local_weights,
                                                                                                                     weights_after_Localtrain,
                                                                                                                     diff_dis_csv_file_path,
                                                                                                                     "distance",
                                                                                                                     False)
        
        # 增加以變化量計算變異數
        dis_variance_Inital_Local_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/dis_variance_Initial_Local_{client_str}"
        # UnAttack_weights_after_Localtrain = weights_after_Localtrain
        # if (not self.bool_Unattack_Judage):
        #     UnAttack_weights_after_Localtrain = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Unattack_AfterLocaltrain.pth")
        #    # 跟最後一次正常未受攻擊的本地模型之距離

        
        # weight_diffs_dis, self.UnAttack_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(Initial_Local_weights,
        #                                                                                                              UnAttack_weights_after_Localtrain,
        #                                                                                                              diff_dis_csv_file_path,
        #                                                                                                              "distance",
        #                                                                                                              False)
        # 類似weight average算法計算閥值 當前回合距離佔20% 上一回合距離佔80%
        self.dis_smooth_Inital_Local = self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis*0.2 + self.Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis*0.8

        #  算每一回合權重距離變化
        # 變化量=(當前可能受到攻擊的距離−上一回合聚合後的未受攻擊距離/上一回合聚合後的未受攻擊距離 )
        if (not self.bool_Unattack_Judage):
            #  self.UnAttack_dis_variation_Inital_Local = EvaluateVariation(self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis,
            #                                                     self.UnAttack_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis)
            self.dis_variation_Inital_Local = EvaluateVariation(self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis,
                                                                self.UnAttack_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis)

            # 增加以變化量計算變異數 不能將攻擊發生期間的變化量納入門檻值計算
            # self.threshold_variance_List.append(self.dis_variation_Inital_Local)
            # self.dis_variance_Inital_Local_threshold = EvaluateVariance(self.threshold_variance_List,dis_variance_Inital_Local_file_path)
            # print(Fore.RESET+Back.GREEN+Style.BRIGHT+f"self.dis_variance_Inital_Local_threshold:\t+{str(self.dis_variance_Inital_Local_threshold)}")

        else:
            self.dis_variation_Inital_Local = EvaluateVariation(self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis,
                                                                self.Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis)
            
            # 增加以變化量計算變異數
            self.threshold_variance_List.append(self.dis_variation_Inital_Local)
            self.dis_variance_Inital_Local_threshold = EvaluateVariance(self.threshold_variance_List,dis_variance_Inital_Local_file_path)
            print(Fore.RESET+Back.GREEN+Style.BRIGHT+f"self.dis_variance_Inital_Local_threshold:\t+{str(self.dis_variance_Inital_Local_threshold)}")
           
            
        
        # 計算平均10 round的門檻值
        self.count_threshold()
        # print(Fore.YELLOW+Style.BRIGHT+Back.LIGHTBLACK_EX+"每10round平均"+self.count_threshold())
    
    ######################################################模型每層加總後求差異（以歐基里德範數(L2 Norm)計算）##############################################        
    def L2_Norm_distance(self,weights_after_Localtrain,After_FedAVG_model):
        # 計算兩個模型的每層權重差距 將每層權重差距值相加（以L2範數計算）
        diff_dis_csv_file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/weight_diffs_dis_{client_str}_Norm.csv"
            
        weight_diffs_dis, self.Norm_Current_Global_vs_Local_total_weight_diff_dis = Calculate_Weight_Diffs_Distance_OR_Absolute(weights_after_Localtrain,
                                                                                                        After_FedAVG_model,
                                                                                                        diff_dis_csv_file_path,
                                                                                                        "distance",
                                                                                                        True)
    
    def Record_file(self):
        Global_vs_Local_file_name = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_FedAVG_weight_diff_{client_str}.csv"
        Previous_and_Current_Local_file_name = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Previous_and_Current_Local_model_weight_diff_{client_str}.csv"
        Inital_Local_vs_AfterLocalTrain_file_name = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Inital_Local_weight_diff_{client_str}.csv"
        # 檢查檔案是否存在
        file_exists_Param_fedavg = os.path.exists(Global_vs_Local_file_name)
        file_exists_Param_Local = os.path.exists(Previous_and_Current_Local_file_name)
        file_exists_Param_Inital_Local = os.path.exists(Inital_Local_vs_AfterLocalTrain_file_name)

        
        if not file_exists_Param_fedavg:
            with open(Global_vs_Local_file_name, "w", newline='') as file:
                file.write("Current_Global_vs_Local,"
                           "Previous_Global_vs_Local,"
                           "Previous_Unattack_Global_vs_Local,"
                           "dis_percent,"
                           "Unattck_dis_percent,"
                           "dis_smooth_Global_Local,"
                           "Unattck_dis_smooth_Global_Local,"
                           "Norm_Current_Global_vs_Local,"
                           "Norm_Previous_Unattack_Global_vs_Local\n")
        # 如果兩個檔案不存在，則創建並寫入表頭
        if not file_exists_Param_Local:
            with open(Previous_and_Current_Local_file_name, "w", newline='') as file:
                file.write("Previous_and_Current_Local,"
                           "Previous_and_Current_Local_Unattack,"
                           "dis_percent,"
                           "Unattck_dis_percent,"
                           "dis_smooth_Previous_and_Current_Local,"
                           "Unattck_dis_smooth_Previous_and_Current_Local\n")
        
        # 如果兩個檔案不存在，則創建並寫入表頭
        if not file_exists_Param_Inital_Local:
            with open(Inital_Local_vs_AfterLocalTrain_file_name, "w", newline='') as file:
                file.write("Inital_Local_vs_AfterLocalTrain,"
                           "Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis,"
                           "UnAttack_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis,"
                           "dis_variation_Inital_Local,"
                           "dis_variance_Inital_Local_threshold,"
                           "UnAttack_dis_variation_Inital_Local,"
                           "dis_each_ten_round_sum,"
                           "dis_each_ten_round_average,"
                           "dis_threshold_Inital_Local,"
                           "dis_variation_Previous_Current_Global,"
                           "dis_smooth_Inital_Local\n")



        with open(Global_vs_Local_file_name, "a+") as file:
                file.write(
                            f"{self.Current_Global_vs_Local_total_weight_diff_dis},"#模型每層差異求總和（以距離計算）
                            f"{self.Record_Previous_Global_vs_Local_total_weight_diff_dis},"#上一回的全局模型與本地端每層差異總和（以距離計算）
                            f"{self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis},"#上一回未受到攻擊的全局模型與本地端每層差異總和（以距離計算）
                            f"{self.dis_variation_Global_Local},"#上一回全局模型與本地端每層差異總和變化百分比（以距離計算）
                            f"{self.Unattck_dis_variation_Global_Local},"#上一回未受到攻擊的的的全局模型與本地端每層差異總和變化百分比（以距離計算）
                            f"{self.dis_smooth_Global_Local},"#類似weight average算法計算閥值 當前回合距離佔20% 上一回合距離佔80%
                            f"{self.Unattck_dis_smooth_Global_Local},"#類似weight average算法計算閥值 當前回合距離佔20% 上一回合未受攻擊模型距離佔80%
                            f"{self.Norm_Current_Global_vs_Local_total_weight_diff_dis},"#模型每層差異求總和（以距離範數計算）
                            f"{self.Norm_Previous_Unattack_Global_vs_Local_total_weight_diff_dis}\n")#上一回未受到攻擊的全局模型與本地端每層差異總和（以距離範數計算）

        with open(Previous_and_Current_Local_file_name, "a+") as file:
                file.write(
                            f"{self.Previous_and_Current_Local_model_weight_diff_dis},"#上一回本地模型與當前本地端模型每層差異總和（以距離計算）
                            f"{self.Previous_and_Current_Local_model_weight_diff_dis_Unattack},"#上一回未受到攻擊的本地模型與當前本地端模型每層差異總和（以距離計算）
                            f"{self.dis_variation_Previous_and_Current_Local_model},"#上一回的本地模型與當前本地端每層差異總和變化百分比（以距離計算）
                            f"{self.Unattck_dis_variation_Previous_and_Current_Local_model},"#上一回的未受到攻擊的本地模型與當前本地端每層差異總和變化百分比（以距離計算）
                            f"{self.dis_smooth_Previous_and_Current_Local_model},"#類似weight average算法計算閥值 當前回合距離佔20% 上一回合距離佔80%
                            f"{self.Unattck_dis_smooth_Previous_and_Current_Local_model}\n")#類似weight average算法計算閥值 當前回合距離佔20% 上一回合未受攻擊模型距離佔80%

        with open(Inital_Local_vs_AfterLocalTrain_file_name, "a+") as file:
                file.write(
                            f"{self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis},"#當前回合未訓練本地模型與本地訓練後本地模型每層差異總和（以距離計算）
                            f"{self.Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis},"#上一回未訓練本地模型與本地訓練後本地模型（以距離計算）
                            f"{self.UnAttack_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis},"#最後一次正常本地模型 當前回合未訓練本地模型與本地訓練後本地模型每層差異總和變化百分比（以距離計算）
                            f"{self.dis_variation_Inital_Local},"#當前回合未訓練本地模型與本地訓練後本地模型每層差異總和變化量（以距離計算）
                            f"{self.dis_variance_Inital_Local_threshold},"#當前回合未訓練本地模型與本地訓練後本地模型每層差異總和變化量門檻值（以距離計算）
                            f"{self.UnAttack_dis_variation_Inital_Local},"#最後一次正常本地模型 當前回合未訓練本地模型與本地訓練後本地模型每層差異總和變化百分比（以距離計算）
                            f"{self.dis_each_ten_round_sum},"#每10回合的當前回合未訓練本地模型與本地訓練後本地模型每層差異總和之加總
                            f"{self.dis_each_ten_round_average},"#每10回合的當前回合未訓練本地模型與本地訓練後本地模型每層差異總和之平均
                            f"{self.dis_threshold_Inital_Local},"#每10回合的當前回合未訓練本地模型與本地訓練後本地模型每層差異總和之平均佔50%+當前回合距離佔50%
                            f"{self.dis_variation_Previous_and_Current_Global_model},"#上一回全局模型模型與與當前全局模型（以距離計算）
                            f"{self.dis_smooth_Inital_Local}\n")#類似weight average算法計算閥值 當前回合距離佔20% 上一回合模型距離佔80%

    def Setting_Adversarial_Attack(self, start_round, end_round,bool_enabel,client_id):
        if bool_enabel:
            print(Fore.BLACK+Style.BRIGHT+Back.YELLOW+f"client_id: {client_id}")
            if (self.global_round >= start_round and self.global_round <= end_round and client_id == "client1"):
                # if choose_dataset == "CICIDS2017":
                    #################################0.1 CICIDS2017 FGSM################################
                    # CICIDS2017 iid Dirichlet 0.1 c1 to FGSM eps 0.01 use 123 feature mapping normal model
                    # print(Fore.BLACK+Style.BRIGHT+Back.YELLOW+f"FGSM_Attack eps =0.01 by genrate by 123_feature Label merge BaseLine normal model")
                    # x_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2017/FGSM_Attack/Npfile/Dirichlet/123_feature/a_0.1/Noniid/train_Dirichlet_client1_a0.1_eps_0.01_Added_Noniid_Label_x.npy", allow_pickle=True)
                    # y_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2017/FGSM_Attack/Npfile/Dirichlet/123_feature/a_0.1/Noniid/train_Dirichlet_client1_a0.1_eps_0.01_Added_Noniid_Label_y.npy", allow_pickle=True)
                    #################################0.5 CICIDS2017 FGSM################################
                    # CICIDS2017 iid Dirichlet 0.5 c1 to FGSM eps 0.01 use 123 feature mapping normal model
                    # print(Fore.BLACK+Style.BRIGHT+Back.YELLOW+f"FGSM_Attack eps =0.01 by genrate by 123_feature Label merge BaseLine normal model")
                    # x_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2017/FGSM_Attack/Npfile/Dirichlet/123_feature/a_0.5/Noniid/train_Dirichlet_client1_a0.5_eps_0.01_Added_Noniid_Label_x.npy", allow_pickle=True)
                    # y_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2017/FGSM_Attack/Npfile/Dirichlet/123_feature/a_0.5/Noniid/train_Dirichlet_client1_a0.5_eps_0.01_Added_Noniid_Label_y.npy", allow_pickle=True)
                    #################################total CICIDS2017 JSMA################################
                    # print(Fore.BLACK+Style.BRIGHT+Back.YELLOW+f"JSMA_Attack theta =0.01 by genrate by 123_feature Label merge BaseLine normal model")
                    # x_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2017/JSMA_Attack/Npfile/Noniid/train_CICIDS2017_theta0.01_gamma_0.05_Added_Noniid_Label_x.npy", allow_pickle=True)
                    # y_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2017/JSMA_Attack/Npfile/Noniid/train_CICIDS2017_theta0.01_gamma_0.05_Added_Noniid_Label_y.npy", allow_pickle=True)

                if choose_dataset == "CICIDS2018":
                    #################################0.5 CICIDS2018 JSMA################################
                    # CICIDS2018 iid Dirichlet 0.5 c1 to JSMA theta 0.5 gamma 0.05
                    # print(Fore.BLACK+Style.BRIGHT+Back.YELLOW+f"JSMA_Attack theta0.5/gamma 0.05 by genrate by 123_feature Label merge BaseLine normal model")
                    # x_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2018/JSMA_Attack/Npfile/Dirichlet/123_feature/a_0.5/Noniid/train_Dirichlet_client1_a0.5_theta0.5_gamma_0.05_Added_Noniid_Label_x.npy", allow_pickle=True)
                    # y_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2018/JSMA_Attack/Npfile/Dirichlet/123_feature/a_0.5/Noniid/train_Dirichlet_client1_a0.5_theta0.5_gamma_0.05_Added_Noniid_Label_y.npy", allow_pickle=True)
                    #################################total CICIDS2018 JSMA################################
                    print(Fore.BLACK+Style.BRIGHT+Back.YELLOW+f"JSMA_Attack theta =0.01 by genrate by 123_feature Label merge BaseLine normal model")
                    x_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2018/JSMA_Attack/Npfile/Noniid/train_CICIDS2018_theta0.01_gamma_0.05_Added_Noniid_Label_x.npy", allow_pickle=True)
                    y_train_attacked = np.load("./Adversarial_Attack_Test/CICIDS2018/JSMA_Attack/Npfile/Noniid/train_CICIDS2018_theta0.01_gamma_0.05_Added_Noniid_Label_y.npy", allow_pickle=True)
                # if choose_dataset == "TONIOT":
                #     #################################0.1 TONIOT PGD################################
                #     # TONIOT iid Dirichlet 0.1 c1 to PGD eps 0.01
                #     print(Fore.BLACK+Style.BRIGHT+Back.YELLOW+f"PGD_Attack eps =0.01 step_eps 0.0002 by genrate by 123_feature Label merge BaseLine normal model")
                #     x_train_attacked = np.load("./Adversarial_Attack_Test/TONIOT/PGD_Attack/Npfile/Dirichlet/123_feature/a_0.1/Noniid/train_Dirichlet_client1_a0.1_esp0.01_step_0.0002_Added_Noniid_Label_x.npy", allow_pickle=True)
                #     y_train_attacked = np.load("./Adversarial_Attack_Test/TONIOT/PGD_Attack/Npfile/Dirichlet/123_feature/a_0.1/Noniid/train_Dirichlet_client1_a0.1_esp0.01_step_0.0002_Added_Noniid_Label_y.npy", allow_pickle=True)

                x_train_attacked = torch.from_numpy(x_train_attacked).type(torch.FloatTensor).to(DEVICE)
                y_train_attacked = torch.from_numpy(y_train_attacked).type(torch.LongTensor).to(DEVICE)
                
                train_data_attacked = TensorDataset(x_train_attacked, y_train_attacked)
                # trainloader = DataLoader(train_data_attacked, batch_size=512, shuffle=True)
                print(Fore.RED+Style.BRIGHT+Back.YELLOW+f"*********************{self.client_id}在第{self.global_round}回合開始使用被攻擊的數據*********************************************")
                return DataLoader(train_data_attacked, batch_size=512, shuffle=True)

            else:
                print(Fore.BLACK+Style.BRIGHT+Back.YELLOW+f"*********************在第{self.global_round}使用正常trainloader*********************************************")
                return self.original_trainloader
        else:
                print(Fore.WHITE+Style.BRIGHT+Back.YELLOW+f"*********************使用正常trainloader*********************************************")
                return self.original_trainloader
        

    # 先強制存指定round的model當測試
    def designate_Round_To_Save_Model(self,designate_Round,model,bool_Global_Or_Local, bool_force_save):
        if (self.global_round == designate_Round):
            # True表示強制儲存
            if bool_force_save:
                if bool_Global_Or_Local: 
                    # True存Local   False存global
                    torch.save(model,f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Force_Save_Last_round_Local_model_unattack_{designate_Round}.pth")
                else:
                    torch.save(model,f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Force_Save_fedavg_unattack_{designate_Round}.pth")
            else:
                if bool_Global_Or_Local:
                    torch.save(model,f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Initial_unattack_{designate_Round}.pth")
                else:
                    torch.save(model,f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/fedavg_unattack_{designate_Round}.pth")

             

    # 載入上一回合聚合後的最後一次未受攻擊汙染的全局模型和本地模型
    def Load_Last_round_Unattack_Model(self,start_attack_round,end_attack_round):
        try:
            if (self.global_round >= start_attack_round and self.global_round <= end_attack_round):
                # 強制觸發條件 bool_Unattack_Judage 以測試
                # self.bool_Unattack_Judage = False
                # 強制載入正常global和local當試測
                self.Last_round_Unattack_After_FedAVG_model = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/fedavg_unattack_{start_attack_round-1}.pth")
                self.Last_round_Local_model_unattack = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Last_round_Local_model_unattack_{start_attack_round-1}.pth")
            # elif(self.global_round < start_attack_round or self.global_round > end_attack_round):
            #     # self.bool_Unattack_Judage = True
        except FileNotFoundError:
            print("Warning: Unattack model not found, using fallback model.")
            self.Last_round_Unattack_After_FedAVG_model = net.state_dict()
            self.Last_round_Local_model_unattack = net.state_dict()
    
    def save_threshold_to_csv(self, filename="threshold_values.csv"):
        # 將 threshold_List 轉換為 pandas DataFrame
        df = pd.DataFrame(self.threshold_List, columns=["Threshold Value"])

        # 將 DataFrame 儲存為 CSV
        df.to_csv(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/{filename}", index=False)
        print(f"Threshold values saved to {filename}")

    # 每10回合計算一次門檻值
    def count_threshold(self):
        # 初始化 threshold_List 並確保每次調用清空
        if not hasattr(self, 'threshold_List'):
            self.threshold_List = []
        # 每次都重置計算總和
        
        self.each_ten_round_sum += self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis
        
        # 每10回合計算一次
        if self.global_round % 10 == 0:
            self.threshold_List.append(self.each_ten_round_sum)
            
            # 使用 threshold_List[-1] 訪問最後一個門檻值
            print(Fore.YELLOW+Style.BRIGHT+f"Last threshold value:"+str({self.threshold_List[-1]}))
            print(Fore.YELLOW+Style.BRIGHT+f"Last threshold value average:"+str({self.threshold_List[-1]/10}))
            self.dis_each_ten_round_sum = self.threshold_List[-1]
            self.dis_each_ten_round_average = self.threshold_List[-1]/10
            self.save_threshold_to_csv()  # 儲存 CSV
            self.dis_threshold_Inital_Local = self.dis_each_ten_round_average*0.35+self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis*0.65
            # 重置總和
            self.each_ten_round_sum = 0
        
        print(Fore.YELLOW+Style.BRIGHT+f"threshold_List:"+str(self.threshold_List))

    # 判斷有沒有攻擊    
    def JudageAttack(self):
        # if self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis > self.dis_threshold_Inital_Local*1.1:
        # if self.dis_variation_Inital_Local > 1:# 不超過原本距離的變化之1倍
        # if self.dis_variation_Inital_Local > 0.5:# 不超過原本距離的變化之1倍
        # if self.dis_variation_Inital_Local > self.dis_variance_Inital_Local_threshold:# 不超過原本距離的變異數之最大值2倍
        if self.dis_variation_Inital_Local > 0.5:# 不超過原本距離的變異數之最大值0.3倍固定測試

            print(Fore.RED+Style.BRIGHT+Back.CYAN+f"global_round_{self.global_round}_occur Attack!!!")
            print(Fore.RED+Style.BRIGHT+Back.CYAN+f"Initial_and_AfterLocalTrain_Local_model_weight_diff_dis:{self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis}")
            self.bool_Unattack_Judage = False
            self.Record_UnAttack_counter += 1
            # UnAttack_Inital_weights = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Initial_Local_model_Before_local_train_{self.global_round-1}.pth")
            #  e.g:當前攻擊發生在第25 round於Local上並在Local train完後上傳至server到聚合後，當client再下載受汙染資料時已是第26 round
            #  所以要在第25 round受到攻擊後就開始傳最後一次正常的Local train的權重
            #  保留上次正常的Local train的model
            UnAttack_AfterLocaltrain_weights = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_model_After_local_train_{self.global_round-1}.pth")
            
            # 表示最初發生攻擊時機點並儲存最後一次正常的模型
            if self.Record_UnAttack_counter == 1:
                # torch.save(UnAttack_Inital_weights,f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Unattack_Initial_Local_model.pth")
                torch.save(UnAttack_AfterLocaltrain_weights,f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Unattack_AfterLocaltrain.pth")
                # 紀錄最後一次正常未受攻擊的距離
                self.UnAttack_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis = self.Record_Initial_and_AfterLocalTrain_Local_model_weight_diff_dis

            print(Fore.YELLOW+Style.BRIGHT+Back.LIGHTBLACK_EX+f"Save Pervious Round UnAttack model After Local Train Model in Round:"+str(self.global_round-1))
            # self.designate_Round_To_Save_Model(self.global_round-1,UnAttack_Inital_weights, False,  False)

        # else:
        #     self.bool_Unattack_Judage = True
        #     self.Record_UnAttack_counter = 0
         
    def fit(self, parameters, config):
       
        # 紀錄全局回合次數
        self.global_round += 1
        print(Fore.YELLOW+Style.BRIGHT+f"Current global round: {self.global_round}")

        #####################################################用accuracy保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################
        # self.By_Accuracy_Save_Previous_Unattack_After_FedAvg_Model_each_alyer_sum_diff()
        #####################################################用accuracy保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################

        # 剛聚合完的global權重 更新客戶端模型參數為新的全局模型參數
        self.set_parameters(parameters)
        
        """
        global test 對每global round剛聚合完的gobal model進行測試 要在Local_train之前測試
        通常第1 round測出來會是0
        """
        # 保存模型剛聚合完的全局模型
        torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/gobal_model_Before_local_train_model_round_{self.global_round}.pth")
        After_FedAVG_model = torch.load(f'./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/gobal_model_Before_local_train_model_round_{self.global_round}.pth')
        # global test測試剛聚合完的全局模型
        accuracy = test(net, global_testloader, start_IDS, client_str,f"global_test",True)
        # local test測試剛聚合完的全局模型
        test_global_inLocaltest_accuracy = test(net, local_testloader, start_IDS, client_str,f"global_model_local_test",True)
        self.Reocrd_global_model_accuracy = accuracy
        print(Fore.RED+Style.BRIGHT+"global_model_accuracy:"+str(accuracy))
        print(Fore.RED+Style.BRIGHT+"Reocrd_global_model_accuracy:"+str(self.Reocrd_global_model_accuracy))
        print(Fore.RED+Style.BRIGHT+"test_global_in_Local_test_accuracy:"+str(test_global_inLocaltest_accuracy))

        # 聚合完的全局模型
        # weights_after_FedAVG = net.state_dict()
        
        # True存Local
        # Fale存global
        self.designate_Round_To_Save_Model(save_model_round,net.state_dict(),False,True)

        ######################################################Fedavg完的模型每層加總總和############################################# 
        # self.Global_Model_each_layer_sum(weights_after_FedAVG)
        #####################################################Fedavg完的模型每層加總總和############################################# 

        #####################################################每層加總總和_保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################
        # self.Save_Previous_Unattack_After_FedAvg_Model_each_layer_sum_diff()
        #####################################################每層加總總和_保存上一回合未受到攻擊FedAVG後的正常模型每層權重總和#####################################################
       
        #  寫入Accuracy文件
        with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/accuracy-gobal_model_{client_str}.csv", "a+") as file:
            file.write(f"{accuracy}\n")
         
        #####################################################對抗式攻擊設定#################################################   
        # True表示設定攻擊 Fasle表示使用正常資料
        # trainloader = self.Setting_Adversarial_Attack(start_attack_round, end_attack_round,True,self.client_id)
        # trainloader = self.Setting_Adversarial_Attack(start_attack_round, end_attack_round,False,self.client_id)
        trainloader = self.Setting_Adversarial_Attack(start_attack_round, end_attack_round,set_attack,self.client_id)
        # 紀錄攻擊開始前後的report
        # +1是因為這邊是初始權重剛下載下來還未Local train，攻擊會在Local train後才生效
        if self.global_round == (start_attack_round):
            Before_Local_train_accuracy = test(net, local_testloader, start_IDS, client_str, "global_model_local_test",True, start_attack_round)

        elif self.global_round == (end_attack_round):
            Before_Local_train_accuracy = test(net, local_testloader, start_IDS, client_str, "global_model_local_test",True, end_attack_round)
    
        else:
            Before_Local_train_accuracy = test(net, local_testloader, start_IDS, client_str, "global_model_local_test",True)
        print(Fore.MAGENTA+Style.BRIGHT+f"Before_Local_train測試準確度: {Before_Local_train_accuracy:.4f}")
        #  寫入Accuracy文件
        with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/accuracy-Before_LocalTrain-Inital_Local_model_{client_str}.csv", "a+") as file:
            file.write(f"{accuracy}\n")
        #####################################################對抗式攻擊設定#################################################
        
        #####################################################本地訓練階段#################################################  
        self.Local_train_accuracy = train(net, trainloader, epochs=num_epochs)
        # 在本地訓練階段後保存模型
        weights_after_Localtrain = net.state_dict()
        torch.save(weights_after_Localtrain, f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_model_After_local_train_{self.global_round}.pth")
        # 載入本地訓練後的模型
        weights_after_Localtrain = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_model_After_local_train_{self.global_round}.pth")
        
        if self.global_round == (start_attack_round):
            accuracy = test(net, local_testloader, start_IDS, client_str, "local_test",True, start_attack_round)

        elif self.global_round == (end_attack_round):
            accuracy = test(net, local_testloader, start_IDS, client_str, "local_test",True, end_attack_round)
        else:
            accuracy = test(net, local_testloader, start_IDS, client_str,f"local_test",True)
        # 寫入Accuracy
        with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/accuracy-local_model_{client_str}.csv", "a+") as file:
            file.write(f"{accuracy}\n")

        self.Local_train_accuracy = accuracy

        # 載入上一回合本地訓練後的本地模型
        # 從第一回合開始取值 避免0值寫入問題
        if self.global_round > 1:
            Previous_round_Local_weights = torch.load(
                f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_model_After_local_train_{self.global_round-1}.pth"
            )
        else:
            Previous_round_Local_weights = net.state_dict()  # 避免第一輪讀取不存在的檔案
        #####################################################本地訓練階段#################################################  

        ######################################################Local train完的模型每層加總總和############################################# 
        # self.Local_Model_each_layer_sum(weights_after_Localtrain)

        # 先強制存第124_round Local當測試
        self.designate_Round_To_Save_Model(save_model_round,net.state_dict(),True,True)

        #放在更新變數之前執行以記錄
        self.Record_Previous_Value()

        # 計算模型每層差異求總和之歐基里得距離距離
        # 用當前回合的全局模型和當前訓練後本地模型
        # 用前一次的本地模型和當前訓練後本地模型
        ####################################################歐基里德距離-模型每層差異求總和#################################################  
        if self.global_round > 1:
            Previous_round_Global_weights = torch.load(f'./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/gobal_model_Before_local_train_model_round_{self.global_round-1}.pth')
        else:
            Previous_round_Global_weights = net.state_dict()  # 避免第一輪讀取不存在的檔案
        
        self.Euclidean_distance(weights_after_Localtrain,After_FedAVG_model,Previous_round_Local_weights,Previous_round_Global_weights)
        ####################################################歐基里德距離-模型每層差異求總和#################################################  
        
        ###################################################未本地訓練階段與本地訓練後差異#################################################
        Initial_Local_weights = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Initial_Local.pth")
        # 確保載入的格式正確
        if not isinstance(Initial_Local_weights, dict):
            print(Fore.YELLOW+Style.BRIGHT+"Error: Initial_Local_weights is not a valid state_dict format!")
        else:
            self.Initial_Local_weights_Euclidean_distance(weights_after_Localtrain, Initial_Local_weights)
        ###################################################未本地訓練階段與本地訓練後差異#################################################

        # 載入上一回合聚合後的最後一次未受攻擊汙染的全局模型和本地模型
        # self.Load_Last_round_Unattack_Model(start_attack_round, end_attack_round)
        if(not self.bool_Unattack_Judage): #fasle表示受到攻擊            
            # 計算模型每層差異求總和之歐基里得距離
            # 用最後一次未受攻擊汙染的全局模型和當前訓練後本地模型
            # 用最後一次未受攻擊汙染的本地模型和當前訓練後本地模型
            #     # 強制載入正常global和local當試測
            #     Last_round_Unattack_After_FedAVG_model = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/fedavg_unattack_124.pth")
            #     Last_round_Local_model_unattack = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Last_round_Local_model_unattack_124.pth")
            self.Last_round_Local_model_unattack = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Unattack_AfterLocaltrain.pth")
            # self.Last_round_Unattack_After_FedAVG_model = torch.load(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Force_Save_fedavg_unattack_{start_attack_round-1}.pth")

            ####################################################歐基里德距離-模型每層差異求總和#################################################  
            self.Unattack_Euclidean_distance(weights_after_Localtrain,self.Last_round_Unattack_After_FedAVG_model,self.Last_round_Local_model_unattack)
            ####################################################歐基里德距離-模型每層差異求總和#################################################  

        else:
            # Global vs Local未受到攻擊就讀當前Local vs 上一回合的值
            self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis = self.Record_Previous_Global_vs_Local_total_weight_diff_dis
            self.Unattck_dis_variation_Global_Local = self.dis_variation_Global_Local
            self.Unattck_dis_smooth_Global_Local = self.dis_smooth_Global_Local
            self.Norm_Previous_Unattack_Global_vs_Local_total_weight_diff_dis = self.Norm_Current_Global_vs_Local_total_weight_diff_dis
                
            #Local vs Previous Local未受到攻擊就讀當前Local vs 上一回合的值
            self.Previous_and_Current_Local_model_weight_diff_dis_Unattack = self.Previous_and_Current_Local_model_weight_diff_dis
            self.Unattck_dis_smooth_Previous_and_Current_Local_model = self.dis_smooth_Previous_and_Current_Local_model
            self.Unattck_dis_variation_Previous_and_Current_Local_model = self.dis_variation_Previous_and_Current_Local_model
            
        
        ######################################################模型每層差異求總和################################################ 
        # 計算兩個模型的每層權重差距 將每層權重差距值相加（以L2範數計算）
        self.L2_Norm_distance(weights_after_Localtrain,After_FedAVG_model)
        ######################################################模型每層差異求總和################################################ 


        ######################################################模型每層加總後求差異##############################################    
        # self.Model_each_layer_sum_diff()
        ######################################################模型每層加總後求差異##############################################    

        
        # 寫入相關參數進行紀錄
        self.Record_file()
        
        # # 前10round 模型未穩定所不看
        ###################################################判斷是否受到攻擊#################################################
        if self.global_round > 10:
            # 改成這邊做評估去判斷權重，避免原本流程fit完才做evaluate造成會上傳初次攻擊髒權重的問題
            self.JudageAttack()
        ###################################################判斷是否受到攻擊#################################################

        
        # 若accuracy大於0.9表示攻擊結束
        # if self.Local_train_accuracy >= 0.9:
        # 不行用accuracy來判斷攻擊是否結束因為 當節點資料量很少並發生攻擊時accuracy也是在0.9左右並持續下降
        # if self.dis_variation_Inital_Local < 1:# 不超過原本距離的變化之1倍表示攻擊結束
        # if self.dis_variation_Inital_Local < self.dis_variance_Inital_Local_threshold:# 變異量不超過原本距離的變異數之最大值2倍表示攻擊結束
        if self.dis_variation_Inital_Local < 0.5:# 變異量不超過原本距離的變異數之最大值2倍表示攻擊結束
            self.bool_Unattack_Judage = True
            self.Record_UnAttack_counter = 0

        print(Fore.YELLOW+Style.BRIGHT+f"Round {self.global_round} -self.bool_Unattack_Judage True表示正常未受到攻擊;False表示受到攻擊"+Fore.RED+Style.BRIGHT+str(self.bool_Unattack_Judage))

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
                                                                          "Initial_and_AfterLocalTrain_Local_model_weight_diff_dis": float(self.Initial_and_AfterLocalTrain_Local_model_weight_diff_dis),
                                                                          "dis_threshold_Inital_Local": float(self.dis_threshold_Inital_Local),
                                                                          "dis_variation": float(self.dis_variation), 
                                                                          "Current_Global_vs_Local_total_weight_diff_dis": float(self.Current_Global_vs_Local_total_weight_diff_dis),
                                                                          "Previous_total_weight_diff_dis": float(self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis)}

    # def evaluate(self, parameters, config):
    # def Customer_evaluate(self, parameters):
    #     # 当前 global round 数
    #     print(f"Evaluating global round: {self.global_round}")
    #     print("client_id",self.client_id)
    #     # local test
    #     # 這邊的測試結果會受到local train的影響
    #     # 保存模型
    #     torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_model_After_local_train_model_{self.global_round}.pth")
    #     accuracy = test(net, local_testloader, start_IDS, client_str,f"local_test",True)
    #     # 寫入Accuracy
    #     with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/accuracy-local_model_{client_str}.csv", "a+") as file:
    #         file.write(f"{accuracy}\n")

    #     self.Local_train_accuracy = accuracy

    #     # # 前10round 模型未穩定所不看
    #     if self.global_round > 10:
    #         self.JudageAttack()
        
    #     # 若accuracy大於0.9表示攻擊結束
    #     if self.Local_train_accuracy >= 0.9:
    #         self.bool_Unattack_Judage = True
    #         self.Record_UnAttack_counter = 0

    #     ###################################################判斷是否受到攻擊#################################################
    #     print(Fore.YELLOW+Style.BRIGHT+"self.bool_Unattack_Judage True表示正常未受到攻擊;False表示受到攻擊"+Fore.RED+Style.BRIGHT+str(self.bool_Unattack_Judage))

    #     # self.set_parameters(parameters)#更新現有的知識#step4 更新model
    #     print(f"Client {self.client_id} returning metrics: {{accuracy: {accuracy}, client_id: {self.client_id}}}")
        
    #     with open(f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Local_train_weight_sum-FedAVG weight_sum_{client_str}.csv", "a+") as file:
    #             file.write(f"{self.Current_total_Local_weight_sum},"
    #                         f"{self.Current_total_FedAVG_weight_sum},"#當前每一回FedAVG後的權重
    #                         f"{self.Previous_Unattack_round_total_FedAVG_weight_sum},"#上一回未受到攻擊FedAVG後的權重
    #                         f"{self.Record_Previous_total_FedAVG_weight_sum},"#實際上一回FedAVG後的權重
    #                         f"{self.current_array[0]},"#當前每一回Local_train_weight_sum-Previous_FedAVG weight_sum
    #                         f"{self.previous_array[0]},"#上一回未受到攻擊Local_train_weight_sum-Previous_FedAVG weight_sum
    #                         f"{self.Current_total_weight_diff_abs},"#模型每層差異求總和（以絕對值計算）  
    #                         f"{self.Previous_total_weight_diff_abs},"#上一回未受到攻擊的全局模型與本地端模型每層差異總和（以絕對值計算）
    #                         f"{self.Current_Global_vs_Local_total_weight_diff_dis},"#模型每層差異求總和（以距離計算）
    #                         f"{self.Record_Previous_Global_vs_Local_total_weight_diff_dis},"#上一回的全局模型與本地端每層差異總和（以距離計算）
    #                         f"{self.dis_variation},"#上一回的全局模型與本地端每層差異總和變化百分比（以距離計算）
    #                         f"{self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis},"#上一回未受到攻擊的全局模型與本地端每層差異總和（以距離計算）
    #                         f"{self.Previous_and_Current_Local_model_weight_diff_dis},"#上一回未受到攻擊的本地模型與當前本地端模型每層差異總和（以距離計算）
    #                         f"{self.Unattck_dis_variation},"#上一回未受到攻擊的全局模型與本地端每層差異總和變化百分比（以距離計算）
    #                         f"{self.Unattck_dis_smooth_Global_Local},"#類似weight average算法計算閥值 當前回合距離佔20% 上一回合未受攻擊模型距離佔80%
    #                         f"{self.Norm_Current_Global_vs_Local_total_weight_diff_dis},"#模型每層差異求總和（以距離範數計算）
    #                         f"{self.Norm_Previous_Unattack_Global_vs_Local_total_weight_diff_dis}\n")#上一回未受到攻擊的全局模型與本地端每層差異總和（以距離範數計算）

        
    #     self.Current_total_Local_weight_sum = float(self.Current_total_Local_weight_sum)  # 將字符串轉換為浮點數
    #     # print("Local_weight_sum before multiplication:", self.Current_total_Local_weight_sum, "of type:", type(self.Current_total_Local_weight_sum))
    #     percentage_five = self.Current_total_Local_weight_sum * 0.05
    #     # # 保留小数点后两位
    #     percentage_five = round(percentage_five, 2)
    #     # print("Local_train_weight_sum_percentage_five\n",percentage_five)
    #     return accuracy, len(local_testloader.dataset), {"global_round": self.global_round,
    #                                                "accuracy": accuracy,
    #                                                "Local_train_accuracy": self.Local_train_accuracy,
    #                                                "client_id": self.client_id,
    #                                                "Local_train_weight_sum":self.Current_total_Local_weight_sum,
    #                                                "Previous_round_FedAVG_weight_sum":self.Previous_Unattack_round_total_FedAVG_weight_sum,
    #                                                "Current_FedAVG_weight_sum":self.Current_total_FedAVG_weight_sum,
    #                                                "Local_train_weight_sum-Current_FedAVG weight_sum":float(self.current_array[0]),
    #                                                 "Current_total_weight_diff_abs": float(self.Current_total_weight_diff_abs),
    #                                                 "Current_Global_vs_Local_total_weight_diff_dis": float(self.Current_Global_vs_Local_total_weight_diff_dis),
    #                                                 "Previous_total_weight_diff_abs": float(self.Previous_total_weight_diff_abs),
    #                                                 "Previous_total_weight_diff_dis": float(self.Previous_Unattack_Global_vs_Local_total_weight_diff_dis)}

# 初始化神经网络模型
net = ChooseUseModel("MLP", x_train.shape[1], labelCount).to(DEVICE)

# 启动Flower客户端
fl.client.start_numpy_client(
    server_address="127.0.0.1:53388",
    # server_address="192.168.1.137:53388",
    client=FlowerClient(),
    
)

#紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime",end_IDS,f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}")