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
from mytoolfunction import ChooseUseModel, getStartorEndtime
from collections import Counter


#CICIIDS2017 or Edge 62個特徵
# labelCount = 15
#CICIIDS2019
# labelCount = 13
#Wustl 41個特徵
labelCount = 5
#Kub 36個特徵
# labelCount = 4
#CICIIDS2017、TONIOT、CICIIDS2019 聯集
# labelCount = 35

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
#python client.py --dataset train_half1 --epochs 50 --method normal
#python client.py --dataset train_half2 --epochs 50 --method normal
#python client.py --dataset train_half3 --epochs 50 --method normal
file, num_epochs,Choose_method = ParseCommandLineArgs(["dataset", "epochs", "method"])
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

# 20240507 after do labelencode and minmax Wustl iid
x_test = np.load(filepath + "\\dataset_AfterProcessed\\Wustl\\x_Resplit_test_20240507.npy", allow_pickle=True)
y_test = np.load(filepath + "\\dataset_AfterProcessed\\Wustl\\y_Resplit_test_20240507.npy", allow_pickle=True)

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
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
    # 學長的參數
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

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
                file.write(str(RecordRecall) + "\n")
        
            # 将总体准确率和其他信息写入 "accuracy-baseline.csv" 文件
            with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}_{str_globalOrlocal}.csv", "a+") as file:
                # file.write(str(RecordAccuracy))
                # file.writelines("\n")
                # 添加标题行
                file.write(f"{client_str}_Accuracy,Time\n")
                # 写入Accuracy数据
                file.write(str(RecordAccuracy) + "\n")

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
        class_names = {
                        0: '0_BENIGN', 
                        1: '1_Bot', 
                        2: '2_DDoS', 
                        3: '3_DoS GoldenEye', 
                        4: '4_DoS Hulk', 
                        5: '5_DoS Slowhttptest', 
                        6: '6_DoS slowloris', 
                        7: '7_FTP-Patator', 
                        8: '8_Heartbleed', 
                        9: '9_Infiltration', 
                        10: '10_PortScan', 
                        11: '11_SSH-Patator', 
                        12: '12_Web Attack Brute Force', 
                        13: '13_Web Attack Sql Injection', 
                        14: '14_Web Attack XSS',
                        15: '15_backdoor',
                        16: '16_dos',
                        17: '17_injection',
                        18: '18_mitm',
                        19: '19_password',
                        20: '20_ransomware',
                        21: '21_scanning',
                        22: '22_xss',
                        23: '23_DrDoS_DNS',
                        24: '24_DrDoS_LDAP',
                        25: '25_DrDoS_MSSQL',
                        26: '26_DrDoS_NTP', 
                        27: '27_DrDoS_NetBIOS',
                        28: '28_DrDoS_SNMP' ,
                        29: '29_DrDoS_SSDP',
                        30: '30_DrDoS_UDP',
                        31: '31_Syn',
                        32: '32_TFTP',
                        33: '33_UDPlag',
                        34: '34_WebDDoS' 
                        }      
        # class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11','12','13','14','15','16','17','18','20','21']
        # class_names = ['0', '1', '2', '3']
        df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names)
        plt.figure(figsize = (20,10))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn', annot_kws={"size": 10})
        plt.title(client_str +"_"+ Choose_method)
        plt.xlabel("prediction",fontsize=12)
        # plt.ylabel("label (ground truth)",fontsize=5,labelpad=0, va='top')
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=11)
        plt.savefig(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_{str_globalOrlocal}_confusion_matrix.png")
        # plt.show()

# 创建用于训练和测试的 DataLoader
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=512, shuffle=True)  # 设置 shuffle 为 True
# test_data 的batch_size要設跟test_data(y_test)的筆數一樣 重要!!!
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
# #############################################################################
# 2. 使用 Flower 集成的代码
# #############################################################################

# 定义Flower客户端类
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):#是知識載入
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)# 剛聚合完的權重
        #global test 對每global round剛聚合完的gobal model進行測試 要在Local_train之前測試
        # 通常第1 round測出來會是0
        # 在训练或测试结束后，保存模型
        torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/Before_local_train_model.pth")
        accuracy = test(net, testloader, start_IDS, client_str,f"global_test",True)
        print("accuracy",accuracy)
                    # 将总体准确率和其他信息写入 "accuracy-baseline.csv" 文件
        with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/accuracy-gobal_model_{client_str}.csv", "a+") as file:
            # file.write(str(RecordAccuracy))
            # file.writelines("\n")
            # 添加标题行
            file.write(f"{client_str}_gobal_model_Accuracy\n")
            # 写入Accuracy数据
            file.write(str(accuracy) + "\n")

        train(net, trainloader, epochs=num_epochs)
        return self.get_parameters(config={}), len(trainloader.dataset), {}#step1上傳給權重，#step2在server做聚合，step3往下傳給server

    def evaluate(self, parameters, config):
        # local test
        # 這邊的測試結果會受到local train的影響
        # 在训练或测试结束后，保存模型
        torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/After_local_train_model.pth")
        accuracy = test(net, testloader, start_IDS, client_str,f"local_test",True)
        self.set_parameters(parameters)#更新現有的知識#step4 更新model
        return accuracy, len(testloader.dataset), {"accuracy": accuracy}

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
        