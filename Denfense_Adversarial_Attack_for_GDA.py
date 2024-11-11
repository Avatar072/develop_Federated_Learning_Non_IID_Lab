import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SaliencyMapMethod
from art.defences.preprocessor import GaussianAugmentation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mytoolfunction import generatefolder, SaveDataToCsvfile, SaveDataframeTonpArray, ChooseUseModel,getStartorEndtime
from mytoolfunction import Load_Model_BasedOnDataset,mergeDataFrameAndSaveToCsv
from collections import Counter, defaultdict
from colorama import Fore, Back, Style, init
# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)

# 定义文件路径和日期
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today().strftime("%Y%m%d")
start_IDS = time.time()

generatefolder(f"./Adversarial_Attack_Denfense/CICIDS2019/", today)
getStartorEndtime("starttime", start_IDS, f"./Adversarial_Attack_Denfense/CICIDS2019/{today}")

# # 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+f"Using device: {device}")

# # 加载CICIDS2017 test after do labelencode and minmax chi_square45 75 25分
# # afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20240502\\doFeatureSelect\\44\\ALLday_test_dataframes_AfterFeatureSelect.csv")
# # 加载TONIOT test
# afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\test_ToN-IoT_dataframes_20240523.csv")

# # 加载CICIDS2019 train after do labelencode and minmax chi_square45 75 25分
afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\20240502\\01_12_train_dataframes_20240502.csv")
# # 加载CICIDS2019 train after do labelencode and minmax chi_square45 75 25分
afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\20240502\\01_12_test_dataframes_20240502.csv")

# # 加载TONIOT client3 train 均勻劃分
# # afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_train_half3_20240523.csv")
# # 加载TONIOT client3 train 隨機劃分
# # afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_train_half3_random_20240523.csv")

# print("Dataset loaded.")

# # 移除字符串类型特征
def RemoveStringTypeValueForJSMA(afterprocess_dataset):
    crop_dataset = afterprocess_dataset.iloc[:, :]
    #cicids2017 normal
    columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Timestamp', 'Protocol']
    #cicids2017 normal chi-square45 後Protocol可能不見
    # columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Timestamp']
    #toniot
    # columns_to_exclude = ['ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto']
    testdata_removestring = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col in columns_to_exclude]]

    # SaveDataframeTonpArray(testdata_removestring, f"./Adversarial_Attack_Test/{today}", f"testdata_removestring", today)
    print(f"Removed string type columns: {columns_to_exclude}")
    return testdata_removestring, undoScalerdataset

# 加载模型
# model = ChooseUseModel("MLP", 77, 12).to(device)
model = Load_Model_BasedOnDataset("CICIDS2019","MLP",83, 12).to(device)

# 將PyTorch模型轉換為ART分類器
classifier = PyTorchClassifier(
    model=model, # PyTorch 模型
    loss=torch.nn.CrossEntropyLoss(), # 交叉熵損失函數
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),# Adam 優化器，學習率為 0.001
    input_shape=(83,),# 輸入特徵的形狀，包含 83 個特徵
    nb_classes=12,# 分類的類別數量，這裡是12個類別
    clip_values=(0.0, 1.0)#將輸入數據裁剪到 [0.0, 1.0] 範圍內
)
print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+f"ART classifier created.")

# 4. 定義 GDA（高斯資料增強）防禦策略
gda = GaussianAugmentation(sigma=0.1, augmentation=True, ratio=1.0)

# 5. 在訓練過程中應用 GDA 增強
x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_20240502.npy", allow_pickle=True)
y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_20240502.npy", allow_pickle=True)
x_train_augmented, y_train_augmented = gda(x_train, y_train)

mergeDataFrameAndSaveToCsv("./Adversarial_Attack_Denfense/",
                           x_train_augmented,
                           y_train_augmented, 
                           f"CICIDS2019/{today}/CICIDS2019_augmented")

#np.save
np.save(f"./Adversarial_Attack_Denfense/CICIDS2019/x_CICIDS2019_train_augmented.npy", x_train_augmented)
np.save(f"./Adversarial_Attack_Denfense/CICIDS2019/y_CICIDS2019_train_augmented.npy", y_train_augmented)

# 6. 訓練增強後的模型
# classifier.fit(x_train_augmented, y_train_augmented, batch_size=128, epochs=5)

# # 紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime", end_IDS, f"./Adversarial_Attack_Denfense/CICIDS2019/{today}")



