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

generatefolder(f"./Adversarial_Attack_Denfense/CICIDS2017/", today)
getStartorEndtime("starttime", start_IDS, f"./Adversarial_Attack_Denfense/CICIDS2017/{today}")

# # 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.GREEN +Back.WHITE+ Style.BRIGHT+f"Using device: {device}")

# # 加载CICIDS2017 test after do labelencode and minmax chi_square45 75 25分
# # afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20240502\\doFeatureSelect\\44\\ALLday_test_dataframes_AfterFeatureSelect.csv")
# # 加载TONIOT test
# afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\test_ToN-IoT_dataframes_20240523.csv")

# # 加载CICIDS2019 train after do labelencode and minmax chi_square45 75 25分
# afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\20240502\\01_12_train_dataframes_20240502.csv")
# # 加载CICIDS2019 train after do labelencode and minmax chi_square45 75 25分
# afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\20240502\\01_12_test_dataframes_20240502.csv")

# # 加载TONIOT client3 train 均勻劃分
# # afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_train_half3_20240523.csv")
# # 加载TONIOT client3 train 隨機劃分
# # afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_train_half3_random_20240523.csv")
# print("Dataset loaded.")

# # 加载CICIDS2017 train after do labelencode and ALL feature minmax 75 25分
afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20250121\\Deleted79features\\10000筆資料\\ALLDay_train_dataframes_Deleted79features_20250121.csv")


# 加载模型
# model = ChooseUseModel("MLP", 77, 12).to(device)
model = Load_Model_BasedOnDataset("CICIDS2017","MLP",83, 12).to(device)

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
# 設定高斯資料增強的參數
gda = GaussianAugmentation(sigma=0.1,  # 標準差，用於控制擾動量
                           augmentation=True,  # 是否啟用資料增強
                           ratio=1.0)  # 新增擾動後資料的比例（1.0 表示全量使用擾動）

# 5. 在訓練過程中應用 GDA 增強
# x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_20240502.npy", allow_pickle=True)
# y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_20240502.npy", allow_pickle=True)

# 20250121 CIC-IDS2017 after do labelencode and except str and drop feature to 79 feature and all featrue minmax 75 25分
print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"with normal After Do labelencode and minmax and drop feature to 79 feature")
x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_train_Deleted79features_20250121.npy", allow_pickle=True)
y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_Deleted79features_20250121.npy", allow_pickle=True)


x_train_augmented, y_train_augmented = gda(x_train, y_train)

mergeDataFrameAndSaveToCsv("./Adversarial_Attack_Denfense/",
                           x_train_augmented,
                           y_train_augmented, 
                           f"CICIDS2017/{today}/CICIDS2017_augmented")

#np.save
# np.save(f"./Adversarial_Attack_Denfense/CICIDS2019/x_CICIDS2019_train_augmented.npy", x_train_augmented)
# np.save(f"./Adversarial_Attack_Denfense/CICIDS2019/y_CICIDS2019_train_augmented.npy", y_train_augmented)

np.save(f"./Adversarial_Attack_Denfense/CICIDS2017/x_CICIDS2017_train_augmented.npy", x_train_augmented)
np.save(f"./Adversarial_Attack_Denfense/CICIDS2017/y_CICIDS2017_train_augmented.npy", y_train_augmented)

# 6. 訓練增強後的模型
# classifier.fit(x_train_augmented, y_train_augmented, batch_size=128, epochs=5)

# # 紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime", end_IDS, f"./Adversarial_Attack_Denfense/CICIDS2019/{today}")



