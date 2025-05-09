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
from art.attacks.evasion import ProjectedGradientDescent,FastGradientMethod,BasicIterativeMethod,SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier
import os
import random
import time
import datetime
from collections import Counter, defaultdict
from sklearn.metrics import classification_report
from mytoolfunction import ChooseUseModel, getStartorEndtime
from adeversarial_config import SettingAderversarialConfig
from colorama import Fore, Back, Style, init
import math

# 定義兩個向量
# vector1 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],[5.0,5.0,5.0]])
# vector2 = torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0],[3.0,4.0,6.0]])

# 定義兩個向量
vector1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 8.0]])
vector2 = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

print("Vector1:", vector1)
print("Vector2:", vector2)

# 1. 計算單個向量的 L2 範數(即長度)
print("\n1. 計算 Vector1 的 L2 範數:")

# 計算每個元素平方的總和，然後取平方根
squared_sum1 = torch.sum(vector1*vector1)  # 將每個元素平方，然後計算總和
norm1 = math.sqrt(squared_sum1.item())  # 將總和轉為標量並取平方根
print(f"手動計算: √({squared_sum1.item():.4f}) = {norm1:.4f}")

# 計算每個元素平方的總和，然後取平方根
squared_sum2 = torch.sum(vector2*vector2)  # 將每個元素平方，然後計算總和
norm2 = math.sqrt(squared_sum2.item())  # 將總和轉為標量並取平方根
print(f"手動計算: √({squared_sum2.item():.4f}) = {norm2:.4f}")


# 使用 torch.norm() 來計算 L2 範數
print(f"使用 torch.norm(): {torch.norm(vector1).item():.4f}")
print(f"使用 torch.norm(): {torch.norm(vector2).item():.4f}")


# 2. 計算兩個向量之間的歐幾里得距離
print("\n2. 計算 Vector1 和 Vector2 之間的歐幾里得距離:")

# 計算兩個向量之間的差異
diff = vector2 - vector1
# 計算差異的平方
squared_diff = diff*diff
# 計算差異平方和
squared_sum_diff = torch.sum(squared_diff)
# 計算平方和的平方根，得到歐幾里得距離
distance = math.sqrt(squared_sum_diff.item())

print("差異向量:", diff)
print("差異平方:", squared_diff)
print(f"平方和: {squared_sum_diff.item():.4f}")
print(f"歐幾里得距離: √{squared_sum_diff.item():.4f} = {distance:.4f}")
print(f"使用 torch.norm()求歐幾里得距離: {torch.norm(vector1 - vector2,p=2).item():.4f}")


# 定義兩個向量
# vector_1 = torch.tensor([1.0, 2.0, 3.0])
# vector_2 = torch.tensor([4.0, 5.0, 6.0])

# 計算歐幾里得距離
euclidean_distance = torch.norm(vector1 - vector2,p=2)

# 計算L2範數 (標準化)
normalized_vector_1 = vector1 / torch.norm(vector1,p=2)
normalized_vector_2 = vector2 / torch.norm(vector2,p=2)

# 計算兩個經過L2正規化的向量之間的歐幾里得距離
normalized_distance = torch.norm(normalized_vector_1 - normalized_vector_2,p=2)

# 輸出結果
print("Euclidean Distance:", euclidean_distance)
print("Normalized Euclidean Distance:", normalized_distance)

print("vector1:", vector1)

print("normalized_vector_1:", normalized_vector_1)

def ema_smooth(data, smoothing_factor=0.1):
    # Initialize EMA with the first data point
    smoothed_data = torch.zeros_like(data)
    smoothed_data[0] = data[0]

    # Apply EMA smoothing for each data point
    for i in range(1, len(data)):
        smoothed_data[i] = smoothing_factor * data[i] + (1 - smoothing_factor) * smoothed_data[i - 1]
    
    return smoothed_data

# Example usage
data = torch.tensor([0.1, 0.15, 0.2, 0.18, 0.25, 0.3, 0.5, 0.4, 0.35])  # Example data
smoothed_data = ema_smooth(data, smoothing_factor=0.1)

print(smoothed_data)
# Plot the original and smoothed data
# plt.figure(figsize=(10, 6))
# plt.plot(data.numpy(), label="Original Data", color='blue', linestyle='--', marker='o')
# plt.plot(smoothed_data.numpy(), label="Smoothed Data (EMA)", color='red', linewidth=2)
# plt.title("Original vs Smoothed Data (EMA)")
# plt.xlabel("Iterations")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.show()
# # 3. 計算向量的 L1 範數（曼哈頓距離）
# print("\n3. 計算 Vector1 的 L1 範數:")
# l1_norm1 = sum(abs(x) for x in vector1)
# print(f"手動計算: {' + '.join(f'|{x}|' for x in vector1)} = {l1_norm1:.4f}")
# print(f"使用 torch.norm(p=1): {torch.norm(vector1, p=1).item():.4f}")

# # 4. 計算兩個向量之間的 L1 距離
# print("\n4. 計算 Vector1 和 Vector2 之間的 L1 距離:")
# l1_distance = sum(abs(x) for x in diff)
# print(f"手動計算: {' + '.join(f'|{x:.2f}|' for x in diff)} = {l1_distance:.4f}")
# print(f"使用 torch.norm(p=1): {torch.norm(vector1 - vector2, p=1).item():.4f}")
# 提取前126行數據
df = pd.read_csv("E:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20250319\\CICIDS2017_use_20250205_data_merge_label_FGSM_eps0.05測試_79_feature\\Inital_Local_weight_diff_client1_count.csv")
# list_
# list_ = df["dis_variation_Inital_Local"].head(10)
# list_ = df["dis_variation_Inital_Local"]

# 累加平均值計算
# 從第11個數字開始計算累加平均值（忽略前10個數字）
# 先過濾掉前10個數字
df_filtered = df["dis_variation_Inital_Local"][10:]

# 再過濾掉第125到第200個數字（注意索引從10開始，所以需要調整索引位置）
# df_filtered = df_filtered.drop(df_filtered.index[124-10:199-10])
# 計算剩餘數據的累加平均值
df["cumulative_mean"] = df_filtered.expanding().mean()

# 計算每次計算的最大累加平均值
df["cumulative_mean_max"] = df["cumulative_mean"].expanding().max()
# print("\ncumulative_mean:")
# print(df)


# 計算每次計算後的最大累積平均值
df["cumulative_mean_max"] = df["cumulative_mean"].expanding().max()

# 計算最大累積平均值與累積平均值之間的差異
df["difference"] = df["cumulative_mean_max"] - df["cumulative_mean"]

# 計算最大累積平均值與累積平均值之間的差異後求平方
df["squared_difference"] = df["difference"] ** 2

# 計算每次計算後的最大 squared_difference
df["max_squared_difference"] = df["squared_difference"].expanding().max()

# 修改公式:Delta=𝑎∗max_squared_difference*K+(1-a)*D為迪利克雷分布的a值0.5或0.1根據距離變化量，給定值2或是1
# 迪利克雷分布的a值0.1 D=1
# 迪利克雷分布的a值0.5 D=2

#  計算 scale_factor
# scale_factor = df["max_squared_difference"] / df["cumulative_mean_max"] 
# df["scale_factor"] = df.apply(lambda row: max(1, row['max_squared_difference'] / row['cumulative_mean_max']) * 3, axis=1)
# df["scale_factor"] = max(1, scale_factor ) * 3
# *k倍
df["Delta_K_2"]=0.5*df["max_squared_difference"]*2+0.5*1
df["Delta_K_3"]=0.5*df["max_squared_difference"]*3+0.5*1
df["Delta_K_4"]=0.5*df["max_squared_difference"]*4+0.5*1
# df["Delta_K_5"]=0.5*df["max_squared_difference"]*5+0.50.1*2
# df["Delta_a=0.1"]=0.5*df["max_squared_difference"]*2+0.5*1
# df["Delta_a=0.5"]=0.5*df["max_squared_difference"]*2+0.5+0.5*2
# df["Delta"]=0.5*df["max_squared_difference"]*2+0.1*2

# 顯示結果
print("\ncumulative_mean, cumulative_mean_max, difference, squared_difference, max_squared_difference:")
print(df)
df.to_csv("./cumulative_mean_max-cumulative_mean_max_teset.csv")



# # 計算EMA，使用span來控制平滑程度
# df['EMA'] = df['dis_variation_Inital_Local'].ewm(span=3, adjust=False).mean()
# # df['EMA'] = df['dis_variation_Inital_Local'].rolling(window=3).mean()  # 简单滑动窗口方法

# # 使用rolling + apply来实现WMA
# weights = [0.5,0.3,0.2]  # 权重可以根据需要调整
# df['WMA'] = df['dis_variation_Inital_Local'].rolling(window=3).apply(lambda x: (x * weights).sum() / sum(weights), raw=False)
# # 查看計算結果
# print(df[['dis_variation_Inital_Local', 'EMA']].head(20))  # 顯示前20行的結果

# # 如果你只想查看EMA的結果
# print(df['EMA'])  # 顯示前20行的EMA結果
# df['EMA'].to_csv("./EMA.csv")
# df['WMA'].to_csv("./WMA.csv")

# # 繪製圖表
# plt.figure(figsize=(10,6))
# plt.plot(df['dis_variation_Inital_Local'], label='Original Data', color='blue', alpha=0.5)
# plt.plot(df['EMA'], label='Exponential Moving Average (EMA)', color='orange', linestyle='--')
# plt.plot(df['WMA'], label='Exponential Moving Average (WMA)', color='red', linestyle='-')
# # 添加標題和標籤
# plt.title('Exponential Moving Average (EMA) of dis_variation_Inital_Local', fontsize=16)
# plt.xlabel('Rounds', fontsize=12)
# plt.ylabel('Value', fontsize=12)

# # 添加圖例
# plt.legend()

# # 顯示圖表
# plt.show()