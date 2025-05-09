import warnings
import os
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from mytoolfunction import generatefolder,SaveDataToCsvfile,SaveDataframeTonpArray,CheckFileExists,splitdatasetbalancehalf,printFeatureCountAndLabelCountInfo
from colorama import Fore, Back, Style, init
# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)

# filepath = "D:\\ToN-IoT-Network\\TON_IoT Datasets\\UNSW-ToN-IoT"
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
# generatefolder(filepath + "\\", "data")
generatefolder(filepath + "\\", "dataset_AfterProcessed")
# generatefolder(filepath + "\\dataset_AfterProcessed\\", "TONIOT_test_and_CICIDS2017_test_combine")
# generatefolder(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\", today)
# generatefolder(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_and_EdgeIIoT_test_combine\\", today)

# generatefolder(filepath + "\\dataset_AfterProcessed\\", "CICIDS2017_and_CICIDS2018_CICIDS2019_test")
# generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017_and_CICIDS2018_CICIDS2019_test\\", today)
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017_and_CICIDS2018_TONIOT_test\\", today)


def clearDirtyData(df):
    # 将每个列中的 "-" 替换为 NaN
    # df.replace("-", pd.NA, inplace=True)
    # 找到不包含NaN、Infinity和"inf"值和"-"值的行
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    return df


### label Encoding And Replace the number of greater than 10,000
def ReplaceMorethanTenthousandQuantity(df):
  
    # 超過提取10000行的只取10000，其餘保留 
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_encoded.csv")
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed.csv")
    # 获取每个标签的出现次数
    label_counts = df['type'].value_counts()
    # 打印提取后的DataFrame
    print(label_counts)
    # 创建一个空的DataFrame来存储结果
    extracted_df = pd.DataFrame()

    # 获取所有不同的标签
    unique_labels = df['type'].unique()

    # 遍历每个标签
    for label in unique_labels:
        # 选择特定标签的行
        label_df = df[df['type'] == label]
    
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
    print(extracted_df['type'].value_counts())
    return extracted_df

# Loading datasets
# cicids2017_testdataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\test_CICIDS2017_dataframes_20240110.csv")
# TONIOT_testdataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\test_ToN-IoT_dataframes_20240110.csv")


# # 加载 .npy 文件

# # 载入ALLDay test chi-square change ip
# cicids2017_ALLDay_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/x_ALLDay_test_cicids2017_AfterFeatureSelect44_change_ip_encode_20240319.npy")
# cicids2017_ALLDay_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/y_ALLDay_test_cicids2017_AfterFeatureSelect44_change_ip_encode_20240319.npy")

# # 载入ALLDay test chi-square normal
# cicids2017_ALLDay_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/x_ALLDay_test_dataframes_AfterFeatureSelect_Noniid_change_ip_20240323.npy")
# cicids2017_ALLDay_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/y_ALLDay_test_dataframes_AfterFeatureSelect_Noniid_change_ip_20240323.npy")

# cicids2017_ALLDay_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/x_Resplit_test_20240506.npy")
# cicids2017_ALLDay_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/y_Resplit_test_20240506.npy")

# # # 20240502 non iid us BaseLine npfile CIC-IDS2017 after do labelencode and minmax chi_square45 75 25分
# cicids2017_ALLDay_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/x_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy")
# cicids2017_ALLDay_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/y_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy")

# # 20240502 non iid us BaseLine npfile CIC-IDS2017 after do labelencode and minmax chi_square45 75 25分
# cicids2017_ALLDay_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/x_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy")
# cicids2017_ALLDay_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/ALLday/y_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy")

# 20250113 CIC-IDS2017 after do labelencode all featrue minmax 75 25分 Non-iid Do PCA
# 20250121 CIC-IDS2017 after do labelencode all featrue minmax 75 25分 Non-iid Do feature to 79
# print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"test with normal After Do labelencode and minmax")
# cicids2017_ALLDay_x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_test_AfterPCA79_20250113.npy", allow_pickle=True)
# cicids2017_ALLDay_y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_AfterPCA79_20250113_ChangeLabelencode.npy", allow_pickle=True)

# cicids2017_ALLDay_x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_test_Deleted79features_20250121.npy", allow_pickle=True)
# cicids2017_ALLDay_y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_AfterDeleted79features_20250121_ChangeLabelencode.npy", allow_pickle=True)

# 20250317 CIC-IDS2017 after do labelencode and except str and drop feature to 79 feature and all featrue minmax 75 25分
# 79 feature use Label meraged BaseLine data do feature mapping to 123 feature

print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2017" +f"test with normal After Do labelencode and minmax and feature mapping")
cicids2017_ALLDay_x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLday_test_featureMapping_20250317.npy", allow_pickle=True)
cicids2017_ALLDay_y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLday_test_featureMapping_20250317.npy", allow_pickle=True)

# 20250113 CIC-IDS2018 after do labelencode all featrue minmax 75 25分 Non-iid Do PCA
# 20250121 CIC-IDS2018 after do labelencode all featrue minmax 75 25分 Non-iid Do feature to 79
# print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f"test with normal After Do labelencode and minmax")
# cicids2018_ALLDay_x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_test_AfterPCA79_20250113.npy", allow_pickle=True)
# cicids2018_ALLDay_y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_AfterPCA79_20250113_ChangeLabelencode.npy", allow_pickle=True)

# cicids2018_ALLDay_x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_test_20250106.npy", allow_pickle=True)
# cicids2018_ALLDay_y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_featureMapping_20250317_ChangeLabelencode.npy", allow_pickle=True)
# 20250317 CIC-IDS2017 after do labelencode and except str and drop feature to 79 feature and all featrue minmax 75 25分
# 79 feature use Label meraged BaseLine data do feature mapping to 123 feature
print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f"test with normal After Do labelencode and minmax and feature mapping")
cicids2018_ALLDay_x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_test_featureMapping_20250317.npy", allow_pickle=True)
cicids2018_ALLDay_y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_featureMapping_20250317_ChangeLabelencode.npy", allow_pickle=True)

# 20250113 CIC-IDS2017 after do labelencode all featrue minmax 75 25分 Non-iid Do PCA
# 20250121 CIC-IDS2017 after do labelencode all featrue minmax 75 25分 Non-iid Do feature to 79
# print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2019" +f"test with normal After Do labelencode and minmax")
# cicids2019_ALLDay_x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\x_01_12_test_AfterPCA79_20250113.npy", allow_pickle=True)
# cicids2019_ALLDay_y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\y_01_12_test_After_ChangeLabelEncode_for_Noniid.npy", allow_pickle=True)

print(Fore.BLUE+Style.BRIGHT+"Loading TONIOT" +f"test with normal After Do labelencode and minmax and feature mapping")                   
TONIOT_x_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Npfile\\x_TONIOT_test_featureMapping_20250317.npy", allow_pickle=True)
TONIOT_y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\Npfile\\y_TONIOT_test_featureMapping_20250317_ChangeLabelEncode_for_Noniid.npy", allow_pickle=True)
                      
# # 载入Monday_and_Firday test chi-square
# cicids2017_Monday_and_Firday_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Monday_and_Firday/x_Monday_and_Firday_test_AfterPCA77_20240317.npy")
# cicids2017_Monday_and_Firday_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Monday_and_Firday/y_Monday_and_Firday_test_AfterPCA77_20240317.npy")

# # 载入Tuesday_and_Wednesday_and_Thursday test chi-square
# cicids2017_Tuesday_and_Wednesday_and_Thursday_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Tuesday_and_Wednesday_and_Thursday/x_Tuesday_and_Wednesday_and_Thursday_test_AfterPCA77_20240317.npy")
# cicids2017_Tuesday_and_Wednesday_and_Thursday_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Tuesday_and_Wednesday_and_Thursday/y_Tuesday_and_Wednesday_and_Thursday_test_AfterPCA77_20240317.npy")

# 载入TONIOT test minmax normal
# TONIOT_x_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/x_TONIOT_test_change_ip_encode_20240317.npy")
# TONIOT_y_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/y_TONIOT_test_change_ip_encode_20240317.npy")

# 20240523 non iid us BaseLine npfile TONIoT after do labelencode and minmax  75 25分
#因non iid所以 y_test要使用ChangeLabelEncode
# TONIOT_x_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/x_test_ToN-IoT_20240523.npy")
# TONIOT_y_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/y_TONIOT_test_After_ChangeLabelEncode_for_Noniid.npy")


# 载入cicids2019 test minmax normal
# cicids2019_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/x_01_12_test_CICIDS2019_AfterFeatureSelect44_20240428.npy")
# cicids2019_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/y_01_12_test_CICIDS2019_AfterFeatureSelect44_20240428.npy")
# cicids2019_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/x_01_12_Resplit_test_20240506.npy")
# cicids2019_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/y_01_12_Resplit_test_20240506.npy")

# 20240520  non iid client3 use EdgeIIoT after do labelencode and minmax chi_square45 75 25分
 #因non iid所以 y_test要使用ChangeLabelEncode
# EdgeIIOT_x_test = np.load(f"./data/dataset_AfterProcessed/EdgeIIoT/x_EdgeIIoT_test_AfterFeatureSelect44_20240520.npy")
# EdgeIIOT_y_test = np.load(f"./data/dataset_AfterProcessed/EdgeIIoT/y_EdgeIIoT_test_After_ChangeLabelEncode_for_Noniid.npy")

# 加载 .npy 文件
# 载入Monday_and_Firday test PCA
# cicids2017_Monday_and_Firday_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Monday_and_Firday/x_Monday_and_Firday_test_AfterPCA38_20240318.npy")
# cicids2017_Monday_and_Firday_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Monday_and_Firday/y_Monday_and_Firday_test_AfterPCA38_20240318.npy")

# # 载入Tuesday_and_Wednesday_and_Thursday test PCA
# cicids2017_Tuesday_and_Wednesday_and_Thursday_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Tuesday_and_Wednesday_and_Thursday/x_Tuesday_and_Wednesday_and_Thursday_test_AfterPCA38_20240318.npy")
# cicids2017_Tuesday_and_Wednesday_and_Thursday_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Tuesday_and_Wednesday_and_Thursday/y_Tuesday_and_Wednesday_and_Thursday_test_AfterPCA38_20240318.npy")


# 载入TONIOT test
# TONIOT_x_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/x_test_ToN-IoT_test_AfterPCA77_20240317.npy")
# TONIOT_y_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/y_test_ToN-IoT_test_AfterPCA77_20240317.npy")


# # 合并 x_test 和 x1_test
# merged_x = np.concatenate((cicids2017_Monday_and_Firday_x_test,
#                            cicids2017_Tuesday_and_Wednesday_and_Thursday_x_test
#                            ), axis=0)

# # 合并 y_test 和 y1_test
# merged_y = np.concatenate((cicids2017_Monday_and_Firday_y_test,
#                            cicids2017_Tuesday_and_Wednesday_and_Thursday_y_test
#                            ), axis=0)

# # # 合并 x_test 和 x1_test
# merged_x = np.concatenate((cicids2017_Monday_and_Firday_x_test,
#                            cicids2017_Tuesday_and_Wednesday_and_Thursday_x_test,
#                            TONIOT_x_test), axis=0)

# # 合并 y_test 和 y1_test
# merged_y = np.concatenate((cicids2017_Monday_and_Firday_y_test,
#                            cicids2017_Tuesday_and_Wednesday_and_Thursday_y_test,
#                            TONIOT_y_test), axis=0)

# # # 合并 x_test 和 x1_test
# merged_x = np.concatenate((cicids2017_ALLDay_x_test,
#                            TONIOT_x_test), axis=0)

# # 合并 y_test 和 y1_test
# merged_y = np.concatenate((cicids2017_ALLDay_y_test,
#                            TONIOT_y_test), axis=0)


# merged_x = np.concatenate((cicids2017_ALLDay_x_test,
#                            TONIOT_x_test,
#                            EdgeIIOT_x_test), axis=0)

# # 合并 y_test 和 y1_test
# merged_y = np.concatenate((cicids2017_ALLDay_y_test,
#                            TONIOT_y_test,
#                            EdgeIIOT_y_test), axis=0)


merged_x = np.concatenate((cicids2017_ALLDay_x_test,
                           cicids2018_ALLDay_x_test,
                           TONIOT_x_test), axis=0)


# 合并 y_test 和 y1_test
merged_y = np.concatenate((cicids2017_ALLDay_y_test,
                           cicids2018_ALLDay_y_test,
                           TONIOT_y_test), axis=0)

# D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\TONIOT_test_and_CICIDS2017_test_combine\20240110
# 保存合并后的数组
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_add_toniot.npy", merged_x)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_add_toniot.npy", merged_y)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_cicids2017_toniot_remove_ip_port.npy", merged_x)
# # np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_cicids2017_toniot_remove_ip_port.npy", merged_y)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_cicids2017_toniot_Chi_square_45_change_ip.npy", merged_x)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_cicids2017_toniot_Chi_square_45_change_ip.npy", merged_y)

# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_cicids2017_toniot_cicids2019_Chi_square_45_change_ip.npy", merged_x)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_cicids2017_toniot_cicids2019_Chi_square_45_change_ip.npy", merged_y)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_cicids2017_toniot_Chi_square_45_change_ip_encode.npy", merged_x)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_cicids2017_toniot_Chi_square_45_change_ip_encode.npy", merged_y)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_cicids2017_toniot_Chi_square_45_change_ts.npy", merged_x)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_cicids2017_toniot_Chi_square_45_change_ts.npy", merged_y)

np.save(f"./data/dataset_AfterProcessed/CICIDS2017_and_CICIDS2018_TONIOT_test/{today}/merged_x_Non_IID_ALL_test.npy", merged_x)
np.save(f"./data/dataset_AfterProcessed/CICIDS2017_and_CICIDS2018_TONIOT_test/{today}/merged_y_Non_IID_ALL_test.npy", merged_y)

print(merged_x.shape)
print(merged_y.shape)

print("按行合并 x 的结果：")
print(merged_x)

print("按行合并 y 的结果：")
print(merged_y)

counter = Counter(merged_y)
print("test",counter)
# 将 NumPy 数组转换为 Pandas DataFrame
merged_df = pd.DataFrame(data=np.column_stack((merged_x, merged_y)), columns=[f'feature_{i}' for i in range(merged_x.shape[1])] + ['Label'])

# 保存为 CSV 文件
# merged_df.to_csv(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/TONIOT_test_and_CICIDS2017_test_merged_change_ip.csv", index=False)
# merged_df.to_csv(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/TONIOT_test_and_CICIDS2017_and_CICIDS2019_test_merged.csv", index=False)

merged_df.to_csv(f"./data/dataset_AfterProcessed/CICIDS2017_and_CICIDS2018_TONIOT_test/{today}/CICIDS2017_and_CICIDS2018_TONIOT_test_merged.csv", index=False)

# # 将 NumPy 数组转换为 Pandas DataFrame
# merged_df = pd.DataFrame(data=np.column_stack((merged_x, merged_y)), columns=[f'feature_{i}' for i in range(merged_x.shape[1])] + ['Label'])

# # 保存为 CSV 文件
# merged_df.to_csv(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/TONIOT_test_and_CICIDS2017_test_merged_change_ts.csv", index=False)

