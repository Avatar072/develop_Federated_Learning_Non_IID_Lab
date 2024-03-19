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

# filepath = "D:\\ToN-IoT-Network\\TON_IoT Datasets\\UNSW-ToN-IoT"
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
# generatefolder(filepath + "\\", "data")
generatefolder(filepath + "\\", "dataset_AfterProcessed")
generatefolder(filepath + "\\dataset_AfterProcessed\\", "TONIOT_test_and_CICIDS2017_test_combine")
generatefolder(filepath + "\\dataset_AfterProcessed\\TONIOT_test_and_CICIDS2017_test_combine\\", today)


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
# # 载入Monday_and_Firday test chi-square
# cicids2017_Monday_and_Firday_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Monday_and_Firday/x_Monday_and_Firday_test_AfterPCA77_20240317.npy")
# cicids2017_Monday_and_Firday_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Monday_and_Firday/y_Monday_and_Firday_test_AfterPCA77_20240317.npy")

# # 载入Tuesday_and_Wednesday_and_Thursday test chi-square
# cicids2017_Tuesday_and_Wednesday_and_Thursday_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Tuesday_and_Wednesday_and_Thursday/x_Tuesday_and_Wednesday_and_Thursday_test_AfterPCA77_20240317.npy")
# cicids2017_Tuesday_and_Wednesday_and_Thursday_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Tuesday_and_Wednesday_and_Thursday/y_Tuesday_and_Wednesday_and_Thursday_test_AfterPCA77_20240317.npy")


# 加载 .npy 文件
# 载入Monday_and_Firday test PCA
cicids2017_Monday_and_Firday_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Monday_and_Firday/x_Monday_and_Firday_test_AfterPCA38_20240318.npy")
cicids2017_Monday_and_Firday_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Monday_and_Firday/y_Monday_and_Firday_test_AfterPCA38_20240318.npy")

# # 载入Tuesday_and_Wednesday_and_Thursday test PCA
cicids2017_Tuesday_and_Wednesday_and_Thursday_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Tuesday_and_Wednesday_and_Thursday/x_Tuesday_and_Wednesday_and_Thursday_test_AfterPCA38_20240318.npy")
cicids2017_Tuesday_and_Wednesday_and_Thursday_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/Tuesday_and_Wednesday_and_Thursday/y_Tuesday_and_Wednesday_and_Thursday_test_AfterPCA38_20240318.npy")


# 载入TONIOT test
# TONIOT_x_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/x_test_ToN-IoT_test_AfterPCA77_20240317.npy")
# TONIOT_y_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/y_test_ToN-IoT_test_AfterPCA77_20240317.npy")


# 合并 x_test 和 x1_test
merged_x = np.concatenate((cicids2017_Monday_and_Firday_x_test,
                           cicids2017_Tuesday_and_Wednesday_and_Thursday_x_test
                           ), axis=0)

# 合并 y_test 和 y1_test
merged_y = np.concatenate((cicids2017_Monday_and_Firday_y_test,
                           cicids2017_Tuesday_and_Wednesday_and_Thursday_y_test
                           ), axis=0)

# # 合并 x_test 和 x1_test
# merged_x = np.concatenate((cicids2017_Monday_and_Firday_x_test,
#                            cicids2017_Tuesday_and_Wednesday_and_Thursday_x_test,
#                            TONIOT_x_test), axis=0)

# # 合并 y_test 和 y1_test
# merged_y = np.concatenate((cicids2017_Monday_and_Firday_y_test,
#                            cicids2017_Tuesday_and_Wednesday_and_Thursday_y_test,
#                            TONIOT_y_test), axis=0)

# D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\TONIOT_test_and_CICIDS2017_test_combine\20240110
# 保存合并后的数组
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_add_toniot.npy", merged_x)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_add_toniot.npy", merged_y)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_cicids2017_toniot_remove_ip_port.npy", merged_x)
# np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_cicids2017_toniot_remove_ip_port.npy", merged_y)
np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x_cicids2017_toniot_PCA_38.npy", merged_x)
np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y_cicids2017_toniot_PCA_38.npy", merged_y)
print("按行合并 x 的结果：")
print(merged_x)

print("按行合并 y 的结果：")
print(merged_y)

counter = Counter(merged_y)
print("test",counter)
# 将 NumPy 数组转换为 Pandas DataFrame
merged_df = pd.DataFrame(data=np.column_stack((merged_x, merged_y)), columns=[f'feature_{i}' for i in range(merged_x.shape[1])] + ['Label'])

# 保存为 CSV 文件
merged_df.to_csv(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/TONIOT_test_and_CICIDS2017_test_merged.csv", index=False)


