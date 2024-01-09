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
cicids2017_testdataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\test_CICIDS2017_dataframes_20240110.csv")
TONIOT_testdataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\test_ToN-IoT_dataframes_20240110.csv")


# 加载 .npy 文件
# 载入第一组数据
cicids2017_x_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/x_test_CICIDS2017_20240110.npy")
cicids2017_y_test = np.load(f"./data/dataset_AfterProcessed/CICIDS2017/y_test_CICIDS2017_20240110.npy")

# 载入第二组数据
TONIOT_x_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/x_test_ToN-IoT_20240110.npy")
TONIOT_y_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/y_test_ToN-IoT_20240110.npy")

# 合并 x_test 和 x1_test
merged_x = np.concatenate((cicids2017_x_test, TONIOT_x_test), axis=0)

# 合并 y_test 和 y1_test
merged_y = np.concatenate((cicids2017_y_test, TONIOT_y_test), axis=0)

# D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\TONIOT_test_and_CICIDS2017_test_combine\20240110
# 保存合并后的数组
np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_x.npy", merged_x)
np.save(f"./data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/{today}/merged_y.npy", merged_y)

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



# Loading datasets
# 第一次訓練後有client1和client2有缺Label如下，所以要補，不然訓練會失敗
# cicids2017 client1少 15 19先要補值
# TONIOT client2 少 4 12要補值

def AddLabelToCICIDS2017(df):
    values_to_insert = ['15', '19']
     # 获取 'Label' 列前的所有列的列名
    columns_before_type = df.columns.tolist()[:df.columns.get_loc('Label')]

    # 將新資料插入 DataFrame
    for value in values_to_insert:
        new_data = {'Label': value}  # 這裡 'another_type' 是範例，您可以根據實際需求修改
        
        # 设置 'type' 列前的所有列的值为0
        for column in columns_before_type:
            new_data[column] = 0

        # 添加新数据到 DataFrame
        df = df.append(new_data, ignore_index=True)
    
    return df

### add 沒有的Label到TONIOT
def AddLabelToTONIOT(df):
    values_to_insert = ['4', '12']
     # 获取 'type' 列前的所有列的列名
    columns_before_type = df.columns.tolist()[:df.columns.get_loc('type')]

    # 將新資料插入 DataFrame
    for value in values_to_insert:
        new_data = {'type': value}  # 這裡 'another_type' 是範例，您可以根據實際需求修改
        
        # 设置 'type' 列前的所有列的值为0
        for column in columns_before_type:
            new_data[column] = 0
        
        # 设置'label'列的值为1因為TONIOT的攻擊在label列都視為1，種類用type表示
        new_data['label'] = 1

        # 添加新数据到 DataFrame
        df = df.append(new_data, ignore_index=True)

    return df

cicids2017_traindataset = pd.read_csv(filepath + f"\\dataset_AfterProcessed\\CICIDS2017\\{today}\\train_CICIDS2017_dataframes_20240110.csv")
TONIOT_traindataset = pd.read_csv(filepath + f"\\dataset_AfterProcessed\\TONIOT\\{today}\\train_ToN-IoT_dataframes_20240110.csv")

cicids2017_traindataset = AddLabelToCICIDS2017(cicids2017_traindataset)
TONIOT_traindataset = AddLabelToTONIOT(TONIOT_traindataset)
# client1少Label 15
SaveDataToCsvfile(cicids2017_traindataset, f"./data/dataset_AfterProcessed/CICIDS2017/{today}", f"train_CICIDS2017_dataframes_addlossvalue_{today}")
SaveDataframeTonpArray(cicids2017_traindataset, f"./data/dataset_AfterProcessed/CICIDS2017/{today}", "train_CICIDS2017_addlossvalue",today)

SaveDataToCsvfile(TONIOT_traindataset, f"./data/dataset_AfterProcessed/TONIOT/{today}", f"train_ToN-IoT_dataframes_addlossvalue_{today}")
SaveDataframeTonpArray(TONIOT_traindataset, f"./data/dataset_AfterProcessed/TONIOT/{today}", "train_ToN-IoT_addlossvalue",today)