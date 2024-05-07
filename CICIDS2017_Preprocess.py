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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from mytoolfunction import SaveDataToCsvfile,printFeatureCountAndLabelCountInfo,CheckFileExists
from mytoolfunction import clearDirtyData,label_Encoding,splitdatasetbalancehalf,spiltweakLabelbalance,SaveDataframeTonpArray,generatefolder
from mytoolfunction import spiltweakLabelbalance_afterOnehot

#############################################################################  variable  ###################
# filepath = "D:\\Labtest20230911\\data"
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"

today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
generatefolder(filepath + "\\", "dataset_AfterProcessed")
# generatefolder(filepath + "\\dataset_AfterProcessed\\", today)
generatefolder(filepath + "\\dataset_AfterProcessed\\", "CICIDS2017")
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\", today)
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\", today)
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\", today)
#############################################################################  variable  ###################

#############################################################################  funcion宣告與實作  ###########

# 加载CICIDS 2017数据集
def writeData(file_path, bool_Rmove_Benign):
    # 读取CSV文件并返回DataFrame
    df = pd.read_csv(file_path,encoding='cp1252',low_memory=False)
    # df = pd.read_csv(file_path)
    # 找到不包含NaN、Infinity和"inf"值的行
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    if bool_Rmove_Benign != False:
        # 找到不包含"BENIGN"值的Label 這邊Label帶進來前面要空格 
        # 下面兩者寫法都行
        # df = df[~(df[' Label']=='BENIGN')]
        # 只留Monday的BENIGN，其他天移掉
        df = df[~df[" Label"].isin(["BENIGN"])]
    return df

### merge多個DataFrame
def mergeData(folder_path, choose_merge_days):
    # 创建要合并的DataFrame列表
    dataframes_to_merge = []

    # 添加每个CSV文件的DataFrame到列表
    if choose_merge_days == "Monday_and_Firday":
        dataframes_to_merge.append(writeData(folder_path + "\\Monday-WorkingHours.pcap_ISCX.csv",False))
        dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Morning.pcap_ISCX.csv",True))
    elif choose_merge_days == "Tuesday_and_Wednesday_and_Thursday":
        dataframes_to_merge.append(writeData(folder_path + "\\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Tuesday-WorkingHours.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Wednesday-workingHours.pcap_ISCX.csv",True))
    elif choose_merge_days == "ALLDay":
        dataframes_to_merge.append(writeData(folder_path + "\\Monday-WorkingHours.pcap_ISCX.csv",False))
        dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Morning.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Tuesday-WorkingHours.pcap_ISCX.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\Wednesday-workingHours.pcap_ISCX.csv",True))

    # 检查特征名是否一致
    if check_column_names(dataframes_to_merge):
        # 特征名一致，可以进行合并
        result = pd.concat(dataframes_to_merge)
        # 使用clearDirtyData函数获取要删除的行的索引列表
        result = clearDirtyData(result)
        
        # 使用DataFrame的drop方法删除包含脏数据的行
        #result = result.drop(list_to_drop)
        return result
    else:
        # 特征名不一致，需要处理这个问题
        print("特征名不一致，请检查并处理特征名一致性")
        return None

### 检查要合并的多个DataFrame的特征名是否一致
def check_column_names(dataframes):
    # 获取第一个DataFrame的特征名列表
    reference_columns = list(dataframes[0].columns)

    # 检查每个DataFrame的特征名是否都与参考特征名一致
    for df in dataframes[1:]:
        if list(df.columns) != reference_columns:
            return False

    return True


### 检查CSV文件是否存在，如果不存在，则合并数据并保存到CSV文件中
def ChecktotalCsvFileIsexists(file,choose_merge_days):
    if not os.path.exists(file):
        # 如果文件不存在，执行数据合并    
        # data = mergeData("D:\\Labtest20230911\\data\\MachineLearningCVE")
        data = mergeData(filepath + "\\CICIDS2017_Original\\TrafficLabelling",choose_merge_days)#完整的資料
        
        # data = clearDirtyData(data)
       
        if data is not None:
            # 去除特征名中的空白和小于ASCII 32的字符
            data.columns = data.columns.str.replace(r'[\s\x00-\x1F]+', '', regex=True)
            # 保存到CSV文件，同时将header设置为True以包括特征名行
            data.to_csv(file, index=False, header=True)
            last_column_index = data.shape[1] - 1
            Label_counts = data.iloc[:, last_column_index].value_counts()
            print(Label_counts)
            print(f"共有 {len(Label_counts)} 个不同的标签")
            print("mergeData complete")
    else:
        print(f"文件 {file} 已存在，不执行合并和保存操作。")

    return file

### label encoding
def label_Encoding(label,df):
    label_encoder = preprocessing.LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])
    df[label].unique()
    return df

### show original label name and after labelenocode
def label_encoding(label, dataset):
    label_encoder = preprocessing.LabelEncoder()
    # original_values = dataset[label].unique()
    
    dataset[label] = label_encoder.fit_transform(dataset[label])
    # encoded_values = dataset[label].unique()
    
      # 获取原始值和编码值的对照关系字典
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    return label_mapping,dataset   

def OneHot_Encoding(feature, data):
    # 建立 OneHotEncoder 實例
    onehotencoder = OneHotEncoder()
    # 使用 OneHotEncoder 進行 One-Hot 編碼
    onehot_encoded_data = onehotencoder.fit_transform(data[[feature]]).toarray()
     # 將 One-Hot 編碼的結果轉換為 DataFrame，並指定欄位名稱
    onehot_encoded_data = pd.DataFrame(onehot_encoded_data, columns=[f"{feature}_{i}" for i in range(onehot_encoded_data.shape[1])])
    # 合併原始 DataFrame 與 One-Hot 編碼的結果
    data = pd.concat([data, onehot_encoded_data], axis=1)
    # 刪除原始特徵列
    data = data.drop(feature, axis=1)
    return data
    
### label Encoding And Replace the number of greater than 10,000
def ReplaceMorethanTenthousandQuantity(df):
  
    # 超過提取10000行的只取10000，其餘保留 
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_encoded.csv")
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed.csv")
    # 获取每个标签的出现次数
    label_counts = df['Label'].value_counts()
    # 打印提取后的DataFrame
    print(label_counts)
    # 创建一个空的DataFrame来存储结果
    extracted_df = pd.DataFrame()

    # 获取所有不同的标签
    unique_labels = df['Label'].unique()

    # 遍历每个标签
    for label in unique_labels:
        # 选择特定标签的行
        label_df = df[df['Label'] == label]
    
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
    print(extracted_df['Label'].value_counts())
    return extracted_df


# CheckCsvFileIsexists檢查file存不存在，若file不存在產生新檔
# ChecktotalCsvFileIsexists(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\CICIDS2017_original.csv")
def GetAtferMergeFinishFilepath(choose_merge_days):
    if choose_merge_days == "Monday_and_Firday":
        Csv_AtferMergeFinish_Filepath = ChecktotalCsvFileIsexists(filepath + 
                                                                  "\\dataset_AfterProcessed\\CICIDS2017\\Monday_and_Firday\\CICIDS2017_Monday_and_Firday.csv",
                                                                  choose_merge_days)
    elif choose_merge_days == "Tuesday_and_Wednesday_and_Thursday":
        Csv_AtferMergeFinish_Filepath = ChecktotalCsvFileIsexists(filepath + 
                                                                  "\\dataset_AfterProcessed\\CICIDS2017\\Tuesday_and_Wednesday_and_Thursday\\CICIDS2017_Tuesday_and_Wednesday_and_Thursday.csv",
                                                                  choose_merge_days)
    elif choose_merge_days == "ALLDay":
        Csv_AtferMergeFinish_Filepath = ChecktotalCsvFileIsexists(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\CICIDS2017_ALLDay.csv",
                                                                  choose_merge_days)

    return Csv_AtferMergeFinish_Filepath
# Loading datasets after megre complete
# mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\CICIDS2017_original.csv")

def LoadingDatasetAfterMegreComplete(choose_merge_days):
    if choose_merge_days == "Monday_and_Firday":
        mergecompelete_dataset = pd.read_csv(GetAtferMergeFinishFilepath(choose_merge_days))
        # print("路徑",GetAtferMergeFinishFilepath(choose_merge_days))
    elif choose_merge_days == "Tuesday_and_Wednesday_and_Thursday":
        mergecompelete_dataset = pd.read_csv(GetAtferMergeFinishFilepath(choose_merge_days))
    elif choose_merge_days == "ALLDay":
        mergecompelete_dataset = pd.read_csv(GetAtferMergeFinishFilepath(choose_merge_days))   
    
    # DoLabelEncoding(mergecompelete_dataset)
    mergecompelete_dataset = ReplaceMorethanTenthousandQuantity(mergecompelete_dataset)
    mergecompelete_dataset = mergecompelete_dataset.drop('FlowID', axis=1)
    # 去除所有非数字、字母和下划线的字符
    mergecompelete_dataset['Label'] = mergecompelete_dataset['Label'].replace({r'[^\w]': ''}, regex=True)

    if(CheckFileExists(filepath + 
                       "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_"+choose_merge_days+"_updated_10000.csv")
                       !=True):
        # dataset.to_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000.csv", index=False)
        mergecompelete_dataset.to_csv(filepath + 
                                      "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_"+choose_merge_days+"_updated_10000.csv",
                                      index=False)
        mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_"+choose_merge_days+"_updated_10000.csv")

    else:
        mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_"+choose_merge_days+"_updated_10000.csv")



    return mergecompelete_dataset


### add TONIOT的Label到CICIDS2017
def AddLabelToCICIDS2017(df,add_mergedays_label_or_dataset_label):
    
    if add_mergedays_label_or_dataset_label == "TONIOT":
            values_to_insert = ['backdoor', 'dos', 'injection', 'mitm', 'password', 
                                'ransomware', 'scanning','xss']
    elif add_mergedays_label_or_dataset_label == "Monday_and_Firday":
            values_to_insert = ['DoSGoldenEye', 'DoSHulk', 'DoSSlowhttptest', 'DoSslowloris', 
                                 'FTPPatator', 'Heartbleed', 'Infiltration', 'SSHPatator', 
                                 'WebAttackBruteForce','WebAttackSqlInjection','WebAttackXSS']
    elif add_mergedays_label_or_dataset_label == "Tuesday_and_Wednesday_and_Thursday":
            values_to_insert = ['BENIGN', 'Bot', 'DDoS', 'PortScan']
     # 获取 'Label' 列前的所有列的列名
    elif add_mergedays_label_or_dataset_label == "CICIDS2019":
            values_to_insert = ['DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP', 
                                'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 
                                'Syn', 'TFTP', 'UDPlag', 'WebDDoS']
    columns_before_type = df.columns.tolist()[:df.columns.get_loc('Label')]

    # 將新資料插入 DataFrame
    for value in values_to_insert:
        new_data = {'Label': value} 
        
        # 设置 'Label' 列前的所有列的值为0
        for column in columns_before_type:
            new_data[column] = 0

        # 添加新数据到 DataFrame
        df = df.append(new_data, ignore_index=True)
    
    # df['Label'] = df['Label'].replace({'BENIGN': 'normal'})
    return df

### train dataframe做就好
def DoAddLabel(df,choose_mergedays_or_dataset,bool_Add_TONIOT_Label):
    if choose_mergedays_or_dataset == "Monday_and_Firday":
        #載入Monday_and_Firday 要add TONIOT和Tuesday_and_Wednesday_and_Thursday的Label
        df = AddLabelToCICIDS2017(df,choose_mergedays_or_dataset)
    elif choose_mergedays_or_dataset == "Tuesday_and_Wednesday_and_Thursday":
        #Tuesday_and_Wednesday_and_Thursday 要add TONIOT和載入Monday_and_Firday的Label
        df = AddLabelToCICIDS2017(df,choose_mergedays_or_dataset)
    elif choose_mergedays_or_dataset == "ALLDay":
        #ALLDay 要add TONIOT的Label
        df = AddLabelToCICIDS2017(df,"TONIOT")
        #ALLDay 要add CICIDS2019的Label
        df = AddLabelToCICIDS2017(df,"CICIDS2019")

    # 當初因CICIDS2017 有做不同星期結合加TONIOT所做的判斷 現在沒用先留著 
    if(bool_Add_TONIOT_Label):
        print(choose_mergedays_or_dataset)
        df = AddLabelToCICIDS2017(df,choose_mergedays_or_dataset)
        df = AddLabelToCICIDS2017(df,"TONIOT")
        

    return df

def LabelMapping(df):
    # 定义您想要的固定编码值的字典映射
    encoding_map = {
        'BENIGN': 0,
        'Bot': 1,
        'DDoS': 2,
        'DoSGoldenEye': 3,
        'DoSHulk': 4,
        'DoSSlowhttptest': 5,
        'DoSslowloris': 6,
        'FTPPatator': 7,
        'Heartbleed': 8,
        'Infiltration': 9,
        'PortScan': 10,
        'SSHPatator': 11,
        'WebAttackBruteForce': 12,
        'WebAttackSqlInjection': 13,
        'WebAttackXSS': 14,
        'backdoor': 15,
        'dos': 16,
        'injection': 17,
        'mitm': 18,
        'password': 19,
        'ransomware': 20,
        'scanning': 21,
        'xss': 22,
        'DrDoS_DNS': 23,
        'DrDoS_LDAP': 24,
        'DrDoS_MSSQL': 25,
        'DrDoS_NTP': 26,
        'DrDoS_NetBIOS': 27,
        'DrDoS_SNMP': 28,
        'DrDoS_SSDP': 29,
        'DrDoS_UDP': 30,
        'Syn': 31,
		'TFTP': 32,
        'UDPlag': 33,
        'WebDDoS': 34
    }
    # 將固定編碼值映射應用到DataFrame中的Label列，直接更新原始的Label列
    df['Label'] = df['Label'].map(encoding_map)
    return df, encoding_map

def DoMinMaxAndLabelEncoding(afterprocess_dataset,choose_merge_days,bool_doencode):
    
    ##除了Label外的特徵做encode
    afterprocess_dataset = label_Encoding('SourceIP',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('SourcePort',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationIP',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationPort',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Protocol',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Timestamp',afterprocess_dataset)
    
    ### extracting features
    #除了Label外的特徵
    crop_dataset=afterprocess_dataset.iloc[:,:-1]
    # 列出要排除的列名，這6個以外得特徵做minmax
    columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
    # 使用条件选择不等于这些列名的列
    doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
    # print(doScalerdataset.info)
    # print(afterprocess_dataset.info)
    # print(undoScalerdataset.info)
    # 開始minmax
    X=doScalerdataset
    X=X.values
    # scaler = preprocessing.StandardScaler() #資料標準化
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    scaler.fit(X)
    X=scaler.transform(X)
    # 将缩放后的值更新到 doScalerdataset 中
    doScalerdataset.iloc[:, :] = X
    # 将排除的列名和选中的特征和 Label 合并为新的 DataFrame
    afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,afterprocess_dataset['Label']], axis = 1)
    print("test")
    # 保存Lable未做label_encoding的DataFrame方便後續Noniid實驗
    if bool_doencode != True:
        # afterminmax_dataset.to_csv(filepath + 
                                #    "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv", index=False)
        if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv")
                           !=True):
            # False 只add 所選擇的星期沒有的Label或只 add TONIOT的Label
            afterminmax_dataset = DoAddLabel(afterminmax_dataset,choose_merge_days,False)
            # True add 所選擇的星期沒有的Label和TONIOT的Label
            # afterminmax_dataset = DoAddLabel(afterminmax_dataset,choose_merge_days,True)

            afterminmax_dataset.to_csv(filepath +
                                       "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv")

        # encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)
        # 固定Label encode值方便後續Noniid實驗
        afterminmax_dataset,encoded_type_values = LabelMapping(afterminmax_dataset)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/encode_and_count_Noniid.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")
    #保存Lable做label_encoding的DataFrame方便後續BaseLine實驗
    else:
        encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)
        # afterminmax_dataset.to_csv(filepath + 
                                #    "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv", index=False)
        
        
        if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")
                           !=True):
            afterminmax_dataset.to_csv(filepath +
                                       "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")
        
        # print("Original Type Values:", original_type_values)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/encode_and_count_baseLine.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")

    return afterminmax_dataset

# Use SpiltIP dataset
def DoMinMaxAndLabelEncodingWithUseIPspilt(afterprocess_dataset,choose_merge_days,bool_doencode):
    
    #這邊就先載入spilt處理好的 有空再優化 媽的
    afterprocess_dataset = pd.read_csv('D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2017\ALLday\CICIDS2017_ALLday_spiltIPtest.csv')



    # # 将IP地址拆分为四个列，并命名为相应的部分
    # afterprocess_dataset[['SourceIP_first', 
    #                       'SourceIP_second', 
    #                       'SourceIP_third', 
    #                       'SourceIP_fourth']] = afterprocess_dataset['SourceIP'].str.split('.', expand=True)
    
    # afterprocess_dataset[['DestinationIP_first', 
    #                       'DestinationIP_second', 
    #                       'DestinationIP_third', 
    #                       'DestinationIP_fourth']] = afterprocess_dataset['DestinationIP'].str.split('.', expand=True)

    ##除了Label外的特徵做encode
    afterprocess_dataset = label_Encoding('SourceIP_first',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('SourceIP_second',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('SourceIP_third',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('SourceIP_fourth',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('SourcePort',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationIP_first',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationIP_second',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationIP_third',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationIP_fourth',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationPort',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Protocol',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Timestamp',afterprocess_dataset)
    
    ### extracting features
    #除了Label外的特徵
    crop_dataset=afterprocess_dataset.iloc[:,:-1]
    # 列出要排除的列名，這6個以外得特徵做minmax
    # columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
    columns_to_exclude = ['SourceIP_first', 'SourceIP_second','SourceIP_third','SourceIP_fourth','SourcePort', 
                          'DestinationIP_first', 'DestinationIP_second','DestinationIP_third','DestinationIP_fourth',
                          'DestinationPort', 'Protocol', 'Timestamp']
    # 使用条件选择不等于这些列名的列
    doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
    # print(doScalerdataset.info)
    # print(afterprocess_dataset.info)
    # print(undoScalerdataset.info)
    # 開始minmax
    X=doScalerdataset
    X=X.values
    # scaler = preprocessing.StandardScaler() #資料標準化
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    scaler.fit(X)
    X=scaler.transform(X)
    # 将缩放后的值更新到 doScalerdataset 中
    doScalerdataset.iloc[:, :] = X
    # 将排除的列名和选中的特征和 Label 合并为新的 DataFrame
    afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,afterprocess_dataset['Label']], axis = 1)
    print("test")
    # 保存Lable未做label_encoding的DataFrame方便後續Noniid實驗
    if bool_doencode != True:
        # afterminmax_dataset.to_csv(filepath + 
                                #    "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv", index=False)
        if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_WithIPSpilt_"+choose_merge_days+".csv")
                           !=True):
            # False 只add 所選擇的星期沒有的Label或只 add TONIOT的Label
            afterminmax_dataset = DoAddLabel(afterminmax_dataset,choose_merge_days,False)
            # True add 所選擇的星期沒有的Label和TONIOT的Label
            # afterminmax_dataset = DoAddLabel(afterminmax_dataset,choose_merge_days,True)

            afterminmax_dataset.to_csv(filepath +
                                       "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_WithIPSpilt_"+choose_merge_days+".csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_WithIPSpilt_"+choose_merge_days+".csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_UndoLabelencode_WithIPSpilt_"+choose_merge_days+".csv")

        # encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)
        # 固定Label encode值方便後續Noniid實驗
        afterminmax_dataset,encoded_type_values = LabelMapping(afterminmax_dataset)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/encode_and_count_Noniid_WithIPSpilt.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")
    #保存Lable做label_encoding的DataFrame方便後續BaseLine實驗
    else:
        encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)
        # afterminmax_dataset.to_csv(filepath + 
                                #    "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv", index=False)
        
        
        if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")
                           !=True):
            afterminmax_dataset.to_csv(filepath +
                                       "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")
        
        # print("Original Type Values:", original_type_values)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/encode_and_count_baseLine.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")

    return afterminmax_dataset

# 手動或自動劃分
def DoSpiltdatasetAutoOrManual(df, bool_Auto,choose_merge_days):
    if bool_Auto:
        train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)
    else:
        train_dataframes,test_dataframes = manualspiltdataset(df,choose_merge_days)

    return train_dataframes,test_dataframes

#手動劃分
def manualspiltdataset(df,choose_merge_days):
    print(choose_merge_days)
    train_dataframes,test_dataframes = pd.DataFrame(),pd.DataFrame()
    if choose_merge_days =="Monday_and_Firday":
            train_dataframes = pd.concat([
                                            df[df['Label'] == 0].iloc[:8000],
                                            df[df['Label'] == 1].iloc[:1565],
                                            df[df['Label'] == 2].iloc[:8000],
                                            df[df['Label'] == 3].iloc[:1],
                                            df[df['Label'] == 4].iloc[:1],
                                            df[df['Label'] == 5].iloc[:1],
                                            df[df['Label'] == 6].iloc[:1],
                                            df[df['Label'] == 7].iloc[:1],
                                            df[df['Label'] == 8].iloc[:1],
                                            df[df['Label'] == 9].iloc[:1],
                                            df[df['Label'] == 10].iloc[:8000],
                                            df[df['Label'] == 11].iloc[:1],
                                            df[df['Label'] == 12].iloc[:1],
                                            df[df['Label'] == 13].iloc[:1],
                                            df[df['Label'] == 14].iloc[:1],
                                            df[df['Label'] == 15].iloc[:1],
                                            df[df['Label'] == 16].iloc[:1],
                                            df[df['Label'] == 17].iloc[:1],
                                            df[df['Label'] == 18].iloc[:1],
                                            df[df['Label'] == 19].iloc[:1],
                                            df[df['Label'] == 20].iloc[:1],
                                            df[df['Label'] == 21].iloc[:1],
                                            df[df['Label'] == 22].iloc[:1]
                                        ], ignore_index=True)

            test_dataframes = pd.concat([
                                            df[df['Label'] == 0].iloc[8000:],
                                            df[df['Label'] == 1].iloc[1565:],
                                            df[df['Label'] == 2].iloc[8000:],
                                            df[df['Label'] == 10].iloc[8000:]
                                        ], ignore_index=True)
    elif choose_merge_days =="Tuesday_and_Wednesday_and_Thursday":
            train_dataframes = pd.concat([
                                            df[df['Label'] == 0].iloc[:1],
                                            df[df['Label'] == 1].iloc[:1],
                                            df[df['Label'] == 2].iloc[:1],
                                            df[df['Label'] == 3].iloc[:8000],
                                            df[df['Label'] == 4].iloc[:8000],
                                            df[df['Label'] == 5].iloc[:4399],
                                            df[df['Label'] == 6].iloc[:4637],
                                            df[df['Label'] == 7].iloc[:6348],
                                            df[df['Label'] == 10].iloc[:1],
                                            df[df['Label'] == 11].iloc[:4718],
                                            df[df['Label'] == 12].iloc[:1206],
                                            df[df['Label'] == 14].iloc[:522],
                                            df[df['Label'] == 15].iloc[:1],
                                            df[df['Label'] == 16].iloc[:1],
                                            df[df['Label'] == 17].iloc[:1],
                                            df[df['Label'] == 18].iloc[:1],
                                            df[df['Label'] == 19].iloc[:1],
                                            df[df['Label'] == 20].iloc[:1],
                                            df[df['Label'] == 21].iloc[:1],
                                            df[df['Label'] == 22].iloc[:1]                                        
                                        ], ignore_index=True)

            test_dataframes = pd.concat([
                                            df[df['Label'] == 3].iloc[8000:],
                                            df[df['Label'] == 4].iloc[8000:],
                                            df[df['Label'] == 5].iloc[4399:],
                                            df[df['Label'] == 6].iloc[4637:],
                                            df[df['Label'] == 7].iloc[6348:],
                                            df[df['Label'] == 11].iloc[4718:],
                                            df[df['Label'] == 12].iloc[1206:],
                                            df[df['Label'] == 14].iloc[522:]                                        
                                        ], ignore_index=True)  
    elif choose_merge_days =="ALLDay":
            print(choose_merge_days)
            train_dataframes = pd.concat([
                                            df[df['Label'] == 0].iloc[:8000],
                                            df[df['Label'] == 1].iloc[:1565],
                                            df[df['Label'] == 2].iloc[:8000],
                                            df[df['Label'] == 3].iloc[:8000],
                                            df[df['Label'] == 4].iloc[:8000],
                                            df[df['Label'] == 5].iloc[:4399],
                                            df[df['Label'] == 6].iloc[:4637],
                                            df[df['Label'] == 7].iloc[:6348],
                                            df[df['Label'] == 8].iloc[:7],
                                            df[df['Label'] == 9].iloc[:24],
                                            df[df['Label'] == 10].iloc[:8000],
                                            df[df['Label'] == 11].iloc[:4718],
                                            df[df['Label'] == 12].iloc[:1206],
                                            df[df['Label'] == 13].iloc[:14],
                                            df[df['Label'] == 14].iloc[:522],
                                            df[df['Label'] == 15].iloc[:1],
                                            df[df['Label'] == 16].iloc[:1],
                                            df[df['Label'] == 17].iloc[:1],
                                            df[df['Label'] == 18].iloc[:1],
                                            df[df['Label'] == 19].iloc[:1],
                                            df[df['Label'] == 20].iloc[:1],
                                            df[df['Label'] == 21].iloc[:1],
                                            df[df['Label'] == 22].iloc[:1]                                        
                                        ], ignore_index=True)

            test_dataframes = pd.concat([
                                            df[df['Label'] == 0].iloc[8000:],
                                            df[df['Label'] == 1].iloc[1565:],
                                            df[df['Label'] == 2].iloc[8000:],
                                            df[df['Label'] == 3].iloc[8000:],
                                            df[df['Label'] == 4].iloc[8000:],
                                            df[df['Label'] == 5].iloc[4399:],
                                            df[df['Label'] == 6].iloc[4637:],
                                            df[df['Label'] == 7].iloc[6348:],
                                            df[df['Label'] == 8].iloc[7:],
                                            df[df['Label'] == 9].iloc[24:],
                                            df[df['Label'] == 10].iloc[8000:],
                                            df[df['Label'] == 11].iloc[4718:],
                                            df[df['Label'] == 12].iloc[1206:],
                                            df[df['Label'] == 13].iloc[14:],
                                            df[df['Label'] == 14].iloc[522:]                                        
                                        ], ignore_index=True)  
            
    return train_dataframes,test_dataframes

# do Labelencode and minmax 
def DoSpiltAllfeatureAfterMinMax(df,choose_merge_days,bool_Noniid):  
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes,"Label")
    
    
    if bool_Noniid !=True:
        if choose_merge_days =="Monday_and_Firday":
            # 篩選test_dataframes中標籤為2、3、5和19的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([2,3,5,19])]
            # test刪除Label相當於2,3,5,19的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([2,3,5,19])]
            # 合併Label2,3,5,19回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
        
        elif choose_merge_days =="Tuesday_and_Wednesday_and_Thursday":
            # Noniid時
            # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
            # encode後對照如下
            # Heartbleed:8、
            # Infiltration:9、
            # Web Attack Sql Injection:13
            # Label encode mode  分別取出Label等於8、9、13的數據 對半分
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)
            # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
            
            # # 篩選test_dataframes中標籤為1和17的行加回去train
            # train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([1,17])]
            # # test刪除Label相當於1和17的行，因為這些是因為noniid要加到train的Label
            # test_dataframes = test_dataframes[~test_dataframes['Label'].isin([1,17])]
            # # 合併Label1和17回去到train
            # train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
        elif choose_merge_days =="ALLDay":
            # Noniid時
            # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
            # encode後對照如下
            # Heartbleed:8、
            # Infiltration:9、
            # Web Attack Sql Injection:13
            # Label encode mode  分別取出Label等於8、9、13的數據 對半分
            
            
            # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
            List_train_Label = []
            List_test_Label = []
            for i in range(15):
                if i == 8 or i == 9 or i ==13:
                    continue
                train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
                List_train_Label.append(train_label_split)
                List_test_Label.append(test_label_split)         
            
            df_train = pd.concat(List_train_Label)
            df_test = pd.concat(List_test_Label)
            SaveDataToCsvfile(df_train, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_train_dataframes_df_{today}")
            SaveDataToCsvfile(df_test,  f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_test_dataframes_df_{today}")
            
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)
            # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
            
            # 篩選test_dataframes中標籤為29,32的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([29,32])]
            # test刪除Label相當於29,32的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([29,32])]
            # # 合併Label29,32回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
    else:
        # BaseLine時
        # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
        # encode後對照如下
        # Heartbleed:8、
        # Infiltration:9、
        # Web Attack Sql Injection:13
        if choose_merge_days =="ALLDay":
            
            # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
            List_train_Label = []
            List_test_Label = []
            for i in range(15):
                if i == 8 or i == 9 or i ==13:
                    continue
                train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
                List_train_Label.append(train_label_split)
                List_test_Label.append(test_label_split)         
            
            train_dataframes = pd.concat(List_train_Label)
            test_dataframes = pd.concat(List_test_Label)
            
            # Label encode mode  分別取出Label等於8、9、13的數據 對6633分
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)

            # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
            print("test",test_dataframes['Label'].value_counts())
    
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/encode_and_count_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_train_dataframes_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_test_dataframes_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_test",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_train",today)

def dofeatureSelect(df, slecet_label_counts,choose_merge_days):
    significance_level=0.05
    if (slecet_label_counts == None):
        slecet_label_counts ='all'

    # 開始ch2特征选择，先分离特征和目标变量
    y = df['Label']  # 目标变量
    X = df.iloc[:, :-1]  # 特征

    # 创建 SelectKBest 模型，选择 f_classif 统计测试方法
    k_best = SelectKBest(score_func=chi2, k=slecet_label_counts)
    X_new = k_best.fit_transform(X, y)

    # 获取被选中的特征的索引
    selected_feature_indices = k_best.get_support(indices=True)

    # 打印被选中的特征的列名
    selected_features = X.columns[selected_feature_indices]
    print("Selected Features:")
    print(selected_features)

    # 印选择的特征的名称、索引和相应的 F 值、p 值
    print("\nSelected Feature Statistics:")
    selected_feature_stats = []
    for idx, feature_idx in enumerate(selected_feature_indices):
        feature_name = selected_features[idx]
        f_value = k_best.scores_[feature_idx]
        p_value = k_best.pvalues_[feature_idx]
        print(f"Name = {feature_name}, F-value = {f_value}, p-value = {p_value}")
        selected_feature_stats.append({
            'Name': feature_name,
            'F-value': f_value,
            'p-value': p_value
        })
        # 判斷 p-值 是否小於显著性水準
        if p_value <= significance_level:
            print(f"Feature {feature_name} is statistically significant.")
        else:
            print(f"Feature {feature_name} is not statistically significant.")

    print("selected特徵數", len(selected_feature_indices))

    # 迴圈遍歷所有特徵，印出相應的統計信息
    print("\nAll Features Statistics:")
    all_feature_stats = []
    for idx, feature_name in enumerate(X.columns):
        f_value = k_best.scores_[idx]
        p_value = k_best.pvalues_[idx]
        print(f"Name = {feature_name}, F-value = {f_value}, p-value = {p_value}")
        all_feature_stats.append({
            'Name': feature_name,
            'F-value': f_value,
            'p-value': p_value
        })
    print("原特徵數", len(X.columns))

    # 將選中特徵的統計信息存儲到 CSV 文件
    selected_feature_stats_df = pd.DataFrame(selected_feature_stats)
    all_feature_stats_df = pd.DataFrame(all_feature_stats)
    SaveDataToCsvfile(selected_feature_stats_df, 
                      f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_selected_feature_stats_{today}")

    SaveDataToCsvfile(all_feature_stats_df, 
                      f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_all_feature_stats_{today}")

    # 将未被选中特徵的統計信息存儲到 CSV 文件
    unselected_feature_indices = list(set(range(len(X.columns))) - set(selected_feature_indices))
    unselected_features = X.columns[unselected_feature_indices]
    unselected_feature_stats = []
    for idx, feature_idx in enumerate(unselected_feature_indices):
        feature_name = unselected_features[idx]
        f_value = k_best.scores_[feature_idx]
        p_value = k_best.pvalues_[feature_idx]
        print(f"Unselected Feature - Name = {feature_name}, F-value = {f_value}, p-value = {p_value}")
        unselected_feature_stats.append({
            'Name': feature_name,
            'F-value': f_value,
            'p-value': p_value
        })
    
    # 將未被選中特徵的統計信息存儲到 CSV 文件
    unselected_feature_stats_df = pd.DataFrame(unselected_feature_stats)
    SaveDataToCsvfile(unselected_feature_stats_df, 
                      f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_unselected_feature_stats_{today}")
    

    # 将 X_new 转换为 DataFrame
    X_new_df = pd.DataFrame(X_new, columns=selected_features)

    # 将选中的特征和 Label 合并为新的 DataFrame
    selected_data = pd.concat([X_new_df, df['Label']], axis=1)
    
    # SaveDataToCsvfile(selected_data, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
    #                   f"{choose_merge_days}_AfterSelected_{slecet_label_counts}_feature_data_{today}")
    return selected_data

# do chi-square and Labelencode and minmax 
def DoSpiltAfterFeatureSelect(df,slecet_label_counts,choose_merge_days,bool_Noniid):
    df = dofeatureSelect(df,slecet_label_counts,choose_merge_days)
    # 自動切
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # 手動劃分資料集!!!!!!!! 注意用手動切資料集 cicids2017訓練結果會他媽的超級差 媽的 爛function
    # train_dataframes, test_dataframes = DoSpiltdatasetAutoOrManual(df, False,choose_merge_days)    
    
    #加toniot的情況
    if bool_Noniid !=True:
        if choose_merge_days =="Monday_and_Firday":
            # 篩選test_dataframes中標籤為2、3、5和19的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([12])]
            # test刪除Label相當於2,3,5,19的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([12])]
            # 合併Label2,3,5,19回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
        
        elif choose_merge_days =="Tuesday_and_Wednesday_and_Thursday":
            # Noniid時
            # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
            # encode後對照如下
            # Heartbleed:8、
            # Infiltration:9、
            # Web Attack Sql Injection:13
            # Label encode mode  分別取出Label等於8、9、13的數據 對半分
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)
            # # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
        elif choose_merge_days =="ALLDay":
            # Noniid時
            # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
            # encode後對照如下
            # Heartbleed:8、
            # Infiltration:9、
            # Web Attack Sql Injection:13
            # Label encode mode  分別取出Label等於8、9、13的數據 對半分
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)
            # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
            
            # 篩選test_dataframes中標籤為29,32的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([29,32])]
            # test刪除Label相當於29,32的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([29,32])]
            # # 合併Label29,32回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
    else:
        # BaseLine時
        # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
        # encode後對照如下
        # Heartbleed:8、
        # Infiltration:9、
        # Web Attack Sql Injection:13
        if choose_merge_days =="Tuesday_and_Wednesday_and_Thursday":
            # Label encode mode  分別取出Label等於8、9、13的數據 對半分
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)

            # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
    
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/encode_and_count_after_chisquare_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")


    SaveDataToCsvfile(train_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}",  
                      f"{choose_merge_days}_train_dataframes_AfterFeatureSelect")
    SaveDataToCsvfile(test_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_test_dataframes_AfterFeatureSelect")
    SaveDataframeTonpArray(test_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                           f"{choose_merge_days}_test_cicids2017_AfterFeatureSelect{slecet_label_counts}",today)
    SaveDataframeTonpArray(train_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                           f"{choose_merge_days}_train_cicids2017_AfterFeatureSelect{slecet_label_counts}",today)




# do PCA and Labelencode and minmax 
def DoSpiltAfterDoPCA(df,number_of_components,choose_merge_days,bool_Noniid):
    # number_of_components=20
    
    crop_dataset=df.iloc[:,:-1]
    # 列出要排除的列名
    columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
    # 使用条件选择不等于这些列名的列
    # number_of_components=77 # 原84個的特徵，扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label' | 84-7 =77
    doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
    # afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,mergecompelete_dataset['Label']], axis = 1)

    print("Original number of features:", len(df.columns) - 1)  # 减去 'Label' 列
    # X = df.drop(columns=['Label'])  # 提取特征，去除 'Label' 列
    X = doScalerdataset
    pca = PCA(n_components=number_of_components)
    columns_array=[]
    for i in range (number_of_components):
        columns_array.append("principal_Component"+str(i+1))
        
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = columns_array)

    finalDf = pd.concat([undoScalerdataset,principalDf, df[['Label']]], axis = 1)
    df=finalDf

    SaveDataToCsvfile(df, 
                      f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                      f"{choose_merge_days}_cicids2017_AfterProcessed_minmax_PCA")

    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # 手動劃分資料集
    # train_dataframes, test_dataframes = DoSpiltdatasetAutoOrManual(df, False,choose_merge_days)
    # printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes,"Label")
    if bool_Noniid !=True:
        if choose_merge_days =="Monday_and_Firday":
            # 篩選test_dataframes中標籤為2、3、5和19的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([12])]
            # test刪除Label相當於2,3,5,19的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([12])]
            # 合併Label2,3,5,19回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])

        elif choose_merge_days =="Tuesday_and_Wednesday_and_Thursday":
            # Noniid時
            # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
            # encode後對照如下
            # Heartbleed:8、
            # Infiltration:9、
            # Web Attack Sql Injection:13
            # Label encode mode  分別取出Label等於8、9、13的數據 對半分
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)

            # # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
        elif choose_merge_days =="ALLDay":
            # Noniid時
            # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
            # encode後對照如下
            # Heartbleed:8、
            # Infiltration:9、
            # Web Attack Sql Injection:13
            # Label encode mode  分別取出Label等於8、9、13的數據 對半分
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)
            # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
            
            # 篩選test_dataframes中標籤為29,32的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([29,32])]
            # test刪除Label相當於29,32的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([29,32])]
            # # 合併Label29,32回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
    else:
        # BaseLine時
        # 單獨把Heartbleed、Infiltration、Web Attack Sql Injection测试集的比例为33%
        # encode後對照如下
        # Heartbleed:8、
        # Infiltration:9、
        # Web Attack Sql Injection:13
        if choose_merge_days =="Tuesday_and_Wednesday_and_Thursday":
            # Label encode mode  分別取出Label等於8、9、13的數據 對半分
            train_label_Heartbleed, test_label_Heartbleed = spiltweakLabelbalance(8,df,0.33)
            train_label_Infiltration, test_label_Infiltration = spiltweakLabelbalance(9,df,0.33)
            train_label_WebAttackSql_Injection, test_label_WebAttackSql_Injection = spiltweakLabelbalance(13,df,0.33)

            # # 刪除Label相當於8、9、13的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
            # 合併Label8、9、13回去
            test_dataframes = pd.concat([test_dataframes, test_label_Heartbleed, test_label_Infiltration, test_label_WebAttackSql_Injection])
            train_dataframes = pd.concat([train_dataframes,train_label_Heartbleed, train_label_Infiltration,train_label_WebAttackSql_Injection])
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/encode_and_count_after_PCA_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                      f"{choose_merge_days}_train_dataframes_AfterPCA{number_of_components}_{today}")
    SaveDataToCsvfile(test_dataframes,
                      f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                      f"{choose_merge_days}_test_dataframes_AfterPCA{number_of_components}_{today}")
    SaveDataframeTonpArray(test_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                           f"{choose_merge_days}_test_AfterPCA{number_of_components}",today)
    SaveDataframeTonpArray(train_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                           f"{choose_merge_days}_train_AfterPCA{number_of_components}",today)

# do split train to half for iid and Labelencode and minmax 
def DoSpilthalfForiid(choose_merge_days):
    if choose_merge_days == "ALLDay":
        df_ALLtrain = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\20240502\\ALLDay_train_dataframes_20240502.csv")
                    # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_half1_Label = []
        List_train_half2_Label = []
        for i in range(15):
            train_half1_label_split, train_half2_label_split = spiltweakLabelbalance(i,df_ALLtrain,0.5)
            List_train_half1_Label.append(train_half1_label_split)
            List_train_half2_Label.append(train_half2_label_split)         
            
        df_train_half1 = pd.concat(List_train_half1_Label)
        df_train_half2 = pd.concat(List_train_half2_Label)
            

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/encode_and_count_iid.csv", "a+") as file:
            label_counts = df_train_half1['Label'].value_counts()
            print("df_train_half1\n", label_counts)
            file.write("df_train_half1_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = df_train_half2['Label'].value_counts()
            print("df_train_half2\n", label_counts)
            file.write("df_train_half2_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(df_train_half1, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half1_{today}")
        SaveDataToCsvfile(df_train_half2,  f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half2_{today}")
        SaveDataframeTonpArray(df_train_half1, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half1",today)
        SaveDataframeTonpArray(df_train_half2, f"./data/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half2",today)
# 開始進行資料劃分主要function
def SelectfeatureUseChiSquareOrPCA(df,choose_merge_days,bool_doChiSquare,bool_doPCA,bool_Noniid):
    if bool_doChiSquare!=False:
        # 選ALL特徵
        # DoSpiltAfterFeatureSelect(df,None)
        #ChiSquare選80個特徵
        # DoSpiltAfterFeatureSelect(df,80,choose_merge_days,bool_Noniid)
        # #ChiSquare選70個特徵
        # DoSpiltAfterFeatureSelect(df,70,choose_merge_days,bool_Noniid)
        # # #ChiSquare選65個特徵
        # DoSpiltAfterFeatureSelect(df,60,choose_merge_days,bool_Noniid)
        # # #ChiSquare選60個特徵
        # DoSpiltAfterFeatureSelect(df,60,choose_merge_days,bool_Noniid)
        # # #ChiSquare選55個特徵
        # DoSpiltAfterFeatureSelect(df,55,choose_merge_days,bool_Noniid)
        # # #ChiSquare選50個特徵
        # DoSpiltAfterFeatureSelect(df,50,choose_merge_days,bool_Noniid)
        # # #ChiSquare選46個特徵
        # DoSpiltAfterFeatureSelect(df,46,choose_merge_days,bool_Noniid)
        # # #ChiSquare選45個特徵
        # DoSpiltAfterFeatureSelect(df,45,choose_merge_days,bool_Noniid)
        # #ChiSquare選44個特徵
        DoSpiltAfterFeatureSelect(df,44,choose_merge_days,bool_Noniid)
        # #ChiSquare選40個特徵
        # DoSpiltAfterFeatureSelect(df,40,choose_merge_days,bool_Noniid)
        # #ChiSquare選38個特徵
        # DoSpiltAfterFeatureSelect(df,38,choose_merge_days,bool_Noniid)
    elif bool_doPCA!=False:
        #  #PCA選77個特徵 總84特徵=77+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSpiltAfterDoPCA(df,77,choose_merge_days,bool_Noniid)
        # #PCA選73個特徵 總80特徵=73+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSpiltAfterDoPCA(df,73,choose_merge_days,bool_Noniid)
        # #PCA選63個特徵 總70特徵=73+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSpiltAfterDoPCA(df,63,choose_merge_days,bool_Noniid)
        # #PCA選53個特徵 總60特徵=53+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSpiltAfterDoPCA(df,53,choose_merge_days,bool_Noniid)
        #PCA選43個特徵 總50特徵=43+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSpiltAfterDoPCA(df,43,choose_merge_days,bool_Noniid)
        # #PCA選38個特徵 總45特徵=38+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        DoSpiltAfterDoPCA(df,38,choose_merge_days,bool_Noniid)
        # #PCA選33個特徵 總40特徵=33+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSpiltAfterDoPCA(df,33,choose_merge_days,bool_Noniid) 

def forBaseLineUseData(choose_merge_days,bool_Noniid):
    if choose_merge_days == "Monday_and_Firday":
        # 載入資料集
        df_Monday_and_Firday=LoadingDatasetAfterMegreComplete(choose_merge_days)
        # 預處理和正規化
        # True for BaseLine
        # False for Noniid
        df_Monday_and_Firday=DoMinMaxAndLabelEncoding(df_Monday_and_Firday,choose_merge_days,bool_Noniid)
        # 一般全部特徵
        # DoSpiltAllfeatureAfterMinMax(df_Monday_and_Firday,choose_merge_days,bool_Noniid)
        # 做ChiSquare
        # SelectfeatureUseChiSquareOrPCA(df_Monday_and_Firday,choose_merge_days,True,False,bool_Noniid)
        # 做PCA
        SelectfeatureUseChiSquareOrPCA(df_Monday_and_Firday,choose_merge_days,False,True,bool_Noniid)
    elif choose_merge_days == "Tuesday_and_Wednesday_and_Thursday":
        df_Tuesday_and_Wednesday_and_Thursday=LoadingDatasetAfterMegreComplete(choose_merge_days)
        # 預處理和正規化
        # True for BaseLine
        # False for Noniid
        df_Tuesday_and_Wednesday_and_Thursday=DoMinMaxAndLabelEncoding(df_Tuesday_and_Wednesday_and_Thursday,choose_merge_days,bool_Noniid)
        # DoSpiltAllfeatureAfterMinMax(df_Tuesday_and_Wednesday_and_Thursday,choose_merge_days,bool_Noniid)
        # # 做ChiSquare
        SelectfeatureUseChiSquareOrPCA(df_Tuesday_and_Wednesday_and_Thursday,choose_merge_days,True,False,bool_Noniid)
        # # 做PCA
        # SelectfeatureUseChiSquareOrPCA(df_Tuesday_and_Wednesday_and_Thursday,choose_merge_days,False,True,bool_Noniid)
    elif choose_merge_days == "ALLDay":

        df_ALLDay=LoadingDatasetAfterMegreComplete(choose_merge_days)
        # True for BaseLine
        # False for Noniid
        df_ALLDay=DoMinMaxAndLabelEncoding(df_ALLDay,choose_merge_days,bool_Noniid)

        # BaseLine測試用
        # df_ALLDay = pd.read_csv('D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2017\ALLday\\20240319\doFeatureSelect\\45_20240124\\cicids2017_AfterProcessed_minmax.csv')

        # MinMax Use with IP spilt
        # df_ALLDay=DoMinMaxAndLabelEncodingWithUseIPspilt(df_ALLDay,choose_merge_days,bool_Noniid)
        # for iid 實驗將ALL train分一半
        DoSpilthalfForiid(choose_merge_days)
        # 一般全部特徵
        # DoSpiltAllfeatureAfterMinMax(df_ALLDay,choose_merge_days,bool_Noniid)
        # 做ChiSquare
        # SelectfeatureUseChiSquareOrPCA(df_ALLDay,choose_merge_days,True,False,bool_Noniid)
        # 做PCA
        # SelectfeatureUseChiSquareOrPCA(df_ALLDay,choose_merge_days,False,True,bool_Noniid)


# True for BaseLine
# False for Noniid
# forBaseLineUseData("Monday_and_Firday",False)
# forBaseLineUseData("Monday_and_Firday",True)
# forBaseLineUseData("Tuesday_and_Wednesday_and_Thursday",False)
# forBaseLineUseData("Tuesday_and_Wednesday_and_Thursday",True)
# forBaseLineUseData("ALLDay",False)
forBaseLineUseData("ALLDay",True)

# DoAllfeatureOrSelectfeature(afterminmax_dataset,False)
# DoAllfeatureOrSelectfeature(afterminmax_dataset,True)






# ###########################################################Don't do one hot mode################################################################################################
# # 要做這邊的話上面OneHot_Encoding Protocol和Label要註解掉
# # Label encode mode  分別取出Label等於8、9、13、14的數據 對半分
# train_label_8, test_label_8 = spiltweakLabelbalance(8,mergecompelete_dataset,0.4)
# train_label_9, test_label_9 = spiltweakLabelbalance(9,mergecompelete_dataset,0.5)
# train_label_13, test_label_13 = spiltweakLabelbalance(13,mergecompelete_dataset,0.5)
# # train_label_14, test_label_14 = spiltweakLabelbalance(14,mergecompelete_dataset,0.5)

# # # 刪除Label相當於8、9、13、14的行
# # test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13, 14])]
# # train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13,14])]
# # # 合併Label8、9、13、14回去
# # test_dataframes = pd.concat([test_dataframes, test_label_8, test_label_9, test_label_13, test_label_14])
# # train_dataframes = pd.concat([train_dataframes,train_label_8, train_label_9,train_label_13,train_label_14])

# # # 刪除Label相當於8、9、13的行
# test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13])]
# train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13])]
# # 合併Label8、9、13回去
# test_dataframes = pd.concat([test_dataframes, test_label_8, test_label_9, test_label_13])
# train_dataframes = pd.concat([train_dataframes,train_label_8, train_label_9,train_label_13])

# label_counts = test_dataframes['Label'].value_counts()
# print("test_dataframes\n", label_counts)
# label_counts = train_dataframes['Label'].value_counts()
# print("train_dataframes\n", label_counts)

# # split train_dataframes各一半
# train_half1,train_half2 = splitdatasetbalancehalf(train_dataframes)

# # 找到train_df_half1和train_df_half2中重复的行
# duplicates = train_half2[train_half2.duplicated(keep=False)]

# # 删除train_df_half2中与train_df_half1重复的行
# train_df_half2 = train_half2[~train_half2.duplicated(keep=False)]

# # train_df_half1和train_df_half2 detail information
# printFeatureCountAndLabelCountInfo(train_half1, train_df_half2)

# SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{today}", f"train_CICIDS2017_dataframes_{today}")
# SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2017/{today}", f"test_CICIDS2017_dataframes_{today}")
# SaveDataToCsvfile(train_half1, f"./data/dataset_AfterProcessed/{today}", f"train_half1_{today}")
# SaveDataToCsvfile(train_half2,  f"./data/dataset_AfterProcessed/{today}", f"train_half2_{today}") 

# SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{today}", "test_CICIDS2017",today)
# SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2017/{today}", "train_CICIDS2017",today)
# SaveDataframeTonpArray(train_half1, f"./data/dataset_AfterProcessed/{today}", "train_half1", today)
# SaveDataframeTonpArray(train_half2, f"./data/dataset_AfterProcessed/{today}", "train_half2", today)

###########################################################one hot mode################################################################################################
# # one hot mode 分別取出Label等於8、9、13的數據 對半分
# def ifspiltweakLabelbalance_AfterOneHot(test_dataframes,train_dataframes):
#     test_label_8,train_label_8 = spiltweakLabelbalance_afterOnehot('Label_8',mergecompelete_dataset,0.5)
#     test_label_9,train_label_9  = spiltweakLabelbalance_afterOnehot('Label_9',mergecompelete_dataset,0.5)
#     test_label_13,train_label_13   = spiltweakLabelbalance_afterOnehot('Label_13',mergecompelete_dataset,0.5)
#     # 取Label不是於Label_8、Label_9、Label_13的列
#     test_dataframes = test_dataframes[(test_dataframes['Label_8'] != 1) & 
#                                       (test_dataframes['Label_9'] != 1) &
#                                       (test_dataframes['Label_13'] != 1)
#                                       ]
    
#     train_dataframes = train_dataframes[(train_dataframes['Label_8'] != 1) & 
#                                         (train_dataframes['Label_9'] != 1) &
#                                         (train_dataframes['Label_13'] != 1)
#                                         ]
    
#     #存回原本的test_dataframes和train_dataframes
#     test_dataframes = pd.concat([test_dataframes, test_label_8, test_label_9, test_label_13])
#     train_dataframes = pd.concat([train_dataframes,train_label_8, train_label_9,train_label_13])
#     # 保存新的 DataFrame 到文件
#     test_dataframes.to_csv("./data/test_test.csv", index=False)
#     train_dataframes.to_csv("./data/test_train.csv", index=False)

#     return test_dataframes, train_dataframes



# def pintLabelcountAfterOneHot(dfname1,dfname2,test_dataframes,train_dataframes):
#     for i in range(0,15):
#         print(f"{str(dfname1)} Label_{i} count",len(test_dataframes[test_dataframes[f'Label_{i}'] == 1]))
#         print(f"{str(dfname2)} Label_{i} count",len(train_dataframes[train_dataframes[f'Label_{i}'] == 1]))
    
# def spilt_half_train_dataframes_AfterOneHot(train_dataframes):
#     df = pd.DataFrame(train_dataframes)

#     # 初始化兩個 DataFrame 以存儲結果
#     train_half1 = pd.DataFrame()
#     train_half2 = pd.DataFrame()

#     # 分割每個標籤
#     for i in range(0,15):
#         label_name = f'Label_{i}'
#         label_data = df[label_name]
#         label_half1, label_half2 = train_test_split(df[label_data == 1], test_size=0.5, random_state=42)

#         # 將每個標籤的一半添加到對應的 DataFrame
#         train_half1 = pd.concat([train_half1, label_half1], axis=0)
#         train_half2 = pd.concat([train_half2, label_half2], axis=0)

#     # 打印存儲結果
#     # print("train_half1:\n", train_half1)
#     # print("\ntrain_half2:\n", train_half2)
#     train_half1.to_csv("./data/train_half1.csv", index=False)
#     train_half2.to_csv("./data/train_half2.csv", index=False)
    
#     return train_half1, train_half2


# test_dataframes, train_dataframes = ifspiltweakLabelbalance_AfterOneHot(test_dataframes,train_dataframes)
# train_half1, train_half2 = spilt_half_train_dataframes_AfterOneHot(train_dataframes)
# pintLabelcountAfterOneHot("test_dataframes","train_dataframes",test_dataframes,train_dataframes)
# pintLabelcountAfterOneHot("train_half1","train_half2",train_half1,train_half2)
# SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/{today}", f"train_dataframes_{today}")
# SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/{today}", f"test_dataframes_{today}")
# SaveDataToCsvfile(train_half1, f"./data/dataset_AfterProcessed/{today}", f"train_half1_{today}")
# SaveDataToCsvfile(train_half2,  f"./data/dataset_AfterProcessed/{today}", f"train_half2_{today}") 

# SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/{today}", "test",today)
# SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/{today}", "train",today)
# SaveDataframeTonpArray(train_half1, f"./data/dataset_AfterProcessed/{today}", "train_half1", today)
# SaveDataframeTonpArray(train_half2, f"./data/dataset_AfterProcessed/{today}", "train_half2", today)


    