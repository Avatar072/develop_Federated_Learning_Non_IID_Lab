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
from mytoolfunction import spiltweakLabelbalance_afterOnehot,DominmaxforStringTypefeature
from colorama import Fore, Back, Style, init
# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)
#############################################################################  variable  ###################
# filepath = "D:\\Labtest20230911\\data"
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"

today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
generatefolder(filepath + "\\", "dataset_AfterProcessed")
# generatefolder(filepath + "\\dataset_AfterProcessed\\", today)
generatefolder(filepath + "\\dataset_AfterProcessed\\", "CICIDS2019")
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\", today)
#############################################################################  variable  ###################

#############################################################################  funcion宣告與實作  ###########

# 加載CICIDS 2019數據集
def writeData(file_path, bool_Rmove_Benign):
    # 讀取CSV文件並返回DataFrame
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
    # 這邊先取超過10000上限 因CICIDS2019的Label數量都很大
    df = ReplaceMorethanTenthousandQuantity(df)
    return df

### merge多個DataFrame
def mergeData(folder_path, choose_merge_days):
    # 創建要合並的DataFrame列表
    dataframes_to_merge = []

    # 添加每個CSV文件的DataFrame到列表
    # 用01-12這天的DDOS因在CICIDS2019是用於訓練的 03-11是用測試驗證
    if choose_merge_days == "01_12":
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\DrDoS_DNS.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\DrDoS_LDAP.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\DrDoS_MSSQL.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\DrDoS_NetBIOS.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\DrDoS_NTP.csv",False))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\DrDoS_SNMP.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\DrDoS_SSDP.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\DrDoS_UDP.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\Syn.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\TFTP.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\01-12\\UDPLag.csv",True))



    # 檢查特徵名是否一致
    if check_column_names(dataframes_to_merge):
        # 特徵名一致，可以進行合併
        result = pd.concat(dataframes_to_merge)
        # 使用clearDirtyData函數獲取要刪除的行的索引列表
        result = clearDirtyData(result)
        
        # 使用DataFrame的drop方法刪除包含臟數據的行
        #result = result.drop(list_to_drop)
        return result
    else:
        # 特徵名不一致，需要處理這個問題
        print("特徵名不一致，請檢查並處理特徵名一致性")
        return None

### 檢查要合併的多個DataFrame的特徵名是否一致
def check_column_names(dataframes):
    # 獲取第一個DataFrame的特徵名列表
    reference_columns = list(dataframes[0].columns)

    # 檢查每個DataFrame的特徵名是否都與參考特徵名一致
    for df in dataframes[1:]:
        if list(df.columns) != reference_columns:
            return False

    return True


### 檢查CSV文件是否存在，如果不存在，則合並數據並保存到CSV文件中
def ChecktotalCsvFileIsexists(file,choose_merge_days):
    if not os.path.exists(file):
        # 如果文件不存在，执行数据合并    
        data = mergeData(filepath + "\\CICIDS2019_Original\\",choose_merge_days)#完整的資料
        
        # data = clearDirtyData(data)
       
        if data is not None:
            # 去除特徵名中的空白和小於ASCII 32的字符
            data.columns = data.columns.str.replace(r'[\s\x00-\x1F]+', '', regex=True)
            # 保存到CSV文件，同時將header設置為True以包括特徵名行
            data.to_csv(file, index=False, header=True)
            last_column_index = data.shape[1] - 1
            Label_counts = data.iloc[:, last_column_index].value_counts()
            print(Label_counts)
            print(f"共有 {len(Label_counts)} 個不同的標籤")
            print("mergeData complete")
    else:
        print(f"文件 {file} 已存在，不執行合併和保存操作。")

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
    
      # 獲取原始值和編碼值的對照關系字典
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
    # 獲取每個標籤的出現次數
    label_counts = df[' Label'].value_counts()
    # 打印提取後的DataFrame
    print(label_counts)
    # 創建一個空的DataFrame來存儲結果
    extracted_df = pd.DataFrame()

    # 獲取所有不同的標籤
    unique_labels = df[' Label'].unique()

    # 遍歷每個標籤
    for label in unique_labels:
        # 選擇特定標籤的行
        label_df = df[df[' Label'] == label]
    
        # 如果標籤的數量超過1萬，提取前1萬行；否則提取所有行
        # if len(label_df) > 10000:
            # label_df = label_df.head(10000)
            
        # 如果標籤的數量超過1萬，隨機提取1萬行；否則提取所有行
        if len(label_df) > 10000:
            label_df = label_df.sample(n=10000, random_state=42)  # 使用指定的隨機種子(random_state)以保證可重現性
    
        # 將結果添加到提取的DataFrame中
        extracted_df = pd.concat([extracted_df, label_df])

    # 將更新後的DataFrame保存到文件
    # SaveDataToCsvfile(extracted_df, "./data/dataset_AfterProcessed","total_encoded_updated_10000")

    # 打印修改後的結果
    print(extracted_df[' Label'].value_counts())
    return extracted_df


# CheckCsvFileIsexists檢查file存不存在，若file不存在產生新檔
# ChecktotalCsvFileIsexists(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\CICIDS2019_original.csv")
def GetAtferMergeFinishFilepath(choose_merge_days):
    if choose_merge_days == "01_12":
        Csv_AtferMergeFinish_Filepath = ChecktotalCsvFileIsexists(filepath + 
                                                                  "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\CICIDS2019_01_12.csv",
                                                                  choose_merge_days)

    return Csv_AtferMergeFinish_Filepath
# Loading datasets after megre complete
# mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\CICIDS2019_original.csv")

def LoadingDatasetAfterMegreComplete(choose_merge_days):
    if choose_merge_days == "01_12":
        mergecompelete_dataset = pd.read_csv(GetAtferMergeFinishFilepath(choose_merge_days))   
    
    # DoLabelEncoding(mergecompelete_dataset)
    # mergecompelete_dataset = ReplaceMorethanTenthousandQuantity(mergecompelete_dataset)
    mergecompelete_dataset = mergecompelete_dataset.drop('FlowID', axis=1)
    mergecompelete_dataset = mergecompelete_dataset.drop('Unnamed:0', axis=1)
    mergecompelete_dataset = mergecompelete_dataset.drop('SimillarHTTP', axis=1)
    mergecompelete_dataset = mergecompelete_dataset.drop('Inbound', axis=1)

    # 去除所有非數字、字母和下劃線的字符
    mergecompelete_dataset['Label'] = mergecompelete_dataset['Label'].replace({r'[^\w]': ''}, regex=True)

    if(CheckFileExists(filepath + 
                       "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_"+choose_merge_days+"_updated_10000.csv")
                       !=True):
        # dataset.to_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000.csv", index=False)
        mergecompelete_dataset.to_csv(filepath + 
                                      "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_"+choose_merge_days+"_updated_10000.csv",
                                      index=False)
        mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_"+choose_merge_days+"_updated_10000.csv")

    else:
        mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_"+choose_merge_days+"_updated_10000.csv")



    return mergecompelete_dataset


### add TONIOT 和 CICIDS2017 的Label到CICIDS2019
def AddLabelToCICIDS2019(df,add_mergedays_label_or_dataset_label):

    
    if add_mergedays_label_or_dataset_label == "TONIOT":
            values_to_insert = ['backdoor', 'dos', 'injection', 'mitm', 'password', 
                                'ransomware', 'scanning','xss']
    elif add_mergedays_label_or_dataset_label == "CICIDS2017":
            values_to_insert = ['Bot', 'DDoS', 'DoSGoldenEye', 'DoSHulk', 'DoSSlowhttptest', 'DoSslowloris', 
                        'FTPPatator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSHPatator', 
                        'WebAttackBruteForce','WebAttackSqlInjection','WebAttackXSS']
    # 獲取 'Label' 列前的所有列的列名
    columns_before_type = df.columns.tolist()[:df.columns.get_loc('Label')]

    # 將新資料插入 DataFrame
    for value in values_to_insert:
        new_data = {'Label': value} 
        
        # 設置 'Label' 列前的所有列的值為0
        for column in columns_before_type:
            new_data[column] = 0

        # 添加新數據到 DataFrame
        df = df.append(new_data, ignore_index=True)
    
    # df['Label'] = df['Label'].replace({'BENIGN': 'normal'})
    return df

### train dataframe做就好
def DoAddLabel(df,choose_mergedays_or_dataset):
    if choose_mergedays_or_dataset == "01_12":
        #add TONIOT 和 CICIDS2017 的Label
        df = AddLabelToCICIDS2019(df,"TONIOT")
        df = AddLabelToCICIDS2019(df,"CICIDS2017")
        
    return df

def LabelMapping(df):
    # 定義您想要的固定編碼值的字典映射
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

# for 二元分類
def LabelMappingBinary(df):
    # 定義您想要的固定編碼值的字典映射
    encoding_map = {
        'BENIGN': 0,
        'DrDoS_DNS': 1,
        'DrDoS_LDAP': 1,
        'DrDoS_MSSQL': 1,
        'DrDoS_NTP': 1,
        'DrDoS_NetBIOS': 1,
        'DrDoS_SNMP': 1,
        'DrDoS_SSDP': 1,
        'DrDoS_UDP': 1,
        'Syn': 1,
		'TFTP': 1,
        'UDPlag': 1,
        'WebDDoS': 1
    }
    # 將固定編碼值映射應用到DataFrame中的Label列，直接更新原始的Label列
    df['Label'] = df['Label'].map(encoding_map)
    return df, encoding_map
def DoMinMaxALLFeature_OR_excpetStringType(df, bool_excpet_Strtype):
    crop_dataset=df.iloc[:,:-1]
    # 列出要排除的列名，這3個以外得特徵做minmax
    columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
    # 使用條件選擇不等於這些列名的列  
    doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
    # print(doScalerdataset.info)
    # print(afterprocess_dataset.info)
    # print(undoScalerdataset.info)
    # 開始minmax
    if bool_excpet_Strtype:
        # 除string type以外特徵都做minmax
        X=doScalerdataset
        X=X.values
        # scaler = preprocessing.StandardScaler() #資料標準化
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
        scaler.fit(X)
        X=scaler.transform(X)
        # 將縮放後的值更新到 doScalerdataset 中
        doScalerdataset.iloc[:, :] = X
        # 將排除的列名和選中的特徵和 Label 合併為新的 DataFrame
        df = pd.concat([undoScalerdataset,doScalerdataset,df['Label']], axis = 1)
    else:
        # 全特徵都做minmax
        X = crop_dataset
        X=X.values
        # scaler = preprocessing.StandardScaler() #資料標準化
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
        scaler.fit(X)
        X=scaler.transform(X)
        # 將縮放後的值更新到 doScalerdataset 中
        crop_dataset.iloc[:, :] = X
        # 將排除的列名和選中的特徵和 Label 合併為新的 DataFrame
        df = pd.concat([crop_dataset,df['Label']], axis = 1)
    return df

def DoMinMaxAndLabelEncoding(afterprocess_dataset,choose_merge_days,bool_doencode):
    
    ##除了Label外的特徵做encode
    afterprocess_dataset = label_Encoding('SourceIP',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('SourcePort',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationIP',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('DestinationPort',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Protocol',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Timestamp',afterprocess_dataset)
    
    # ### extracting features
    # #除了Label外的特徵
    # crop_dataset=afterprocess_dataset.iloc[:,:-1]
    # # 列出要排除的列名，這6個以外得特徵做minmax
    # columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
    # # 使用條件選擇不等於這些列名的列
    # doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    # undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
    # # print(doScalerdataset.info)
    # # print(afterprocess_dataset.info)
    # # print(undoScalerdataset.info)
    # # 開始minmax
    # X=doScalerdataset
    # X=X.values
    # # scaler = preprocessing.StandardScaler() #資料標準化
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    # scaler.fit(X)
    # X=scaler.transform(X)
    # # 將縮放後的值更新到 doScalerdataset 中
    # doScalerdataset.iloc[:, :] = X
    # # 將排除的列名和選中的特徵和 Label 合併為新的 DataFrame
    # afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,afterprocess_dataset['Label']], axis = 1)
    # print("test")

    # True為除string type以外特徵都做minmax
    # False為全特徵都做minmax
    afterminmax_dataset = DoMinMaxALLFeature_OR_excpetStringType(afterprocess_dataset,False)
    # 保存Lable未做label_encoding的DataFrame方便後續Noniid實驗
    if bool_doencode != True:
        # afterminmax_dataset.to_csv(filepath + 
                                #    "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv", index=False)
        if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv")
                           !=True):
            afterminmax_dataset = DoAddLabel(afterminmax_dataset,choose_merge_days)

            afterminmax_dataset.to_csv(filepath +
                                       "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_UndoLabelencode_"+choose_merge_days+".csv")

        # encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)
        # 固定Label encode值方便後續Noniid實驗
        afterminmax_dataset,encoded_type_values = LabelMapping(afterminmax_dataset)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/encode_and_count_Noniid.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")
    #保存Lable做label_encoding的DataFrame方便後續BaseLine實驗
    else:
        encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)
        # afterminmax_dataset.to_csv(filepath + 
                                #    "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv", index=False)
        
        
        if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")
                           !=True):
            afterminmax_dataset.to_csv(filepath +
                                       "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv")
        
        # print("Original Type Values:", original_type_values)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/encode_and_count_baseLine.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")

    return afterminmax_dataset

def DoMinMaxAndBinaryLabelEncoding(afterprocess_dataset,choose_merge_days):
    
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
    # 使用條件選擇不等於這些列名的列
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
    # 將縮放後的值更新到 doScalerdataset 中
    doScalerdataset.iloc[:, :] = X
    # 將排除的列名和選中的特徵和 Label 合並為新的 DataFrame
    afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,afterprocess_dataset['Label']], axis = 1)
    print("test")

    #二元分類Label encode
    afterminmax_dataset,encoded_type_values = LabelMappingBinary(afterminmax_dataset)
        
    if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoBinaryLabelencode_"+choose_merge_days+".csv")
                           !=True):
        afterminmax_dataset.to_csv(filepath +
                                    "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoBinaryLabelencode_"+choose_merge_days+".csv", index=False)
                
        afterminmax_dataset = pd.read_csv(filepath +
                                            "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoBinaryLabelencode_"+choose_merge_days+".csv")

    else:
        afterminmax_dataset = pd.read_csv(filepath +
                                            "\\dataset_AfterProcessed\\CICIDS2019\\"+choose_merge_days+"\\CICIDS2019_AfterProcessed_DoBinaryLabelencode_"+choose_merge_days+".csv")
        
        # print("Original Type Values:", original_type_values)
    print("Encoded Type Values:", encoded_type_values)
    with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/Binaryencode_and_count_baseLine.csv", "a+") as file:
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

# Base劃分
def DoBaselinesplit(df,train_dataframes,test_dataframes):
    # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
    List_train_Label = []
    List_test_Label = []
    for i in range(13):
        if i == 12:
            continue
        train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
        List_train_Label.append(train_label_split)
        List_test_Label.append(test_label_split)         

    train_dataframes = pd.concat(List_train_Label)
    test_dataframes = pd.concat(List_test_Label)
    # encode後對照如下
    # WebDDoS:12
    # Label encode mode  分別取出Label等於12的數據 對6633分
    train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(12,df,0.33)
    # # 刪除Label相當於12的行
    test_dataframes = test_dataframes[~test_dataframes['Label'].isin([12])]
    train_dataframes = train_dataframes[~train_dataframes['Label'].isin([12])]
    # 合併Label12回去
    test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
    train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])
    return train_dataframes,test_dataframes

# do Labelencode and minmax 
def DoSpiltAllfeatureAfterMinMax(df,choose_merge_days,bool_Noniid):  
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes,"Label")
    
    
    if bool_Noniid !=True:
        if choose_merge_days =="01_12":
            # Noniid時
            # encode後對照如下
            # WebDDoS:34
            train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(34,df,0.33)
            # # 刪除Label相當於34的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([34])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([34])]
            # 合併Label 34回去
            test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])   

            # 篩選test_dataframes中標籤為2,14,20,22的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([2,14,20,22])]
            # test刪除Label相當於2,14,20,22的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([2,14,20,22])]
            # # 合併Label2,14,20,22回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
    else:
        # BaseLine時
        if choose_merge_days =="01_12":
            # # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
            # List_train_Label = []
            # List_test_Label = []
            # for i in range(13):
            #     if i == 12:
            #         continue
            #     train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
            #     List_train_Label.append(train_label_split)
            #     List_test_Label.append(test_label_split)         
            
            # train_dataframes = pd.concat(List_train_Label)
            # test_dataframes = pd.concat(List_test_Label)
            # # encode後對照如下
            # # WebDDoS:12
            # # Label encode mode  分別取出Label等於12的數據 對6633分
            # train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(12,df,0.33)
            # # # 刪除Label相當於12的行
            # test_dataframes = test_dataframes[~test_dataframes['Label'].isin([12])]
            # train_dataframes = train_dataframes[~train_dataframes['Label'].isin([12])]
            # # 合併Label12回去
            # test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            # train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])
            train_dataframes, test_dataframes= DoBaselinesplit(df,train_dataframes,test_dataframes)            
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/encode_and_count_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}", f"{choose_merge_days}_train_dataframes_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}", f"{choose_merge_days}_test_dataframes_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}", f"{choose_merge_days}_test",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}", f"{choose_merge_days}_train",today)

# do Binary and minmax 
def DoBinarySpiltAllfeatureAfterMinMax(df,choose_merge_days,bool_Noniid):  
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.42, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%    

    if bool_Noniid:
        # BaseLine時
        if choose_merge_days =="01_12":
            # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
            List_train_Label = []
            List_test_Label = []
            for i in range(2):
                train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
                List_train_Label.append(train_label_split)
                List_test_Label.append(test_label_split)         
            
            train_dataframes = pd.concat(List_train_Label)
            test_dataframes = pd.concat(List_test_Label)
         
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/encode_and_count_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/Bainary/{today}", f"{choose_merge_days}_train_dataframes_Bainary_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/Bainary/{today}", f"{choose_merge_days}_test_dataframes_Bainary_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/Bainary/{today}", f"{choose_merge_days}_test_Bainary",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/Bainary/{today}", f"{choose_merge_days}_train_Bainary",today)

def dofeatureSelect(df, slecet_label_counts,choose_merge_days):
    significance_level=0.05
    if (slecet_label_counts == None):
        slecet_label_counts ='all'

    # 開始ch2特徵選擇，先分離特徵和目標變量
    y = df['Label']  # 目标变量
    X = df.iloc[:, :-1]  # 特徵

    # 創建 SelectKBest 模型，選擇 f_classif 統計測試方法
    k_best = SelectKBest(score_func=chi2, k=slecet_label_counts)
    X_new = k_best.fit_transform(X, y)

    # 獲取被選中的特徵的索引
    selected_feature_indices = k_best.get_support(indices=True)

    # 打印被選中的特徵的列名
    selected_features = X.columns[selected_feature_indices]
    print("Selected Features:")
    print(selected_features)

    # 印選擇的特徵的名稱、索引和相應的 F 值、p 值
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
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_selected_feature_stats_{today}")

    SaveDataToCsvfile(all_feature_stats_df, 
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_all_feature_stats_{today}")

    # 將未被選中特徵的統計信息存儲到 CSV 文件
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
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_unselected_feature_stats_{today}")
    

    # 將 X_new 轉換為 DataFrame
    X_new_df = pd.DataFrame(X_new, columns=selected_features)

    # 將選中的特徵和 Label 合並為新的 DataFrame
    selected_data = pd.concat([X_new_df, df['Label']], axis=1)
    
    # SaveDataToCsvfile(selected_data, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
    #                   f"{choose_merge_days}_AfterSelected_{slecet_label_counts}_feature_data_{today}")
    return selected_data

# do chi-square and Labelencode and minmax 
def DoSpiltAfterFeatureSelect(df,slecet_label_counts,choose_merge_days,bool_Noniid):
    df = dofeatureSelect(df,slecet_label_counts,choose_merge_days)
    # 自動切
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # 手動劃分資料集!!!!!!!! 注意用手動切資料集 CICIDS2019訓練結果會他媽的超級差 媽的 爛function
    # train_dataframes, test_dataframes = DoSpiltdatasetAutoOrManual(df, False,choose_merge_days)    
    
    #加toniot的情況
    if bool_Noniid !=True:
            # Noniid時
            # encode後對照如下
            # WebDDoS:34
            train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(34,df,0.33)
            # # 刪除Label相當於34的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([34])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([34])]
            # 合併Label 34回去
            test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])   

            # 篩選test_dataframes中標籤為2,14,20,22的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([2,14,20,22])]
            # test刪除Label相當於2,14,20,22的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([2,14,20,22])]
            # # 合併Label2,14,20,22回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
    else:
        # BaseLine時
        if choose_merge_days =="01_12":
            # encode後對照如下
            # WebDDoS:12
            train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(12,df,0.33)
            # # 刪除Label相當於12的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([12])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([12])]
            # 合併Label12回去
            test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])

    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/encode_and_count_after_chisquare_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")


    SaveDataToCsvfile(train_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}",  
                      f"{choose_merge_days}_train_dataframes_AfterFeatureSelect")
    SaveDataToCsvfile(test_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_test_dataframes_AfterFeatureSelect")
    SaveDataframeTonpArray(test_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                           f"{choose_merge_days}_test_CICIDS2019_AfterFeatureSelect{slecet_label_counts}",today)
    SaveDataframeTonpArray(train_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                           f"{choose_merge_days}_train_CICIDS2019_AfterFeatureSelect{slecet_label_counts}",today)



# do PCA to all feature or excpetStringType
def DoPCA_ALLFeature_OR_excpetStringType(df,number_of_components ,bool_excpet_Strtype):

    # number_of_components=20
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    crop_dataset=df.iloc[:,:-1]
    # 列出要排除的列名
    columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
    # 使用條件選擇不等於這些列名的列
    # number_of_components=77 # 原84個的特徵，扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label' | 84-7 =77
    doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
    # afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,mergecompelete_dataset['Label']], axis = 1)

    print("Original number of features:", len(df.columns) - 1)  # 减去 'Label' 列
    # X = df.drop(columns=['Label'])  # 提取特徵，去除 'Label' 列
    if bool_excpet_Strtype:
        str_filename = "excpet_Strtype" 
        # 除string type以外特徵都做minmax
        X = doScalerdataset
        pca = PCA(n_components=number_of_components)
        columns_array=[]
        for i in range (number_of_components):
            columns_array.append("principal_Component"+str(i+1))
            
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = columns_array)

        finalDf = pd.concat([undoScalerdataset,principalDf, df[['Label']]], axis = 1)
    else:
        str_filename = "ALLMINMAX" 
        # 全特徵都做minmax
        X = crop_dataset
        pca = PCA(n_components=number_of_components)
        columns_array=[]
        for i in range (number_of_components):
            columns_array.append("principal_Component"+str(i+1))
            
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = columns_array)

        finalDf = pd.concat([principalDf, df[['Label']]], axis = 1)
        
    df=finalDf
    SaveDataToCsvfile(df, 
                      f"./data/dataset_AfterProcessed/CICIDS2019/01_12/{today}/doPCA/{number_of_components}", 
                      f"CICIDS2019_AfterProcessed_{str_filename}_minmax_PCA")
    return df


# do PCA and Labelencode and minmax 
def DoSpiltAfterDoPCA(df,number_of_components,choose_merge_days,bool_Noniid):
    # # number_of_components=20
    
    # crop_dataset=df.iloc[:,:-1]
    # # 列出要排除的列名
    # columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
    # # 使用條件選擇不等於這些列名的列
    # # number_of_components=77 # 原84個的特徵，扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label' | 84-7 =77
    # doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    # undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
    # # afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,mergecompelete_dataset['Label']], axis = 1)

    # print("Original number of features:", len(df.columns) - 1)  # 减去 'Label' 列
    # # X = df.drop(columns=['Label'])  # 提取特徵，去除 'Label' 列
    # X = doScalerdataset
    # pca = PCA(n_components=number_of_components)
    # columns_array=[]
    # for i in range (number_of_components):
    #     columns_array.append("principal_Component"+str(i+1))
        
    # principalComponents = pca.fit_transform(X)
    # principalDf = pd.DataFrame(data = principalComponents
    #             , columns = columns_array)

    # finalDf = pd.concat([undoScalerdataset,principalDf, df[['Label']]], axis = 1)
    # df=finalDf

    # SaveDataToCsvfile(df, 
    #                   f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
    #                   f"{choose_merge_days}_CICIDS2019_AfterProcessed_minmax_PCA")

    # True為除string type以外特徵都做PCA
    # False為全特徵都做PCA
    df = DoPCA_ALLFeature_OR_excpetStringType(df,number_of_components ,False)
    # df = DoPCA_ALLFeature_OR_excpetStringType(df,number_of_components ,True)
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # 手動劃分資料集
    # train_dataframes, test_dataframes = DoSpiltdatasetAutoOrManual(df, False,choose_merge_days)
    # printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes,"Label")
    if bool_Noniid !=True:
        if choose_merge_days =="01_12":
            # Noniid時
            # encode後對照如下
            # WebDDoS:34
            train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(34,df,0.33)
            # # 刪除Label相當於34的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([34])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([34])]
            # 合併Label 34回去
            test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])   

            # 篩選test_dataframes中標籤為2,14,20,22的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([2,14,20,22])]
            # test刪除Label相當於2,14,20,22的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([2,14,20,22])]
            # # 合併Label2,14,20,22回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
    else:
        # BaseLine時
        if choose_merge_days =="01_12":
            # # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
            # List_train_Label = []
            # List_test_Label = []
            # for i in range(13):
            #     if i == 12:
            #         continue
            #     train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
            #     if train_label_split is None or test_label_split is None:
            #         print(Fore.RED+Style.BRIGHT+f"No data available for label {i}. Skipping this label.")
            #     List_train_Label.append(train_label_split)
            #     List_test_Label.append(test_label_split)         
            
            # train_dataframes = pd.concat(List_train_Label)
            # test_dataframes = pd.concat(List_test_Label)
            # # encode後對照如下
            # # WebDDoS:12
            # # Label encode mode  分別取出Label等於12的數據 對6633分
            # train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(12,df,0.33)
            # # # 刪除Label相當於12的行
            # test_dataframes = test_dataframes[~test_dataframes['Label'].isin([12])]
            # train_dataframes = train_dataframes[~train_dataframes['Label'].isin([12])]
            # # 合併Label12回去
            # test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            # train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])
            train_dataframes, test_dataframes= DoBaselinesplit(df,train_dataframes,test_dataframes)
    
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/encode_and_count_after_PCA_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                      f"{choose_merge_days}_train_dataframes_AfterPCA{number_of_components}_{today}")
    SaveDataToCsvfile(test_dataframes,
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                      f"{choose_merge_days}_test_dataframes_AfterPCA{number_of_components}_{today}")
    SaveDataframeTonpArray(test_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                           f"{choose_merge_days}_test_AfterPCA{number_of_components}",today)
    SaveDataframeTonpArray(train_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                           f"{choose_merge_days}_train_AfterPCA{number_of_components}",today)

# do split train to half for iid and Labelencode and minmax 
def DoSpilthalfForiid(choose_merge_days):
    if choose_merge_days == "01_12":
        df_ALLtrain = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\20240502\\01_12_train_dataframes_20240502.csv")
                    # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_half1_Label = []
        List_train_half2_Label = []
        for i in range(13):
            train_half1_label_split, train_half2_label_split = spiltweakLabelbalance(i,df_ALLtrain,0.5)
            List_train_half1_Label.append(train_half1_label_split)
            List_train_half2_Label.append(train_half2_label_split)         
            
        df_train_half1 = pd.concat(List_train_half1_Label)
        df_train_half2 = pd.concat(List_train_half2_Label)
            

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/encode_and_count_iid.csv", "a+") as file:
            label_counts = df_train_half1['Label'].value_counts()
            print("df_train_half1\n", label_counts)
            file.write("df_train_half1_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = df_train_half2['Label'].value_counts()
            print("df_train_half2\n", label_counts)
            file.write("df_train_half2_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(df_train_half1, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half1_{today}")
        SaveDataToCsvfile(df_train_half2,  f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half2_{today}")
        SaveDataframeTonpArray(df_train_half1, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half1",today)
        SaveDataframeTonpArray(df_train_half2, f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half2",today)

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
        #  #PCA選79個特徵 總80特徵=79+'Label'
        DoSpiltAfterDoPCA(df,79,choose_merge_days,bool_Noniid)
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
        # DoSpiltAfterDoPCA(df,38,choose_merge_days,bool_Noniid)
        # #PCA選33個特徵 總40特徵=33+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSpiltAfterDoPCA(df,33,choose_merge_days,bool_Noniid) 

# do Pearson_Correlation_Coefficient feautre select 
def dofeatureSelect_Pearson_Correlation_Coefficient(df,choose_merge_days):
    # 設定相關性閾值
    # 設定一個相關性閾值為 0.9。這意味著如果兩個特徵之間的相關係數超過 0.9，
    # 就會認為它們具有高度相關性，其中一個特徵將被移除。
    threshold = 0.9

    # 計算相關性矩陣
    # 使用 pandas 的 .corr() 方法計算 DataFrame 的相關性矩陣。
    # 這會生成一個對稱矩陣，描述每個特徵之間的相關性。
    correlation_matrix = df.corr().abs()

    # 找到上三角矩陣來避免重複
    # 使用 np.triu() 函數提取上三角矩陣，避免重複計算相同的相關性值（因為矩陣是對稱的）。
    # 這樣可以只考慮每對特徵中的一個相關性值。
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # 找到相關性超過閾值的特徵
    # 將上三角矩陣中，相關性超過 0.9 的特徵收集到 features_to_drop 列表中。
    # 這些特徵將會被移除，因為它們與其他特徵高度相關，對模型可能帶來多重共線性問題。
    features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # 移除高度相關的特徵 
    #  DataFrame.drop() 移除高度相關的特徵，產生維度較低的新 DataFrame df_reduced。
    df_reduced = df.drop(columns=features_to_drop)

    # 將選中的特徵和 Label 合併為新的 DataFrame
    selected_data = pd.concat([df_reduced, df['Label']], axis=1)

    print(Fore.GREEN+Style.BRIGHT+f"原始特徵數量: {df.shape[1]}")
    # 查看篩選後的特徵
    print(df_reduced.head())
    print(Fore.GREEN+Style.BRIGHT+f"移除冗餘特徵後的特徵數量: {df_reduced.shape[1]}")

    SaveDataToCsvfile(selected_data, 
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/Pearson", 
                      f"{choose_merge_days}_selected_feature_stats_{today}")


    return selected_data,int(df_reduced.shape[1])

# do Pearson and Labelencode and minmax 
def DoSpiltAfterPearsonFeatureSelect(df,choose_merge_days,bool_Noniid):
    df,slecet_label_counts = dofeatureSelect_Pearson_Correlation_Coefficient(df,choose_merge_days)
    # 自動切
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.25, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # 手動劃分資料集!!!!!!!! 注意用手動切資料集 CICIDS2019訓練結果會他媽的超級差 媽的 爛function
    # train_dataframes, test_dataframes = DoSpiltdatasetAutoOrManual(df, False,choose_merge_days)    
    
    #加toniot的情況
    if bool_Noniid !=True:
            # Noniid時
            # encode後對照如下
            # WebDDoS:34
            train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(34,df,0.33)
            # # 刪除Label相當於34的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([34])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([34])]
            # 合併Label 34回去
            test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])   

            # 篩選test_dataframes中標籤為2,14,20,22的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['Label'].isin([2,14,20,22])]
            # test刪除Label相當於2,14,20,22的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([2,14,20,22])]
            # # 合併Label2,14,20,22回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
    else:
        # BaseLine時
        if choose_merge_days =="01_12":
            # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
            List_train_Label = []
            List_test_Label = []
            for i in range(13):
                if i == 12:
                    continue
                train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
                if train_label_split is None or test_label_split is None:
                    print(Fore.RED+Style.BRIGHT+f"No data available for label {i}. Skipping this label.")
                List_train_Label.append(train_label_split)
                List_test_Label.append(test_label_split)         
            
            train_dataframes = pd.concat(List_train_Label)
            test_dataframes = pd.concat(List_test_Label)
            # encode後對照如下
            # WebDDoS:12
            # Label encode mode  分別取出Label等於12的數據 對6633分
            train_label_WebDDoS, test_label_WebDDoS = spiltweakLabelbalance(12,df,0.33)
            # # 刪除Label相當於12的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([12])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([12])]
            # 合併Label12回去
            test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])            

    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/encode_and_count_after_Pearson_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")


    SaveDataToCsvfile(train_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/Pearson/{slecet_label_counts}",  
                      f"{choose_merge_days}_train_dataframes_AfterPearsonFeatureSelect")
    SaveDataToCsvfile(test_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/Pearson/{slecet_label_counts}", 
                      f"{choose_merge_days}_test_dataframes_AfterPearsonFeatureSelect")
    SaveDataframeTonpArray(test_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/Pearson/{slecet_label_counts}", 
                           f"{choose_merge_days}_test_CICIDS2019_AfterPearsonFeatureSelect{slecet_label_counts}",today)
    SaveDataframeTonpArray(train_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2019/{choose_merge_days}/{today}/doFeatureSelect/Pearson/{slecet_label_counts}", 
                           f"{choose_merge_days}_train_CICIDS2019_AfterPearsonFeatureSelect{slecet_label_counts}",today)

# 針對string type 做minmax
def RedoCICIDS2019stringtypeMinMaxfortrainORtest(afterprocess_dataset,bool_tain_OR_test):
    #除了Label外的特徵
    crop_dataset=afterprocess_dataset.iloc[:,:-1]
    columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Timestamp', 'Protocol']
    # columns_to_exclude = ['ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto']
    testdata_removestring = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col in columns_to_exclude]]
    doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    # 補string type 做minmax
    undoScalerdataset = DominmaxforStringTypefeature(undoScalerdataset)
    
    # 將排除的列名和選中的特徵和 Label 合並為新的 DataFrame
    afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,afterprocess_dataset['Label']], axis = 1)

    if bool_tain_OR_test:
        afterminmax_dataset.to_csv(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/CICIDS2019_AfterProcessed_DoLabelencode_ALLMinmax_train.csv", index=False)
        SaveDataframeTonpArray(afterminmax_dataset, f"./data/dataset_AfterProcessed/CICIDS2019/01_12/{today}", f"01_12_train_dataframes_ALLMinmax", today)
    else:
        afterminmax_dataset.to_csv(f"./data/dataset_AfterProcessed/CICIDS2019/01_12/CICIDS2019_AfterProcessed_DoLabelencode_ALLMinmax_test.csv", index=False)
        SaveDataframeTonpArray(afterminmax_dataset, f"./data/dataset_AfterProcessed/CICIDS2019/01_12/{today}", f"01_12_test_dataframes_ALLMinmax", today)

    return afterprocess_dataset

    # 對已劃分好的tain和test的Strig type做完label ecnode後補做minmax
    # afterprocess_dataset_train = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\20240502\\01_12_train_dataframes_20240502.csv")
    # # 加载CICIDS2019 test after do labelencode and minmax  75 25分
    # afterprocess_dataset_test = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\20240502\\01_12_test_dataframes_20240502.csv")
    # print("Dataset loaded.")

    # afterprocess_dataset_train = RedoMinMaxfortrainORtest(afterprocess_dataset_train,True)
    # afterprocess_dataset_test = RedoMinMaxfortrainORtest(afterprocess_dataset_test,False)

def forBaseLineUseData(choose_merge_days,bool_Noniid):
    if choose_merge_days == "01_12":
        # 載入資料集
        df_01_12=LoadingDatasetAfterMegreComplete(choose_merge_days)
        # 預處理和正規化
        # True for BaseLine
        # False for Noniid
        df_01_12=DoMinMaxAndLabelEncoding(df_01_12,choose_merge_days,bool_Noniid)
        # for iid 實驗將ALL train分一半
        # DoSpilthalfForiid(choose_merge_days)
        # 一般全部特徵
        # DoSpiltAllfeatureAfterMinMax(df_01_12,choose_merge_days,bool_Noniid)
        # 做ChiSquare
        # SelectfeatureUseChiSquareOrPCA(df_01_12,choose_merge_days,True,False,bool_Noniid)
        # 做PCA
        SelectfeatureUseChiSquareOrPCA(df_01_12,choose_merge_days,False,True,bool_Noniid)
        # 依據皮爾森相關係數進行特徵篩選
        # df_01_12 = DoSpiltAfterPearsonFeatureSelect(df_01_12,choose_merge_days,bool_Noniid)


def forBaseLineUseBinaryData(choose_merge_days,bool_Noniid):

    if choose_merge_days == "01_12":
        # 載入資料集
        df_01_12=LoadingDatasetAfterMegreComplete(choose_merge_days)
        # 預處理和正規化
        # True for BaseLine
        # False for Noniid
        # 二元分類
        # df_01_12=DoMinMaxAndBinaryLabelEncoding(df_01_12,choose_merge_days)
# True for BaseLine
# False for Noniid
# forBaseLineUseData("01_12",False)
forBaseLineUseData("01_12",True)
# forBaseLineUseBinaryData("01_12",True)

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

# SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{today}", f"train_CICIDS2019_dataframes_{today}")
# SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2019/{today}", f"test_CICIDS2019_dataframes_{today}")
# SaveDataToCsvfile(train_half1, f"./data/dataset_AfterProcessed/{today}", f"train_half1_{today}")
# SaveDataToCsvfile(train_half2,  f"./data/dataset_AfterProcessed/{today}", f"train_half2_{today}") 

# SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{today}", "test_CICIDS2019",today)
# SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2019/{today}", "train_CICIDS2019",today)
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


    