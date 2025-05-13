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
from mytoolfunction import SaveDataToCsvfile,CheckFileExists
from mytoolfunction import clearDirtyData,label_Encoding,splitdatasetbalancehalf,splitweakLabelbalance,SaveDataframeTonpArray,generatefolder
from mytoolfunction import splitweakLabelbalance_afterOnehot,DominmaxforStringTypefeature
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
generatefolder(filepath + "\\dataset_AfterProcessed\\", "CICIDS2018")
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\", today)
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
    # 這邊先取超過10000上限 因CICIDS2018的Label數量都很大
    df = ReplaceMorethanTenthousandQuantity(df)
    return df

### merge多個DataFrame
def mergeData(folder_path, choose_merge_days):
    # 創建要合並的DataFrame列表
    dataframes_to_merge = []

    # 添加每個CSV文件的DataFrame到列表
    # 用Data這天的DDOS因在CICIDS2018是用於訓練的 03-11是用測試驗證
    if choose_merge_days == "csv_data":
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",False))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",True))
        dataframes_to_merge.append(writeData(folder_path + "\\csv_data\\Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",True))



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
        data = mergeData(filepath + "\\CICIDS2018_Original\\",choose_merge_days)#完整的資料
        
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
# ChecktotalCsvFileIsexists(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\CICIDS2018_original.csv")
def GetAtferMergeFinishFilepath(choose_merge_days):
    if choose_merge_days == "csv_data":
        Csv_AtferMergeFinish_Filepath = ChecktotalCsvFileIsexists(filepath + 
                                                                  "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\merged_data.csv",
                                                                  choose_merge_days)

    return Csv_AtferMergeFinish_Filepath

def LoadingDatasetAfterMegreComplete(choose_merge_days):
    if choose_merge_days == "csv_data":
        mergecompelete_dataset = pd.read_csv(GetAtferMergeFinishFilepath(choose_merge_days))   
    
    # mergecompelete_dataset = ReplaceMorethanTenthousandQuantity(mergecompelete_dataset)
    mergecompelete_dataset = mergecompelete_dataset.drop('FlowID', axis=1)
    mergecompelete_dataset = mergecompelete_dataset.drop('Unnamed:0', axis=1)
    mergecompelete_dataset = mergecompelete_dataset.drop('SimillarHTTP', axis=1)
    mergecompelete_dataset = mergecompelete_dataset.drop('Inbound', axis=1)

    # 去除所有非數字、字母和下劃線的字符
    mergecompelete_dataset['Label'] = mergecompelete_dataset['Label'].replace({r'[^\w]': ''}, regex=True)

    if(CheckFileExists(filepath + 
                       f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\sampled_data_max_10000_per_label.csv")
                       !=True):
        mergecompelete_dataset.to_csv(filepath + 
                                      f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\sampled_data_max_10000_per_label.csv",
                                      index=False)
        mergecompelete_dataset = pd.read_csv(filepath + f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\sampled_data_max_10000_per_label.csv")

    else:
        mergecompelete_dataset = pd.read_csv(filepath + f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\sampled_data_max_10000_per_label.csv")



    return mergecompelete_dataset


### add TONIOT 和 CICIDS2017 的Label到CICIDS2018
def AddLabelToCICIDS2018(df,add_mergedays_label_or_dataset_label):

    
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
    if choose_mergedays_or_dataset == "csv_data":
        #add TONIOT 和 CICIDS2017 的Label
        df = AddLabelToCICIDS2018(df,"TONIOT")
        df = AddLabelToCICIDS2018(df,"CICIDS2017")
        
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

def DoMinMaxALLFeature_OR_excpetStringType(df, bool_excpet_Strtype):
    crop_dataset=df.iloc[:,:-1]
    # 列出要排除的列名，這3個以外得特徵做minmax
    columns_to_exclude = ['Dst Port', 'Protocol', 'Timestamp']
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
    # afterprocess_dataset = label_Encoding('SourceIP',afterprocess_dataset)
    # afterprocess_dataset = label_Encoding('SourcePort',afterprocess_dataset)
    # afterprocess_dataset = label_Encoding('DestinationIP',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Dst Port',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Protocol',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('Timestamp',afterprocess_dataset)
    
    # True為除string type以外特徵都做minmax
    # False為全特徵都做minmax
    afterprocess_dataset = DoMinMaxALLFeature_OR_excpetStringType(afterprocess_dataset,False)
    print("test")
    # 保存Lable未做label_encoding的DataFrame方便後續Noniid實驗
    if bool_doencode != True:
        if(CheckFileExists(filepath + 
                           f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\CICIDS2018_AfterProcessed_UndoLabelencode.csv")
                           !=True):
            afterminmax_dataset = DoAddLabel(afterminmax_dataset,choose_merge_days)

            afterminmax_dataset.to_csv(filepath +
                                       f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\CICIDS2018_AfterProcessed_UndoLabelencode.csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\CICIDS2018_AfterProcessed_UndoLabelencode.csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\CICIDS2018_AfterProcessed_UndoLabelencode.csv")

        # encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)
        # 固定Label encode值方便後續Noniid實驗
        afterminmax_dataset,encoded_type_values = LabelMapping(afterminmax_dataset)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/encode_and_count_Noniid.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")
    #保存Lable做label_encoding的DataFrame方便後續BaseLine實驗
    else:
        encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)

        if(CheckFileExists(filepath + 
                           f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\CICIDS2018_AfterProcessed_DoLabelencode.csv")
                           !=True):
            afterminmax_dataset.to_csv(filepath +
                                       f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\CICIDS2018_AfterProcessed_DoLabelencode.csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\CICIDS2018_AfterProcessed_DoLabelencode.csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              f"\\dataset_AfterProcessed\\CICIDS2018\\{choose_merge_days}\\CICIDS2018_AfterProcessed_DoLabelencode.csv")
        
        # print("Original Type Values:", original_type_values)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/encode_and_count_baseLine.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")

    return afterminmax_dataset

def DoBaselinesplit(df,train_dataframes,test_dataframes):
    List_train_Label = []
    List_test_Label = []
    for i in range(15):
        if i == 13:
            continue
        train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
        List_train_Label.append(train_label_split)
        List_test_Label.append(test_label_split)         

    train_dataframes = pd.concat(List_train_Label)
    test_dataframes = pd.concat(List_test_Label)
    # encode後對照如下
    # SQL_Injection :15
    # Label encode mode  分別取出Label等於12的數據 對6633分
    train_label_SQL_Injection, test_label_SQL_Injection = splitweakLabelbalance(13,df,0.33)
    # # 刪除Label相當於15的行
    test_dataframes = test_dataframes[~test_dataframes['Label'].isin([13])]
    train_dataframes = train_dataframes[~train_dataframes['Label'].isin([13])]
    # 合併Label15回去
    test_dataframes = pd.concat([test_dataframes, test_label_SQL_Injection])
    train_dataframes = pd.concat([train_dataframes,train_label_SQL_Injection])
    return train_dataframes,test_dataframes

def DoBaselinesplitAfter_LabelMerge(df,train_dataframes,test_dataframes):
    List_train_Label = []
    List_test_Label = []
    for i in range(8):
        train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
        List_train_Label.append(train_label_split)
        List_test_Label.append(test_label_split)         

    train_dataframes = pd.concat(List_train_Label)
    test_dataframes = pd.concat(List_test_Label)
    return train_dataframes,test_dataframes
# do Labelencode and minmax 
def DoSplitAllfeatureAfterMinMax(df,choose_merge_days,bool_Noniid):  
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes,"Label")
    
    
    if bool_Noniid !=True:
        if choose_merge_days =="csv_data":
            print("Do Nothing")
    else:
        # BaseLine時
        if choose_merge_days =="csv_data":
            train_dataframes, test_dataframes = DoBaselinesplit(df,train_dataframes,test_dataframes)
        #     # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        #     List_train_Label = []
        #     List_test_Label = []
        #     for i in range(15):
        #         if i == 13:
        #             continue
        #         train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
        #         List_train_Label.append(train_label_split)
        #         List_test_Label.append(test_label_split)         
            
        #     train_dataframes = pd.concat(List_train_Label)
        #     test_dataframes = pd.concat(List_test_Label)
        #     # encode後對照如下
        #     # SQL_Injection :13
        #     # Label encode mode  分別取出Label等於12的數據 對6633分
        #     train_label_SQL_Injection, test_label_SQL_Injection = splitweakLabelbalance(13,df,0.33)
        #     # # 刪除Label相當於13的行
        #     test_dataframes = test_dataframes[~test_dataframes['Label'].isin([13])]
        #     train_dataframes = train_dataframes[~train_dataframes['Label'].isin([13])]
        #     # 合併Label12回去
        #     test_dataframes = pd.concat([test_dataframes, test_label_SQL_Injection])
        #     train_dataframes = pd.concat([train_dataframes,train_label_SQL_Injection])            
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/encode_and_count_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}", f"{choose_merge_days}_train_dataframes_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}", f"{choose_merge_days}_test_dataframes_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}", f"{choose_merge_days}_test",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}", f"{choose_merge_days}_train",today)

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
                      f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_selected_feature_stats_{today}")

    SaveDataToCsvfile(all_feature_stats_df, 
                      f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
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
                      f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_unselected_feature_stats_{today}")
    

    # 將 X_new 轉換為 DataFrame
    X_new_df = pd.DataFrame(X_new, columns=selected_features)

    # 將選中的特徵和 Label 合並為新的 DataFrame
    selected_data = pd.concat([X_new_df, df['Label']], axis=1)
    
    # SaveDataToCsvfile(selected_data, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
    #                   f"{choose_merge_days}_AfterSelected_{slecet_label_counts}_feature_data_{today}")
    return selected_data

# do chi-square and Labelencode and minmax 
def DoSplitAfterFeatureSelect(df,slecet_label_counts,choose_merge_days,bool_Noniid):
    df = dofeatureSelect(df,slecet_label_counts,choose_merge_days)
    # 自動切
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%    
    #加toniot的情況
    if bool_Noniid !=True:
            # Noniid時
            # encode後對照如下
            # WebDDoS:34
            train_label_WebDDoS, test_label_WebDDoS = splitweakLabelbalance(34,df,0.33)
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
        if choose_merge_days =="csv_data":
            # encode後對照如下
            # WebDDoS:12
            train_label_WebDDoS, test_label_WebDDoS = splitweakLabelbalance(12,df,0.33)
            # # 刪除Label相當於12的行
            test_dataframes = test_dataframes[~test_dataframes['Label'].isin([12])]
            train_dataframes = train_dataframes[~train_dataframes['Label'].isin([12])]
            # 合併Label12回去
            test_dataframes = pd.concat([test_dataframes, test_label_WebDDoS])
            train_dataframes = pd.concat([train_dataframes,train_label_WebDDoS])

    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/encode_and_count_after_chisquare_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")


    SaveDataToCsvfile(train_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}",  
                      f"{choose_merge_days}_train_dataframes_AfterFeatureSelect")
    SaveDataToCsvfile(test_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_test_dataframes_AfterFeatureSelect")
    SaveDataframeTonpArray(test_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                           f"{choose_merge_days}_test_CICIDS2018_AfterFeatureSelect{slecet_label_counts}",today)
    SaveDataframeTonpArray(train_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                           f"{choose_merge_days}_train_CICIDS2018_AfterFeatureSelect{slecet_label_counts}",today)



# do PCA to all feature or excpetStringType
def DoPCA_ALLFeature_OR_excpetStringType(df,number_of_components ,bool_excpet_Strtype):

    # number_of_components=20
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    crop_dataset=df.iloc[:,:-1]
    # 列出要排除的列名
    columns_to_exclude = ['Dst Port', 'Protocol', 'Timestamp']
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
                      f"./data/dataset_AfterProcessed/CICIDS2018/csv/{today}/doPCA/{number_of_components}", 
                      f"CICIDS2018_AfterProcessed_{str_filename}_minmax_PCA")

    return df
# do PCA and Labelencode and minmax 
def DoSplitAfterDoPCA(df,number_of_components,choose_merge_days,bool_Noniid):
    # True為除string type以外特徵都做PCA
    # False為全特徵都做PCA
    df = DoPCA_ALLFeature_OR_excpetStringType(df,number_of_components ,False)
    # df = DoPCA_ALLFeature_OR_excpetStringType(df,number_of_components ,True)
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示將數據集分成測試集的比例為20%
    # printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes,"Label")
    if bool_Noniid !=True:
        if choose_merge_days =="csv_data":
            # Noniid時
            print("Nothing")
    else:
        # BaseLine時
        if choose_merge_days =="csv_data":
            train_dataframes, test_dataframes = DoBaselinesplit(df,train_dataframes,test_dataframes)
            #     # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
            # List_train_Label = []
            # List_test_Label = []
            # for i in range(15):
            #     if i == 13:
            #         continue
            #     train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
            #     List_train_Label.append(train_label_split)
            #     List_test_Label.append(test_label_split)         
            
            # train_dataframes = pd.concat(List_train_Label)
            # test_dataframes = pd.concat(List_test_Label)
            # # encode後對照如下
            # # SQL_Injection :13
            # # Label encode mode  分別取出Label等於13的數據 對6633分
            # train_label_SQL_Injection, test_label_SQL_Injection = splitweakLabelbalance(13,df,0.33)
            # # # 刪除Label相當於12的行
            # test_dataframes = test_dataframes[~test_dataframes['Label'].isin([13])]
            # train_dataframes = train_dataframes[~train_dataframes['Label'].isin([13])]
            # # 合併Label12回去
            # test_dataframes = pd.concat([test_dataframes, test_label_SQL_Injection])
            # train_dataframes = pd.concat([train_dataframes,train_label_SQL_Injection])   

    
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/encode_and_count_after_PCA_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, 
                      f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                      f"{choose_merge_days}_train_dataframes_AfterPCA{number_of_components}_{today}")
    SaveDataToCsvfile(test_dataframes,
                      f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                      f"{choose_merge_days}_test_dataframes_AfterPCA{number_of_components}_{today}")
    SaveDataframeTonpArray(test_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                           f"{choose_merge_days}_test_AfterPCA{number_of_components}",today)
    SaveDataframeTonpArray(train_dataframes, 
                           f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}/doPCA/{number_of_components}", 
                           f"{choose_merge_days}_train_AfterPCA{number_of_components}",today)

# do split train to half for iid and Labelencode and minmax 
def DoSplithalfForiid(choose_merge_days):
    if choose_merge_days == "csv_data":
        df_ALLtrain = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\20240502\\csv_data_train_dataframes_20240502.csv")
                    # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_half1_Label = []
        List_train_half2_Label = []
        for i in range(13):
            train_half1_label_split, train_half2_label_split = splitweakLabelbalance(i,df_ALLtrain,0.5)
            List_train_half1_Label.append(train_half1_label_split)
            List_train_half2_Label.append(train_half2_label_split)         
            
        df_train_half1 = pd.concat(List_train_half1_Label)
        df_train_half2 = pd.concat(List_train_half2_Label)
            

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/encode_and_count_iid.csv", "a+") as file:
            label_counts = df_train_half1['Label'].value_counts()
            print("df_train_half1\n", label_counts)
            file.write("df_train_half1_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = df_train_half2['Label'].value_counts()
            print("df_train_half2\n", label_counts)
            file.write("df_train_half2_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(df_train_half1, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half1_{today}")
        SaveDataToCsvfile(df_train_half2,  f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half2_{today}")
        SaveDataframeTonpArray(df_train_half1, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half1",today)
        SaveDataframeTonpArray(df_train_half2, f"./data/dataset_AfterProcessed/CICIDS2018/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half2",today)

# 開始進行資料劃分主要function
def SelectfeatureUseChiSquareOrPCA(df,choose_merge_days,bool_doChiSquare,bool_doPCA,bool_Noniid):
    if bool_doChiSquare!=False:
        # 選ALL特徵
        # DoSplitAfterFeatureSelect(df,None)
        #ChiSquare選80個特徵
        # DoSplitAfterFeatureSelect(df,80,choose_merge_days,bool_Noniid)
        # #ChiSquare選70個特徵
        # DoSplitAfterFeatureSelect(df,70,choose_merge_days,bool_Noniid)
        # # #ChiSquare選65個特徵
        # DoSplitAfterFeatureSelect(df,60,choose_merge_days,bool_Noniid)
        # # #ChiSquare選60個特徵
        # DoSplitAfterFeatureSelect(df,60,choose_merge_days,bool_Noniid)
        # # #ChiSquare選55個特徵
        # DoSplitAfterFeatureSelect(df,55,choose_merge_days,bool_Noniid)
        # # #ChiSquare選50個特徵
        # DoSplitAfterFeatureSelect(df,50,choose_merge_days,bool_Noniid)
        # # #ChiSquare選46個特徵
        # DoSplitAfterFeatureSelect(df,46,choose_merge_days,bool_Noniid)
        # # #ChiSquare選45個特徵
        # DoSplitAfterFeatureSelect(df,45,choose_merge_days,bool_Noniid)
        # #ChiSquare選44個特徵
        DoSplitAfterFeatureSelect(df,44,choose_merge_days,bool_Noniid)
        # #ChiSquare選40個特徵
        # DoSplitAfterFeatureSelect(df,40,choose_merge_days,bool_Noniid)
        # #ChiSquare選38個特徵
        # DoSplitAfterFeatureSelect(df,38,choose_merge_days,bool_Noniid)
    elif bool_doPCA!=False:
        ##PCA選79個特徵 總80特徵=79+扣掉'Label'
        DoSplitAfterDoPCA(df,79,choose_merge_days,bool_Noniid)
        #  #PCA選77個特徵 總84特徵=77+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSplitAfterDoPCA(df,77,choose_merge_days,bool_Noniid)
        # #PCA選73個特徵 總80特徵=73+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSplitAfterDoPCA(df,73,choose_merge_days,bool_Noniid)
        # #PCA選63個特徵 總70特徵=73+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSplitAfterDoPCA(df,63,choose_merge_days,bool_Noniid)
        # #PCA選53個特徵 總60特徵=53+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSplitAfterDoPCA(df,53,choose_merge_days,bool_Noniid)
        #PCA選43個特徵 總50特徵=43+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSplitAfterDoPCA(df,43,choose_merge_days,bool_Noniid)
        # #PCA選38個特徵 總45特徵=38+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSplitAfterDoPCA(df,38,choose_merge_days,bool_Noniid)
        # #PCA選33個特徵 總40特徵=33+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSplitAfterDoPCA(df,33,choose_merge_days,bool_Noniid) 

# 針對string type 做minmax
def RedoCICIDS2018stringtypeMinMaxfortrainORtest(afterprocess_dataset,bool_tain_OR_test):
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
        afterminmax_dataset.to_csv(f"./data/dataset_AfterProcessed/CICIDS2018/csv_data/CICIDS2018_AfterProcessed_DoLabelencode_ALLMinmax_train.csv", index=False)
        SaveDataframeTonpArray(afterminmax_dataset, f"./data/dataset_AfterProcessed/CICIDS2018/csv_data/{today}", f"csv_data_train_dataframes_ALLMinmax", today)
    else:
        afterminmax_dataset.to_csv(f"./data/dataset_AfterProcessed/CICIDS2018/csv_data/CICIDS2018_AfterProcessed_DoLabelencode_ALLMinmax_test.csv", index=False)
        SaveDataframeTonpArray(afterminmax_dataset, f"./data/dataset_AfterProcessed/CICIDS2018/csv_data/{today}", f"csv_data_test_dataframes_ALLMinmax", today)

    return afterprocess_dataset

    # 對已劃分好的tain和test的Strig type做完label ecnode後補做minmax
    # afterprocess_dataset_train = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\20240502\\csv_data_train_dataframes_20240502.csv")
    # # 加载CICIDS2018 test after do labelencode and minmax  75 25分
    # afterprocess_dataset_test = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\20240502\\csv_data_test_dataframes_20240502.csv")
    # print("Dataset loaded.")

    # afterprocess_dataset_train = RedoMinMaxfortrainORtest(afterprocess_dataset_train,True)
    # afterprocess_dataset_test = RedoMinMaxfortrainORtest(afterprocess_dataset_test,False)

def forBaseLineUseData(choose_merge_days,bool_Noniid):
    if choose_merge_days == "csv_data":
        # 載入資料集
        # df_csv_data=LoadingDatasetAfterMegreComplete(choose_merge_days)
        # 預處理和正規化
        # True for BaseLine
        # False for Noniid
        # df_csv_data=DoMinMaxAndLabelEncoding(df_csv_data,choose_merge_days,bool_Noniid)
        # for iid 實驗將ALL train分一半
        # DoSplithalfForiid(choose_merge_days)
        # 一般全部特徵
        # 只接用昱陞預處理好的去做劃分
        df_csv_data = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\sampled_data_max_10000_per_label.csv")
        # DoSplitAllfeatureAfterMinMax(df_csv_data,choose_merge_days,bool_Noniid)
        # 做ChiSquare
        # SelectfeatureUseChiSquareOrPCA(df_csv_data,choose_merge_days,True,False,bool_Noniid)
        # 做PCA
        SelectfeatureUseChiSquareOrPCA(df_csv_data,choose_merge_days,False,True,bool_Noniid)

# True for BaseLine
# False for Noniid
# forBaseLineUseData("csv_data",False)
# forBaseLineUseData("csv_data",True)
# forBaseLineUseBinaryData("csv_data",True)

# DoAllfeatureOrSelectfeature(afterminmax_dataset,False)
# DoAllfeatureOrSelectfeature(afterminmax_dataset,True)


