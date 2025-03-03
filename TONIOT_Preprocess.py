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
from mytoolfunction import generatefolder,SaveDataToCsvfile,SaveDataframeTonpArray,CheckFileExists,splitweakLabelbalance,DominmaxforStringTypefeature

# filepath = "D:\\ToN-IoT-Network\\TON_IoT Datasets\\UNSW-ToN-IoT"
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
# generatefolder(filepath + "\\", "data")
generatefolder(filepath + "\\", "dataset_AfterProcessed")
generatefolder(filepath + "\\dataset_AfterProcessed\\", "TONIOT")
generatefolder(filepath + "\\dataset_AfterProcessed\\TONIOT\\", today)

def remove_nan_and_inf(df):
    df = df.dropna(how='any', axis=0, inplace=False)
    inf_condition = (df == np.inf).any(axis=1)
    df = df[~inf_condition]
    return df

def clearDirtyData(df):
    # 将每个列中的 "-" 替换为 NaN
    # df.replace("-", pd.NA, inplace=True)
    # 找到不包含NaN、Infinity和"inf"值和"-"值的行
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
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



# 将每列中的 "-" 替换为最常见值
def RealpaceSymboltoTheMostCommonValue(dataset):
    # 列出每个列的最常见值
    # most_common_values = dataset.mode().iloc[0]
    # 列出每个列除了 "-" 以外的最常见值
    most_common_values = dataset.apply(lambda col: col[col != "-"].mode().iloc[0] if any(col != "-") else "-", axis=0)

    # 将每列中的 "-" 替换为最常见值
    for column in dataset.columns:
        most_common_value = most_common_values[column]
        dataset[column].replace("-", most_common_value, inplace=True)

    # 打印替换后的结果
    print(dataset)
    # 打印结果
    print(most_common_values)
    return dataset

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
    df['type'] = df['type'].map(encoding_map)
    return df, encoding_map


def LoadingDatasetAfterMegreComplete(dataset):

    dataset = clearDirtyData(dataset)
    dataset = RealpaceSymboltoTheMostCommonValue(dataset)
    if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed.csv")!=True):
        #存将每列中的 "-" 替换为最常见值後的csv
        dataset.to_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed.csv", index=False)
    else:
        dataset= pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed.csv")


        dataset = ReplaceMorethanTenthousandQuantity(dataset)

    if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000.csv")!=True):
        dataset.to_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000.csv", index=False)
        afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000.csv")

    else:
        afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000.csv")

    return afterprocess_dataset
### add 沒有的Label到TONIOT
def DoAddLabelToTONIOT(df,add_mergedays_label_or_dataset_label):
    if add_mergedays_label_or_dataset_label == "CICIDS2017":
            values_to_insert = ['Bot', 'DoSGoldenEye', 'DoSHulk', 'DoSSlowhttptest', 'DoSslowloris', 
                                'FTPPatator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSHPatator', 
                                'WebAttackBruteForce','WebAttackSqlInjection','WebAttackXSS']
    
    elif add_mergedays_label_or_dataset_label == "CICIDS2019":
            values_to_insert = ['DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP', 
                                'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 
                                'Syn', 'TFTP', 'UDPlag', 'WebDDoS']
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
    
    df['type'] = df['type'].replace({'normal': 'BENIGN'})
    df['type'] = df['type'].replace({'ddos': 'DDoS'})

    return df

def DoRenameLabel(df):
    df['type'] = df['type'].replace({'normal': 'BENIGN'})
    df['type'] = df['type'].replace({'ddos': 'DDoS'})

    # 將 'type' 列重命名為 'Label'
    # df.rename(columns={'type': 'Label'}, inplace=True)
    return df

def DoMinMaxAndLabelEncoding(afterprocess_dataset,bool_doencode):
    ### label encoding
    def label_Encoding(label):
        label_encoder = preprocessing.LabelEncoder()
        afterprocess_dataset[label] = label_encoder.fit_transform(afterprocess_dataset[label])
        afterprocess_dataset[label].unique()

    ##除了Label外的特徵做encode
   # 等標籤對其後 在對特徵做處理
    label_Encoding('src_ip')
    label_Encoding('src_port')
    label_Encoding('dst_ip')
    label_Encoding('dst_port')
    label_Encoding('proto')
    label_Encoding('ts')
    label_Encoding('service')
    label_Encoding('conn_state')
    # 需要做label_Encoding的欄位
    label_Encoding('dns_query')
    label_Encoding('dns_AA')
    label_Encoding('dns_RD')
    label_Encoding('dns_RA')
    label_Encoding('dns_rejected')
    label_Encoding('ssl_version')
    label_Encoding('ssl_cipher')
    label_Encoding('ssl_resumed')
    label_Encoding('ssl_established')
    label_Encoding('ssl_subject')
    label_Encoding('ssl_issuer')
    # label_Encoding('http_trans_depth')
    label_Encoding('http_method')
    label_Encoding('http_uri')
    # label_Encoding('http_version')
    label_Encoding('http_user_agent')
    label_Encoding('http_orig_mime_types')
    label_Encoding('http_resp_mime_types')
    label_Encoding('weird_name')
    # label_Encoding('weird_addl')
    label_Encoding('weird_notice')
    
    ## extracting features
    # 除了type外的特徵
    crop_dataset=afterprocess_dataset.iloc[:,:-1]
    # 列出要排除的列名，這6個以外得特徵做minmax
    columns_to_exclude = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'ts']
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
    afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,afterprocess_dataset['type']], axis = 1)
    
    # 保存Lable未做label_encoding的DataFrame方便後續Noniid實驗
    if bool_doencode != True:
        if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000_add_Label_and_UnDoLabelencode_and_minmax.csv")
                           !=True):

            afterminmax_dataset = DoAddLabelToTONIOT(afterminmax_dataset,"CICIDS2017")
            afterminmax_dataset = DoAddLabelToTONIOT(afterminmax_dataset,"CICIDS2019")
            afterminmax_dataset.to_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000_add_Label_and_UnDoLabelencode_and_minmax.csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000_add_Label_and_UnDoLabelencode_and_minmax.csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000_add_Label_and_UnDoLabelencode_and_minmax.csv")

         # encoded_type_values, afterminmax_dataset = label_encoding("type", afterminmax_dataset)
        # # print("Original Type Values:", original_type_values)
        # print("Encoded Type Values:", encoded_type_values)
        afterminmax_dataset,encoded_type_values = LabelMapping(afterminmax_dataset)

        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/TONIOT/encode_and_count_Noniid.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")
    #保存Lable做label_encoding的DataFrame方便後續BaseLine實驗
    else:
        # Rename Label方便後續Noniid實驗
        afterminmax_dataset = DoRenameLabel(afterminmax_dataset)
        encoded_type_values, afterminmax_dataset = label_encoding("type", afterminmax_dataset)
        # afterminmax_dataset.to_csv(filepath + 
                                #    "\\dataset_AfterProcessed\\CICIDS2017\\"+choose_merge_days+"\\CICIDS2017_AfterProcessed_DoLabelencode_"+choose_merge_days+".csv", index=False)
        
        
        if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000_add_Label_and_DoLabelencode_and_minmax.csv")
                           !=True):
            afterminmax_dataset.to_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000_add_Label_and_DoLabelencode_and_minmax.csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000_add_Label_and_DoLabelencode_and_minmax.csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\Train_Test_Network_AfterProcessed_updated_10000_add_Label_and_DoLabelencode_and_minmax.csv")

        # print("Original Type Values:", original_type_values)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/TONIOT/encode_and_count_baseLine.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")

    return afterminmax_dataset

# do Labelencode and minmax 
def DoSplitAllfeatureAfterMinMax(df,bool_Noniid):  
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    #不能用手動劃分 因其劃分時不是random沒有將資料打亂會造成資料誤判
    # train_dataframes, test_dataframes = manualsplitdataset(df)#test_size=0.2表示将数据集分成测试集的比例为20%

    if bool_Noniid !=True:

            # 篩選test_dataframes中標籤為5,10,14,26,34的行加回去train
            train_dataframes_add = test_dataframes[test_dataframes['type'].isin([5,10,14,26,34])]
            # test刪除Label相當於5,10,14,26,34的行，因為這些是因為noniid要加到train的Label
            test_dataframes = test_dataframes[~test_dataframes['type'].isin([5,10,14,26,34])]
            # # 合併Label5,10,14,26,34回去到train
            train_dataframes = pd.concat([train_dataframes,train_dataframes_add])
    else:
        # BaseLine時
            # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
            List_train_Label = []
            List_test_Label = []
            for i in range(10):
                train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
                List_train_Label.append(train_label_split)
                List_test_Label.append(test_label_split)         

            train_dataframes = pd.concat(List_train_Label)
            test_dataframes = pd.concat(List_test_Label)

            print("test",test_dataframes['type'].value_counts())
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/TONIOT/encode_and_count_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['type'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['type'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/{today}", f"train_ToN-IoT_dataframes_{today}")
    SaveDataToCsvfile(test_dataframes, f"./data/dataset_AfterProcessed/TONIOT/{today}", f"test_ToN-IoT_dataframes_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/TONIOT/{today}", "test_ToN-IoT",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/{today}", "train_ToN-IoT",today)


def DoSplitAfterDoPCA(df,number_of_components,bool_Noniid,bool_add_cicids2017feature):
    
    # add cicids2017 chisquare select 45出來除IP外的的39個特徵
    if(bool_add_cicids2017feature):
        columns = [
                'FlowDuration', 'FwdPacketLengthMin', 'BwdPacketLengthMax', 'BwdPacketLengthMin', 'BwdPacketLengthMean', 'BwdPacketLengthStd',
                'FlowPackets/s', 'FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FwdIATTotal', 'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin',
                'BwdIATTotal', 'BwdIATMean', 'BwdIATMax', 'FwdPSHFlags', 'BwdPackets/s', 'MinPacketLength', 'MaxPacketLength', 'PacketLengthMean',
                'PacketLengthStd', 'PacketLengthVariance', 'FINFlagCount', 'SYNFlagCount', 'PSHFlagCount', 'ACKFlagCount', 'URGFlagCount', 
                'Down/UpRatio', 'AveragePacketSize', 'AvgBwdSegmentSize', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'ActiveMin',
                'IdleMean', 'IdleStd', 'IdleMax', 'IdleMin'
            ]

        
        # 创建新列的DataFrame，所有值都是0
        df_cicids2017feature = pd.DataFrame(0, index=df.index, columns=columns)
        df = pd.concat([df.iloc[:,:-1],df_cicids2017feature,df[['type']]], axis = 1)
        
    crop_dataset=df.iloc[:,:-1]
    # 列出要排除的列名
    columns_to_exclude = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'ts']
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

    finalDf = pd.concat([undoScalerdataset,principalDf, df[['type']]], axis = 1)
    df=finalDf

    SaveDataToCsvfile(df, 
                      f"./data/dataset_AfterProcessed/TONIOT/{today}/doPCA/{number_of_components}", 
                      f"TONIOT_AfterProcessed_minmax_PCA")

    train_dataframes, test_dataframes = train_test_split(df, test_size=0.25, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    # BaseLine時
            # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
    List_train_Label = []
    List_test_Label = []
    for i in range(10):
        train_label_split, test_label_split = splitweakLabelbalance(i,df,0.25)
        List_train_Label.append(train_label_split)
        List_test_Label.append(test_label_split)         

        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)

        print("test",test_dataframes['type'].value_counts())
    # printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes,"Label")
    if bool_Noniid !=True:

            label_counts = test_dataframes['type'].value_counts()
            print("test_dataframes\n", label_counts)
            label_counts = train_dataframes['type'].value_counts()
            print("train_dataframes\n", label_counts)
    else:
        # BaseLine時
            label_counts = test_dataframes['type'].value_counts()
            print("test_dataframes\n", label_counts)
            label_counts = train_dataframes['type'].value_counts()
            print("train_dataframes\n", label_counts)


    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/{today}/doPCA/{number_of_components}", 
                      f"train_ToN-IoT_dataframes_AfterPCA{number_of_components}_{today}")
    SaveDataToCsvfile(test_dataframes, f"./data/dataset_AfterProcessed/TONIOT/{today}/doPCA/{number_of_components}", 
                      f"test_ToN-IoT_dataframes_AfterPCA{number_of_components}_{today}")
    
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/TONIOT/{today}/doPCA/{number_of_components}", 
                           f"test_ToN-IoT_test_AfterPCA{number_of_components}",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/TONIOT/{today}/doPCA/{number_of_components}", 
                           f"train_ToN-IoT_train_AfterPCA{number_of_components}",today)


# do split train to half for iid and Labelencode and minmax 
def DoSplitthrildClientForiid():
        # 75 train分 均勻分3份
        df_ALLtrain = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_20240523.csv")
                    # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_half1_Label = []
        List_train_half2_Label = []
        List_train_half3_Label = []

        for i in range(10):
            # 第一次拆分：将数据拆分成 33.3% 和 66.7%
            train_half1_label_split, train_half2_label_split = splitweakLabelbalance(i,df_ALLtrain,0.3333)
            # 第二次拆分：将 66.7% 的数据再拆分成 50% 和 50%（即 33.3% 和 33.3%）
            train_half1_label_split_half1,train_half1_label_split_half2 = splitweakLabelbalance(i,train_half1_label_split,0.5)

            List_train_half1_Label.append(train_half1_label_split_half1)
            List_train_half2_Label.append(train_half2_label_split)    
            List_train_half3_Label.append(train_half1_label_split_half2)      
            
        df_train_half1 = pd.concat(List_train_half1_Label)
        df_train_half2 = pd.concat(List_train_half2_Label)
        df_train_half3 = pd.concat(List_train_half3_Label)
            

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/TONIOT/encode_and_count_iid.csv", "a+") as file:
            label_counts = df_train_half1['type'].value_counts()
            print("df_train_half1\n", label_counts)
            file.write("df_train_half1_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = df_train_half2['type'].value_counts()
            print("df_train_half2\n", label_counts)
            file.write("df_train_half2_label_counts\n")
            file.write(str(label_counts) + "\n")

            label_counts = df_train_half3['type'].value_counts()
            print("df_train_half3\n", label_counts)
            file.write("df_train_half2_label_counts\n")
            file.write(str(label_counts) + "\n")


        SaveDataToCsvfile(df_train_half1, f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half1_20240523")
        SaveDataToCsvfile(df_train_half2,  f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half2_20240523")
        SaveDataToCsvfile(df_train_half3,  f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half3_20240523")

        SaveDataframeTonpArray(df_train_half1, f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half1",20240523)
        SaveDataframeTonpArray(df_train_half2, f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half2",20240523)
        SaveDataframeTonpArray(df_train_half3, f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half3",20240523)



# do split train to half for iid and Labelencode and minmax 
import random
def DoRandomSplitthrildClientForiid():
        # 75 train分3份
        df_ALLtrain = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_20240523.csv")
                    # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
        List_train_half1_Label = []
        List_train_half2_Label = []
        List_train_half3_Label = []
        
        # random 劃分
        # 随机调整比例，除了标签5
        first_random_size_dict = {}
        second_random_size_dict = {}
        total_first_random_size = 0
        total_second_random_size = 0
        # 数据划分
        for i in range(10):
            if i == 5:
                first_random_size = 0.3333
                second_random_size = 0.5 
            else:
            # 获取标签 i 对应的随机比例
                # first_random_size = first_random_size_dict.get(i, 0.5)
                # second_random_size = second_random_size_dict.get(i, 0.5)
                first_random_size = round(random.uniform(0.2, 0.7), 4)
                # 生成0.2到1之间的随机比例
                # second_random_size = round(random.uniform(0.5, 1), 4)

                # # 确保两个比例的和不超过1
                # if first_random_size + second_random_size <= 1:
                #     first_random_size_dict[i] = first_random_size
                #     second_random_size_dict[i] = second_random_size
                #     break  # 退出循环
            
            # 计算所有其他标签的比例总和
            total_first_random_size = sum(first_random_size_dict.values())
            total_second_random_size = sum(second_random_size_dict.values())
            # 输出结果
            print("First Random Size Dict:", total_first_random_size)
            print("Second Random Size Dict:", total_second_random_size)

            # 第一次拆分
            train_half1_label_split, train_half2_label_split = splitweakLabelbalance(i, df_ALLtrain, first_random_size)
            # 第二次拆分
            train_half1_label_split_half1, train_half1_label_split_half2 = splitweakLabelbalance(i, train_half1_label_split, first_random_size)

            # 保存到相应的列表中
            List_train_half1_Label.append(train_half1_label_split_half1)
            List_train_half2_Label.append(train_half2_label_split)
            List_train_half3_Label.append(train_half1_label_split_half2)

        df_train_half1 = pd.concat(List_train_half1_Label)
        df_train_half2 = pd.concat(List_train_half2_Label)
        df_train_half3 = pd.concat(List_train_half3_Label)
            

        # 紀錄資料筆數
        with open(f"./data/dataset_AfterProcessed/TONIOT/encode_and_count_iid.csv", "a+") as file:
            label_counts = df_train_half1['type'].value_counts()
            print("df_train_half1\n", label_counts)
            file.write("df_train_half1_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = df_train_half2['type'].value_counts()
            print("df_train_half2\n", label_counts)
            file.write("df_train_half2_label_counts\n")
            file.write(str(label_counts) + "\n")

            label_counts = df_train_half3['type'].value_counts()
            print("df_train_half3\n", label_counts)
            file.write("df_train_half2_label_counts\n")
            file.write(str(label_counts) + "\n")

        # random 劃分
        SaveDataToCsvfile(df_train_half1, f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half1_random_20240523")
        SaveDataToCsvfile(df_train_half2,  f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half2_random_20240523")
        SaveDataToCsvfile(df_train_half3,  f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_train_half3_random_20240523")

        SaveDataframeTonpArray(df_train_half1, f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_random_train_half1",20240523)
        SaveDataframeTonpArray(df_train_half2, f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_random_train_half2",20240523)
        SaveDataframeTonpArray(df_train_half3, f"./data/dataset_AfterProcessed/TONIOT/20240523", f"train_ToN-IoT_dataframes_random_train_half3",20240523)




#手動劃分
def manualsplitdataset(df):
        train_dataframes = pd.concat([
                                            df[df['type'] == 0].iloc[:8000],
                                            df[df['type'] == 1].iloc[:1],
                                            df[df['type'] == 2].iloc[:8000],
                                            df[df['type'] == 3].iloc[:1],
                                            df[df['type'] == 4].iloc[:1],
                                            df[df['type'] == 5].iloc[:1],
                                            df[df['type'] == 6].iloc[:1],
                                            df[df['type'] == 7].iloc[:1],
                                            df[df['type'] == 8].iloc[:1],
                                            df[df['type'] == 9].iloc[:1],
                                            df[df['type'] == 10].iloc[:1],
                                            df[df['type'] == 11].iloc[:1],
                                            df[df['type'] == 12].iloc[:1],
                                            df[df['type'] == 13].iloc[:1],
                                            df[df['type'] == 14].iloc[:1],
                                            df[df['type'] == 15].iloc[:8000],
                                            df[df['type'] == 16].iloc[:8000],
                                            df[df['type'] == 17].iloc[:8000],
                                            df[df['type'] == 18].iloc[:827],
                                            df[df['type'] == 19].iloc[:8000],
                                            df[df['type'] == 20].iloc[:8000],
                                            df[df['type'] == 21].iloc[:8000],
                                            df[df['type'] == 22].iloc[:8000],
                                        ], ignore_index=True)

        test_dataframes = pd.concat([
                                            df[df['type'] == 0].iloc[8000:],
                                            df[df['type'] == 2].iloc[8000:],
                                            df[df['type'] == 15].iloc[8000:],
                                            df[df['type'] == 16].iloc[8000:],
                                            df[df['type'] == 17].iloc[8000:],
                                            df[df['type'] == 18].iloc[827:],
                                            df[df['type'] == 19].iloc[8000:],
                                            df[df['type'] == 20].iloc[8000:],
                                            df[df['type'] == 21].iloc[8000:],
                                            df[df['type'] == 22].iloc[8000:],
                                        ], ignore_index=True)
        
        return train_dataframes,test_dataframes

def forBaseLineUseData(bool_Noniid):
        # Loading datasets
        df = pd.read_csv(filepath + "\\TONTOT_Original\\Train_Test_Network.csv")
        # 載入資料集預處理和正規化
        df=LoadingDatasetAfterMegreComplete(df)
        # True for BaseLine
        # False for Noniid
        df=DoMinMaxAndLabelEncoding(df,bool_Noniid)
        # 一般全部特徵
        DoSplitAllfeatureAfterMinMax(df,bool_Noniid)
        # PCA
        DoSplitAfterDoPCA(df,77,bool_Noniid,True)


# True for BaseLine
# False for Noniid
# forBaseLineUseData(False)
# forBaseLineUseData(True)
# DoSplitthrildClientForiid()
# DoRandomSplitthrildClientForiid()

# 針對string type 做minmax
def RedoTONIOTstringtypeMinMaxfortrainORtest(afterprocess_dataset,bool_tain_OR_test):
    #除了Label外的特徵
    crop_dataset=afterprocess_dataset.iloc[:,:-1]
    columns_to_exclude = ['ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto']
    testdata_removestring = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col in columns_to_exclude]]
    doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    # 補string type 做minmax
    undoScalerdataset = DominmaxforStringTypefeature(undoScalerdataset)
    
    # 将排除的列名和选中的特征和 Label 合并为新的 DataFrame
    afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,afterprocess_dataset['type']], axis = 1)

    if bool_tain_OR_test:
        afterminmax_dataset.to_csv(f"./data/dataset_AfterProcessed/TONIOT/20240523/train_ToN-IoT_AfterProcessed_DoLabelencode_ALLMinmax_train.csv", index=False)
        SaveDataframeTonpArray(afterminmax_dataset, f"./data/dataset_AfterProcessed/TONIOT/{today}", f"ToN-IoT_train_dataframes_ALLMinmax", today)
    else:
        afterminmax_dataset.to_csv(f"./data/dataset_AfterProcessed/TONIOT/20240523//test_ToN-IoT_AfterProcessed_DoLabelencode_ALLMinmax_test.csv", index=False)
        SaveDataframeTonpArray(afterminmax_dataset, f"./data/dataset_AfterProcessed/TONIOT/{today}", f"ToN-IoT_test_dataframes_ALLMinmax", today)

    return afterprocess_dataset

# 對已劃分好的tain和test的Strig type做完label ecnode後補做minmax
# # 加载TONIOT test after do labelencode and minmax  75 25分
afterprocess_dataset_train = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_20240523.csv")
afterprocess_dataset_test = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\test_ToN-IoT_dataframes_20240523.csv")
print("Dataset loaded.")    
afterprocess_dataset_train = RedoTONIOTstringtypeMinMaxfortrainORtest(afterprocess_dataset_train,True)
afterprocess_dataset_test = RedoTONIOTstringtypeMinMaxfortrainORtest(afterprocess_dataset_test,False)

