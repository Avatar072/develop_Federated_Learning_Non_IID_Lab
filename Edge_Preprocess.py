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
from mytoolfunction import SaveDataToCsvfile,printFeatureCountAndLabelCountInfo,CheckFileExists,ReplaceMorethanTenthousandQuantity
from mytoolfunction import clearDirtyData,label_Encoding,splitdatasetbalancehalf,spiltweakLabelbalance,SaveDataframeTonpArray,generatefolder
from mytoolfunction import SaveDataToCsvfile,ChooseDataSetNpFile,CheckFileExists,DoReStoreNpFileToCsv,ResotreTrainAndTestToCSVandReSplit

#############################################################################  variable  ###################
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")

generatefolder(filepath + "\\dataset_AfterProcessed\\", "EdgeIIoT")
generatefolder(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\", today)

# restore np file to csv
# ResotreTrainAndTestToCSVandReSplit("EdgeIIoT",filepath)

def DoSpilthalfForiid(choose_merge_days):
    if choose_merge_days == "EdgeIIoT":
        df_ALLtrain = pd.read_csv(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\20240507\\Resplit_train_dataframes_20240507.csv")
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
        with open(f"./data/dataset_AfterProcessed/{choose_merge_days}/encode_and_count_iid.csv", "a+") as file:
            label_counts = df_train_half1['Label'].value_counts()
            print("df_train_half1\n", label_counts)
            file.write("df_train_half1_label_counts\n")
            file.write(str(label_counts) + "\n")
            
            label_counts = df_train_half2['Label'].value_counts()
            print("df_train_half2\n", label_counts)
            file.write("df_train_half2_label_counts\n")
            file.write(str(label_counts) + "\n")

        SaveDataToCsvfile(df_train_half1, f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half1_{today}")
        SaveDataToCsvfile(df_train_half2,  f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half2_{today}")
        SaveDataframeTonpArray(df_train_half1, f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half1",today)
        SaveDataframeTonpArray(df_train_half2, f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}", f"{choose_merge_days}_train_half2",today)

# DoSpilthalfForiid("EdgeIIoT")

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
        # EdgeIIoT
        'DDoS_UDP': 23,
        'DDoS_ICMP': 24,
        'SQL_injection': 25,
        'Vulnerability_scanner': 26,
        'DDoS_TCP': 27,
        'DDoS_HTTP': 28,
        'Uploading': 29,
        'Fingerprinting': 30
        # CICIDS2019
        # 'DrDoS_DNS': 23,
        # 'DrDoS_LDAP': 24,
        # 'DrDoS_MSSQL': 25,
        # 'DrDoS_NTP': 26,
        # 'DrDoS_NetBIOS': 27,
        # 'DrDoS_SNMP': 28,
        # 'DrDoS_SSDP': 29,
        # 'DrDoS_UDP': 30,
        # 'Syn': 31,
		# 'TFTP': 32,
        # 'UDPlag': 33,
        # 'WebDDoS': 34
    }
    # 將固定編碼值映射應用到DataFrame中的Label列，直接更新原始的Label列
    df['Label'] = df['Label'].map(encoding_map)
    return df, encoding_map

### show original label name and after labelenocode
def label_encoding(label, dataset):
    label_encoder = preprocessing.LabelEncoder()
    # original_values = dataset[label].unique()
    
    dataset[label] = label_encoder.fit_transform(dataset[label])
    # encoded_values = dataset[label].unique()
    
      # 获取原始值和编码值的对照关系字典
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    return label_mapping,dataset   

def DoCheckFileExistAndSaveCsv(dataset, file_name):
    if(CheckFileExists(filepath + f"\\dataset_AfterProcessed\\EdgeIIoT\\ML-EdgeIIoT-dataset_{file_name}.csv")!=True):
        #存過濾到dirty data後的csv
        dataset.to_csv(filepath + f"\\dataset_AfterProcessed\\EdgeIIoT\\ML-EdgeIIoT-dataset_{file_name}.csv", index=False)
    else:
        # # 讀取 CSV 文件並設置 dtype 選項，將每個column type強制只轉成string
        # 載入資料集
        dataset= pd.read_csv(filepath + f"\\dataset_AfterProcessed\\EdgeIIoT\\ML-EdgeIIoT-dataset_{file_name}.csv", low_memory=False, dtype='str')
    
    return dataset

def clearDirtyData(df):
    # 找到不包含NaN、Infinity和"inf"值和"-"值的行
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    return df
def DoRenameLabel(df):
    df['Attack_type'] = df['Attack_type'].replace({'Normal': 'BENIGN'})
    df['Attack_type'] = df['Attack_type'].replace({'MITM': 'mitm'})
    df['Attack_type'] = df['Attack_type'].replace({'Password': 'password'})
    df['Attack_type'] = df['Attack_type'].replace({'Ransomware': 'ransomware'})
    df['Attack_type'] = df['Attack_type'].replace({'Backdoor': 'backdoor'})
    df['Attack_type'] = df['Attack_type'].replace({'Port_Scanning': 'PortScan'})
    df['Attack_type'] = df['Attack_type'].replace({'XSS': 'xss'})

    # 將 'Attack_type' 列重命名為 'Label'
    df.rename(columns={'Attack_type': 'Label'}, inplace=True)
    return df

def LoadingDataset(df_original_dataset):
    # 過濾到dirty data 找到不包含NaN、Infinity和"inf"值的行
    df_original_dataset = clearDirtyData(df_original_dataset)
    # 取資料上限10000筆
    df_original_dataset =ReplaceMorethanTenthousandQuantity(df_original_dataset,'Attack_type')
    # 存未Rename的csv 防止會用到
    df_original_dataset = DoCheckFileExistAndSaveCsv(df_original_dataset,"after_processed_UnRename")
    # Rename Label方便後續Noniid實驗
    df_original_dataset = DoRenameLabel(df_original_dataset)
    df_original_dataset = df_original_dataset.astype(str)
    df_original_dataset = DoCheckFileExistAndSaveCsv(df_original_dataset,"after_processed_Rename_and_10000")

    return df_original_dataset



def DoMinMaxAndLabelEncoding(afterprocess_dataset,bool_doencode):
    
    ##除了Label外的特徵做encode
    afterprocess_dataset = label_Encoding('frame.time',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('ip.src_host',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('arp.src.proto_ipv4',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('ip.dst_host',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('arp.dst.proto_ipv4',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('icmp.transmit_timestamp',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('http.file_data',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('http.request.uri.query',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('http.request.method',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('http.referer',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('http.request.full_uri',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('http.request.version',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('tcp.ack',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('tcp.ack_raw',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('tcp.dstport',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('tcp.options',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('tcp.payload',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('tcp.srcport',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('udp.port',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('dns.qry.name.len',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('mqtt.conack.flags',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('mqtt.msg',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('mqtt.protoname',afterprocess_dataset)
    afterprocess_dataset = label_Encoding('mqtt.topic',afterprocess_dataset)

    ### extracting features
    #除了Label外的特徵
    crop_dataset=afterprocess_dataset.iloc[:,:-1]
    # 列出要排除的列名，這9個以外得特徵做minmax
    columns_to_exclude = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.dst.proto_ipv4', 'icmp.transmit_timestamp','arp.src.proto_ipv4', 'tcp.dstport', 'tcp.srcport', 'udp.port']

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
    # print("test")
    # 保存Lable未做label_encoding的DataFrame方便後續Noniid實驗
    if bool_doencode != True:

        if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AfterProcessed_UndoLabelencode.csv")
                           !=True):
            # False 只add 所選擇的星期沒有的Label或只 add TONIOT的Label
            # afterminmax_dataset = DoAddLabel(afterminmax_dataset,False)
            # True add 所選擇的星期沒有的Label和TONIOT的Label
            # afterminmax_dataset = DoAddLabel(afterminmax_dataset,True)

            afterminmax_dataset.to_csv(filepath +
                                       "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AfterProcessed_UndoLabelencode.csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AfterProcessed_UndoLabelencode.csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AfterProcessed_UndoLabelencode.csv")

        # 固定Label encode值方便後續Noniid實驗
        afterminmax_dataset,encoded_type_values = LabelMapping(afterminmax_dataset)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/EdgeIIoT/encode_and_count_Noniid.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")
    #保存Lable做label_encoding的DataFrame方便後續BaseLine實驗
    else:
        encoded_type_values, afterminmax_dataset = label_encoding("Label", afterminmax_dataset)
        if(CheckFileExists(filepath + 
                           "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AfterProcessed_DoLabelencode.csv")
                           !=True):
            afterminmax_dataset.to_csv(filepath +
                                       "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AfterProcessed_DoLabelencode.csv", index=False)
                
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AfterProcessed_DoLabelencode.csv")

        else:
            afterminmax_dataset = pd.read_csv(filepath +
                                              "\\dataset_AfterProcessed\\EdgeIIoT\\EdgeIIoT_AfterProcessed_DoLabelencode.csv")
        
        # print("Original Type Values:", original_type_values)
        print("Encoded Type Values:", encoded_type_values)
        with open(f"./data/dataset_AfterProcessed/EdgeIIoT/encode_and_count_baseLine.csv", "a+") as file:
            file.write("Encoded Type Values\n")
            file.write(str(encoded_type_values) + "\n")

    return afterminmax_dataset


# do Labelencode and minmax 
def DoSpiltAllfeatureAfterMinMax(df,bool_Noniid):  
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.25, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%    
    # 把Label encode mode  分別取出Label的數據分 train:75% test:25%
    if bool_Noniid !=True:
        List_train_Label = []
        List_test_Label = []
        for i in range(31):# 從 0 到 30 的迴圈 因為做Noniid共有30個Label
            # 跳過不是 0、10、15 和不在 18 到 20 之間、不在 22 到 30 之間
            # 如果 i 不在 [0, 10, 15] 列表中，並且 i 不在 18 到 20 之間，並且 i 不在 22 到 30 之間，那麼執行 continue 跳過當前迴圈
            # 只有當 i 同時不滿足這三個條件時才會進入 continue 跳過迴圈
            if i not in [0, 10, 15] and not (18 <= i <= 20) and not (22 <= i <= 30):
                continue
            train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
            List_train_Label.append(train_label_split)
            List_test_Label.append(test_label_split)         

        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)

        print("test",test_dataframes['Label'].value_counts())

    else:
        List_train_Label = []
        List_test_Label = []
        for i in range(15):
            train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
            List_train_Label.append(train_label_split)
            List_test_Label.append(test_label_split)         

        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)

        print("test",test_dataframes['Label'].value_counts())
        # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/EdgeIIoT/encode_and_count_resplit.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
            
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")

    SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/EdgeIIoT/{today}", f"EdgeIIoT_train_dataframes_{today}")
    SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/EdgeIIoT/{today}", f"EdgeIIoT_test_dataframes_{today}")
    SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/EdgeIIoT/{today}", f"EdgeIIoT_test",today)
    SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/EdgeIIoT/{today}", f"EdgeIIoT_train",today)

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
                      f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_selected_feature_stats_{today}")

    SaveDataToCsvfile(all_feature_stats_df, 
                      f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
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
                      f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_unselected_feature_stats_{today}")
    

    # 将 X_new 转换为 DataFrame
    X_new_df = pd.DataFrame(X_new, columns=selected_features)

    # 将选中的特征和 Label 合并为新的 DataFrame
    selected_data = pd.concat([X_new_df, df['Label']], axis=1)
    
    # SaveDataToCsvfile(selected_data, f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
    #                   f"{choose_merge_days}_AfterSelected_{slecet_label_counts}_feature_data_{today}")
    return selected_data



# do chi-square and Labelencode and minmax 
def DoSpiltAfterFeatureSelect(df,slecet_label_counts,choose_merge_days,bool_Noniid):
    df = dofeatureSelect(df,slecet_label_counts,choose_merge_days)
    # 自動切
    train_dataframes, test_dataframes = train_test_split(df, test_size=0.25, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
    if bool_Noniid !=True:
        if choose_merge_days =="EdgeIIoT":
                List_train_Label = []
                List_test_Label = []
                for i in range(31):# 從 0 到 30 的迴圈 因為做Noniid共有30個Label
                    # 跳過不是 0、10、15 和不在 18 到 20 之間、不在 22 到 30 之間
                    # 如果 i 不在 [0, 10, 15] 列表中，並且 i 不在 18 到 20 之間，並且 i 不在 22 到 30 之間，那麼執行 continue 跳過當前迴圈
                    # 只有當 i 同時不滿足這三個條件時才會進入 continue 跳過迴圈
                    if i not in [0, 10, 15] and not (18 <= i <= 20) and not (22 <= i <= 30):
                        continue
                    train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
                    List_train_Label.append(train_label_split)
                    List_test_Label.append(test_label_split)         

                train_dataframes = pd.concat(List_train_Label)
                test_dataframes = pd.concat(List_test_Label)

                print("test",test_dataframes['Label'].value_counts())
    else:
        # BaseLine時
        List_train_Label = []
        List_test_Label = []
        for i in range(15):
            train_label_split, test_label_split = spiltweakLabelbalance(i,df,0.25)
            List_train_Label.append(train_label_split)
            List_test_Label.append(test_label_split)         

        train_dataframes = pd.concat(List_train_Label)
        test_dataframes = pd.concat(List_test_Label)

        print("test",test_dataframes['Label'].value_counts())
    # 紀錄資料筆數
    with open(f"./data/dataset_AfterProcessed/{choose_merge_days}/encode_and_count_after_chisquare_{bool_Noniid}.csv", "a+") as file:
        label_counts = test_dataframes['Label'].value_counts()
        print("test_dataframes\n", label_counts)
        file.write("test_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")
        
        label_counts = train_dataframes['Label'].value_counts()
        print("train_dataframes\n", label_counts)
        file.write("train_dataframes_label_counts\n")
        file.write(str(label_counts) + "\n")


    SaveDataToCsvfile(train_dataframes, 
                      f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}",  
                      f"{choose_merge_days}_train_dataframes_AfterFeatureSelect")
    SaveDataToCsvfile(test_dataframes, 
                      f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                      f"{choose_merge_days}_test_dataframes_AfterFeatureSelect")
    SaveDataframeTonpArray(test_dataframes, 
                           f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                           f"{choose_merge_days}_test_AfterFeatureSelect{slecet_label_counts}",today)
    SaveDataframeTonpArray(train_dataframes, 
                           f"./data/dataset_AfterProcessed/{choose_merge_days}/{today}/doFeatureSelect/{slecet_label_counts}", 
                           f"{choose_merge_days}_train_AfterFeatureSelect{slecet_label_counts}",today)


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
        # DoSpiltAfterDoPCA(df,38,choose_merge_days,bool_Noniid)
        # #PCA選33個特徵 總40特徵=33+扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label'
        # DoSpiltAfterDoPCA(df,33,choose_merge_days,bool_Noniid) 

        DoSpiltAfterFeatureSelect(df,44,choose_merge_days,bool_Noniid)


def forBaseLineUseData(bool_Noniid):
       
        # 載入資料集讀取 CSV 文件並設置 dtype 選項，強制column轉成str type
        df_Edge_IIoT = pd.read_csv(filepath + "\\EdgeIIoT_Original\\ML-EdgeIIoT-dataset.csv", low_memory=False, dtype='str')
        # 檢查整個 DataFrame 中每列的數據類型
        print(df_Edge_IIoT.dtypes)
        # 載入轉存str type的dataset
        df_Edge_IIoT=LoadingDataset(df_Edge_IIoT)
        # # True for BaseLine
        # # False for Noniid
        # 載入資料集預處理和正規化
        df_Edge_IIoT=DoMinMaxAndLabelEncoding(df_Edge_IIoT,bool_Noniid)
        # 一般全部特徵
        # DoSpiltAllfeatureAfterMinMax(df_Edge_IIoT,bool_Noniid)
        # 做ChiSquare
        # SelectfeatureUseChiSquareOrPCA(df_Edge_IIoT,True,False,bool_Noniid)
        SelectfeatureUseChiSquareOrPCA(df_Edge_IIoT,"EdgeIIoT",True,False,bool_Noniid)
        # 做PCA
forBaseLineUseData(True)
# forBaseLineUseData(False)