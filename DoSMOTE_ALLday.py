import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import imblearn # Oversample with SMOTE and random undersample for imbalanced dataset
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
# We import seaborn to make nice plots.
import seaborn as sns
from sklearn.manifold import TSNE
from numpy import where
from mytoolfunction import  ChooseLoadNpArray, ChooseTrainDatastes, ParseCommandLineArgs,generatefolder
from mytoolfunction import splitdatasetbalancehalf,SaveDataToCsvfile,SaveDataframeTonpArray


filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
desired_sample_count = 4000
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# k_neighbors = 1  # 调整k_neighbors的值  label 8要設因為樣本只有2個
######################## Choose Dataset ##############################
# 根據參數選擇dataset
# python DoSMOTE_ALLday.py --dataset train_half1
# python DoSMOTE_ALLday.py --dataset total_train
# python DoSMOTE_ALLday.py --dataset train_half1 --epochs 10000 --weaklabel 13
# python DoSMOTE_ALLday.py --dataset train_half1 --method normal
# file, num_epochs, weakLabel= ParseCommandLineArgs(["dataset", "epochs","weaklabel"])
file, num_epochs,Choose_method = ParseCommandLineArgs(["dataset", "epochs", "method"])
print(f"Dataset: {file}")
print(f"Number of epochs: {num_epochs}")
print(f"Choose_method: {Choose_method}")
# ChooseLoadNpArray function  return x_train、y_train 和 client_str and Choose_method
x_train, y_train, client_str = ChooseLoadNpArray(filepath, file, Choose_method)
print(f"client_str: {client_str}")

#feature and label count
# print(x_train.shape[1])
# print(len(np.unique(y_train)))

# 打印原始类别分布
counter = Counter(y_train)
print(counter)

# 生成到原本Train數的原始筆數的50%
sampling_strategy_Label_BENIGN    = {0: 12053}
sampling_strategy_Label_DDoS    = {2: 12053}
sampling_strategy_Label_DoS_GoldenEye = {3: 12000}
sampling_strategy_Label_lDoS_Hulk  = {4: 12000}
sampling_strategy_Label_DoS_Slowhttptest   = {5: 6519} 
sampling_strategy_Label_DoS_slowloris  = {6: 6956}
sampling_strategy_Label_FTP_Patator = {7: 10062}
sampling_strategy_Label_Heartbleed = {8: 11} 
sampling_strategy_Label_Infiltration= {9: 36}
sampling_strategy_LabeSSH_Patator  = {11: 7077}
sampling_strategy_Label_Web_Attack_Brute_Force  = {12: 1809} 
sampling_strategy_Label_Web_Attack_Sql_Injection = {13: 21}
sampling_strategy_Label_Web_Attack_XSS = {14: 783}

def spilttrainhalfAfterSMOTE(X_res,y_res):
    # 使用 pd.DataFrame 將 X_res 和 y_res 水平合併
    column_names = ["principal_Component" + str(i) for i in range(1, 64)] + ["Label"]
    combined_df = pd.DataFrame(np.column_stack((X_res, y_res)), columns=column_names)
    # 找到不包含NaN、Infinity和"inf"值的行
    combined_df = combined_df[~combined_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # combined_df.to_csv("./data/combined_df.csv", index=False)
    # 顯示合併後的 DataFrame
    print(combined_df)
    # split train_dataframes各一半
    train_half1,train_half2 = splitdatasetbalancehalf(combined_df)
    # 找到train_df_half1和train_df_half2中重复的行
    duplicates = train_half2[train_half2.duplicated(keep=False)]

    # 删除train_df_half2中与train_df_half1重复的行
    train_df_half2 = train_half2[~train_half2.duplicated(keep=False)]
    SaveDataToCsvfile(train_half1, f"./ALL_Label/SMOTE/{today}/{client_str}/{file}_AfterSMOTEspilthalf", f"train_half1_AfterSMOTEspilt_{today}")
    SaveDataToCsvfile(train_half2,  f"./ALL_Label/SMOTE/{today}/{client_str}/{file}_AfterSMOTEspilthalf", f"train_half2_AfterSMOTEspilt_{today}") 
    SaveDataframeTonpArray(train_half1, f"./ALL_Label/SMOTE/{today}/{client_str}/{file}_AfterSMOTEspilthalf", "train_half1_AfterSMOTEspilt", today)
    SaveDataframeTonpArray(train_half2, f"./ALL_Label/SMOTE/{today}/{client_str}/{file}_AfterSMOTEspilthalf", "train_half2_AfterSMOTEspilt", today)

def SMOTEParameterSet(choose_strategy, choose_k_neighbors,x_train, y_train, Label_encode,choose_merge_days):
        
        oversample = SMOTE(sampling_strategy = choose_strategy, k_neighbors = choose_k_neighbors, random_state = 42)
        
        X_res, y_res = oversample.fit_resample(x_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # # 獲取原始數據中標籤為 weakLabel 的索引
        Label_indices_Original = np.where(y_train == Label_encode)[0]

        # 獲取原始數據中標籤為 weakLabel 的數據點
        x_train_Label_Oringinal = x_train[Label_indices_Original]

        # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
        Label_indices_SMOTE = np.where(y_res == Label_encode)[0]

        # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
        X_resampled_Label_SMOTE = X_res[Label_indices_SMOTE]

        plt.scatter(x_train_Label_Oringinal[:, 0], 
                x_train_Label_Oringinal[:, 1], 
                c='red', marker='o', s=20, 
                label=f'Original Samples (Label {Label_encode}): {len(x_train_Label_Oringinal)})')
        plt.legend()
        plt.savefig(f"{filepath}/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/ALL_Label/SMOTE/{today}/{client_str}/Label{Label_encode}/SMOTE_Samples_Original_Label_{Label_encode}.png")

        plt.show()
        # 繪制SMOTE採樣後的數據中的標籤 weakLabel
        plt.scatter(X_resampled_Label_SMOTE[:, 0], 
                    X_resampled_Label_SMOTE[:, 1], 
                    c='blue', marker='x', s=36, 
                    label=f'SMOTE Samples (Label {Label_encode}: {len(X_resampled_Label_SMOTE)})')
        # 添加圖例
        plt.legend()
        plt.savefig(f"{filepath}/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/ALL_Label/SMOTE/{today}/{client_str}/Label{Label_encode}/SMOTE_Samples_After_SMOTE_Label_{Label_encode}.png")   
        plt.show()
        return X_res, y_res

def DoALLWeakLabel(x_train,y_train, ChooseLabel, bool_choose_default_k_neighbors,choose_merge_days):
    
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\', client_str)
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', f'{ChooseLabel}')
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label0")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label2")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label3")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label4")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label5")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label6")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label7")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label8")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label9")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label10")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label11")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label12")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label13")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label14")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "ALLday_ALL_Label")

    x_train = x_train.real #去除复數 因為做完統計百分比PCA後會有
    # Assuming y_train contains the labels
    unique_labels = np.unique(y_train)

    # Choose a colormap with at least 15 distinct colors
    cmap = plt.get_cmap('tab20')

    for label in unique_labels:
        row_ix = np.where(y_train == label)[0]
        plt.scatter(x_train[row_ix, 0], x_train[row_ix, 1], label=f'{label}', color=cmap(label))

    plt.legend()
    plt.show()    

    print('Original dataset shape %s' % Counter(y_train))
    y_train = y_train.astype(int) 
     # Start Do SMOTE
    if(bool_choose_default_k_neighbors):# k_neighbors  use default 5
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_BENIGN,5, x_train, y_train, 0,choose_merge_days)
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_DDoS,5, x_train, y_train, 2,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_DDoS,5, X_res, y_res, 2,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_DoS_GoldenEye,5, x_train, y_train, 3,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_lDoS_Hulk, 5, X_res, y_res, 4,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_DoS_Slowhttptest, 5, X_res, y_res, 5,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_DoS_slowloris, 5, x_train, y_train, 6,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_DoS_slowloris, 5, X_res, y_res, 6,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_FTP_Patator, 5, X_res, y_res, 7,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_Heartbleed, 5, X_res, y_res, 8,choose_merge_days)
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_Infiltration,5, X_res, y_res, 9,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_DDoS, 5, X_res, y_res, 10,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_LabeSSH_Patator, 5, X_res, y_res, 11,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_Web_Attack_Brute_Force, 5, X_res, y_res, 12,choose_merge_days)
        #X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_Web_Attack_Sql_Injection, 5, X_res, y_res, 13,choose_merge_days)
        # X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_Web_Attack_XSS, 5, X_res, y_res, 14,choose_merge_days)
        # y_res = y_res.astype(int) 
        # spilttrainhalfAfterSMOTE(X_res,y_res)
    else: 
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_Heartbleed, 2, x_train, y_train, 8,choose_merge_days)
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_Infiltration, 5, X_res, y_res, 9,choose_merge_days)
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label_Web_Attack_Sql_Injection,4, X_res, y_res, 13,choose_merge_days)

  

    print('After SMOTE dataset shape %s' % Counter(y_res)) 
    np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\x_{file}_SMOTE_{ChooseLabel}_{today}.npy", X_res)
    np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\y_{file}_SMOTE_{ChooseLabel}_{today}.npy", y_res)


def DoALL_Label(x_train,y_train):
    generatefolder(filepath, "ALL_Label")
    generatefolder(f'{filepath}' + '\\ALL_Label\\', "SMOTE")
    generatefolder(f'{filepath}' + '\\ALL_Label\\SMOTE\\', today)
    # 对ALL　Label进行SMOTE
    # 打印原始类别分布
    # counter = Counter(y_train)
    # print(counter)
    for i in range(0, 15):# 因為有15個Label
        print(i)
        sampling_strategy_ALL = {i: 10000}
        oversample = SMOTE(sampling_strategy=sampling_strategy_ALL, random_state=42)
        x_train, y_train = oversample.fit_resample(x_train, y_train)
        print("ALL Label SMOTE", Counter(y_train))
    
    np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\x_{file}_SMOTE_ALL_Label.npy", x_train)
    np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\y_{file}_SMOTE_ALL_Label.npy", y_train)


def BorderLineParameterSet(choose_strategy, choosekind, choose_k_neighbors,choose_m_neighbors,x_train, y_train, Label_encode,choose_merge_days):
        
        oversample = BorderlineSMOTE(sampling_strategy = choose_strategy, kind = choosekind, 
                                        k_neighbors = choose_k_neighbors, m_neighbors = choose_m_neighbors,
                                        random_state = 42)
        
        X_res, y_res = oversample.fit_resample(x_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # # 獲取原始數據中標籤為 weakLabel 的索引
        Label_indices_Original = np.where(y_train == Label_encode)[0]

        # 獲取原始數據中標籤為 weakLabel 的數據點
        x_train_Label_Oringinal = x_train[Label_indices_Original]

        # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
        Label_indices_SMOTE = np.where(y_res == Label_encode)[0]

        # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
        X_resampled_Label_SMOTE = X_res[Label_indices_SMOTE]

        plt.scatter(x_train_Label_Oringinal[:, 0], 
                x_train_Label_Oringinal[:, 1], 
                c='red', marker='o', s=20, 
                label=f'Original Samples (Label {Label_encode}): {len(x_train_Label_Oringinal)})')
        plt.legend()
        plt.savefig(f"{filepath}/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/ALL_Label/BorderlineSMOTE/{choosekind}/{today}/{client_str}/Label{Label_encode}/BorederlineSMOTE_{choosekind}_Samples_Original_Label_{Label_encode}.png")
        plt.show()
        # 繪制SMOTE採樣後的數據中的標籤 weakLabel
        plt.scatter(X_resampled_Label_SMOTE[:, 0], 
                    X_resampled_Label_SMOTE[:, 1], 
                    c='blue', marker='x', s=36, 
                    label=f'Borderline SMOTE {choosekind} Samples (Label {Label_encode}: {len(X_resampled_Label_SMOTE)})')
        # 添加圖例
        plt.legend()
        plt.savefig(f"{filepath}/dataset_AfterProcessed/CICIDS2017/{choose_merge_days}/ALL_Label/BorderlineSMOTE/{choosekind}/{today}/{client_str}/Label{Label_encode}/BorederlineSMOTE_{choosekind}_Samples_Label_{Label_encode}.png")
        plt.show()
        return X_res, y_res


def DoBorederlineSMOTE(x_train, y_train,choosekind,ChooseLable,choose_merge_days):
    # 產生存檔分類用資料夾
    # generatefolder(filepath, "ALL_Label")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\', "BorderlineSMOTE")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\', choosekind)
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\', today)
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\', client_str)
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', f'{ChooseLable}')
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label0")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label2")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label3")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label4")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label5")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label6")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label7")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label8")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label9")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label10")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label11")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label12")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label13")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "Label14")
    generatefolder(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\', "ALLday_ALL_Label")

    x_train = x_train.real #去除复數 因為做完統計百分比PCA後會有
    # 將標籤列轉換為整數型別
    y_train = y_train.astype(int)
    # Assuming y_train contains the labels
    unique_labels = np.unique(y_train)

    # Choose a colormap with at least 15 distinct colors
    cmap = plt.get_cmap('tab20')

    for label in unique_labels:
        row_ix = np.where(y_train == label)[0]
        plt.scatter(x_train[row_ix, 0], x_train[row_ix, 1], label=f'{label}', color=cmap(label))

    plt.legend()
    plt.show()
    
    print('Original dataset shape %s' % Counter(y_train))
    # y_res = y_res.astype(int)
    y_train = y_train.astype(int)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_BENIGN, choosekind, 5, 10, x_train, y_train, 0,choose_merge_days)
    X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_DDoS, choosekind, 5, 10, x_train, y_train, 2,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_DDoS, choosekind, 5, 10, X_res, y_res, 2,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_DoS_GoldenEye, choosekind, 5, 10, x_train, y_train, 3,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_lDoS_Hulk, choosekind, 5, 10, X_res, y_res, 4,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_DoS_Slowhttptest, choosekind, 5, 10, X_res, y_res, 5,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_DoS_slowloris, choosekind, 5, 10, X_res, y_res, 6,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_DoS_slowloris, choosekind, 5, 10, x_train, y_train, 6,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_FTP_Patator, choosekind, 5, 10, X_res, y_res, 7,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_Heartbleed, choosekind, 5, 15, X_res, y_res, 8,choose_merge_days)
    X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_Infiltration, choosekind, 5, 10, X_res, y_res, 9,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_DDoS, choosekind, 5, 10, X_res, y_res, 10,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_LabeSSH_Patator, choosekind, 5, 10, X_res, y_res, 11,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_Web_Attack_Brute_Force, choosekind, 5, 10, X_res, y_res, 12,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_Web_Attack_Sql_Injection, choosekind, 5, 30, X_res, y_res, 13,choose_merge_days)
    # X_res, y_res = BorderLineParameterSet(sampling_strategy_Label_Web_Attack_XSS, choosekind, 5, 10, X_res, y_res, 14,choose_merge_days)


    print('Afterr BorderLine SMOTE dataset shape %s' % Counter(y_res))
#    D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2017\ALLday\ALL_Label\BorderlineSMOTE\borderline-1\20240317\client2\ALLday_ALL_Label
    np.save(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\ALLday_ALL_Label\\x_{choosekind}_{today}.npy', X_res)
    np.save(f'{filepath}\\dataset_AfterProcessed\\CICIDS2017\\{choose_merge_days}\\ALL_Label\\BorderlineSMOTE\\{choosekind}\\{today}\\{client_str}\\ALLday_ALL_Label\\y_{choosekind}_{today}.npy', y_res)



 
def scatter(x, colors):  
    # 設置 seaborn 的風格和上下文
    sns.set_style('darkgrid')  
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})  
    
    # 手動指定顏色
    custom_colors = ['red', 'blue', 'green', 'orange', 'purple', 
                     'yellow', 'pink', 'brown', 'cyan', 'magenta', 
                     'black', 'gray', 'olive', 'navy', 'teal']
 
    # 創建散點圖
    f = plt.figure(figsize=(12, 12))  
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20, alpha=0.8,
                    c=[custom_colors[i] for i in colors.astype(np.int)])  # 手動指定顏色
    
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    ax.axis('off')
    ax.axis('tight')
 
    # 為每個數字添加標籤
    txts = []
    for i in range(15):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=12)  
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=2, foreground="w"),  
            PathEffects.Normal()])
        txts.append(txt)
    
    # 創建圖例
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Label ' + str(i),
                          markerfacecolor=custom_colors[i], markersize=10) for i in range(15)]
    # 添加圖例，並調整位置到右上方
    # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    
    # 添加圖例標題
    plt.title('Label Colors')

    return f, ax, sc, txts

def ShowTSNEPicture():
    # 使用 t-SNE 降維並獲取投影
    digits_proj = TSNE(
                        n_components=2,  # 降維後的維度為2
                        verbose=1,       # 打印運行情況
                        n_iter=1000,     # 迭代次數設置為1000
                        perplexity=30,    # Perplexity，用於控制局部關係的參數
                        early_exaggeration=12,  # 控制早期夸大的程度，影響投影的布局
                        learning_rate=200,     # 學習率，控制嵌入空間的距離
                        random_state=42       # 隨機種子以確保可重現性
                        ).fit_transform(x_train)


    # 可視化投影
    scatter(digits_proj, y_train)
    plt.savefig('digits_tsne-generated.png', dpi=120)
    plt.show()

# ShowTSNEPicture()

DoALLWeakLabel(x_train,y_train,"ALLday_ALL_Label", True,"ALLday")
# DoALL_Label(x_train,y_train)
DoBorederlineSMOTE(x_train, y_train,"borderline-1","ALLday_ALL_Label","ALLday")
DoBorederlineSMOTE(x_train, y_train,"borderline-2","ALLday_ALL_Label","ALLday")

############################################################n參數說明#################################################################################
# #一次SMOTE只SMOTE一個weaklabel
# sampling_strategy = {weakLabel: desired_sample_count}   # weakLabel和設置desired_sample_count為希望稱生成的樣本數量
#                                                         # SMOTE進行過抽樣時，所請求的樣本數應不小於原始類別中的樣本數。

# oversample = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
# # 參數說明：
# # ratio：用於指定重抽樣的比例，如果指定字元型的值，可以是'minority'，表示對少數類別的樣本進行抽樣、'majority'，表示對多數類別的樣本進行抽樣、
# # 'not minority'表示採用欠採樣方法、'all'表示採用過採樣方法，
# # 默認為'auto'，等同於'all'和'not minority';如果指定字典型的值，其中鍵為各個類別標籤，值為類別下的樣本量;
# # random_state：用於指定隨機數生成器的種子，預設為None，表示使用預設的隨機數生成器;
# # k_neighbors：指定近鄰個數，預設為5個;
# # m_neighbors：指定從近鄰樣本中隨機挑選的樣本個數，預設為10個;
# # kind：用於指定SMOTE演算法在生成新樣本時所使用的選項，預設為'regular'，表示對少數類別的樣本進行隨機採樣，也可以是'borderline1'、'borderline2'和'svm';
# # svm_estimator：用於指定SVM分類器，預設為sklearn.svm.SVC，該參數的目的是利用支援向量機分類器生成支援向量，然後再生成新的少數類別的樣本;

# oversample = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
