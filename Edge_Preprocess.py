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
from mytoolfunction import SaveDataToCsvfile,ChooseDataSetNpFile,CheckFileExists,DoReStoreNpFileToCsv,ResotreTrainAndTestToCSVandReSplit

#############################################################################  variable  ###################
# filepath = "D:\\Labtest20230911\\data"
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")

generatefolder(filepath + "\\dataset_AfterProcessed\\", "Edge")
generatefolder(filepath + "\\dataset_AfterProcessed\\Edge\\", today)

ResotreTrainAndTestToCSVandReSplit("Edge",filepath)

def DoSpilthalfForiid(choose_merge_days):
    if choose_merge_days == "Edge":
        df_ALLtrain = pd.read_csv(filepath + "\\dataset_AfterProcessed\\Edge\\20240507\\Resplit_train_dataframes_20240507.csv")
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

DoSpilthalfForiid("Edge")