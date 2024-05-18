import numpy as np
import pandas as pd
import time
import datetime
from mytoolfunction import generatefolder
from mytoolfunction import clearDirtyData,label_Encoding,splitdatasetbalancehalf,spiltweakLabelbalance,SaveDataframeTonpArray,generatefolder
from mytoolfunction import SaveDataToCsvfile,ChooseDataSetNpFile,CheckFileExists,DoReStoreNpFileToCsv,ResotreTrainAndTestToCSVandReSplit


filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")


def DoAddLabelToTrainData(Str_ChooseDataset, Int_add_Label_count=None):
    # 載入已有的特徵數據和標籤數據
    if Str_ChooseDataset == "CICIDS2017":
        # 20240323 non iid client1 use cicids2017 ALLday  chi-square_45 change ip encode
        x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_Resplit_train_20240506.npy", allow_pickle=True)
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_Resplit_train_20240506.npy", allow_pickle=True)
        
        generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\", today)
        save_filename = filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\" + today + "\\CICIDS2017_AddedLabel"
        add_Labels = np.array([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34])
        # add_Labels = np.array([23,24,25,26,27,28,29,30,31,32,33,34])
  

    elif Str_ChooseDataset == "TONIOT":
        # # 20240323 non iid client2 use TONIOT change ts change ip encode
        x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_TONIOT_train_change_ts_change_ip_20240317.npy", allow_pickle=True)
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_TONIOT_train_change_ts_change_ip_20240317.npy", allow_pickle=True)
    
        x_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/x_TONIOT_test_change_ts_change_ip_20240317.npy")
        y_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/y_TONIOT_test_change_ts_change_ip_20240317.npy")


        generatefolder(filepath + "\\dataset_AfterProcessed\\TONIOT\\", today)
        save_filename = filepath + "\\dataset_AfterProcessed\\TONIOT\\" + today + "\\TONIIOT_AddedLabel"
        # add_Labels = np.array([1,3,4,5,6,7,8,9,10,11,12,13,14,23,24,25,26,27,28,29,30,31,32,33,34])
        add_Labels = np.array([23,24,25,26,27,28,29,30,31,32,33,34])  


    elif Str_ChooseDataset == "CICIDS2019":

        x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_Resplit_train_20240506.npy", allow_pickle=True)
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_Resplit_train_20240506.npy", allow_pickle=True)

   
        print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\", today))
        save_filename = filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\" + today + "\\CICIDS2019_AddedLabel"
        add_Labels = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])

    
    if Int_add_Label_count != None:
        # 二維寫法如下範例 注意括號
        # e.g:
        # add_Labels  = np.array([[50, 51, 52, 53]])
        # 一維寫法如下範例 注意括號
        # 創建新的Label資料 np.array中寫要新增的Label encode編號
        # e.g:
        # 處理當Int_add_Label_count當入參的情況
        # 賦值給add_Labels比免填空值
        add_Labels = np.array([50, 51, 52, 53])  
    else:
    # if Int_add_Label_count is None:
        # 處理當Int_add_Label_count沒當入參的情況
        # 直接讀選擇資料集要插入的add_Labels長度
        Int_add_Label_count = len(add_Labels)
        print("No additional label count provided.")

    ###########打印選擇資料吉和相關資訊###########
    print("ChooseDataset為:", Str_ChooseDataset)
    print("add_Labels 的個數為:", Int_add_Label_count)
    
    ###########打印原本x_train相關資訊###########
    print("x 的形狀為", x_train.shape)
    print("x 的維度為", x_train.ndim)
    print("x 包含了", x_train.shape[0], "行和", x_train.shape[1], "列的", x_train.ndim, "維數組")
    print("x 數組", x_train.shape[0], "個樣本，每個樣本有", x_train.shape[1], "個特徵")
    print("x_train 的形狀:", y_train.shape)
    ###########打印原本y_train相關資訊###########
    print("y 的形狀為", y_train.shape)
    print("y 的維度為", y_train.ndim)
    print("y 包含了", y_train.shape[0], "個元素(標籤)")
    print("y_train 的形狀:", y_train.shape)



    # 要新增的Label數量
    # Int_add_Label_count = 4
    # x_train.shape[1]是特徵數量
    # 使用np.zero將要新增的Label的特徵補0
    add_feature = np.zeros((Int_add_Label_count, x_train.shape[1]))

    # TONIOT的第43個欄位 攻擊都是1 normal是0
    if Str_ChooseDataset == "TONIOT":
        # 將特定列填充為1，但排除Label encode值等於0的情況
        column_index = 43  # 要插入的index欄位，是第43列
        for i in range(Int_add_Label_count):
            if add_Labels[i] != 0:  # 排除Label encode值等於0的情況
                add_feature[i, column_index] = 1
    
    # 使用垂直堆疊將新的特徵數據補零後追加到已有特徵數據的末尾
    x_Added = np.vstack((x_train, add_feature))


    print("add_Labels 的形狀:", add_Labels.shape)
    # 將新的標籤數據添加到 y_train 的末尾
    y_Added = np.hstack((y_train, add_Labels))
    print("新增add_Labels後y_train的形狀:", y_Added.shape)
    # 將新的特徵數據和標籤數據保存到新的文件中
    np.save(f'{save_filename}_x.npy', x_Added)
    np.save(f'{save_filename}_y.npy', y_Added)
    print("新增Label後x_train的形狀為", x_Added.shape)
    print("新增Labely_train的形狀為", y_Added.shape)


    # 將特徵數據和標籤數據合併成一個 DataFrame
    columns_x = [f'feature_{i}' for i in range(x_train.shape[1])]
    df_x = pd.DataFrame(x_Added, columns=columns_x)
    df_y = pd.DataFrame(y_Added, columns=['Label'])
    # 合併 x 和 y DataFrame
    df_combined = pd.concat([df_x, df_y], axis=1)
    # 將 DataFrame 轉換為 CSV 文件並保存
    df_combined.to_csv(f'{save_filename}.csv', index=False)


# mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\CICIDS2017_original.csv")

ResotreTrainAndTestToCSVandReSplit("CICIDS2017",filepath)
# ResotreTrainAndTestToCSVandReSplit("CICIDS2019",filepath)

#
# DoAddLabelToTrainData("CICIDS2017")
# DoAddLabelToTrainData("CICIDS2019")
#
# DoAddLabelToTrainData("TONIOT")
#
# DoAddLabelToTrainData("CICIDS2019",4)