import numpy as np
import pandas as pd
import time
import datetime
from mytoolfunction import generatefolder
from mytoolfunction import clearDirtyData,label_Encoding,splitdatasetbalancehalf,splitweakLabelbalance,SaveDataframeTonpArray,generatefolder
from mytoolfunction import SaveDataToCsvfile,ChooseDataSetNpFile,CheckFileExists,DoReStoreNpFileToCsv,ResotreTrainAndTestToCSVandReSplit
from colorama import Fore, Back, Style, init
# 初始化 colorama（Windows 系統中必須）
init(autoreset=True)

filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")


#########
# TONIOT和EdgeIIOT用baseline的Np file先將轉原Label encode的值轉換調，使用下面這兩個function
# EdgeIIOTNpfileChangeLabelEncodeValue和TONIoTNpfileChangeLabelEncodeValue

# 在用轉換掉Label encode值得npfile去add label，
# 使用DoAddLabelToTrainData
#########


def DoAddLabelToTrainData(Str_ChooseDataset, Int_add_Label_count=None):
    # 載入已有的特徵數據和標籤數據
    if Str_ChooseDataset == "CICIDS2017":
        # 20240323 non iid client1 use cicids2017 ALLday  chi-square_45 change ip encode
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_Resplit_train_20240506.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_Resplit_train_20240506.npy", allow_pickle=True)
        
        # # 20240502 non iid us BaseLine npfile CIC-IDS2017 after do labelencode and minmax chi_square45 75 25分
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLday_train_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLday_train_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)    
        
        # 20250113 CIC-IDS2017 after do labelencode  all featrue minmax 75 25分 do PCA
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_train_AfterPCA79_20250113.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_AfterPCA79_20250113.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_AfterPCA79_20250113_ChangeLabelencode.npy", allow_pickle=True)
   
        # 20250121 CIC-IDS2017 after do labelencode  all featrue minmax 75 25分 do feature drop to 79 feature
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\x_ALLDay_train_Deleted79features_20250121.npy", allow_pickle=True)
        # # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_Deleted79features_20250121.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_AfterDeleted79features_20250121_ChangeLabelencode.npy", allow_pickle=True)
   
        # 20250122 CIC-IDS2017 after do labelencode  all featrue minmax 75 25分 do feature drop to 79 feature DO FGSM esp0.05
        # D:\develop_Federated_Learning_Non_IID_Lab\Adversarial_Attack_Test\CICIDS2017\FGSM_Attack\Npfile
        x_train = np.load(f"./\\Adversarial_Attack_Test\\CICIDS2017\\FGSM_Attack\\Npfile\\x_train_CICIDS2017_eps0.05.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\Adversarial_Attack_Test\\CICIDS2017\\FGSM_Attack\\Npfile\\y_train_CICIDS2017_eps0.05.npy", allow_pickle=True)
        y_train = np.load(f"./\\Adversarial_Attack_Test\\CICIDS2017\\FGSM_Attack\\Npfile\\y_ALLDay_train_AfterFGSM_eps.05_20250122_ChangeLabelencode.npy", allow_pickle=True)
   
        print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\Npfile\\", "Noniid"))
        print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\Npfile\\Noniid\\", today))

        # save_filename = filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\Npfile\\Noniid\\" + today + "\\CICIDS2017_AddedLabel_Noniid"
        
        # do FGSM
        save_filename = filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\Npfile\\Noniid\\" + today + "\\CICIDS2017_AddedLabel_Noniid_FGSM"
        # Add TONIOT CICIDS2019
        # add_Labels = np.array([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34])
        # Add TONIOT EdgeIIoT
        # add_Labels = np.array([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
        # add_Labels = np.array([23,24,25,26,27,28,29,30,31,32,33,34])
        # Add CICIDS2018 CICIDS2019
        add_Labels = np.array([11,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
  

    elif Str_ChooseDataset == "TONIOT":
        # # 20240323 non iid client2 use TONIOT change ts change ip encode
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_TONIOT_train_change_ts_change_ip_20240317.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_TONIOT_train_change_ts_change_ip_20240317.npy", allow_pickle=True)
    
        # x_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/x_TONIOT_test_change_ts_change_ip_20240317.npy")
        # y_test = np.load(f"./data/dataset_AfterProcessed/TONIOT/y_TONIOT_test_change_ts_change_ip_20240317.npy")

        # 20240523 non iid us BaseLine npfile TONIoT after do labelencode and minmax  75 25分
        #因non iid所以 y_train要使用ChangeLabelEncode
        x_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\x_train_ToN-IoT_20240523.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_20240523.npy", allow_pickle=True)  
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_TONIOT_train_After_ChangeLabelEncode_for_Noniid.npy", allow_pickle=True)

        generatefolder(filepath + "\\dataset_AfterProcessed\\TONIOT\\", today)
        save_filename = filepath + "\\dataset_AfterProcessed\\TONIOT\\" + today + "\\TONIIOT_AddedLabel"
        # Add CICIDS2017 CICIDS2019
        # add_Labels = np.array([1,3,4,5,6,7,8,9,10,11,12,13,14,23,24,25,26,27,28,29,30,31,32,33,34])
        # Add CICIDS2019
        # add_Labels = np.array([23,24,25,26,27,28,29,30,31,32,33,34])
        # Add EdgeIIoT  
        add_Labels = np.array([23,24,25,26,27,28,29,30])  
        # Add CICIDS2017 EdgeIIoT
        add_Labels = np.array([1,3,4,5,6,7,8,9,10,11,12,13,14,23,24,25,26,27,28,29,30])


    elif Str_ChooseDataset == "CICIDS2018":

        # 20250121 CIC-IDS2018 after do labelencode and all featrue minmax 75 25分
        print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f" with normal After Do labelencode and minmax")
        x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_train_20250106.npy", allow_pickle=True)    
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_train_20250106.npy", allow_pickle=True)
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_train_After_20250121_ChangeLabelencode.npy", allow_pickle=True)
    
        # # 20250113 CIC-IDS2018 after do labelencode and all featrue minmax 75 25分 Do PCA
        # print(Fore.BLUE+Style.BRIGHT+"Loading CICIDS2018" +f" with normal After Do labelencode and minmax and PCA")
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\x_csv_data_train_AfterPCA79_20250113.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_train_AfterPCA79_20250113.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_train_AfterPCA79_20250113_ChangeLabelencode.npy", allow_pickle=True)
    
        print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\", "Noniid"))
        print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\Noniid\\", today))
        save_filename = filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\Noniid\\" + today + "\\CICIDS2018_AddedLabel_Noniid"
        # Add CICIDS2017 CICIDS2019
        add_Labels = np.array([9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    elif Str_ChooseDataset == "CICIDS2019":
        # 20240506 non iid client3 use CICIDS2019 mimmax 75 25 分
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_Resplit_train_20240506.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_Resplit_train_20240506.npy", allow_pickle=True)
        
        # 20250113 CIC-IDS2019 after do labelencode all featrue minmax 75 25分 Do PCA
        # print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2019 after do labelencode do pca" +f" with normal attack type")
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\x_01_12_train_AfterPCA79_20250113.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\y_01_12_train_AfterPCA79_20250113.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\y_01_12_train_After_ChangeLabelEncode_for_Noniid.npy", allow_pickle=True)
    
        # 20250121 01-12 and 03-11 merge ALLDay
        #  CIC-IDS2019 after do labelencode all featrue minmax 75 25分 Do feature drop to 79 feature
        print(Fore.GREEN+Style.BRIGHT+"Loading CICIDS2019 after do labelencode Do feature drop" +f" with normal attack type")
        x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\x_ALLDay_train_Deleted79features_20250120.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\y_ALLDay_train_Deleted79features_20250120.npy", allow_pickle=True)
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\y_ALLDay_train_After_ChangeLabelEncode_for_Noniid.npy", allow_pickle=True)
    

        # 20250121 01-12
        # print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\", "Noniid"))
        # print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\Noniid\\", today))
        # save_filename = filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\Noniid\\" + today + "\\CICIDS2019_AddedLabel_Noniid"
       
        # 20250121 01-12 and 03-11 merge ALLDay
        print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\", "Noniid"))
        print(generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\Noniid\\", today))     
        save_filename = filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\Noniid\\" + today + "\\CICIDS2019_AddedLabel_Noniid"
        # Add CICIDS2017 CICIDS2018
        add_Labels = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

    elif Str_ChooseDataset == "EdgeIIOT":
        # 20240519 EdgeIIoT after do labelencode and minmax  75 25分
        # x_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_train_20240519.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_20240519.npy", allow_pickle=True)    
    
        # x_test = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_test_20240519.npy", allow_pickle=True)
        # y_test = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_test_20240519.npy", allow_pickle=True)
   
        # 20240520  non iid client3 use EdgeIIoT after do labelencode and minmax chi_square45 75 25分
        #因non iid所以 y_train要使用ChangeLabelEncode
        x_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\x_EdgeIIoT_train_AfterFeatureSelect44_20240520.npy", allow_pickle=True)
        # y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_AfterFeatureSelect44_20240520.npy", allow_pickle=True)    
        y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_After_ChangeLabelEncode_for_Noniid.npy", allow_pickle=True)


        print(generatefolder(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\", today))
        save_filename = filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\" + today + "\\EdgeIIoT_AddedLabel"
        # Add TONIOT CICIDS2017
        add_Labels = np.array([1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,21])

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
    if Str_ChooseDataset == "TONIOT" or Str_ChooseDataset == "EdgeIIOT":
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

# ResotreTrainAndTestToCSVandReSplit("CICIDS2017",filepath)
# ResotreTrainAndTestToCSVandReSplit("CICIDS2019",filepath)

#
# DoAddLabelToTrainData("CICIDS2017")
# DoAddLabelToTrainData("CICIDS2019")
#
# DoAddLabelToTrainData("TONIOT")
#
# DoAddLabelToTrainData("CICIDS2019",4)

# DoAddLabelToTrainData("EdgeIIOT")

def EdgeIIOTNpfileChangeLabelEncodeValue():
    y_train = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_AfterFeatureSelect44_20240520.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_test_AfterFeatureSelect44_20240520.npy", allow_pickle=True)
    # EdgeIIOT将 y_train和y_test進行Label encode值替換 for Noniid實驗
    # 0: 'BENIGN',  				改encode		'BENIGN': 0,
    # 1: 'DDoS_HTTP', 				改encode		'DDoS_HTTP': 28, 
    # 2: 'DDoS_ICMP', 				改encode		'DDoS_ICMP': 24, 
    # 3: 'DDoS_TCP',				改encode		'DDoS_TCP': 27,
    # 4: 'DDoS_UDP', 				改encode		'DDoS_UDP': 23,
    # 5: 'Fingerprinting', 			改encode		'Fingerprinting': 30
    # 6: 'PortScan', 				改encode		'PortScan': 10, 
    # 7: 'SQL_injection', 			改encode		'SQL_injection': 25,
    # 8: 'Uploading', 				改encode		'Uploading': 29,
    # 9: 'Vulnerability_scanner',	改encode 	    'Vulnerability_scanner': 26,
    # 10: 'backdoor', 				改encode		'backdoor': 15, 
    # 11: 'mitm',					改encode		'mitm': 18,
    # 12: 'password',				改encode		'password': 19,
    # 13: 'ransomware', 			改encode		'ransomware': 20,
    # 14: 'xss'						改encode		'xss': 22, 

    y_train_After_Change = np.where(y_train == 1, 28,
                                    np.where(y_train == 2, 24,
                                    np.where(y_train == 3, 27,
                                    np.where(y_train == 4, 23,
                                    np.where(y_train == 5, 30,
                                    np.where(y_train == 6, 10,
                                    np.where(y_train == 7, 25,
                                    np.where(y_train == 8, 29,
                                    np.where(y_train == 9, 26,
                                    np.where(y_train == 10, 15,
                                    np.where(y_train == 11, 18,
                                    np.where(y_train == 12, 19,
                                    np.where(y_train == 13, 20,
                                    np.where(y_train == 14, 22,
                                    y_train))))))))))))))
    
    y_test_After_Change = np.where(y_test == 1, 28,
                                    np.where(y_test == 2, 24,
                                    np.where(y_test == 3, 27,
                                    np.where(y_test == 4, 23,
                                    np.where(y_test == 5, 30,
                                    np.where(y_test == 6, 10,
                                    np.where(y_test == 7, 25,
                                    np.where(y_test == 8, 29,
                                    np.where(y_test == 9, 26,
                                    np.where(y_test == 10, 15,
                                    np.where(y_test == 11, 18,
                                    np.where(y_test == 12, 19,
                                    np.where(y_test == 13, 20,
                                    np.where(y_test == 14, 22,
                                    y_test))))))))))))))

    #np.save
    np.save(f"{filepath}\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_train_After_ChangeLabelEncode_for_Noniid.npy", y_train_After_Change)
    np.save(f"{filepath}\\dataset_AfterProcessed\\EdgeIIoT\\y_EdgeIIoT_test_After_ChangeLabelEncode_for_Noniid.npy", y_test_After_Change)

    y_train_unique_values = np.unique(y_train_After_Change)
    print("唯一值数量:", len(y_train_unique_values))
    print("唯一值:", y_train_unique_values)
    y_test_unique_values = np.unique(y_test_After_Change)
    print("唯一值数量:", len(y_test_unique_values))
    print("唯一值:", y_test_unique_values)

def TONIoTNpfileChangeLabelEncodeValue():
    y_train = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_train_ToN-IoT_20240523.npy", allow_pickle=True)  
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\TONIOT\\y_test_ToN-IoT_20240523.npy", allow_pickle=True)   

    #  0: 'BENIGN',           改encode	    0: 'BENIGN', 
    #  1: 'DDoS',             改encode		2: 'DDoS', 
    #  2: 'backdoor',         改encode		15: 'backdoor', 
    #  3: 'dos',              改encode		16: 'dos', 
    #  4: 'injection',        改encode		17: 'injection', 
    #  5: 'mitm',             改encode		18: 'mitm', 
    #  6: 'password',         改encode		19: 'password', 
    #  7: 'ransomware',       改encode		20: 'ransomware',
    #  8: 'scanning',         改encode		21: 'scanning', 
    #  9: 'xss',              改encode 	    22: 'xss', 

    y_train_After_Change = np.where(y_train == 1, 2,
                                    np.where(y_train == 2, 15,
                                    np.where(y_train == 3, 16,
                                    np.where(y_train == 4, 17,
                                    np.where(y_train == 5, 18,
                                    np.where(y_train == 6, 19,
                                    np.where(y_train == 7, 20,
                                    np.where(y_train == 8, 21,
                                    np.where(y_train == 9, 22,
                                    y_train)))))))))
    
    y_test_After_Change = np.where(y_test == 1, 2,
									np.where(y_test == 2, 15,
                                    np.where(y_test == 3, 16,
                                    np.where(y_test == 4, 17,
                                    np.where(y_test == 5, 18,
                                    np.where(y_test == 6, 19,
                                    np.where(y_test == 7, 20,
                                    np.where(y_test == 8, 21,
                                    np.where(y_test == 9, 22,
                                    y_test)))))))))

    #np.save
    np.save(f"{filepath}\\dataset_AfterProcessed\\TONIOT\\y_TONIOT_train_After_ChangeLabelEncode_for_Noniid.npy", y_train_After_Change)
    np.save(f"{filepath}\\dataset_AfterProcessed\\TONIOT\\y_TONIOT_test_After_ChangeLabelEncode_for_Noniid.npy", y_test_After_Change)

    y_train_unique_values = np.unique(y_train_After_Change)
    print("唯一值数量:", len(y_train_unique_values))
    print("唯一值:", y_train_unique_values)
    y_test_unique_values = np.unique(y_test_After_Change)
    print("唯一值数量:", len(y_test_unique_values))
    print("唯一值:", y_test_unique_values)


def CICIDS2017NpfileChangeLabelEncodeValue():
    # 20250113 CIC-IDS2017 after do labelencode and except str and PCA all featrue minmax 75 25分
    # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_AfterPCA79_20250113.npy", allow_pickle=True)
    # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_AfterPCA79_20250113.npy", allow_pickle=True)
    
    # 20250121 CIC-IDS2017 after do labelencode and all featrue minmax 75 25分 DropFeature to 79 featrue
    # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_Deleted79features_20250121.npy", allow_pickle=True)
    # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_Deleted79features_20250121.npy", allow_pickle=True)
    
    # 20250122 CIC-IDS2017 after do labelencode  all featrue minmax 75 25分 do feature drop to 79 feature DO FGSM esp0.05
    y_train = np.load(filepath + "\\Adversarial_Attack_Test\\CICIDS2017\\FGSM_Attack\\Npfile\\y_train_CICIDS2017_eps0.05.npy", allow_pickle=True)

    # CICIDS2017将 y_train和y_test進行Label encode值替換 for Noniid實驗
    # 0: BENIGN                        改encode	BENIGN              0	
    # 1: Bot                           改encode	Bot                 1	
    # 2: DDoS                          改encode	DDoS                2	
    # 3: DoS GoldenEye                 改encode	DoS GoldenEye       3	
    # 4: DoS Hulk                      改encode	DoS Hulk            4	
    # 5: DoS Slowhttptest              改encode	DoS Slowhttptest    5	
    # 6: DoS slowloris                 改encode	DoS slowloris       6	
    # 7: FTP-Patator                   改encode	Infilteration		7	
    # 8: Heartbleed                    改encode	Web Attack			8	
    # 9: Infiltration                  改encode Heartbleed          9	
    # 10:PortScan                      改encode	PortScan            10	
    # 11:SSH-Patator                   改encode	FTP-BruteForce		11	
    # 12:Web Attack Brute Force        改encode	FTP-Patator         12	
    # 13:Web Attack Sql Injection      改encode	SSH-Bruteforce		13	
    # 14:Web Attack XSS                改encode	SSH-Patator         14	
                                                # DrDoS_DNS			15
                                                # DrDoS_LDAP		16
                                                # DrDoS_MSSQL		17
                                                # DrDoS_NTP			18
                                                # DrDoS_NetBIOS		19
                                                # DrDoS_SNMP		20
                                                # DrDoS_SSDP		21
                                                # DrDoS_UDP			22
                                                # Syn				23
                                                # TFTP				24
                                                # UDPlag			25
                                                # WebDDoS			26

    y_train_After_Change = np.where(y_train == 1, 1,
                                    np.where(y_train == 2, 2,
                                    np.where(y_train == 3, 3,
                                    np.where(y_train == 4, 4,
                                    np.where(y_train == 5, 5,
                                    np.where(y_train == 6, 6,
                                    np.where(y_train == 7, 12,
                                    np.where(y_train == 8, 9,
                                    np.where(y_train == 9, 7,
                                    np.where(y_train == 10, 10,
                                    np.where(y_train == 11, 14,
                                    np.where(y_train == 12, 8,
                                    np.where(y_train == 13, 8,
                                    np.where(y_train == 14, 8,
                                    y_train))))))))))))))
    
    # y_test_After_Change = np.where(y_test == 1, 1,
    #                                 np.where(y_test == 2, 2,
    #                                 np.where(y_test == 3, 3,
    #                                 np.where(y_test == 4, 4,
    #                                 np.where(y_test == 5, 5,
    #                                 np.where(y_test == 6, 6,
    #                                 np.where(y_test == 7, 12,
    #                                 np.where(y_test == 8, 9,
    #                                 np.where(y_test == 9, 7,
    #                                 np.where(y_test == 10, 10,
    #                                 np.where(y_test == 11, 14,
    #                                 np.where(y_test == 12, 8,
    #                                 np.where(y_test == 13, 8,
    #                                 np.where(y_test == 14, 8,
    #                                 y_test))))))))))))))

    #np.savey_ALLDay_train_AfterPCA79_20250113
    # y_ALLDay_test_AfterPCA79_20250113
    # np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_AfterPCA79_20250113_ChangeLabelencode.npy", y_train_After_Change)
    # np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_AfterPCA79_20250113_ChangeLabelencode.npy", y_test_After_Change)

    #np.savey_ALLDay_train_After do feature drop to 79 feature
    # np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_AfterDeleted79features_20250121_ChangeLabelencode.npy", y_train_After_Change)
    # np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_test_AfterDeleted79features_20250121_ChangeLabelencode.npy", y_test_After_Change)

    #np.savey_ALLDay_train_After do feature drop to 79 feature FGSM eps 0.5
    np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\Npfile\\y_ALLDay_train_AfterFGSM_eps.05_20250122_ChangeLabelencode.npy", y_train_After_Change)

    y_train_unique_values = np.unique(y_train_After_Change)
    print("唯一值数量:", len(y_train_unique_values))
    print("唯一值:", y_train_unique_values)
    # y_test_unique_values = np.unique(y_test_After_Change)
    # print("唯一值数量:", len(y_test_unique_values))
    # print("唯一值:", y_test_unique_values)


def CICIDS2018NpfileChangeLabelEncodeValue():
    # 20250121 CIC-IDS2018 after do labelencode and all featrue minmax 75 25分
    y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_train_20250106.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_20250106.npy", allow_pickle=True)
    
    # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_train_AfterPCA79_20250113.npy", allow_pickle=True)
    # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_AfterPCA79_20250113.npy", allow_pickle=True)
    
    # EdgeIIOT将 y_train和y_test進行Label encode值替換 for Noniid實驗
    # 0: BENIGN                        改encode	BENIGN              0	
    # 1: Bot                           改encode	Bot                 1	
    # 2: Brute Force -Web              改encode	DDoS                2	
    # 3: Brute Force -XSS              改encode	DoS GoldenEye       3	
    # 4: DDOS attack-HOIC              改encode	DoS Hulk            4	
    # 5: DDOS attack-LOIC-UDP          改encode	DoS Slowhttptest    5	
    # 6: DDoS attacks-LOIC-HTTP        改encode	DoS slowloris       6	
    # 7: DoS attacks-GoldenEye         改encode	Infilteration		7	
    # 8: DoS attacks-Hulk              改encode	Web Attack			8	
    # 9: DoS attacks-SlowHTTPTest      改encode Heartbleed          9	
    # 10:DoS attacks-Slowloris         改encode	PortScan            10	
    # 11:FTP-BruteForce                改encode	FTP-BruteForce		11	
    # 12:Infilteration                 改encode	FTP-Patator         12	
    # 13:SQL Injection                 改encode	SSH-Bruteforce		13	
    # 14:SSH-Bruteforce                改encode	SSH-Patator         14	
                                                # DrDoS_DNS			15
                                                # DrDoS_LDAP		16
                                                # DrDoS_MSSQL		17
                                                # DrDoS_NTP			18
                                                # DrDoS_NetBIOS		19
                                                # DrDoS_SNMP		20
                                                # DrDoS_SSDP		21
                                                # DrDoS_UDP			22
                                                # Syn				23
                                                # TFTP				24
                                                # UDPlag			25
                                                # WebDDoS			26

    y_train_After_Change = np.where(y_train == 1, 1,
                                    np.where(y_train == 2, 8,
                                    np.where(y_train == 3, 8,
                                    np.where(y_train == 4, 2,
                                    np.where(y_train == 5, 2,
                                    np.where(y_train == 6, 2,
                                    np.where(y_train == 7, 3,
                                    np.where(y_train == 8, 4,
                                    np.where(y_train == 9, 5,
                                    np.where(y_train == 10,6,
                                    np.where(y_train == 11,11,
                                    np.where(y_train == 12,7,
                                    np.where(y_train == 13,8,
                                    np.where(y_train == 14,13,
                                    y_train))))))))))))))
    
    y_test_After_Change = np.where(y_test == 1, 1,
                                    np.where(y_test == 2, 8,
                                    np.where(y_test == 3, 8,
                                    np.where(y_test == 4, 2,
                                    np.where(y_test == 5, 2,
                                    np.where(y_test == 6, 2,
                                    np.where(y_test == 7, 3,
                                    np.where(y_test == 8, 4,
                                    np.where(y_test == 9, 5,
                                    np.where(y_test == 10,6,
                                    np.where(y_test == 11,11,
                                    np.where(y_test == 12,7,
                                    np.where(y_test == 13,8,
                                    np.where(y_test == 14,13,
                                    y_test))))))))))))))

    #np.savey_csv_data_train_AfterDo minmax_20250121
    np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_train_After_20250121_ChangeLabelencode.npy", y_train_After_Change)
    np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_After_20250121_ChangeLabelencode.npy", y_test_After_Change)

    #np.savey_csv_data_train_AfterPCA79_20250113
    # np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_train_AfterPCA79_20250113_ChangeLabelencode.npy", y_train_After_Change)
    # np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2018\\Npfile\\y_csv_data_test_AfterPCA79_20250113_ChangeLabelencode.npy", y_test_After_Change)

    y_train_unique_values = np.unique(y_train_After_Change)
    print("唯一值数量:", len(y_train_unique_values))
    print("唯一值:", y_train_unique_values)
    y_test_unique_values = np.unique(y_test_After_Change)
    print("唯一值数量:", len(y_test_unique_values))
    print("唯一值:", y_test_unique_values)

def CICIDS2019NpfileChangeLabelEncodeValue():
    # y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\y_01_12_train_AfterPCA79_20250113.npy", allow_pickle=True)
    # y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\y_01_12_test_AfterPCA79_20250113.npy", allow_pickle=True)
    # CICIDS2019 79 feature
    y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\y_ALLDay_train_Deleted79features_20250120.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\y_ALLDay_test_Deleted79features_20250120.npy", allow_pickle=True)

    # CICIDS2019 y_train和y_test進行Label encode值替換 for Noniid實驗
	# 0: BENIGN                     改encode	BENIGN              0
	# 1: DrDoS_DNS                  改encode	Bot                 1
	# 2: DrDoS_LDAP                 改encode	DDoS                2
	# 3: DrDoS_MSSQL                改encode	DoS GoldenEye       3
	# 4: DrDoS_NTP                  改encode	DoS Hulk            4
	# 5: DrDoS_NetBIOS              改encode	DoS Slowhttptest    5
	# 6: DrDoS_SNMP                 改encode	DoS slowloris       6
	# 7: DrDoS_SSDP                 改encode	Infilteration		7
	# 8: DrDoS_UDP                  改encode	Web Attack			8
	# 9: Syn                        改encode 	Heartbleed          9
	# 10:TFTP                       改encode	PortScan            10
	# 11:UDPlag                     改encode	FTP-BruteForce		11
	# 12:WebDDoS                    改encode	FTP-Patator         12
									# 改encode	SSH-Bruteforce		13
									# 改encode	SSH-Patator         14
											    # DrDoS_DNS	        15
                                                # DrDoS_LDAP	    16
                                                # DrDoS_MSSQL	    17
                                                # DrDoS_NTP	        18
                                                # DrDoS_NetBIOS	    19
                                                # DrDoS_SNMP	    20
                                                # DrDoS_SSDP	    21
                                                # DrDoS_UDP	        22
                                                # LDAP	            23
                                                # MSSQL	            24
                                                # NetBIOS	        25
                                                # Portmap	        26
                                                # Syn	            27
                                                # TFTP	            28
                                                # UDP	            29
                                                # UDPLag	        30
                                                # WebDDoS	        31


    y_train_After_Change = np.where(y_train == 1, 15,
                                    np.where(y_train == 2, 16,
                                    np.where(y_train == 3, 17,
                                    np.where(y_train == 4, 18,
                                    np.where(y_train == 5, 19,
                                    np.where(y_train == 6, 20,
                                    np.where(y_train == 7, 21,
                                    np.where(y_train == 8, 22,
                                    np.where(y_train == 9, 23,
                                    np.where(y_train == 10,24,
                                    np.where(y_train == 11,25,
                                    np.where(y_train == 12,26,
                                    np.where(y_train == 13,27,
                                    np.where(y_train == 14,28,
                                    np.where(y_train == 15,29,
                                    np.where(y_train == 16,30,
                                    np.where(y_train == 17,31,        
                                    y_train)))))))))))))))))
    
    y_test_After_Change = np.where(y_test == 1, 15,
                                    np.where(y_test == 2, 16,
                                    np.where(y_test == 3, 17,
                                    np.where(y_test == 4, 18,
                                    np.where(y_test == 5, 19,
                                    np.where(y_test == 6, 20,
                                    np.where(y_test == 7, 21,
                                    np.where(y_test == 8, 22,
                                    np.where(y_test == 9, 23,
                                    np.where(y_test == 10,24,
                                    np.where(y_test == 11,25,
                                    np.where(y_test == 12,26,
                                    np.where(y_test == 13,27,
                                    np.where(y_test == 14,28,
                                    np.where(y_test == 15,29,
                                    np.where(y_test == 16,30,
                                    np.where(y_test == 17,31, 
                                    y_test)))))))))))))))))

    #np.save
    # 01-12
    # np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\y_01_12_train_After_ChangeLabelEncode_for_Noniid.npy", y_train_After_Change)
    # np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2019\\01_12\\Npfile\\y_01_12_test_After_ChangeLabelEncode_for_Noniid.npy", y_test_After_Change)
    # 01-12 and 03-11 ALLDay
    np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\Noniid\\y_ALLDay_train_After_ChangeLabelEncode_for_Noniid.npy", y_train_After_Change)
    np.save(f"{filepath}\\dataset_AfterProcessed\\CICIDS2019\\ALLDay\\Npfile\\Noniid\\y_ALLDay_test_After_ChangeLabelEncode_for_Noniid.npy", y_test_After_Change)


    y_train_unique_values = np.unique(y_train_After_Change)
    print("唯一值数量:", len(y_train_unique_values))
    print("唯一值:", y_train_unique_values)
    y_test_unique_values = np.unique(y_test_After_Change)
    print("唯一值数量:", len(y_test_unique_values))
    print("唯一值:", y_test_unique_values)
# EdgeIIOTNpfileChangeLabelEncodeValue()
# TONIoTNpfileChangeLabelEncodeValue()

# CICIDS2017NpfileChangeLabelEncodeValue()
# CICIDS2018NpfileChangeLabelEncodeValue()
# CICIDS2019NpfileChangeLabelEncodeValue()
DoAddLabelToTrainData("CICIDS2017")
# DoAddLabelToTrainData("CICIDS2018")
# DoAddLabelToTrainData("CICIDS2019")