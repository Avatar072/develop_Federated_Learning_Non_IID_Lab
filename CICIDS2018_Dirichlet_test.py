import numpy as np
import pandas as pd
import datetime
import time
from mytoolfunction import generatefolder,SaveDataframeTonpArray


# 設定檔案路徑（請根據實際情況調整）
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
csv_path = filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\20250106\\csv_data_train_dataframes_20250106.csv"
# D:\develop_Federated_Learning_Non_IID_Lab\data\dataset_AfterProcessed\CICIDS2018\csv_data\20250106
today = datetime.date.today()
today = today.strftime("%Y%m%d")
current_time = time.strftime("%Hh%Mm%Ss", time.localtime())
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
generatefolder(filepath + "\\", "dataset_AfterProcessed")
generatefolder(filepath + "\\dataset_AfterProcessed\\", "CICIDS2018")
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\", today)
generatefolder(filepath + f"\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\{today}\\", current_time)


# 讀取 CSV 檔案
df_ALLtrain = pd.read_csv(csv_path)

# 依照 Label 分群，計算每個 Label 的筆數
df_grouped = df_ALLtrain.groupby("Label").size().reset_index(name="count")
print("原始資料各 Label 筆數：")
print(df_grouped)

# 設定 Dirichlet 分布的 alpha 值
# 20250205
alpha = 0.5
# alpha = 0.1
# alpha = 5.0

# 用來儲存分配後，各 client 的資料索引
client_indices = {"Client1": [], "Client2": [], "Client3": []}

# 對每個 Label 進行獨立的 Dirichlet 分配
for label, group in df_ALLtrain.groupby("Label"):
    # 取得該 Label 的所有資料索引
    indices = group.index.tolist()
    n_samples = len(indices)
    # 產生兩個權重，這兩個權重的和為 1
    np.random.seed(42)  # 使用任意固定數字作為種子 # 每次執行時得到完全相同的分配結果
    weights = np.random.dirichlet([alpha, alpha, alpha])
    print("權重0:",weights[0])
    print("權重1:",weights[1])
    print("權重2:",weights[2])


# 根據權重計算分配給每個客戶端的筆數
    n_client1 = int(np.round(weights[0] * n_samples))
    n_client2 = int(np.round(weights[1] * n_samples))
    
    # 確保分配的總數不超過樣本總數
    # 避免由於四捨五入導致分配總數超過樣本總數
    if n_client1 + n_client2 > n_samples:
        # 如果超過，按比例縮減
        excess = (n_client1 + n_client2) - n_samples
        # 根據權重比例縮減
        if weights[0] >= weights[1]:
            n_client1 -= excess
        else:
            n_client2 -= excess
    
    # 客戶端3取剩餘所有樣本，確保總數正確
    n_client3 = n_samples - n_client1 - n_client2
    
    # 隨機打亂這個 Label 的資料索引
    np.random.shuffle(indices)
    
    # 分別存入兩個 client 的索引列表中
    client_indices["Client1"].extend(indices[:n_client1])
    client_indices["Client2"].extend(indices[n_client1:n_client1+n_client2])
    client_indices["Client3"].extend(indices[n_client1+n_client2:])

    
    print(f"Label: {label}, 總筆數: {n_samples}, 權重: {weights}, Client1: {n_client1}, Client2: {n_client2}, Client3: {n_client3}")

# 根據分配好的索引，從原始 DataFrame 取出各 client 的資料
df_client1 = df_ALLtrain.loc[client_indices["Client1"]].reset_index(drop=True)
df_client2 = df_ALLtrain.loc[client_indices["Client2"]].reset_index(drop=True)
df_client3 = df_ALLtrain.loc[client_indices["Client3"]].reset_index(drop=True)


# 檢查分配後，每個 client 各 Label 的分佈
print("\nClient1 各 Label 筆數：")
print(df_client1['Label'].value_counts())
print(sum(df_client1['Label'].value_counts()))

print("\nClient2 各 Label 筆數：")
print(df_client2['Label'].value_counts())
print(sum(df_client2['Label'].value_counts()))

print("\nClient3 各 Label 筆數：")
print(df_client3['Label'].value_counts())
print(sum(df_client3['Label'].value_counts()))

# filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
df_client1.to_csv(filepath + f"\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\{today}\\{current_time}\\client1.csv", index=False)
df_client2.to_csv(filepath + f"\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\{today}\\{current_time}\\client2.csv", index=False)
df_client3.to_csv(filepath + f"\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\{today}\\{current_time}\\client3.csv", index=False)

SaveDataframeTonpArray(df_client1, f"{filepath}\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\{today}\\{current_time}\\", "Dirichlet_client1",today)
SaveDataframeTonpArray(df_client2, f"{filepath}\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\{today}\\{current_time}\\", "Dirichlet_client2",today)
SaveDataframeTonpArray(df_client3, f"{filepath}\\dataset_AfterProcessed\\CICIDS2018\\csv_data\\Dirichlet\\{today}\\{current_time}\\", "Dirichlet_client3",today)