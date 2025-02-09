import numpy as np
import pandas as pd
import datetime
from mytoolfunction import generatefolder,SaveDataframeTonpArray


# 設定檔案路徑（請根據實際情況調整）
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
csv_path = filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20250121\\Deleted79features\\10000筆資料\\ALLDay_train_dataframes_Deleted79features_20250121.csv"

today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
generatefolder(filepath + "\\", "dataset_AfterProcessed")
generatefolder(filepath + "\\dataset_AfterProcessed\\", "CICIDS2017")
generatefolder(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\Dirichlet\\", today)

# 讀取 CSV 檔案
df_ALLtrain = pd.read_csv(csv_path)

# 依照 Label 分群，計算每個 Label 的筆數
df_grouped = df_ALLtrain.groupby("Label").size().reset_index(name="count")
print("原始資料各 Label 筆數：")
print(df_grouped)

# 設定 Dirichlet 分布的 alpha 值
alpha = 0.5

# 用來儲存分配後，各 client 的資料索引
client_indices = {"Client1": [], "Client2": []}

# 對每個 Label 進行獨立的 Dirichlet 分配
for label, group in df_ALLtrain.groupby("Label"):
    # 取得該 Label 的所有資料索引
    indices = group.index.tolist()
    n_samples = len(indices)
    # 產生兩個權重，這兩個權重的和為 1
    weights = np.random.dirichlet([alpha, alpha])
    print("權重0:",weights[0])
    print("權重1:",weights[1])

    # 根據權重計算分配給 Client1 的筆數，四捨五入後確保總數一致
    n_client1 = int(np.round(weights[0] * n_samples))
    # 為避免數值誤差，Client2 的筆數由剩餘數決定
    # 作用是確保分配給 Client1 的樣本數 n_client1 不會超過當前 Label 的總樣本數 n_samples
    n_client1 = min(n_client1, n_samples)  # 取 n_client1 與 n_samples 兩者中的最小值 避免超出
    n_client2 = n_samples - n_client1
    
    # 隨機打亂這個 Label 的資料索引
    np.random.shuffle(indices)
    
    # 分別存入兩個 client 的索引列表中
    client_indices["Client1"].extend(indices[:n_client1])
    client_indices["Client2"].extend(indices[n_client1:])
    
    print(f"Label: {label}, 總筆數: {n_samples}, 權重: {weights}, Client1: {n_client1}, Client2: {n_client2}")

# 根據分配好的索引，從原始 DataFrame 取出各 client 的資料
df_client1 = df_ALLtrain.loc[client_indices["Client1"]].reset_index(drop=True)
df_client2 = df_ALLtrain.loc[client_indices["Client2"]].reset_index(drop=True)

# 檢查分配後，每個 client 各 Label 的分佈
print("\nClient1 各 Label 筆數：")
print(df_client1['Label'].value_counts())

print("\nClient2 各 Label 筆數：")
print(df_client2['Label'].value_counts())
# filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
df_client1.to_csv(filepath + f"\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\Dirichlet\\{today}\\client1.csv", index=False)
df_client2.to_csv(filepath + f"\\dataset_AfterProcessed\\CICIDS2017\\ALLDay\\Dirichlet\\{today}\\client2.csv", index=False)
SaveDataframeTonpArray(df_client1, f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/Dirichlet/{today}", "Dirichlet_client1",today)
SaveDataframeTonpArray(df_client2, f"./data/dataset_AfterProcessed/CICIDS2017/ALLDay/Dirichlet/{today}", "Dirichlet_client2",today)