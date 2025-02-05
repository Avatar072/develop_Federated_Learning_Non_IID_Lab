import numpy as np
import pandas as pd

# 模擬數據表內容（假設您已將數據轉換為 pandas DataFrame）
data = {
    "Labe": [
        "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", 
        "DoS slowloris", "FTP-Patator", "Heartbleed", "Infiltration", "PortScan",
        "SSH-Patator", "Web Attack Brute Force", "Web Attack Sql Injection", "Web Attack XSS"
    ],
    "count上限": [10000, 1956, 10000, 10000, 10000, 5499, 5796, 7935, 11, 36, 10000, 5897, 1507, 21, 652]
}

df = pd.DataFrame(data)

# 設定迪利克雷分布的 alpha 值
alpha = 0.5

# 初始化分配結果
client1_counts = []
client2_counts = []

# 基於迪利克雷分布進行數據分配
for count in df["count上限"]:
    # 生成兩個權重
    weights = np.random.dirichlet([alpha, alpha])
    # 分配數據
    client1_counts.append(int(weights[0] * count))
    client2_counts.append(int(weights[1] * count))

# 將結果新增至 DataFrame
df["Client1"] = client1_counts
df["Client2"] = client2_counts

# 確保分配後的總數與原數量一致
df["分配總數驗證"] = df["Client1"] + df["Client2"]

# 輸出到終端
print(df)