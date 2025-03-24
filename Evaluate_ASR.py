import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 計算對抗攻擊成功的樣本數量
# df = pd.read_csv("D:\\develop_Federated_Learning_Non_IID_Lab\\Adversarial_Attack_Test\\CICIDS2017\\FGSM_Attack\\20250207\\successful_attacks_eps_0.05.csv")
# df = pd.read_csv("D:\\develop_Federated_Learning_Non_IID_Lab\\Adversarial_Attack_Test\\CICIDS2017\\FGSM_Attack\\20250305\\successful_attacks_eps_0.05.csv")
# df = pd.read_csv("D:\\develop_Federated_Learning_Non_IID_Lab\\Adversarial_Attack_Test\\CICIDS2017\\FGSM_Attack\\20250317\\successful_attacks_eps_0.05.csv")
df = pd.read_csv("D:\\develop_Federated_Learning_Non_IID_Lab\\Adversarial_Attack_Test\\CICIDS2017\\FGSM_Attack\\20250319\\79_featurebaseLine_mergeLabel_normal_model_to_genrate\\successful_attacks_eps_0.05.csv")

success_count = len(df[df['original_class'] != df['adversarial_class']])

# 總樣本數量
total_count = len(df)

# 計算攻擊成功率
# 攻擊成功率= 原始類別與對抗類別不同的樣本數量/總樣本數量
attack_success_rate = success_count / total_count* 100
print(f"成功攻擊樣本: {success_count}")
print(f"總樣本: {total_count}")
print(f"攻擊成功率: {attack_success_rate :.2f}%")

# 存儲變數到 DataFrame
summary_data = {
    'success_count': [success_count],
    'total_count': [total_count],
    'attack_success_rate': [attack_success_rate]
}

summary_df = pd.DataFrame(summary_data)

# 檢查 df 的列名
print(df.columns)

# 統計每個樣本被攻擊後變成的類別數量
change_counts = df.groupby(['original_class', 'adversarial_class']).size().reset_index(name='count')
# change_counts.to_csv("./test_asr.csv")


# 計算每個類別的對抗攻擊成功的樣本數量
misclassified_samples_count = df[df['original_class'] != df['adversarial_class']].groupby('original_class').size()

# 計算每個原始類別的樣本數
original_class_counts = df['original_class'].value_counts()
print("\n每個類別原始類別的樣本數:")

print(original_class_counts)
# 顯示每個原始類別的對抗攻擊成功樣本數
print("\n每個類別的對抗攻擊成功樣本數量:")
print(misclassified_samples_count)

# 將原始類別數量和對抗攻擊成功樣本數轉換為 DataFrame
original_class_counts_df = pd.DataFrame(list(original_class_counts.items()), columns=['original_class', 'original_count'])
misclassified_samples_count_df = pd.DataFrame(list(misclassified_samples_count.items()), columns=['original_class', 'attack_success_count'])

# 合併 DataFrame
merged_df = pd.merge(original_class_counts_df, misclassified_samples_count_df, on='original_class', how='left')

# 按 original_class 排序
merged_df = merged_df.sort_values(by='original_class', ascending=True)

# 1. 計算每個類別的攻擊成功率（誤分類比例）
# 每個類別的攻擊成功率 (for each class)：該類別被錯誤分類的樣本數/該類別被錯誤分類的樣本數 
# 即:攻擊成功率= 對抗類別 != 原始類別的樣本數/該類別的總樣本數

# 2. 計算每個類別的轉換率（類別轉換為其他類別的比例）
# 這裡可以計算每個類別的樣本數變化
# 轉換率是:指某個原始類別（original_class）的樣本在對抗攻擊後轉換成其他類別（adversarial_class）的比例
# 轉換率 (for each class)= 該原始類別轉換成其他類別的樣本數/該原始類別的總樣本數
misclassified_counts = df[df['original_class'] != df['adversarial_class']].groupby('original_class').size()

conversion_rate_per_class = misclassified_counts / original_class_counts* 100

# 顯示結果
print("\n每個類別的轉換率：")
print(conversion_rate_per_class)

# 存儲到同一個 Excel 文件中，將變數存到另外一個 sheet
with pd.ExcelWriter('./test_asr.xlsx') as writer:
    # 存儲攻擊成功率相關數據到另一個 sheet
    summary_df.to_excel(writer, sheet_name='Attack_Success_Rate', index=False)
    # 存儲 change_counts 到一個 sheet
    change_counts.to_excel(writer, sheet_name='Change_Counts', index=False)
    # 存儲每個類別的轉換率（類別轉換為其他類別的比例）
    conversion_rate_per_class.to_excel(writer, sheet_name='conversion_rate_per_class', index=True)
    # 存儲每個原始類別的對抗攻擊成功樣本數
    misclassified_samples_count.to_excel(writer, sheet_name='misclassified_samples_count', index=True)

    merged_df.to_excel(writer, sheet_name='Class_Counts_Attack_Success', header="f",index=False)

# 顯示結果
print(change_counts)

 
