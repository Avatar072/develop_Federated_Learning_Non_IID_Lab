import numpy as np
import pandas as pd
import os


def EvaluateVariance(list_,file_path):
    add_value_List = []
    mean_value_List = []
    variance_List = []
    variance_square_List = []
    variance_square_Threshold_List = []

    mean_value = 0
    add_value = 0
    counter = 0
    mean_value_List_max = 0
    mean_value_List_min = 0
    variance_List_max = 0
    variance_square_Threshold_List_max = 0
    for idx, val in enumerate(list_):
        counter +=1
        # if idx < len(list_) - 1:  # 檢查是否不是最後一個元素
        #     add_value += list_[idx+1]
        add_value += list_[idx]
        mean_value = add_value/counter
        print(f"round: {counter}")
        print(f"list_[{idx}]: {list_[idx]}")
        if idx < len(list_) - 1:  # 檢查是否不是最後一個元素
            print(f"will add next value\nlist_[{idx+1}]: {list_[idx+1]}")
        
        # # 取小數點後兩位
        add_value  = round(add_value, 4)
        mean_value = round(mean_value, 4)
        print(f"current added_value: {add_value:.5f}")
        print(f"current mean_value: {mean_value:.5f}")
        # 找累加值
        add_value_List.append(add_value)
        # 找累加平均值
        mean_value_List.append(mean_value)

        # 目前最大值=是指目前平均值裡的最大值
        mean_value_List_max = np.max(mean_value_List)
        mean_value_List_min = np.min(mean_value_List)
        # 變異量=目前平均值最大值-累加平均值 後求平方    
        variance = np.abs(mean_value_List_max-mean_value)
        variance_square = variance ** 2# ** 是 Python 的算術操作符，用來對每個數字進行平方運算
        print(f"variance: {variance:.5f}")
        print(f"variance_square: {variance_square:.5f}")
        variance  = round(variance, 5)
        variance_List.append(variance)
        variance_List_max = np.max(variance_List)

        variance_square = round(variance_square, 5)
        variance_square_List.append(variance_square)
        # variance_square_threshold = variance_square*2 #*2表示允許範圍值
        variance_square_threshold = variance_square*3 #*3表示允許範圍值
        variance_square_threshold = round(variance_square_threshold, 5)
        variance_square_Threshold_List.append(variance_square_threshold)
        variance_square_Threshold_List_max = np.max(variance_square_Threshold_List)

    # 找出
    # 平均值為累加的平均值
    # 目前最大值=是指目前平均值裡的最大值
    # 變異量=目前平均值最大值-累加平均值 後求平方
    # 變異量不能累加 抓最大值的變異量


    print(f"add_value_List: {add_value_List}")
    print(f"mean_value_List: {mean_value_List}")
    print(f"mean_value_List_max: {mean_value_List_max:.5f}")
    print(f"mean_value_List_min: {mean_value_List_min:.5f}")

    print(f"variance_List: {variance_List}")
    print(f"variance_List_max: {variance_List_max}")
    print(f"variance_square_List: {variance_square_List}")
    print(f"variance_square_Threshold_List: {variance_square_Threshold_List}")
    print(f"variance_square_Threshold_List_max: {variance_square_Threshold_List_max}")



    # file_path = f"./FL_AnalyseReportfolder/{today}/{current_time}/{client_str}/{Choose_method}/Initial_Local_weights_{client_str}"

    # mean_value_file_name = file_path+"_mean_value.csv"
    # variance_file_name = file_path+"_variance.csv"
    # variance_square_file_name = file_path+"variance_square.csv"
    # variance_square_threshold_file_name = file_path+"variance_square_threshold.csv"

    # file_exists_Param_mean_value = os.path.exists(mean_value_file_name)
    # file_exists_variance_file_name = os.path.exists(variance_file_name)
    # file_exists_variance_square_file_name = os.path.exists(variance_square_file_name)
    # file_exists_variance_square_threshold_file_name = os.path.exists(variance_square_threshold_file_name)


    # if not file_exists_Param_mean_value:
    #     with open(mean_value_file_name, "w", newline='') as file:
    #         file.write("mean_value,mean_value_List_min,mean_value_List_max\n")

    # if not file_exists_variance_file_name:
    #     with open(variance_file_name, "w", newline='') as file:
    #         file.write("variance\n")
    
    # if not file_exists_variance_square_file_name:
    #     with open(variance_square_file_name, "w", newline='') as file:
    #         file.write("variance_square\n")
    
    # if not file_exists_variance_square_threshold_file_name:
    #     with open(variance_square_threshold_file_name, "w", newline='') as file:
    #         file.write("variance_square_threshold\n")

    # # 寫入文件
    # with open(mean_value_file_name, "a+") as file:
    #     for mean_value in mean_value_List:
    #         file.write(f"{mean_value},{mean_value_List_min},{mean_value_List_max}\n")

    # with open(variance_file_name, "a+") as file:
    #     for variance in variance_List:
    #         file.write(f"{variance}\n")

    # with open(variance_square_file_name, "a+") as file:
    #     for variance_square in variance_square_List:
    #         file.write(f"{variance_square}\n")

    # with open(variance_square_threshold_file_name, "a+") as file:
    #     for variance_square_threshold in variance_square_Threshold_List:
    #         file.write(f"{variance_square_threshold}\n")

    # 檔案路徑
    import csv
    file_name = file_path + "_combined_data.csv"

    # 檢查文件是否存在
    file_exists = os.path.exists(file_name)
   # 如果文件不存在，則創建並寫入標題
    if not file_exists:
        with open(file_name, "a+", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["mean_value", "mean_value_List_min", "mean_value_List_max", "variance", "variance_square", "variance_square_threshold"])
    # 打開文件進行追加寫入
    with open(file_name, "a+", newline='') as file:
        writer = csv.writer(file)
        
        # 讀取現有行並轉換為集合，用來檢查是否有重複的資料
        file.seek(0)  # 移動文件指針到開頭
        # 檢查每一筆資料是否已存在於檔案中
        existing_rows = set(tuple(row) for row in csv.reader(file))
        
        # 寫入資料
        # for i in range(len(mean_value_List)):
        #     data = [
        #         mean_value_List[i],
        #         mean_value_List_min,
        #         mean_value_List_max,
        #         variance_List[i],
        #         variance_square_List[i],
        #         variance_square_Threshold_List[i]
        #     ]
        #     # 如果資料不存在，就寫入
        #     if tuple(data) not in existing_rows:
        #         writer.writerow(data)
        #         existing_rows.add(tuple(data))
        # 取得最新資料
        new_data = [
            mean_value_List[-1],  # 假設您要的是最後一筆資料
            mean_value_List_min,
            mean_value_List_max,
            variance_List[-1],
            variance_square_List[-1],
            variance_square_Threshold_List[-1]
        ]   
        # 如果資料不存在，就寫入
        if tuple(new_data) not in existing_rows:
            writer.writerow(new_data) 
            print("Recode new data in csv")


    return variance_square_Threshold_List_max


# 測試區
# file_path = "./1245"

# # 假設數據
# df = pd.read_csv("E:\\develop_Federated_Learning_Non_IID_Lab\\FL_AnalyseReportfolder\\20250319\\CICIDS2017_use_20250205_data_merge_label_FGSM_eps0.05測試_79_feature\\Inital_Local_weight_diff_client1.csv")
# # list_
# # list_ = df["dis_variation_Inital_Local"].head(10)
# list_ = df["dis_variation_Inital_Local"].head(126)
# test = EvaluateVariance(list_,file_path)