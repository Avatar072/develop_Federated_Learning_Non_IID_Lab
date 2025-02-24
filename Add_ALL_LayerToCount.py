import torch
import numpy as np
import csv
import os
from mytoolfunction import CheckFileExists
import pandas as pd
'''
寫法優點
函數加載了一個保存的模型狀態字典。這個字典包含了模型所有層的權重和偏置。
不需要預先知道模型的結構或層的名稱。
避免了將不同大小的張量填充到相同大小的問題。
'''
folder_path = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"

# 加載了一個保存的模型狀態字典。這個字典包含了模型所有層的權重和偏置。
# # After Local train 150 globla round
# client1_state_dict = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240728_150_G/client1/normal/After_local_train_weight.pth')
# client2_state_dict = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G/client2/normal/After_local_train_weight.pth')
# client3_state_dict = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G/client3/normal/After_local_train_weight.pth')

# # After FedAVG 150 global round Before_local_train_model
# client1_state_dict_after_FedAVG = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240728_150_G/client1/normal/Before_local_train_model.pth')
# client2_state_dict_after_FedAVG = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G/client2/normal/Before_local_train_model.pth')
# client3_state_dict_after_FedAVG = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G/client3/normal/Before_local_train_model.pth')


# # After JSMA_ATTACK
# client1_state_dict_after_JSMA_ATTACK = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G_C3_JSMA/client1/normal/After_local_train_weight.pth')
# client2_state_dict_after_JSMA_ATTACK = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G_C3_JSMA/client2/normal/After_local_train_weight.pth')
# # only client3 do JSMA!!!!!!
# client3_state_dict_after_JSMA_ATTACK = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G_C3_JSMA/client3/normal/After_local_train_weight.pth')

# # After JSMA_ATTACK FedAVG 150 global round Before_local_train_model
# client1_state_dict_after_JSMA_ATTACK_FedAVG = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G_C3_JSMA/client1/normal/Before_local_train_model.pth')
# client2_state_dict_after_JSMA_ATTACK_FedAVG = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G_C3_JSMA/client2/normal/Before_local_train_model.pth')
# # only client3 do JSMA!!!!!!
# client3_state_dict_after_JSMA_ATTACK_FedAVG = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G_C3_JSMA/client3/normal/Before_local_train_model.pth')

# 迭代狀態字典中的每個參數：
'''
這個循環遍歷狀態字典中的每個參數張量。
client1_state_dict.values() 返回字典中所有的值，
這些值就是各個層的參數張量。
'''
# param_tensor.flatten(): 將參數張量展平成一維張量。
# .sum(): 計算這個一維張量的所有元素之和。
# 然後，這個和被加到total_weight_sum 上

#將模型每層加總求總和
def DoCountModelWeightSum(dict_file,bool_UseABS,str_client):
    # 初始化總權重和：
    total_weight_sum = 0
    total_abs_weight_sum = 0 
    # if str_client == "client1":
    for param_tensor in dict_file.values():
        # 将每个参数张量展平并求和
        total_weight_sum += param_tensor.flatten().sum()

        # 将每个参数张量展平，取绝对值，然后求和
        total_abs_weight_sum += param_tensor.abs().flatten().sum()
    
    if bool_UseABS:
        # 取绝对值
        # print(f"{str_client}_Total absolute weight sum of the model:", total_abs_weight_sum)
        # 取绝对值絕對值總和更能代表模型的整體特徵
        # 將張量轉移到 CPU 並轉換為 Python 數字：
        total_abs_weight_sum = f"{total_abs_weight_sum:.6f}"
        print(f"{str_client}_\n"+f"Total absolute weight sum of the model: {total_abs_weight_sum}\n")

        return total_abs_weight_sum
    
    else:
        print(f"{str_client}_Total weight sum of the model:", total_weight_sum)
        # 將張量轉移到 CPU 並轉換為 Python 數字：
        # 將張量轉移到 CPU 並轉換為 Python 數字：
        total_weight_sum = f"{total_weight_sum:.6f}"
        print(f"{str_client}_Total weight sum of the model: {total_weight_sum}")
        return abs(float(total_weight_sum))




def evaluateWeightDifferences(str_state,weights1, weights2):

    # 调试信息，检查 weights1 和 weights2 的类型和内容
    print(f"weights1 类型: {type(weights1)}, 值: {weights1}")
    print(f"weights2 类型: {type(weights2)}, 值: {weights2}")

    # 確保 weights1 和 weights2 是 NumPy 數組
    if isinstance(weights1, torch.Tensor):
        weights1 = weights1.cpu().numpy()
    if isinstance(weights2, torch.Tensor):
        weights2 = weights2.cpu().numpy()

    # 如果 weights1 是元組，取第一個元素
    if isinstance(weights1, tuple):
        weights1 = weights1[0]
    if isinstance(weights2, tuple):
        weights2 = weights2[0]
    
    # 確保 weights1 和 weights2 都是數值
    weights1 = float(weights1)
    weights2 = float(weights2)

    differences = np.abs(weights1 - weights2)  # 直接計算絕對值差異
    average_difference = np.mean(weights1 - weights2)
    max_difference = np.max(weights1 - weights2)
    min_difference = np.min(weights1 - weights2)

    print(f"{str_state}相減後的差（絕對值）:", differences)
    print(f"{str_state}平均差異:", average_difference)
    print(f"{str_state}最大差異:", max_difference)
    print(f"{str_state}最小差異:", min_difference)
    return abs(differences) ,average_difference,max_difference,min_difference


# 計算兩個 state_dict 之間每一層的權重差距 以距離來看
'''
各種權重差距算法
torch.norm() 是 PyTorch 中的一個函數，用來計算張量的範數（也就是距離度量）。這個函數根據你指定的 p 值來計算不同類型的範數。
e.g
# torch.norm(input, p='fro', dim=None, keepdim=False, dtype=None, out=None)
input: 需要計算範數的張量。
p: 用來指定範數的類型。
p=1：計算 L1 範數（曼哈頓距離）。
p=2：計算 L2 範數（歐幾里得距離，默認值）。
p=float('inf')：計算 L∞ 範數（最大值範數）。
p='fro'：計算 Frobenius 範數（矩陣的L2範數），適用於2D張量（矩陣）。
dim: 指定要沿著哪個維度計算範數。這對於多維張量非常有用。例如，如果是2D張量，指定 dim=(0,1) 表示對所有元素計算範數。
keepdim: 如果設置為 True，將保留輸出的維度，默認為 False。
dtype: 指定返回的數據類型。
out: 可以選擇將結果寫入到一個已存在的張量中。
///////////////////////////////////////////////////////////////
e.g
# 創建兩個張量
# tensor1 = torch.tensor([1.0, 2.0, 3.0])
# tensor2 = torch.tensor([4.0, 6.0, 8.0])

# # L2範數（歐幾里得距離） 
# l2_diff = torch.norm(tensor1 - tensor2, p=2)
# print(f"L2範數差距（歐幾里得距離）: {l2_diff}")

# 假設我們有一個小型張量：
# 計算過程將是：

 展平: [1, 2, 3, 4, 5, 6]
# 平方: [1, 4, 9, 16, 25, 36]
# 求和: 1 + 4 + 9 + 16 + 25 + 36 = 91
# 平方根: √91 ≈ 9.539
# param1 = torch.tensor([[1, 2, 3],[4, 5, 6]]).to(device)
# torch.norm() 函數只能對浮點數（例如 float32 或 float64）或複數類型的張量進行計算
# 將張量轉換成浮點數類型
param1 = param1.float()
distance_param1 = torch.norm(param1, p=2).item()
print(f"{key}層的distance_param1\n",distance_param1)   

# # L1範數（曼哈頓距離）
# l1_diff = torch.norm(tensor1 - tensor2, p=1)
# print(f"L1範數差距（曼哈頓距離）: {l1_diff}")

# # L∞範數（最大值距離）
# l_inf_diff = torch.norm(tensor1 - tensor2, p=float('inf'))
# print(f"L∞範數差距（最大值距離）: {l_inf_diff}")

'''

# def Calculate_Weight_Diffs_Distance_OR_Absolute(state_dict1, state_dict2, file_path, Str_abs_Or_dis):
#     weight_diff_List = []
#     total_weight_diff = 0  # 初始化總和變量

#     if state_dict1 is None:
#             print(f"{state_dict1} is None")
#     if state_dict2 is None:
#             print(f"{state_dict2} is None")

#     for (name1, param1), (name2, param2) in zip(state_dict1.items(), state_dict2.items()):
#         # 確保比較的名稱和參數是一致的
#         if 'weight' in name1 and 'weight' in name2:
#             # 確保兩個層的形狀一致，才能比較
#             if param1.shape == param2.shape:
#                 # 打印每層的權重來檢查差異
#                 # print(f"正在比較層: {name1}")
#                 # print(f"權重1: {param1}")
#                 # print(f"權重2: {param2}")
#                 if Str_abs_Or_dis == "distance":
#                     # 計算每層對應權重的 L2 範數(歐幾里得範數)差距 後再將差值加總
#                     weight_diff = torch.norm(param1 - param2, p=2).item()
#                     print(f'{name1}層的距離權重差距',weight_diff)
#                     weight_diff_List.append((name1, weight_diff))

#                     # 加到 weight_diff_List 時立即計算總和
#                     total_weight_diff += weight_diff

#                 elif Str_abs_Or_dis == "absolute":
#                     # 計算兩個 state_dict 之間每一層的權重絕對差值 後再將差值加總
#                     # 計算每層對應權重的絕對差值
#                     # torch.abs() 計算的是逐元素的絕對值
#                     # 計算權重的逐元素絕對值差異。這意味著每個權重之間的差異都被單獨計算並展示出來
#                     # 會顯示很多元素計算結果出來，
#                     # 而這些差異取決於每個權重的具體數值變化，可能在不同的層次之間有較大的變動。

#                     abs_diff = torch.abs(param1 - param2)
#                     print(f'個元素絕對差異',torch.sum(abs_diff))
#                     # 計算個元素絕對差異的總和
#                     element_sum_difference = torch.sum(abs_diff).item()
#                     print(f'{name1}層的權重絕對值差距',element_sum_difference)
#                     # 累加所有層的絕對差異總
#                     total_weight_diff += element_sum_difference  
#                     average_difference = torch.mean(abs_diff).item()
#                     max_difference = torch.max(abs_diff).item()
#                     min_difference = torch.min(abs_diff).item()
#                     weight_diff_List.append({
#                                             'layer': name1,
#                                             'element_abs_difference': element_sum_difference,
#                                             'total_sum_diff': total_weight_diff,
#                                             'average_difference': average_difference,
#                                             'max_difference': max_difference,
#                                             'min_difference': min_difference
#                                              })
#             else:
#                 print(f"警告: {name1} 和 {name2} 的形狀不一致，無法比較")
    
#     # 確認 weight_diff_List 是否正確填充數據
#     # print("weight_diff_List 檢查:", weight_diff_List)
#     # 最後輸出結果
#     print(f"所有層的權重差距總和: {total_weight_diff}")
#     # 寫入文件
#     with open(file_path, "a+") as file:
#         if Str_abs_Or_dis == "distance":
#             file.write("layer,distance_difference,total_sum_diff\n")
#             for layer_name, diff in weight_diff_List:
#                 file.write(f"{layer_name},{diff},{total_weight_diff}\n")
#         elif Str_abs_Or_dis == "absolute":
#             file.write("layer,element_abs_difference,total_sum_diff,average_difference,max_difference,min_difference\n")
#             for diff_info in weight_diff_List:
#                 file.write(f"{diff_info['layer']},"
#                            f"{diff_info['element_abs_difference']},"
#                            f"{diff_info['total_sum_diff']},"
#                            f"{diff_info['average_difference']},"
#                            f"{diff_info['max_difference']},"
#                            f"{diff_info['min_difference']}\n")

#     print(f"weight_diffs 已經保存到 {file_path}")

#     return weight_diff_List, total_weight_diff

# 檢查數據是否存在異常情況（比如 0 值或 None）
def check_data_validity(layer_diffs):
    for i, diff in enumerate(layer_diffs):
        if diff is None or diff == 0:
            print(f"Warning: Layer {i} contains invalid data (None or 0).")
            return False
    return True

def Calculate_Weight_Diffs_Distance_OR_Absolute(state_dict1, state_dict2, file_path, Str_abs_Or_dis, bool_use_Norm):
    weight_diff_List = []
    weight_diff_Norm_List = []
    total_weight_diff = 0  # 初始化總和變量
    total_weight_diff_Norm = 0  # 初始化總和變量 for L2範數
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用 GPU 或 CPU
    
    if state_dict1 is None:
        print(f"{state_dict1} is None")
    if state_dict2 is None:
        print(f"{state_dict2} is None")

    # 確保兩個 state_dict 都是字典
    if isinstance(state_dict1, dict) and isinstance(state_dict2, dict):
        for key in state_dict1:
            if key in state_dict2:  # 確保 state_dict2 中也有相同的鍵
                param1 = state_dict1[key].to(device)  # 將 param1 移動到指定設備 local
                param2 = state_dict2[key].to(device)  # 將 param2 移動到指定設備 fedavg
                # print(f"{key}層的param1\n",param1)
                # distance_param1 = torch.norm(param1, p=2).item()
                # print(f"{key}層的distance_param1\n",distance_param1)                
                # print("param2\n",param2)
                if 'weight' in key:
                    # 確保兩個層的形狀一致，才能比較
                    if param1.shape == param2.shape:
                        if Str_abs_Or_dis == "distance":
                            # 計算每層對應權重的 L2 範數(歐幾里得範數)差距
                            # 這樣是先各每一層向量(64*64)陣列裡的差異再求距離
                            # distance_param1 = torch.norm(param2, p=2).item()
                            weight_diff_param1 = torch.norm(param1, p=2).item()
                            weight_diff_param2 = torch.norm(param2, p=2).item()
                            # print(f'{key}層param1的歐幾里得範數', weight_diff_param1)
                            # print(f'{key}層param2的歐幾里得範數', weight_diff_param2)
                            weight_diff = torch.norm(param1 - param2, p=2).item()
                            weight_diff_Norm= abs(weight_diff_param1-weight_diff_param2)
                            # print(f'{key}層的距離權重差距(以歐幾里得距離)', weight_diff)
                            # print(f'{key}層的距離權重差距(以歐幾里得範數)', weight_diff_Norm)
                            # True表示記錄歐基里得範數
                            if bool_use_Norm:
                                weight_diff_Norm_List.append((key, 
                                                          weight_diff_param1,
                                                          weight_diff_param2,
                                                          weight_diff_Norm))
                                total_weight_diff_Norm += weight_diff_Norm

                            else:
                                weight_diff_List.append((key, weight_diff))
                                total_weight_diff += weight_diff

                        elif Str_abs_Or_dis == "absolute":
                            # 計算每層對應權重的絕對差值
                            abs_diff = torch.abs(param1 - param2)
                            element_sum_difference = torch.sum(abs_diff).item()
                            print(f'{key}層的權重絕對值差距', element_sum_difference)
                            total_weight_diff += element_sum_difference 
                            print(f"累加後的 total_weight_diff: {total_weight_diff}") 
                            average_difference = torch.mean(abs_diff).item()
                            max_difference = torch.max(abs_diff).item()
                            min_difference = torch.min(abs_diff).item()
                            weight_diff_List.append({
                                'layer': key,
                                'element_abs_difference': element_sum_difference,
                                'total_sum_diff': total_weight_diff,
                                'average_difference': average_difference,
                                'max_difference': max_difference,
                                'min_difference': min_difference
                            })
                    else:
                        print(f"警告: {key} 的形狀不一致，無法比較")
            else:
                print(f"警告: {key} 在 state_dict2 中不存在")

    # 最後輸出結果
    # print(f"所有層的權重差距總和: {total_weight_diff}")

    # 確保 weight_diff_Norm_List 不為空，否則設置默認值
    if len(weight_diff_Norm_List) == 0 and bool_use_Norm:
        print("Warning: weight_diff_Norm_List is empty, filling with default values.")
       

    # 確保 weight_diff_List 不為空，否則設置默認值
    if len(weight_diff_List) == 0 and not bool_use_Norm:
        print("Warning: weight_diff_List is empty, filling with default values.")

    # 寫入文件
    # 檢查檔案是否存在
    file_exists = os.path.exists(file_path)
    # 如果檔案不存在，使用 'w' 模式寫入表頭，並寫入數據
    if not file_exists:
        with open(file_path, "w", newline='') as file:
            if Str_abs_Or_dis == "distance":
                file.write("layer1.weight,fc2.weight,fc3.weight,fc4.weight,layer5.weight,total_sum_diff\n")

    # 讀取現有的CSV檔（如果存在）
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["layer1.weight", "fc2.weight", "fc3.weight", "fc4.weight", "layer5.weight", "total_sum_diff"])

    #使用 'a+' 模式追加寫入各層權重差異數據
    # with open(file_path, "a+") as file:
    if Str_abs_Or_dis == "distance":
        if bool_use_Norm:
            # 每個元素是 (key,weight_diff_param1,weight_diff_param2 weight_diff_Norm)
            # 取得所有層的 weight_diff_Norm
            layer_diffs = [weight_diff_Norm for _,_,_,weight_diff_Norm in weight_diff_Norm_List]  
            # 打印 layer_diffs 和 total_weight_diff_Norm 的長度與 df.columns 的長度
            # print(f"layer_diffs: {layer_diffs}, total_weight_diff_Norm: {total_weight_diff_Norm}")
            # print(f"Number of columns in df: {len(df.columns)}")
            # print(f"Data to add length: {len(layer_diffs + [total_weight_diff_Norm])}")
            if len(layer_diffs + [total_weight_diff_Norm]) == len(df.columns):
                # 添加新的行
                new_row = pd.DataFrame([layer_diffs + [total_weight_diff_Norm]], columns=df.columns)
                df = pd.concat([df, new_row], ignore_index=True)
                # 寫入CSV文件
                df.to_csv(file_path, index=False)
            else:
                print(f"Error: Data length {len(layer_diffs + [total_weight_diff_Norm])} does not match column length {len(df.columns)}")
        else:
            # 每個元素是 (key, diff)
            # 取得所有層的 weight_diff
            # for i, item in enumerate(weight_diff_List):
            #     print(f"Item {i} has {len(item)} values: {item}")    
            layer_diffs = [diff for _, diff in weight_diff_List] 
            # print(f"layer_diffs: {layer_diffs}, total_weight_diff: {total_weight_diff}")
            # print(f"Number of columns in df: {len(df.columns)}")
            # print(f"Data to add length: {len(layer_diffs + [total_weight_diff])}")
            if len(layer_diffs + [total_weight_diff]) == len(df.columns):
                # 添加新的行
                new_row = pd.DataFrame([layer_diffs + [total_weight_diff]], columns=df.columns)
                df = pd.concat([df, new_row], ignore_index=True)
                # 寫入CSV文件
                df.to_csv(file_path, index=False)
            else:
                print(f"Error: Data length {len(layer_diffs + [total_weight_diff])} does not match column length {len(df.columns)}")





    # 舊寫法
    # with open(file_path, "a+") as file:
    #     if Str_abs_Or_dis == "distance":
    #         if bool_use_Norm:
    #             file.write("'Layer','Weight_Diff_Param 1','Weight_Diff_Param 2','Weight_Diff_Norm'\n")
    #             for layer_name, weight_diff_param1, weight_diff_param2, weight_diff_Norm in weight_diff_List:
    #                 file.write(f"{layer_name},{weight_diff_param1},{weight_diff_param2},{weight_diff_Norm},{total_weight_diff_Norm}\n")
    #         else:
    #             file.write("layer,distance_difference,total_sum_diff\n")
    #             for layer_name, diff in weight_diff_List:
    #                 file.write(f"{layer_name},{diff},{total_weight_diff}\n")
    #     elif Str_abs_Or_dis == "absolute":
    #         file.write("layer,element_abs_difference,total_sum_diff,average_difference,max_difference,min_difference\n")
    #         for diff_info in weight_diff_List:
    #             file.write(f"{diff_info['layer']},"
    #                        f"{diff_info['element_abs_difference']},"
    #                        f"{diff_info['total_sum_diff']},"
    #                        f"{diff_info['average_difference']},"
    #                        f"{diff_info['max_difference']},"
    #                        f"{diff_info['min_difference']}\n")



    #使用 'a+' 模式追加寫入L2範數
    if Str_abs_Or_dis == "distance" and bool_use_Norm:
        # for i, item in enumerate(weight_diff_Norm_List):
            # print(f"Item {i} has {len(item)} values: {item}")

        # 去掉 .csv 後綴
        remove_str_file_path = file_path.replace('.csv', '')
        Param_Local_file_name = remove_str_file_path + "_L2_Local.csv"
        Param_fedavg_file_name = remove_str_file_path + "_L2_fedavg.csv"
        
        # 檢查檔案是否存在
        file_exists_Param_Local = os.path.exists(Param_Local_file_name)
        file_exists_Param_fedavg = os.path.exists(Param_fedavg_file_name)

        # 如果兩個檔案不存在，則創建並寫入表頭
        if not file_exists_Param_Local:
            with open(Param_Local_file_name, "w", newline='') as file:
                file.write("layer1.weight,fc2.weight,fc3.weight,fc4.weight,layer5.weight\n")

        if not file_exists_Param_fedavg:
            with open(Param_fedavg_file_name, "w", newline='') as file:
                file.write("layer1.weight,fc2.weight,fc3.weight,fc4.weight,layer5.weight\n")

        # 生成 Local 權重差異的列表
        layer_diffs_Local = [diff_param_Local for _, diff_param_Local, _, _ in weight_diff_Norm_List]
        print("len(layer_diffs_Local)", len(layer_diffs_Local))
        
        # 將 Local 權重差異寫入檔案
        with open(Param_Local_file_name, "a+") as file:
            file.write(",".join(map(str, layer_diffs_Local)) + f"\n")

        # 生成 Fedavg 權重差異的列表
        layer_diffs_fedavg = [diff_param_fedavg for _, _, diff_param_fedavg, _ in weight_diff_Norm_List]
        print("len(layer_diffs_fedavg)", len(layer_diffs_fedavg))

        # 將 Fedavg 權重差異寫入檔案
        with open(Param_fedavg_file_name, "a+") as file:
            file.write(",".join(map(str, layer_diffs_fedavg)) + f"\n")

        
        print(f"weight_diffs 已經保存到 {file_path}")
    # if bool_use_Norm:
                                
        return weight_diff_Norm_List, total_weight_diff_Norm

    else:
        print(f"weight_diffs 已經保存到 {file_path}")                       
        return weight_diff_List, total_weight_diff


    


# 計算兩個模型的每層權重差距（以距離）
# file_path = f"{folder_path}/weight_diffs_dis_test.csv"
# weight_diffs_dis, total_weight_diff= Calculate_Weight_Diffs_Distance_OR_Absolute(client1_state_dict, 
#                                                                     client1_state_dict_after_FedAVG,
#                                                                     file_path,
#                                                                     "distance")
# # 計算兩個模型的每層權重差距（絕對值）
# file_path = f"{folder_path}/weight_diffs_abs_test.csv"
# weight_diffs_abs, total_sum_diff_abs = Calculate_Weight_Diffs_Distance_OR_Absolute(client1_state_dict, 
#                                                                     client1_state_dict_after_FedAVG,
#                                                                     file_path,
#                                                                     "absolute")

# total_FedAVG_abs_weight_sum=0
# client1_state_dict = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240801_使用random_第一次/client1/normal/After_local_train_weight.pth')
# total_FedAVG_abs_weight_sum = DoCountModelWeightSum(client1_state_dict,False,"After_FedAVG")   

# # 正常情况和JSMA攻击后的权重差值
# normal_diffs = np.array([11.74, 11.71, 14.62])
# jsma_diffs = np.array([11.08, 9.99, 18.91])

# # 计算平均值和标准差
# normal_mean = np.mean(normal_diffs)
# normal_std = np.std(normal_diffs)
# jsma_mean = np.mean(jsma_diffs)
# jsma_std = np.std(jsma_diffs)

# # 打印结果
# print("Normal Weight Diffs Mean:", normal_mean)
# print("Normal Weight Diffs STD:", normal_std)
# print("JSMA Weight Diffs Mean:", jsma_mean)
# print("JSMA Weight Diffs STD:", jsma_std)

# threshold = normal_mean + 2 * normal_std
# print("Threshold:", threshold)


# 創建一個大小為10的數組，初始值為0

# def client1Toevaluate():
#     # 計算權重之間的差異的絕對值
#     evaluate_client1_array = np.zeros(8)
#     #print("After Local train 150 global round")
#     evaluate_client1_array[0] = DoCountModelWeightSum(client1_state_dict,True,
#                                                       "client1_After Local train 150 global round")
#     evaluate_client1_array[1] = DoCountModelWeightSum(client1_state_dict_after_FedAVG,True,
#                                                       "client1_After FedAVG 150 global round")
#     evaluate_client1_array[2] = DoCountModelWeightSum(client1_state_dict_after_JSMA_ATTACK,True,
#                                                       "client1_After Local train 150 global round"+
#                                                       "After JSMA_ATTACK only client3 do JSMA")
    
#     evaluate_client1_array[3] = DoCountModelWeightSum(client1_state_dict_after_JSMA_ATTACK_FedAVG,True,
#                                                       "client1_After FedAVG 150 global round"+
#                                                       "After JSMA_ATTACK only client3 do JSMA")
    
#     # ######################以一個client上傳的weight與Fedavg後的weight相減後的差（絕對值）之平均差、最大差值、最小差值
#     evaluateWeightDifferences("Local-FeaAVG",evaluate_client1_array[0],evaluate_client1_array[1])
#     evaluateWeightDifferences("JSMA_Local-JSMA_FeaAVG",evaluate_client1_array[2],evaluate_client1_array[3])
#     evaluateWeightDifferences("Local-JSMA_Local",evaluate_client1_array[0],evaluate_client1_array[2])
#     #####################加上JSMA後，與Fedavg後的weight相減後的差（絕對值）之平均差、最大差值、最小差值
#     evaluateWeightDifferences("FeaAVG-JSMA_FeaAVG",evaluate_client1_array[1],evaluate_client1_array[3])


 


# def client2Toevaluate():
#     # 計算權重之間的差異的絕對值
#     evaluate_client2_array = np.zeros(8)
#     #print("After Local train 150 global round")
#     evaluate_client2_array[0] = DoCountModelWeightSum(client2_state_dict,True,
#                                                       "client2_After Local train 150 global round")
#     evaluate_client2_array[1] = DoCountModelWeightSum(client2_state_dict_after_FedAVG,True,
#                                                       "client2_After FedAVG 150 global round")
#     evaluate_client2_array[2] = DoCountModelWeightSum(client2_state_dict_after_JSMA_ATTACK,True,
#                                                       "client2_After Local train 150 global round"+
#                                                       "After JSMA_ATTACK only client3 do JSMA")
    
#     evaluate_client2_array[3] = DoCountModelWeightSum(client2_state_dict_after_JSMA_ATTACK_FedAVG,True,
#                                                       "client2_After FedAVG 150 global round"+
#                                                       "After JSMA_ATTACK only client3 do JSMA")
    
#     # ######################以一個client上傳的weight與Fedavg後的weight相減後的差（絕對值）之平均差、最大差值、最小差值
#     evaluateWeightDifferences("Local-FeaAVG",evaluate_client2_array[0],evaluate_client2_array[1])
#     evaluateWeightDifferences("JSMA_Local-JSMA_FeaAVG",evaluate_client2_array[2],evaluate_client2_array[3])
#     evaluateWeightDifferences("Local-JSMA_Local",evaluate_client2_array[0],evaluate_client2_array[2])
#     #####################加上JSMA後，與Fedavg後的weight相減後的差（絕對值）之平均差、最大差值、最小差值
#     evaluateWeightDifferences("FeaAVG-JSMA_FeaAVG",evaluate_client2_array[1],evaluate_client2_array[3])


# def client3Toevaluate():
#     # 計算權重之間的差異的絕對值
#     evaluate_client3_array = np.zeros(8)
#     #print("After Local train 150 global round")
#     evaluate_client3_array[0] = DoCountModelWeightSum(client3_state_dict,True,
#                                                       "client3_After Local train 150 global round")
#     evaluate_client3_array[1] = DoCountModelWeightSum(client3_state_dict_after_FedAVG,True,
#                                                       "client3_After FedAVG 150 global round")
#     evaluate_client3_array[2] = DoCountModelWeightSum(client3_state_dict_after_JSMA_ATTACK,True,
#                                                       "client3_After Local train 150 global round"+
#                                                       "After JSMA_ATTACK only client3 do JSMA")
    
#     evaluate_client3_array[3] = DoCountModelWeightSum(client3_state_dict_after_JSMA_ATTACK_FedAVG,True,
#                                                       "client3_After FedAVG 150 global round"+
#                                                       "After JSMA_ATTACK only client3 do JSMA")
    
#     # ######################以一個client上傳的weight與Fedavg後的weight相減後的差（絕對值）之平均差、最大差值、最小差值
#     evaluateWeightDifferences("Local-FeaAVG",evaluate_client3_array[0],evaluate_client3_array[1])
#     evaluateWeightDifferences("JSMA_Local-JSMA_FeaAVG",evaluate_client3_array[2],evaluate_client3_array[3])
#     evaluateWeightDifferences("Local-JSMA_Local",evaluate_client3_array[0],evaluate_client3_array[2])
#     #####################加上JSMA後，與Fedavg後的weight相減後的差（絕對值）之平均差、最大差值、最小差值
#     evaluateWeightDifferences("FeaAVG-JSMA_FeaAVG",evaluate_client3_array[1],evaluate_client3_array[3])

# client1Toevaluate()
# client2Toevaluate()
# client3Toevaluate()


############
# 另列寫法
# # 從狀態字典中提取所有五層的權重並將其展平。
# # 展平的目的是將原本可能是多维的權重矩陣轉換成一維向量，這樣可以簡化接下來的操作，如加法。
# weights_layer1 = client1_state_dict['layer1.weight'].flatten()
# weights_fc2 = client1_state_dict['fc2.weight'].flatten()
# weights_fc3 = client1_state_dict['fc3.weight'].flatten()
# weights_fc4 = client1_state_dict['fc4.weight'].flatten()
# weights_layer5 = client1_state_dict['layer5.weight'].flatten()

# # 獲取當前權重所在的裝置信息，確保所有操作都在同一裝置上進行，避免裝置不一致錯誤。
# device = weights_layer1.device

# # 為了能將不同層的權重相加，需要將它們擴展到相同的長度。
# # 计算所有权重中的最大长度，以便统一向量长度。
# max_length = max(weights_layer1.size(0), weights_fc2.size(0), weights_fc3.size(0), weights_fc4.size(0), weights_layer5.size(0))

# # 使用torch.cat进行拼接，拼接原始权重向量和必要长度的零向量，使所有权重向量长度一致。
# # 这里创建的零向量在同一设备上（CPU或GPU），确保数据一致性。
# weights_layer1 = torch.cat((weights_layer1, torch.zeros(max_length - weights_layer1.size(0), device=device)), 0)
# weights_fc2 = torch.cat((weights_fc2, torch.zeros(max_length - weights_fc2.size(0), device=device)), 0)
# weights_fc3 = torch.cat((weights_fc3, torch.zeros(max_length - weights_fc3.size(0), device=device)), 0)
# weights_fc4 = torch.cat((weights_fc4, torch.zeros(max_length - weights_fc4.size(0), device=device)), 0)
# weights_layer5 = torch.cat((weights_layer5, torch.zeros(max_length - weights_layer5.size(0), device=device)), 0)

# # 將所有擴展後的權重相加。
# total_weights = weights_layer1 + weights_fc2 + weights_fc3 + weights_fc4 + weights_layer5

# # 打印或使用總權重
# print("Total weights summed across all layers:")
# print(total_weights.flatten().sum())
