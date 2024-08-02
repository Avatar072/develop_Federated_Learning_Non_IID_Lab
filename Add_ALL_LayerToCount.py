import torch
import numpy as np

'''
寫法優點
函數加載了一個保存的模型狀態字典。這個字典包含了模型所有層的權重和偏置。
不需要預先知道模型的結構或層的名稱。
避免了將不同大小的張量填充到相同大小的問題。
'''

# 加載了一個保存的模型狀態字典。這個字典包含了模型所有層的權重和偏置。
# # After Local train 150 globla round
# client1_state_dict = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G/client1/normal/After_local_train_weight.pth')
# client2_state_dict = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G/client2/normal/After_local_train_weight.pth')
# client3_state_dict = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G/client3/normal/After_local_train_weight.pth')

# # After FedAVG 150 global round Before_local_train_model
# client1_state_dict_after_FedAVG = torch.load('D:/develop_Federated_Learning_Non_IID_Lab/FL_AnalyseReportfolder/20240726_150_G/client1/normal/Before_local_train_model.pth')
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
        return float(total_weight_sum)




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
    return differences ,average_difference,max_difference,min_difference


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
