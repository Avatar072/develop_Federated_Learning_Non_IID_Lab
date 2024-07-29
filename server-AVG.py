import os
from typing import List, Tuple, Dict, Any

import flwr as fl
from flwr.common import Metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from mytoolfunction import ChooseUseModel

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.layer1 = nn.Linear(44, 512)
        self.layer1 = nn.Linear(83, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        # self.layer5 = nn.Linear(512, 35)
        self.layer5 = nn.Linear(512, 15)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.layer5(x)
        return x

# 初始化模型和優化器
# model = MLP()
#CICIIDS2017 or Edge 62個特徵
# labelCount = 15
#TONIOT 44個特徵
labelCount = 10
#CICIIDS2019
# labelCount = 13
#Wustl 41個特徵
# labelCount = 5
#Kub 36個特徵
# labelCount = 4
#CICIIDS2017、TONIOT、CICIIDS2019 聯集
# labelCount = 35
#CICIIDS2017、TONIOT、EdgwIIOT 聯集
# labelCount = 31

model = ChooseUseModel("MLP", 44, labelCount)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

############# 凍結 #####################
# for name, param in model.named_parameters():

#     if 'fc2.weight' in name:
#         param.requires_grad = False
#     if 'fc2.bias' in name:
#         param.requires_grad = False
#     if 'fc3.weight' in name:
#         param.requires_grad = False
#     if 'fc3.bias' in name:
#         param.requires_grad = False
#     if 'fc4.weight' in name:
#         param.requires_grad = False
#     if 'fc4.bias' in name:
#         param.requires_grad = False
#     if 'fc5.weight' in name:
#         param.requires_grad = False
#     if 'fc5.bias' in name:
#         param.requires_grad = False
#     if 'fc6.weight' in name:
#         param.requires_grad = False
#     if 'fc6.bias' in name:
#         param.requires_grad = False
    # if 'fc7.weight' in name:
    #     param.requires_grad = False
    # if 'fc7.bias' in name:
    #     param.requires_grad = False

        
        # print("name: ", name)
        # print("requires_grad: ", param.requires_grad)

weights_values = []
for param in model.parameters():
    weights_values.append(param.data.numpy())

# 將模型和優化器放入 FLOWER 的 Model
# initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())
initial_parameters = fl.common.ndarrays_to_parameters(weights_values)

# Define metric aggregation function
# 原本
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#     print("metrics\n",metrics)
#     print("accuracies\n",accuracies)
#     print("examples\n",examples)
#     print("accuracy\n",sum(accuracies) / sum(examples))
#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}

#可以顯示出client id的版本
# def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
#     accuracies = []
#     examples = []
#     client_ids = []  # 用于存储每个客户端的ID

#     # 遍历所有客户端提交的数据
#     for num_examples, m in metrics:
#         if "client_id" in m:
#             client_ids.append(m["client_id"])  # 将client_id添加到列表中
#             if m["client_id"] == "client1":
#                 print("client1\n",m["Local_train_weight_sum-FedAVG weight_sum"])    
#         else:
#             client_ids.append("Unknown")  # 如果没有client_id，记录为"Unknown"

#         if "accuracy" in m:
#             accuracies.append(num_examples * m["accuracy"])  # 计算加权准确度
#             examples.append(num_examples)

#     print("Metrics:\n", metrics)
#     print("Client IDs:\n", client_ids)
#     print("Accuracies:\n", accuracies)
#     print("Examples:\n", examples)
    
#     if not examples:
#         return {"accuracy": 0}  # 防止除以零

#     weighted_accuracy = sum(accuracies) / sum(examples)
#     return {"accuracy": weighted_accuracy, "client_ids": client_ids}

#選第三個客戶端的平均每輪權重差最大者14.62
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]], threshold: float = 14.62) -> Dict[str, float]:
#三個客戶端的平均每輪權重差加起來設立閥值11.74 + 11.71 + 14.62 = 38.07
# def weighted_average(metrics: List[Tuple[int, Dict[str, float]]], threshold: float = 38.07) -> Dict[str, float]:

    accuracies = []
    examples = []
    client_ids = []  # 用于存储每个客户端的ID
    valid_weights = []  # 用于存储符合阈值条件的权重

    # 遍历所有客户端提交的数据
    for num_examples, m in metrics:
        client_id = m.get("client_id", "Unknown")
        client_ids.append(client_id)  # 将client_id添加到列表中
        weight_diff = m.get("Local_train_weight_sum-FedAVG weight_sum", 0)  # 获取权重差异
        
        # 检查权重差异是否超过阈值
        if weight_diff <= threshold:
            if "accuracy" in m:
                accuracies.append(num_examples * m["accuracy"])  # 计算加权准确度
                examples.append(num_examples)
                valid_weights.append(weight_diff)  # 添加有效权重
        else:
            print(f"{client_id} excluded due to weight difference: {weight_diff}")

    print("Metrics:\n", metrics)
    print("Client IDs:\n", client_ids)
    print("Valid Weights (within threshold):\n", valid_weights)
    print("Accuracies:\n", accuracies)
    print("Examples:\n", examples)
    
    if not examples:
        return {"accuracy": 0, "client_ids": client_ids}  # 防止除以零

    weighted_accuracy = sum(accuracies) / sum(examples)
    return {"accuracy": weighted_accuracy}

# Define strategy
strategy = fl.server.strategy.FedAvg(initial_parameters = initial_parameters, evaluate_metrics_aggregation_fn=weighted_average, 
    min_fit_clients = 3, min_evaluate_clients = 3, min_available_clients = 3)
    # min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2)


# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:53388",
    # server_address="127.0.0.1:8080",
    # server_address="192.168.1.137:53388",

    # config=fl.server.ServerConfig(num_rounds=2),
    config=fl.server.ServerConfig(num_rounds=150),

    strategy=strategy,
)
