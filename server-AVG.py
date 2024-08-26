import os
from typing import List, Tuple, Dict, Optional

import flwr as fl
from flwr.common import Metrics
from flwr.common import Parameters, FitRes, EvaluateRes
from flwr.server import client_proxy, client_manager

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
#             # if m["client_id"] == "client1":
#             #     print("client1\n",m["Local_train_weight_sum-FedAVG weight_sum"])    
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
#     # return {"accuracy": weighted_accuracy, "client_ids": client_ids}
#     return {"accuracy": weighted_accuracy}

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    
    print("Received metrics:", metrics)  # 新增：打印輸入的 metrics
    weighted_sum = 0
    total_examples = 0  # 重置为0
    Local_train_accuracies_List = []
    client_ids = []
    valid_weights = []

    for num_examples, m in metrics:
        print("num_examples:",num_examples)
        client_id = m.get("client_id", "Unknown")
        client_ids.append(client_id)
        Local_train_accuracy = m.get("Local_train_accuracy", 0)
        Local_train_accuracies_List.append(Local_train_accuracy)
        global_round = m.get("global_round", 0)
        round = global_round
        Local_weight_sum = m.get("Local_train_weight_sum", 0)
        weight_diff = m.get("Local_train_weight_sum-Previous_FedAVG weight_sum", 0)
        threshold = Local_weight_sum * 0.05

        #前面10round不看
        if global_round > 10:
            #權重差異超過當前本地權重總和的5%就要過濾掉  或 Local_train_accuracy大於90%才能列入計算
            if weight_diff <= threshold or Local_train_accuracy > 0.8:
                if "accuracy" in m and Local_train_accuracy > 0.8:#Local_train_accuracy大於90%才能列入計算
                    print(m["accuracy"])
                    weighted_sum += num_examples * m["accuracy"]
                    total_examples += num_examples  # 只在满足条件时累加
                    valid_weights.append(weight_diff)
            else:
                print(f"{client_id} excluded due to weight difference: {weight_diff}")
        # else:
        #     # 前10轮所有客户端数据都参与计算
        #     if "accuracy" in m:
        #         weighted_sum += num_examples * m["accuracy"]
        #         total_examples += num_examples
        #         valid_weights.append(weight_diff)

    print("Client IDs:", client_ids)
    print("Valid Weights (within threshold):", valid_weights)
    print("Total examples:", total_examples)
    print("Client IDs, Local train Accuracy:", list(zip(client_ids, Local_train_accuracies_List)))
    
    if total_examples == 0:
        return {"accuracy": 0, "client_ids": client_ids}

    print("weighted_sum:", weighted_sum)
    print("total_examples:", total_examples)
    weighted_accuracy = weighted_sum / total_examples
    print("client_ids:", client_ids)
    return {"accuracy": weighted_accuracy, "client_ids": client_ids}


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_min_fit_clients = self.min_fit_clients
        self.initial_min_evaluate_clients = self.min_evaluate_clients
        self.initial_min_available_clients = self.min_available_clients
        self.round_counter = 0

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Optional[Dict[str, Scalar]]]:
        self.round_counter = rnd
        print(f"第 {rnd} 輪：正在處理總共 {len(results)} 個客戶端")
        
        # 從結果中提取指標
        metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        # 計算加權平均
        weighted_avg_result = self.weighted_average(metrics)
        # 獲取有效客戶端 ID 列表
        valid_client_ids = weighted_avg_result.get("Valid client_ids", [])
        
        # 過濾出有效的客戶端結果
        filtered_results = []
        for client, res in results:
            metric_client_id = res.metrics.get("client_id", "Unknown")
            if metric_client_id in valid_client_ids:
                filtered_results.append((client, res))
            print(f"客戶端{metric_client_id} "f"{client.cid}：{'有效' if metric_client_id in valid_client_ids else '無效'}")
        
        # 計算有效客戶端數量
        valid_clients_count = len(filtered_results)
        print(f"第 {rnd} 輪：{len(results)} 個客戶端中有 {valid_clients_count} 個有效客戶端")
        
        # 調整最小客戶端數量
        self.adjust_min_clients(valid_clients_count)

        # 根據有效客戶端數量決定使用哪些結果進行聚合
        if valid_clients_count >= self.min_fit_clients:
            print(f"第 {rnd} 輪：使用 {valid_clients_count} 個有效客戶端進行聚合")
            aggregate_result = super().aggregate_fit(rnd, filtered_results, failures)
        else:
            print(f"第 {rnd} 輪：有效客戶端數量不足，使用所有 {len(results)} 個客戶端進行聚合")
            aggregate_result = super().aggregate_fit(rnd, results, failures)
        
        print(f"第 {rnd} 輪：聚合完成")
        return aggregate_result

    def adjust_min_clients(self, valid_clients_count: int):
        if self.round_counter <= 10:
            self.min_fit_clients = self.initial_min_fit_clients
            self.min_evaluate_clients = self.initial_min_evaluate_clients
            self.min_available_clients = self.initial_min_available_clients
        else:
            new_min = valid_clients_count
            self.min_fit_clients = new_min
            self.min_evaluate_clients = new_min
            self.min_available_clients = new_min

        print(f"第{self.round_counter}輪: 調整後的min_fit_clients = {self.min_fit_clients}, "
              f"min_evaluate_clients = {self.min_evaluate_clients}, "
              f"min_available_clients = {self.min_available_clients}")

    def weighted_average(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        # print("Received metrics:", metrics)  # 新增：打印輸入的 metrics
        weighted_sum = 0
        total_examples = 0  # 重置为0
        Local_train_accuracies_List = []
        client_ids = []
        valid_client_ids = []
        valid_weights = []

        for num_examples, m in metrics:
            print("num_examples:",num_examples)
            client_id = m.get("client_id", "Unknown")
            client_ids.append(client_id)
            Local_train_accuracy = m.get("Local_train_accuracy", 0)
            Local_train_accuracies_List.append(Local_train_accuracy)
            global_round = m.get("global_round", 0)
            Local_weight_sum = m.get("Local_train_weight_sum", 0)
            weight_diff = m.get("Local_train_weight_sum-Previous_FedAVG weight_sum", 0)
            threshold = Local_weight_sum * 0.05

            #前面10round不看
            if global_round > 10:
                #權重差異超過當前本地權重總和的5%就要過濾掉  或 Local_train_accuracy大於90%才能列入計算
                if weight_diff <= threshold or Local_train_accuracy > 0.8:
                    if "accuracy" in m and Local_train_accuracy > 0.8:  # Local_train_accuracy大於90%才能列入計算
                        print(m["accuracy"])
                        weighted_sum += num_examples * m["accuracy"]
                        total_examples += num_examples  # 只在满足条件时累加
                        valid_weights.append(weight_diff)
                        valid_client_ids.append(client_id)
                else:
                    print(f"{client_id} excluded due to weight difference: {weight_diff}")

        print("Valid Weights (within threshold):", valid_weights)
        print("Total examples:", total_examples)
        print("Client IDs, Local train Accuracy:", list(zip(client_ids, Local_train_accuracies_List)))

        if total_examples == 0:
            return {"accuracy": 0, "client_ids": client_ids}

        weighted_accuracy = weighted_sum / total_examples
        return {
            "accuracy": weighted_accuracy,
            "client_ids": client_ids,
            "Total_examples": total_examples,
            "Valid Weights": valid_weights,
            "Valid client_ids": valid_client_ids
        }




# 使用自定義策略啟動Flower伺服器
strategy = CustomFedAvg(
    initial_parameters=initial_parameters,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    # evaluate_metrics_aggregation_fn=weighted_average,
)

# Define strategy
# strategy = fl.server.strategy.FedAvg(initial_parameters = initial_parameters, 
#                                      evaluate_metrics_aggregation_fn=weighted_average, 
#                                      min_fit_clients = 3, 
#                                      min_evaluate_clients = 3,
#                                      min_available_clients = 3)


# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:53388",
    # server_address="127.0.0.1:8080",
    # server_address="192.168.1.137:53388",

    # config=fl.server.ServerConfig(num_rounds=20),
    config=fl.server.ServerConfig(num_rounds=150),

    strategy=strategy,
)
