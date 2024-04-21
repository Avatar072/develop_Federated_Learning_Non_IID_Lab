from typing import List, Tuple

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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(44, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.layer5 = nn.Linear(512, 23)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.layer5(x)
        return x

# 初始化模型和優化器
model = MLP()
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
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print(examples)

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(initial_parameters = initial_parameters, evaluate_metrics_aggregation_fn=weighted_average, 
    min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2)

# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:53388",
    # server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)
