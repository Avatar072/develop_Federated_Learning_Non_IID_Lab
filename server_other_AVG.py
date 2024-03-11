'''
引入套件
'''
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import f1_score, recall_score, precision_score

'''
Step 1. Build Global Model (建立全域模型)
'''

# Define the PyTorch model class
class MLP_Model(nn.Module):
    def __init__(self, input_size):
        super(MLP_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 22)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x

'''
Step 2. Start server and run the strategy (套用所設定的策略，啟動Server)
'''
csv_filename = 'D:/flower_test/server_accuracy/evaluation_results.csv'
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
dftest = pd.read_csv("D:/develop_Federated_Learning_Non_IID_Lab/data/dataset_AfterProcessed/TONIOT_test_and_CICIDS2017_test_combine/20240310/TONIOT_test_and_CICIDS2017_test_merged.csv")    
x_columns = dftest.columns.drop(dftest.filter(like='Label').columns)
x_test = torch.tensor(dftest[x_columns].values.astype('float32'))
y_test = torch.tensor(dftest.filter(like='Label').values.squeeze(), dtype=torch.int64)
print(y_test)

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = MLP_Model(input_size=44)
    #model.summary()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_fn=get_eval_fn(model),
        # initial_parameters=fl.common.weights_to_parameters(model.state_dict()),
    )

    # Start Flower server for ten rounds of federated learning
    fl.server.start_server(server_address="127.0.0.1:53388", config=fl.server.ServerConfig(num_rounds=5), strategy=strategy) #windows

'''
[Model Hyperparameter](Client-side, train strategy)
'''
def fit_config(rnd: int):
    config = {
        "batch_size": 256,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config

'''
[Model Hyperparameter](Client-side, evaluate strategy)
'''
def evaluate_config(rnd: int):
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

'''
[Model Hyperparameter](Server-side, evaluate strategy) 
'''

def get_eval_fn(model):
    round_counter = 0

    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Round', 'Loss', 'Accuracy', 'Recall_weighted', 'Recall_macro']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            csv_writer.writeheader()

    def evaluate(self, parameters, config) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        nonlocal round_counter
        # model.load_state_dict(parameters)
        model.eval()

        # Predict on the test set
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            outputs = model(x_test)
            loss = criterion(outputs, y_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            y_test_class = np.argmax(y_test)
            y_test_pred_class = np.argmax(outputs)
            true_labels = y_test_class
            predicted_labels = y_test_pred_class
            recall = recall_score(y_test.numpy(), predicted.numpy(), average='weighted')
            recall1 = recall_score(y_test.numpy(), predicted.numpy(), average='macro')
            class_names = {
                0: 'Bot', 1: 'DDoS', 2: 'DoSGoldenEye', 3: 'DoSHulk', 4: 'DoSSlowhttptest', 5: 'DoSslowloris', 
                6: 'FTPPatator', 7: 'Heartbleed', 8: 'Infiltration', 9: 'PortScan', 10: 'SSHPatator', 
                11: 'WebAttackBruteForce', 12: 'WebAttackSqlInjection', 13: 'WebAttackXSS', 14: 'backdoor', 
                15: 'dos', 16: 'injection', 17: 'mitm', 18: 'normal', 19: 'password', 20: 'ransomware', 
                21: 'scanning'
            }

            f1_scores = []
            precision_scores = []
            recall_scores = []
            label_counts = {label: (y_test_class == label).sum() for label in range(22)}

            for class_idx in range(22):  # Assuming 22 classes
                true_labels = (y_test_class == class_idx)
                predicted_labels = (y_test_pred_class == class_idx)
                f1 = f1_score(true_labels, predicted_labels, zero_division=1)
                precision = precision_score(true_labels, predicted_labels, zero_division=1)
                recall = recall_score(true_labels, predicted_labels, zero_division=1)
        
                f1_scores.append(f1)
                precision_scores.append(precision)
                recall_scores.append(recall)


            results_table = pd.DataFrame({
                'Class': [class_names[i] for i in range(22)],
                'Precision': precision_scores,
                'Recall': recall_scores,
                'F1-Score': f1_scores,
                'Label Count': [label_counts[i] for i in range(22)]
        
                })

            print("Results Table:")
            print(results_table)

            round_counter += 1

            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                csv_writer.writerow({
                    'Round': round_counter,
                    'Loss': loss.item(),
                    'Accuracy': accuracy,
                    'Recall_weighted': recall,
                    'Recall_macro': recall1
                })

            print(f'Recall: {recall}')
        return loss.item(), {"accuracy": accuracy, "recall_weighted": recall, "recall_macro": recall1}

    return evaluate


'''
main
'''
if __name__ == "__main__":
    main()
