import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from mytoolfunction import generatefolder

'''
直接訓練正常資料和JSMA攻擊後的資料去訓練鑑別器
主要作用是在鑑別器其功用是分別出正常流量跟JSMA攻擊後的流量
最大用意是要查出JSMA攻擊後的流量
所以gloss 和 dloss要差距越大不能收斂，dloss要越大
'''

# 檢查是否可以使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義判別器模型
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# 訓練判別器函數，包括對抗樣本和正常樣本
def train_discriminator_with_adversarial(discriminator, normal_loader, adversarial_loader, epochs=5):
    criterion = nn.BCELoss()  # 使用二元交叉熵損失
    optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)  # 使用RMSprop優化器

    losses = []  # 記錄每個epoch的損失
    for epoch in range(epochs):
        epoch_loss = 0  # 初始化epoch損失
        normal_iter = iter(normal_loader)
        adversarial_iter = iter(adversarial_loader)

        for _ in range(len(normal_loader)):
            try:
                normal_data, normal_labels = next(normal_iter)
                adv_data, adv_labels = next(adversarial_iter)
            except StopIteration:
                break

            normal_data, normal_labels = normal_data.float().to(DEVICE), normal_labels.float().to(DEVICE)
            adv_data, adv_labels = adv_data.float().to(DEVICE), adv_labels.float().to(DEVICE)
            
            combined_data = torch.cat([normal_data, adv_data])
            combined_labels = torch.cat([normal_labels, adv_labels]).unsqueeze(1)

            optimizer.zero_grad()
            outputs = discriminator(combined_data)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(normal_loader)  # 計算平均損失
        losses.append(avg_loss)  # 記錄平均損失
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    return losses

# 繪製混淆矩陣
def draw_confusion_matrix(y_true, y_pred, plot_confusion_matrix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(plot_confusion_matrix)
    plt.close()

# 測試模型性能
def test(net, testloader, start_time, client_str, plot_confusion_matrix, label_count, today, choose_method):
    criterion = nn.BCELoss()
    correct = 0
    total = 0
    loss = 0.0
    ave_loss = 0.0

    net.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = net(data)
            batch_loss = criterion(outputs, labels.view(-1, 1)).item()
            loss += batch_loss
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.view(-1, 1)).sum().item()

            ave_loss = ave_loss * 0.9 + batch_loss * 0.1

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"./Adversarial_Attack_Test/{today}/baseline_report_{client_str}.csv", header=True)

    recalls = [report[str(i)]['recall'] if str(i) in report else 0.0 for i in range(label_count)]
    f1_scores = [report[str(i)]['f1-score'] if str(i) in report else 0.0 for i in range(label_count)]

    with open(f"./Adversarial_Attack_Test/{today}/metrics_baseline_{client_str}.csv", "a+") as file:
        file.write("Epoch,Accuracy,Average Loss,Time," + ",".join([f"Recall_{i}" for i in range(label_count)]) + "," + ",".join([f"F1_{i}" for i in range(label_count)]) + "\n")
        file.write(f"{len(testloader)}, {accuracy:.4f}, {ave_loss:.4f}, {time.time() - start_time}," + ",".join([f"{recall:.4f}" for recall in recalls]) + "," + ",".join([f"{f1:.4f}" for f1 in f1_scores]) + "\n")

    draw_confusion_matrix(y_true, y_pred, plot_confusion_matrix)

    print(f"測試準確度: {accuracy:.4f}")
    print(f"測試損失: {ave_loss:.4f}")

    return accuracy, ave_loss, recalls, f1_scores

# 客戶端訓練代碼示例
def client_training(client_id, normal_data, normal_labels, adv_data, adv_labels, test_data, test_labels, input_size, hidden_size, label_count, epochs=5, plot_confusion_matrix='confusion_matrix.png', today='today', choose_method='method'):
    if len(normal_data) != len(normal_labels):
        raise ValueError(f"Normal data and labels size mismatch: {len(normal_data)} vs {len(normal_labels)}")
    if len(adv_data) != len(adv_labels):
        raise ValueError(f"Adversarial data and labels size mismatch: {len(adv_data)} vs {len(adv_labels)}")

    print(f"Normal data shape: {normal_data.shape}")
    print(f"Normal labels shape: {normal_labels.shape}")
    print(f"Adversarial data shape: {adv_data.shape}")
    print(f"Adversarial labels shape: {adv_labels.shape}")

    min_size = min(len(adv_data), len(adv_labels))
    adv_data = adv_data[:min_size]
    adv_labels = adv_labels[:min_size]

    normal_dataset = TensorDataset(torch.tensor(normal_data, dtype=torch.float32), torch.tensor(normal_labels, dtype=torch.float32).view(-1))
    adversarial_dataset = TensorDataset(torch.tensor(adv_data, dtype=torch.float32), torch.tensor(adv_labels, dtype=torch.float32).view(-1))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32).view(-1))
    
    normal_loader = DataLoader(normal_dataset, batch_size=512, shuffle=True)
    adversarial_loader = DataLoader(adversarial_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    discriminator = Discriminator(input_size, hidden_size).to(DEVICE)
    losses = train_discriminator_with_adversarial(discriminator, normal_loader, adversarial_loader, epochs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Training Loss')
    plt.legend()
    plt.savefig(f'./Adversarial_Attack_Test/{today}/loss_curve_{client_id}.png')
    plt.close()

    accuracy, total_loss, recalls, f1_scores = test(discriminator, test_loader, time.time(), client_id, plot_confusion_matrix, label_count, today, choose_method)
    print(f'Client {client_id} - Accuracy: {accuracy:.4f}, Average Loss: {total_loss:.4f}')
    
    torch.save(discriminator.state_dict(), f'./Adversarial_Attack_Test/{today}/discriminator_client_{client_id}.pth')
    print(f'Client {client_id} - Model saved as discriminator_client_{client_id}.pth')

    return discriminator.state_dict()

# 示例用法
if __name__ == "__main__":
    x_train_RemoveString_np = np.load("./Adversarial_Attack_Test/20240722_FL_cleint3_use_train_.0.05_0.02/x_testdata_removestring_20240722.npy", allow_pickle=True)
    y_train_RemoveString_np = np.load("./Adversarial_Attack_Test/20240722_FL_cleint3_use_train_.0.05_0.02/y_testdata_removestring_20240722.npy", allow_pickle=True)

    adversarial_samples = np.load("./Adversarial_Attack_Test/20240725_FL_cleint3_use_train_.0.05_0.02/x_DoJSMA_adversarialsample_removestring_20240725.npy", allow_pickle=True)
    adversarial_labels = np.load("./Adversarial_Attack_Test/20240725_FL_cleint3_use_train_.0.05_0.02/y_DoJSMA_adversarialsample_removestring_20240725.npy", allow_pickle=True)

    x_test_RemoveString_np = np.load("./Adversarial_Attack_Test/20240721_0.5_0.5/x_testdata_removestring_20240721.npy", allow_pickle=True)
    y_test_RemoveString_np = np.load("./Adversarial_Attack_Test/20240721_0.5_0.5/y_testdata_removestring_20240721.npy", allow_pickle=True)

    num_clients = 1
    input_size = 38
    hidden_size = 512
    label_count = 10
    today = '20240725'
    choose_method = 'JSMA'

    generatefolder(f"./Adversarial_Attack_Test/", today)
    for client_id in range(num_clients):
        normal_data = x_train_RemoveString_np
        normal_labels = y_train_RemoveString_np
        adv_data = adversarial_samples
        adv_labels = adversarial_labels
        test_data = x_test_RemoveString_np
        test_labels = y_test_RemoveString_np

        print(f"Client {client_id} - Normal data size: {normal_data.shape}, Normal labels size: {normal_labels.shape}")
        print(f"Client {client_id} - Adversarial data size: {adv_data.shape}, Adversarial labels size: {adv_labels.shape}")
        
        client_training(client_id, normal_data, normal_labels, adv_data, adv_labels, test_data, test_labels, input_size, hidden_size, label_count, epochs=5, plot_confusion_matrix=f'confusion_matrix_client_{client_id}.png', today=today, choose_method=choose_method)
