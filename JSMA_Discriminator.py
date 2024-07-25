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
from collections import Counter

# 检查是否可以使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        # 使用两层全连接层的神经网络
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 输入层到隐藏层
            nn.ReLU(True),  # 激活函数
            nn.Linear(hidden_size, 1),  # 隐藏层到输出层
            nn.Sigmoid()  # Sigmoid激活函数
        )

    def forward(self, x):
        return self.fc(x)

# 训练判别器函数，包括对抗样本和正常样本
def train_discriminator_with_adversarial(discriminator, normal_loader, adversarial_loader, epochs=5):
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)  # 使用RMSprop优化器
    losses = []  # 记录每个 epoch 的损失值
    
    for epoch in range(epochs):
        epoch_loss = 0
        # 从正常和对抗样本加载器中获取数据
        for (normal_data, normal_labels), (adv_data, adv_labels) in zip(normal_loader, adversarial_loader):
            # 将数据移到设备（CPU或GPU）
            normal_data, normal_labels = normal_data.float().to(DEVICE), normal_labels.float().to(DEVICE)
            adv_data, adv_labels = adv_data.float().to(DEVICE), adv_labels.float().to(DEVICE)
            
            # 合并正常样本和对抗样本
            combined_data = torch.cat([normal_data, adv_data])
            combined_labels = torch.cat([normal_labels, adv_labels]).unsqueeze(1)  # 调整标签大小

            # 重置梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = discriminator(combined_data)
            # 计算损失
            loss = criterion(outputs, combined_labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累积损失
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(normal_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    return losses

# 绘制混淆矩阵
def draw_confusion_matrix(y_true, y_pred, plot_confusion_matrix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(plot_confusion_matrix)
    plt.close()

# 测试模型性能
def test(net, testloader, start_time, client_str, plot_confusion_matrix, label_count, today, choose_method):
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    correct = 0
    total = 0
    loss = 0.0
    ave_loss = 0.0  # 初始化 ave_loss

    net.eval()  # 设置模型为评估模式
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, labels in testloader:
            # 将数据移到设备（CPU或GPU）
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            # 前向传播
            outputs = net(data)  # 使用 squeeze() 来移除最后一个维度
            # 计算损失
            batch_loss = criterion(outputs, labels.view(-1, 1)).item()  # 调整目标大小
            loss += batch_loss
            # 预测结果
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.view(-1, 1)).sum().item()  # 调整目标大小

            # 计算滑动平均损失
            ave_loss = ave_loss * 0.9 + batch_loss * 0.1

            # 将标签和预测结果转换为 NumPy 数组
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算准确度
    accuracy = correct / total
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    # 生成分类报告
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"./Adversarial_Attack_Test/{today}/baseline_report_{client_str}.csv", header=True)

    # 计算每个类别的召回率和 F1 分数
    recalls = [report[str(i)]['recall'] if str(i) in report else 0.0 for i in range(label_count)]
    f1_scores = [report[str(i)]['f1-score'] if str(i) in report else 0.0 for i in range(label_count)]

    # 记录总体准确度、召回率和 F1 分数
    with open(f"./Adversarial_Attack_Test/{today}/metrics_baseline_{client_str}.csv", "a+") as file:
        file.write("Epoch,Accuracy,Average Loss,Time," + ",".join([f"Recall_{i}" for i in range(label_count)]) + "," + ",".join([f"F1_{i}" for i in range(label_count)]) + "\n")
        file.write(f"{len(testloader)}, {accuracy:.4f}, {ave_loss:.4f}, {time.time() - start_time}," + ",".join([f"{recall:.4f}" for recall in recalls]) + "," + ",".join([f"{f1:.4f}" for f1 in f1_scores]) + "\n")

    # 画混淆矩阵
    draw_confusion_matrix(y_true, y_pred, plot_confusion_matrix)

    print(f"测试准确度: {accuracy:.4f}")
    print(f"测试损失: {ave_loss:.4f}")

    return accuracy, ave_loss, recalls, f1_scores

# 客户端训练代码示例
def client_training(client_id, normal_data, normal_labels, adv_data, adv_labels, test_data, test_labels, input_size, hidden_size, label_count, epochs=5, plot_confusion_matrix='confusion_matrix.png', today='today', choose_method='method'):
    # 检查数据集大小是否匹配
    if len(normal_data) != len(normal_labels):
        raise ValueError(f"Normal data and labels size mismatch: {len(normal_data)} vs {len(normal_labels)}")
    if len(adv_data) != len(adv_labels):
        raise ValueError(f"Adversarial data and labels size mismatch: {len(adv_data)} vs {len(adv_labels)}")

    # 打印数据集形状
    print(f"Normal data shape: {normal_data.shape}")
    print(f"Normal labels shape: {normal_labels.shape}")
    print(f"Adversarial data shape: {adv_data.shape}")
    print(f"Adversarial labels shape: {adv_labels.shape}")

    # 修正对抗样本和标签的大小不匹配问题
    min_size = min(len(adv_data), len(adv_labels))
    adv_data = adv_data[:min_size]
    adv_labels = adv_labels[:min_size]

    # 创建数据集和数据加载器
    normal_dataset = TensorDataset(torch.tensor(normal_data, dtype=torch.float32), torch.tensor(normal_labels, dtype=torch.float32).view(-1))
    adversarial_dataset = TensorDataset(torch.tensor(adv_data, dtype=torch.float32), torch.tensor(adv_labels, dtype=torch.float32).view(-1))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32).view(-1))
    
    normal_loader = DataLoader(normal_dataset, batch_size=32, shuffle=True)
    adversarial_loader = DataLoader(adversarial_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化判别器并移动到设备（CPU或GPU）
    discriminator = Discriminator(input_size, hidden_size).to(DEVICE)
    # 训练判别器
    losses = train_discriminator_with_adversarial(discriminator, normal_loader, adversarial_loader, epochs)
    
    # 测试判别器并获取性能指标
    accuracy, total_loss, recalls, f1_scores = test(discriminator, test_loader, time.time(), client_id, plot_confusion_matrix, label_count, today, choose_method)
    print(f'Client {client_id} - Accuracy: {accuracy:.4f}, Average Loss: {total_loss:.4f}')
    
    # 保存训练好的判别器模型
    torch.save(discriminator.state_dict(), f'./Adversarial_Attack_Test/{today}/discriminator_client_{client_id}.pth')
    print(f'Client {client_id} - Model saved as discriminator_client_{client_id}.pth')

    # 绘制训练损失曲线
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(f'./Adversarial_Attack_Test/{today}/training_loss_curve_{client_id}.png')
    plt.close()

    return discriminator.state_dict()

# 示例用法
if __name__ == "__main__":
    # 加载数据集
    # 替换为真实的正常样本
    x_train_RemoveString_np = np.load("./Adversarial_Attack_Test/20240722_FL_cleint3_use_train_.0.05_0.02/x_testdata_removestring_20240722.npy", allow_pickle=True)
    y_train_RemoveString_np = np.load("./Adversarial_Attack_Test/20240722_FL_cleint3_use_train_.0.05_0.02/y_testdata_removestring_20240722.npy", allow_pickle=True)

    # 替换为真实的对抗样本
    adversarial_samples = np.load("./Adversarial_Attack_Test/20240725_FL_cleint3_use_train_.0.05_0.02/x_DoJSMA_adversarialsample_removestring_20240725.npy", allow_pickle=True)
    adversarial_labels = np.load("./Adversarial_Attack_Test/20240725_FL_cleint3_use_train_.0.05_0.02/y_DoJSMA_adversarialsample_removestring_20240725.npy", allow_pickle=True)

    # 加载测试数据
    x_test_RemoveString_np = np.load("./Adversarial_Attack_Test/20240721_0.5_0.5/x_testdata_removestring_20240721.npy", allow_pickle=True)
    y_test_RemoveString_np = np.load("./Adversarial_Attack_Test/20240721_0.5_0.5/y_testdata_removestring_20240721.npy", allow_pickle=True)

    counter = Counter(y_train_RemoveString_np)
    print("y_train_RemoveString_np筆數",counter)

    counter = Counter(adversarial_labels)
    print("adversarial_labels筆數",counter)

    counter = Counter(y_test_RemoveString_np)
    print("y_test_RemoveString_np筆數",counter)
    # 假设有5个客户端
    num_clients = 1
    input_size = 38  # 根据你的数据特征数设置
    hidden_size = 512
    label_count = 10  # 假设有10个类别
    today = '20240725'
    choose_method = 'JSMA'

    generatefolder(f"./Adversarial_Attack_Test/", today)
    for client_id in range(num_clients):
        # 替换为每个客户端的实际数据
        normal_data = x_train_RemoveString_np
        normal_labels = y_train_RemoveString_np
        adv_data = adversarial_samples
        adv_labels = adversarial_labels
        test_data = x_test_RemoveString_np  # 使用实际的测试数据
        test_labels = y_test_RemoveString_np  # 使用实际的测试标签

        # 打印正常样本和标签的大小
        print(f"Client {client_id} - Normal data size: {normal_data.shape}, Normal labels size: {normal_labels.shape}")
        # 打印对抗样本和标签的大小
        print(f"Client {client_id} - Adversarial data size: {adv_data.shape}, Adversarial labels size: {adv_labels.shape}")
        
        client_training(client_id, normal_data, normal_labels, adv_data, adv_labels, test_data, test_labels, input_size, hidden_size, label_count, epochs=5, plot_confusion_matrix=f'confusion_matrix_client_{client_id}.png', today=today, choose_method=choose_method)
