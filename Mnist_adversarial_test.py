import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import os
import random
from torch.utils.data import TensorDataset, DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # Max Pooling Layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # For MNIST (28x28 input size)
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch, losses):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, all_preds, all_labels

def fgsm_attack(data, epsilon, data_grad):
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def defense(device, train_loader, test_loader, epochs, Temp, epsilons):
    # 初始化兩個防禦網路模型
    modelF = Net().to(device)
    optimizerF = optim.Adam(modelF.parameters(), lr=0.0001, betas=(0.9, 0.999))
    schedulerF = optim.lr_scheduler.ReduceLROnPlateau(optimizerF, mode='min', factor=0.1, patience=3)

    modelF1 = Net().to(device)
    optimizerF1 = optim.Adam(modelF1.parameters(), lr=0.0001, betas=(0.9, 0.999))
    schedulerF1 = optim.lr_scheduler.ReduceLROnPlateau(optimizerF1, mode='min', factor=0.1, patience=3)

    criterion = nn.NLLLoss()

    # 訓練第一個防禦模型
    lossF = []
    for epoch in range(1, epochs + 1):
        modelF.train()
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizerF.zero_grad()
            output = modelF(data)
            loss = criterion(output / Temp, target)
            loss.backward()
            optimizerF.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        lossF.append(avg_loss)
        schedulerF.step(avg_loss)

    # 繪製第一個防禦模型的訓練損失曲線
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, epochs + 1), lossF, "*-")
    plt.title("Network F")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.show()

    # 生成軟標籤資料集
    soft_labels = []
    modelF.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            soft_label = F.log_softmax(modelF(data) / Temp, dim=1)  # 使用溫度縮放的軟標籤
            soft_labels.append((data.cpu(), soft_label.cpu()))

    # 建立軟標籤資料集和 DataLoader
    soft_data = torch.cat([x[0] for x in soft_labels], dim=0)
    soft_targets = torch.cat([x[1] for x in soft_labels], dim=0)
    soft_train_loader = DataLoader(TensorDataset(soft_data, soft_targets), batch_size=train_loader.batch_size, shuffle=True)

    # 訓練第二個防禦模型（基於軟標籤資料集）
    lossF1 = []
    for epoch in range(1, epochs + 1):
        modelF1.train()
        epoch_loss = 0
        for data, soft_target in soft_train_loader:
            data, soft_target = data.to(device), soft_target.to(device)
            optimizerF1.zero_grad()
            output = modelF1(data)
            loss = F.kl_div(output, soft_target, reduction='batchmean')
            loss.backward()
            optimizerF1.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(soft_train_loader)
        lossF1.append(avg_loss)
        schedulerF1.step(avg_loss)

    # 繪製第二個防禦模型的訓練損失曲線
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, epochs + 1), lossF1, "*-")
    plt.title("Network F'")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.show()

    # 使用測試集測試防禦效果
    modelF1.eval()
    for attack in ("fgsm", "ifgsm", "mifgsm"):
        accuracies = []
        examples = []
        for eps in epsilons:
            acc, ex = test_with_attack(modelF1, device, test_loader, eps)
            accuracies.append(acc)
            examples.append(ex)
        
        # 繪製準確率曲線
        plt.figure(figsize=(5, 5))
        plt.plot(epsilons, accuracies, "*-")
        plt.title(f"{attack} Defense - Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()

        # 顯示對抗樣本
        cnt = 0
        plt.figure(figsize=(8, 10))
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons), len(examples[0]), cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
                orig, adv, ex = examples[i][j]
                plt.title(f"{orig} -> {adv}")
                plt.imshow(ex, cmap="gray")
        plt.tight_layout()
        plt.show()

def test_with_attack(model, device, test_loader, epsilon):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 對每張圖片逐一進行攻擊
        for i in range(len(data)):
            data_i = data[i:i+1]  # 單獨處理每張圖片
            target_i = target[i:i+1]
            data_i.requires_grad = True
            output = model(data_i)
            init_pred = output.max(1, keepdim=True)[1]
            
            # 如果初始預測錯誤，跳過這張圖片
            if init_pred.item() != target_i.item():
                continue
            loss = F.nll_loss(output, target_i)
            model.zero_grad()
            loss.backward()
            data_grad = data_i.grad.data

            # 使用 FGSM 攻擊生成對抗樣本
            perturbed_data = fgsm_attack(data_i, epsilon, data_grad)
            
            # 重新預測對抗樣本
            output = model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]

            all_preds.append(final_pred.item())
            all_labels.append(target_i.item())

            if final_pred.item() == target_i.item():
                correct += 1
                # 收集對抗樣本
                if epsilon == 0 and len(adv_examples) < 5:
                    adv_ex = perturbed_data.detach().squeeze().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.detach().squeeze().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                
    final_acc = correct / float(len(test_loader.dataset))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {final_acc}")

    # Plot confusion matrix for adversarial attack
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for FGSM Attack (Epsilon: {epsilon})")
    plt.show()

    return final_acc, adv_examples
# 以ART實作的FGSM攻擊
# 以ART實作的FGSM攻擊 原版
# def art_fgsm_attack(model, device, test_loader, epsilon):
#     # 用 ART 建立 PyTorchClassifier
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adadelta(model.parameters())
#     classifier = PyTorchClassifier(
#         model=model,
#         loss=criterion,
#         optimizer=optimizer,
#         input_shape=(1, 28, 28),
#         nb_classes=10,
#         device_type="gpu" if torch.cuda.is_available() else "cpu"
#     )
    
#     # 建立 FGSM 攻擊器
#     attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    
#     correct = 0
#     all_preds = []
#     all_labels = []
#     adv_examples = []
    
#     for data, target in test_loader:
#         data, target = data.to(device), target.to(device)
        
#         # 將資料轉為 numpy 格式，並進行 FGSM 攻擊
#         data_np = data.cpu().numpy()
#         adv_data = attack.generate(x=data_np)
        
#         # 確認 adv_data 是 numpy array，並正確轉為 tensor
#         adv_data_tensor = torch.tensor(adv_data).to(device)
#         output = model(adv_data_tensor)
#         pred = output.argmax(dim=1, keepdim=True)
        
#         # 記錄預測結果與標籤
#         all_preds.extend(pred.cpu().numpy())
#         all_labels.extend(target.cpu().numpy())
        
#         correct += pred.eq(target.view_as(pred)).sum().item()
        
#         # 收集前5個對抗樣本
#         if len(adv_examples) < 5:
#             for i in range(min(5 - len(adv_examples), len(data))):
#                 adv_ex = adv_data_tensor[i].detach().squeeze().cpu().numpy()
#                 adv_examples.append((target[i].item(), pred[i].item(), adv_ex))
    
#     final_acc = correct / len(test_loader.dataset)
#     print(f"[ART FGSM] Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {final_acc}")
    
#     # 繪製混淆矩陣
#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(8, 8))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title(f"Confusion Matrix for ART FGSM Attack (Epsilon: {epsilon})")
#     plt.show()

#     # 確認生成的對抗樣本
#     if len(adv_examples) == 0:
#         print("Warning: No adversarial examples were generated. Please check the attack parameters.")
    
#     return final_acc, adv_examples

# 以ART實作的FGSM攻擊
# 選擇 20 個像素在每次迭代中添加大小為 0.01 的擾動
def art_fgsm_attack_paper(model, device, test_loader, epsilon):
    # 用 ART 建立 PyTorchClassifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 優化器
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    
    # 建立 FGSM 攻擊器
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    
    correct = 0
    all_preds = []
    all_labels = []
    adv_examples = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 將資料轉為 numpy 格式，並進行 FGSM 攻擊
        data_np = data.cpu().numpy()
        adv_data = attack.generate(x=data_np)

        # 轉換回 PyTorch tensor
        adv_data_tensor = torch.tensor(adv_data).to(device)
        
        # 僅選擇每張圖像中梯度絕對值最大的 20 個像素進行擾動
        for i in range(data.size(0)):  # 遍歷每個 batch 中的圖像
            data_grad = adv_data_tensor[i] - data[i]  # 計算每張圖像的擾動梯度
            flat_grad = data_grad.view(-1)
            _, top_indices = torch.topk(flat_grad.abs(), 20)  # 選擇 20 個梯度最大的像素索引
            
            # 在選定像素位置添加擾動幅度 0.01
            for idx in top_indices:
                adv_data_tensor.view(data.size(0), -1)[i, idx] += 0.01 * flat_grad[idx].sign()
        
        # 限制擾動在 [0, 1] 範圍內
        adv_data_tensor = torch.clamp(adv_data_tensor, 0, 1)
        
        output = model(adv_data_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        
        # 記錄預測結果與標籤
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 收集前 5 個對抗樣本
        if len(adv_examples) < 5:
            for i in range(min(5 - len(adv_examples), len(data))):
                adv_ex = adv_data_tensor[i].detach().squeeze().cpu().numpy()
                adv_examples.append((target[i].item(), pred[i].item(), adv_ex))
    
    final_acc = correct / len(test_loader.dataset)
    print(f"[ART FGSM] Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {final_acc}")
    
    # 繪製混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for ART FGSM Attack (Epsilon: {epsilon})")
    plt.show()

    # 確認生成的對抗樣本
    if len(adv_examples) == 0:
        print("Warning: No adversarial examples were generated. Please check the attack parameters.")
    
    return final_acc, adv_examples
def plot_loss_curve(losses):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.show()

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# def plot_adversarial_examples(examples, epsilons):
#     plt.figure(figsize=(10, 10))
#     cnt = 0
#     for i, epsilon in enumerate(epsilons):
#         for j, (orig, adv, ex) in enumerate(examples[i]):
#             cnt += 1
#             plt.subplot(len(epsilons), len(examples[0]), cnt)
#             plt.xticks([], [])
#             plt.yticks([], [])
#             if j == 0:
#                 plt.ylabel(f"Eps: {epsilon}", fontsize=14)
#             plt.title(f"{orig} -> {adv}")
#             plt.imshow(ex, cmap="gray")
#     plt.tight_layout()
#     plt.show()
def plot_adversarial_examples(examples, epsilons):
    plt.figure(figsize=(10, 10))
    cnt = 0
    for i, epsilon in enumerate(epsilons):
        # 隨機從 examples 中抽樣顯示，避免顯示的都是未干擾圖片
        sampled_examples = random.sample(examples[i], min(5, len(examples[i])))
        for j, (orig, adv, ex) in enumerate(sampled_examples):
            cnt += 1
            plt.subplot(len(epsilons), 5, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilon}", fontsize=14)
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST FGSM Attack Example')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=False)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # model = Net().to(device)
    # Initialize the model for MNIST (1 input channel for grayscale)
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # 修改後
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    losses = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, losses)
        scheduler.step()

    plot_loss_curve(losses)
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    accuracy, all_preds, all_labels = test(model, device, test_loader)
    plot_confusion_matrix(all_labels, all_preds)

    # epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,1.0]
    epsilons = [0,0.01 ,0.05, 0.1, 0.15, 0.2, 0.25, 0.3,1.0]
    # accuracies_FGSM = []
    # examples_FGSM = []

    # print("\nSelf-Implemented FGSM Attack Results:")
    # for eps in epsilons:
    #     acc, ex = test_with_attack(model, device, test_loader, eps)
    #     accuracies_FGSM.append(acc)
    #     examples_FGSM.append(ex)

    # plt.figure(figsize=(5,5))
    # plt.plot(epsilons, accuracies_FGSM, "*-")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Accuracy")
    # plt.title("Self-Implemented FGSM Attack - Accuracy vs Epsilon")
    # plt.show()

    # plot_adversarial_examples(examples_FGSM, epsilons)

    accuracies_ART = []
    examples_ART = []

    print("\nART FGSM Attack Results:")
    for eps in epsilons:
        # acc, ex = art_fgsm_attack(model, device, test_loader, eps)
        acc, ex = art_fgsm_attack_paper(model, device, test_loader, eps)
        accuracies_ART.append(acc)
        examples_ART.append(ex)
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies_ART, "*-")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("ART FGSM Attack - Accuracy vs Epsilon")
    plt.show()

    plot_adversarial_examples(examples_ART, epsilons)
    
    #防禦方法
    # Temp=100
    # epochs=10
    # epsilons=[0,0.007,0.01,0.02,0.03,0.05,0.1,0.2,0.3]
    # defense(device,train_loader,test_loader,epochs,Temp,epsilons)

if __name__ == '__main__':
    main()
