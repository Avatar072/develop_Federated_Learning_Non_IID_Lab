import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import os
import random
import time
import datetime
from collections import Counter, defaultdict

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# class Net(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = self.fc4(x)
#         return F.log_softmax(x, dim=1)
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        # 使用 input_size 和 output_size 參數來設置網絡的輸入和輸出維度
        self.layer1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.layer5 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.layer5(x)
        return x
def generatefolder(path, foldername):
    if not os.path.exists(os.path.join(path, foldername)):
        os.makedirs(os.path.join(path, foldername))

def save_to_csv(data, filepath):
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

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
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    losses.append(epoch_loss / len(train_loader))

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
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, all_preds, all_labels

def plot_results(losses, accuracies, confusion_mtx, save_dir):
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def pgd_attack_evaluation(model, device, test_loader, classifier, attack, save_dir, epsilon):
    model.eval()
    successful_attacks = []
    accuracies = []
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data_np = data.cpu().numpy()
        
        # Generate adversarial examples
        x_adv = attack.generate(x=data_np)
        
        # Get predictions for adversarial examples
        predictions = classifier.predict(x_adv)
        adv_preds = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(adv_preds == target.cpu().numpy())
        accuracies.append(accuracy)
        
        # Record successful attacks
        for i in range(len(target)):
            if adv_preds[i] != target[i].cpu().numpy():
                successful_attacks.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'original_class': target[i].item(),
                    'adversarial_class': adv_preds[i]
                })
    
    # Save results
    avg_accuracy = np.mean(accuracies)
    print(f'Average accuracy under PGD attack (ε={epsilon}): {avg_accuracy:.4f}')
    
    attack_results = pd.DataFrame(successful_attacks)
    attack_results.to_csv(os.path.join(save_dir, f'successful_attacks_eps_{epsilon}.csv'), index=False)
    
    return avg_accuracy, successful_attacks

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='PGD Attack on CICIDS2017')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=True)
    args = parser.parse_args()

    # Create save directory
    today = datetime.date.today().strftime("%Y%m%d")
    save_dir = f"./Adversarial_Attack_Test/{today}"
    generatefolder("./Adversarial_Attack_Test/", today)

    # Load CICIDS2019 dataset
    filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"

    # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
    x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_20240502.npy", allow_pickle=True)
    y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_20240502.npy", allow_pickle=True)
    # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
    x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_20240502.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_20240502.npy", allow_pickle=True)
    # 轉換為張量
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    
    # 創建數據加載器
    # dataset = TensorDataset(X_tensor, y_tensor)
    # train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)
    # 創建用於訓練和測試的數據加載器
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=500, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    # 初始化模型
    input_size = x_train.shape[1]  # 特徵數量
    num_classes = len(np.unique(y_train))  # 類別數量
    model = MLP(input_size, num_classes).to(device)
    
    # 每層神經元512下所訓練出來的model
    model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2019\\BaseLine_After_local_train_model_bk.pth'
    # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2019\\BaseLine_After_local_train_model_e500CandW.pth'
    model.load_state_dict(torch.load(model_path))

    # 設定優化器和調度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 訓練模型
    losses = []
    for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, epoch, losses)
        scheduler.step()

    # 測試模型
    accuracy, all_preds, all_labels = test(model, device, test_loader)
    
    # 創建 ART 分類器

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(input_size,),
        nb_classes=num_classes,
        clip_values=(0.0, 1.0)
    )

    # 設定 PGD 攻擊
    epsilons = [0.1, 0.2, 0.3]
    for epsilon in epsilons:
        attack = ProjectedGradientDescent(
            estimator=classifier,
            eps=epsilon,
            eps_step=0.1,
            max_iter=100,
            targeted=False,
            num_random_init=0
        )
        
        # 執行攻擊並評估
        acc, successful_attacks = pgd_attack_evaluation(
            model, device, test_loader, classifier, attack, save_dir, epsilon
        )

    # 保存模型和結果
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        
    # 繪製結果
    confusion_mtx = confusion_matrix(all_labels, all_preds)
    plot_results(losses, [accuracy], confusion_mtx, save_dir)

if __name__ == '__main__':
    main()