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
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

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

# def PGD_attack(data, epsilon, data_grad):
#     perturbed_data = data + epsilon * data_grad.sign()
#     perturbed_data = torch.clamp(perturbed_data, 0, 1)
#     return perturbed_data

def pgd_attack(data, epsilon, model, loss_fn, num_steps=10, alpha=0.01, random_start=True):
    """
    Performs PGD attack on the input data
    
    Args:
        data: Input data to be perturbed
        epsilon: Maximum perturbation allowed (L-inf norm)
        model: Neural network model
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
        num_steps: Number of PGD iterations
        alpha: Step size for each iteration
        random_start: Whether to start with random perturbation
        
    Returns:
        perturbed_data: Adversarially perturbed data
    """
    # Make a copy of the input data
    perturbed_data = data.clone().detach()
    
    # Start from a random point within epsilon ball if random_start is True
    if random_start:
        perturbed_data = perturbed_data + torch.zeros_like(perturbed_data).uniform_(-epsilon, epsilon)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

    # Get the original predictions to determine target labels
    with torch.no_grad():
        original_output = model(data)
        target = original_output.argmax(dim=1)

    for step in range(num_steps):
        # Set requires_grad attribute of tensor
        perturbed_data.requires_grad = True
        
        # Forward pass
        output = model(perturbed_data)
        
        # Calculate loss
        loss = loss_fn(output, target)
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Calculate gradients
        loss.backward()
        
        # Get the sign of the gradients
        sign_data_grad = perturbed_data.grad.sign()
        
        # Create new tensor for the update
        with torch.no_grad():
            # Perform step
            perturbed_data = perturbed_data.detach() + alpha * sign_data_grad
            
            # Project back to epsilon ball
            delta = perturbed_data - data
            delta = torch.clamp(delta, -epsilon, epsilon)
            perturbed_data = data + delta
            
            # Clamp values to valid range [0,1]
            perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data

def test_with_attack(model, device, test_loader, epsilon):
    """
    Test the model with PGD attack
    """
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    adv_examples = []

    # 創建loss函數
    criterion = nn.CrossEntropyLoss()
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 對每張圖片逐一進行攻擊
        for i in range(len(data)):
            data_i = data[i:i+1]  # 單獨處理每張圖片
            target_i = target[i:i+1]
            
            # 獲取初始預測
            with torch.no_grad():
                output = model(data_i)
                init_pred = output.max(1, keepdim=True)[1]
            
            # 如果初始預測錯誤，跳過這張圖片
            if init_pred.item() != target_i.item():
                continue

            # 對單個數據點進行攻擊
            perturbed_data = pgd_attack(
                data=data_i,
                epsilon=epsilon,
                model=model,
                loss_fn=criterion,
                num_steps=10,
                alpha=0.01,
                random_start=True
            )

            # 重新預測對抗樣本
            with torch.no_grad():
                output = model(perturbed_data)
                final_pred = output.max(1, keepdim=True)[1]

            all_preds.append(final_pred.item())
            all_labels.append(target_i.item())

            if final_pred.item() == target_i.item():
                correct += 1
                # 收集對抗樣本
                if len(adv_examples) < 5:
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
    plt.title(f"Confusion Matrix for PGD Attack (Epsilon: {epsilon})")
    plt.show()

    return final_acc, adv_examples

# 以ART實作的PGD攻擊
def art_PGD_attack(model, device, test_loader, epsilon):
    # 用 ART 建立 PyTorchClassifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    
    # 建立 PGD 攻擊器
    # attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    # attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=0.01, max_iter=10)
    attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=0.05, max_iter=10)
    correct = 0
    all_preds = []
    all_labels = []
    adv_examples = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 將資料轉為 numpy 格式，並進行 PGD 攻擊
        data_np = data.cpu().numpy()
        adv_data = attack.generate(x=data_np)
        
        # 確認 adv_data 是 numpy array，並正確轉為 tensor
        adv_data_tensor = torch.tensor(adv_data).to(device)
        output = model(adv_data_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        
        # 記錄預測結果與標籤
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 收集前5個對抗樣本
        if len(adv_examples) < 5:
            for i in range(min(5 - len(adv_examples), len(data))):
                adv_ex = adv_data_tensor[i].detach().squeeze().cpu().numpy()
                adv_examples.append((target[i].item(), pred[i].item(), adv_ex))
    
    final_acc = correct / len(test_loader.dataset)
    print(f"[ART PGD] Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {final_acc}")
    
    # 繪製混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for ART PGD Attack (Epsilon: {epsilon})")
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
            plt.title(f"original{orig} -> adversarial{adv}")
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST PGD Attack Example')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1)
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

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
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

    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,1.0]
    accuracies_PGD = []
    examples_PGD = []

    # print("\nSelf-Implemented PGD Attack Results:")
    # for eps in epsilons:
    #     acc, ex = test_with_attack(model, device, test_loader, eps)
    #     accuracies_PGD.append(acc)
    #     examples_PGD.append(ex)

    # plt.figure(figsize=(5,5))
    # plt.plot(epsilons, accuracies_PGD, "*-")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Accuracy")
    # plt.title("Self-Implemented PGD Attack - Accuracy vs Epsilon")
    # plt.show()

    # plot_adversarial_examples(examples_PGD, epsilons)

    accuracies_ART = []
    examples_ART = []

    print("\nART PGD Attack Results:")
    for eps in epsilons:
        acc, ex = art_PGD_attack(model, device, test_loader, eps)
        accuracies_ART.append(acc)
        examples_ART.append(ex)

    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies_ART, "*-")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("ART PGD Attack - Accuracy vs Epsilon")
    plt.show()

    plot_adversarial_examples(examples_ART, epsilons)

if __name__ == '__main__':
    main()
