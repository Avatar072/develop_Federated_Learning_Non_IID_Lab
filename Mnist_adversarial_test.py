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
import torchvision
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

    # Display some MNIST test images and their predictions
    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0].cpu(), cmap='gray', interpolation='none')
        plt.title(f"True: {target[i].item()} Pred: {pred[i].item()}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return accuracy, all_preds, all_labels

def fgsm_attack(data, epsilon, data_grad):
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

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

def plot_adversarial_examples(examples, epsilons):
    plt.figure(figsize=(10, 10))
    cnt = 0
    for i, epsilon in enumerate(epsilons):
        for j, (orig, adv, ex) in enumerate(examples[i]):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
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
    accuracies_FGSM = []
    examples_FGSM = []

    for eps in epsilons:
        acc, ex = test_with_attack(model, device, test_loader, eps)
        accuracies_FGSM.append(acc)
        examples_FGSM.append(ex)

    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies_FGSM, "*-")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("FGSM Attack - Accuracy vs Epsilon")
    plt.show()

    plot_adversarial_examples(examples_FGSM, epsilons)

if __name__ == '__main__':
    main()
