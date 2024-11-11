import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm  # 用於顯示進度條
import matplotlib.pyplot as plt

# 資料預處理與加載
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 將數據轉換為 numpy 格式以供 k-means 使用
train_data = np.array([data.numpy().reshape(-1) for data, _ in train_loader.dataset])
test_data = np.array([data.numpy().reshape(-1) for data, _ in test_loader.dataset])

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # MNIST 有 10 個類別

    def forward(self, x):
        x = self.model(x)
        return x

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = SimpleResNet().to(device)

# 定義優化器和損失函數
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 評估 k-means 的性能
def evaluate_kmeans(kmeans, data, labels):
    predictions = kmeans.predict(data)
    majority_votes = np.zeros_like(predictions)
    for i in range(kmeans.n_clusters):
        mask = (predictions == i)
        if mask.sum() > 0:
            majority_votes[mask] = np.bincount(labels[mask]).argmax()
    accuracy = np.mean(majority_votes == labels)
    return accuracy

# 更新 train_model 函數中的圖像處理
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            # 將 1 通道擴展為 3 通道
            images = images.repeat(1, 3, 1, 1)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            # 將 1 通道擴展為 3 通道
            data = data.repeat(1, 3, 1, 1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, all_preds, all_labels

train_model(resnet_model, train_loader, criterion, optimizer, epochs=5)
accuracy, all_preds, all_labels = test(resnet_model, device, test_loader)

# ART 的 PyTorchClassifier
classifier = PyTorchClassifier(
    model=resnet_model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0, 1)
)

# 設置 I-FGSM 攻擊，設定步幅 α = ε/4 並逐步增加 ε
num_steps = 40  # 定義步數# β=40
epsilons = np.linspace(0.01, 1.0, num=num_steps)
cln_acc_list = []  # 儲存乾淨樣本準確率
adv_acc_list = []  # 儲存對抗樣本準確率

# 測試資料擾動固定為1，α = ε/4=0.25
test_attack = FastGradientMethod(estimator=classifier, eps=1.0, eps_step=0.25)
# 將 numpy 數組轉換為 PyTorch 張量，然後擴展通道
test_tensor = torch.tensor(test_data).view(-1, 1, 28, 28).repeat(1, 3, 1, 1).to(device)
# 生成對抗性測試資料
test_adv_data = test_attack.generate(x=test_tensor.cpu().numpy())
# 修正：在生成對抗性樣本後，確保形狀是單通道格式並且展平為 784
test_adv_data = test_adv_data[:, 0, :, :].reshape(test_adv_data.shape[0], -1)

# 顯示設定 eps_step 的進度條
# 使用 `with` 語句來創建進度條
with tqdm(epsilons, desc='Generating Adversarial Examples', unit='step') as pbar:
    for current_eps in epsilons:
        # 設置 I-FGSM 攻擊，設定步幅 α = ε/4
        eps_step = current_eps / 4
        attack = FastGradientMethod(estimator=classifier, eps=current_eps, eps_step=eps_step)

        # 顯示當前步驟的 epsilon 和 eps_step
        print(f"Current epsilon: {current_eps:.4f}, eps_step: {eps_step:.4f},num: {num_steps}")
        # 更新進度條描述以顯示當前 epsilon 和 eps_step
        pbar.set_postfix({'epsilon': f'{current_eps:.4f}', 'eps_step': f'{eps_step:.4f}'})

        # 將 numpy 數組轉換為 PyTorch 張量，然後擴展通道
        train_tensor = torch.tensor(train_data).view(-1, 1, 28, 28).repeat(1, 3, 1, 1).to(device)

        # 生成對抗性樣本
        train_adv_data = attack.generate(x=train_tensor.cpu().numpy())
        

        # 修正：在生成對抗性樣本後，確保形狀是單通道格式並且展平為 784
        train_adv_data = train_adv_data[:, 0, :, :].reshape(train_adv_data.shape[0], -1)

        # 將對抗性樣本與乾淨樣本混合，達到 η = 1/2
        # mixed_train_data = np.vstack((train_adv_data[:train_adv_data.shape[0]//2], train_data[:train_data.shape[0]//2]))
        # 將對抗性樣本與乾淨樣本混合，達到 η = 1/6
        # mixed_train_data = np.vstack((
        #     train_adv_data[:train_adv_data.shape[0]//6],  # 選取對抗性樣本的 1/6
        #     train_data[:train_data.shape[0] * 5//6]       # 選取乾淨樣本的 5/6
        # ))

        # 將對抗性樣本與乾淨樣本混合，達到 η = 1/3
        mixed_train_data = np.vstack((
            train_adv_data[:train_adv_data.shape[0]//3],  # 選取對抗性樣本的 1/3
            train_data[:train_data.shape[0] * 2//3]       # 選取乾淨樣本的 2/3
        ))
        # # 使用混合數據進行 k-means 訓練
        kmeans = KMeans(n_clusters=856, init='k-means++', max_iter=300, random_state=42)
        kmeans.fit(mixed_train_data)

        # 使用原始乾淨樣本與對抗樣本進行 k-means 訓練
        # kmeans.fit(train_data)  # 使用原始數據訓練 k-means

        # 計算 ClnAcc 和 AdvAcc
        cln_accuracy = evaluate_kmeans(kmeans, test_data, test_dataset.targets.numpy())
        adv_accuracy = evaluate_kmeans(kmeans, test_adv_data, test_dataset.targets.numpy())

        cln_acc_list.append(cln_accuracy * 100)
        adv_acc_list.append(adv_accuracy * 100)


# 繪製圖表
plt.figure(figsize=(10, 6))
plt.plot(epsilons, cln_acc_list, label='ClnAcc', color='blue', marker='o')
plt.plot(epsilons, adv_acc_list, label='AdvAcc', color='red', marker='o')
plt.xlabel('Epsilon (ε)', fontsize=14)
plt.ylabel('Testing Accuracy (%)', fontsize=14)
plt.title('Testing Accuracy vs. Epsilon (ε)', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
print(f'ClnAcc List: {cln_acc_list}')
print(f'AdvAcc List: {adv_acc_list}')

# 測試評估函數
# def test_with_fixed_epsilon(classifier, kmeans, test_data, test_labels, epsilon=1.0):
#     attack = FastGradientMethod(estimator=classifier, eps=epsilon, eps_step=epsilon/4)
#     test_tensor = torch.tensor(test_data).view(-1, 1, 28, 28).repeat(1, 3, 1, 1).to(device)
#     test_adv_data = attack.generate(x=test_tensor.cpu().numpy())
    
#     # 提取第一個通道並展平
#     test_adv_data = test_adv_data[:, 0, :, :].reshape(test_adv_data.shape[0], -1)

#     # 驗證數據形狀是否正確
#     assert test_adv_data.shape[1] == 784, "test_adv_data features should be 784"

#     # 評估 k-means 的性能
#     adv_accuracy = evaluate_kmeans(kmeans, test_adv_data, test_labels)
#     print(f'Adversarial Test Accuracy with epsilon={epsilon}: {adv_accuracy:.2%}')
#     return adv_accuracy


# 執行固定 epsilon = 1 的測試
# adv_test_accuracy = test_with_fixed_epsilon(classifier, kmeans, test_data, test_dataset.targets.numpy(), epsilon=1.0)

# 計算測試準確性
clean_accuracy = evaluate_kmeans(kmeans, test_data, test_dataset.targets.numpy())
adv_accuracy = evaluate_kmeans(kmeans, test_adv_data, test_dataset.targets.numpy())

print(f'kmeans Clean Test Accuracy: {clean_accuracy:.2%}')
print(f'kmeans Adversarial Test Accuracy: {adv_accuracy:.2%}')
# print(f'Adversarial Test Accuracy: {adv_test_accuracy:.2%}')
