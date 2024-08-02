import torch
import torch.nn as nn
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SaliencyMapMethod
import numpy as np
import matplotlib.pyplot as plt

# 定义你的MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# 假设你有训练好的PyTorch模型的路径
model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\20240522\\BaseLine\\CICIDS2017_chi45\\normal\\BaseLine_After_local_train_model.pth'

# 加载预训练的模型，重命名参数键
model = MLP(input_size=44, output_size=15)
pretrained_dict = torch.load(model_path)
model_dict = model.state_dict()

# 重命名预训练模型的键
rename_dict = {
    'layer1.weight': 'fc1.weight',
    'layer1.bias': 'fc1.bias',
    'layer5.weight': 'fc5.weight',
    'layer5.bias': 'fc5.bias'
}

# 更新模型字典
pretrained_dict = {rename_dict[k]: v for k, v in pretrained_dict.items() if k in rename_dict}
model_dict.update(pretrained_dict)

# 加载更新后的模型参数
model.load_state_dict(model_dict)

model.eval()

# 将PyTorch模型转换为ART分类器
classifier = PyTorchClassifier(
    model=model,
    loss=None,  # 如果模型已经训练好且不需要进一步训练，则不需要指定损失函数和优化器
    optimizer=None,
    input_shape=(44,),  # 输入数据的形状，根据你的数据特征数调整
    nb_classes=15,  # 分类器的类别数
    clip_values=None  # 在文本数据中，不需要进行像素范围的裁剪
)

classifier = PyTorchClassifier(
    model=model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
    input_shape=(44,),
    nb_classes=15,
    clip_values=(0.0, 1.0)
)
# 使用JSMA攻击方法生成对抗性样本并评估模型的鲁棒性
attack = SaliencyMapMethod(classifier=classifier, theta=0.5, gamma=0.1, verbose=True)

# 假设你有测试数据，可以用于生成对抗性样本
x_test = torch.rand(10, 44)  # 示例随机生成测试数据，使用torch.rand生成PyTorch张量

# 示例随机生成对应标签
y_test = np.random.randint(0, 15, size=10)

x_test_adv_jsma = attack.generate(x=x_test.numpy())

# 输出生成的对抗性样本及其标签
for i in range(len(x_test_adv_jsma)):
    print(f'Adversarial Sample {i+1}:')
    print(x_test_adv_jsma[i])
    print(f'Original Label: {y_test[i]}')

# 保存对抗性样本数据
np.save('./Adversarial_Attack_Test/adversarial_samples.npy', x_test_adv_jsma)

# 绘制对抗性样本的趋势图
fig, axs = plt.subplots(len(x_test_adv_jsma), 1, figsize=(10, 6 * len(x_test_adv_jsma)))


for i in range(len(x_test_adv_jsma)):
    axs[i].plot(x_test_adv_jsma[i], label='Adversarial Sample')
    axs[i].plot(x_test[i], label='Original Sample')
    axs[i].set_title(f'Adversarial Sample {i+1} vs Original Sample')
    axs[i].legend()

plt.tight_layout()
plt.show()

predictions_jsma = classifier.predict(x_test_adv_jsma)
accuracy_adv_jsma = np.sum(np.argmax(predictions_jsma, axis=1) == y_test) / len(y_test)
print(f'Accuracy on adversarial samples (JSMA): {accuracy_adv_jsma * 100}%')

