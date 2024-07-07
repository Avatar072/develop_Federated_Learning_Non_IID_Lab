import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SaliencyMapMethod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mytoolfunction import generatefolder
from mytoolfunction import ChooseUseModel, getStartorEndtime


# 定义文件路径和日期
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
start_IDS = time.time()
today = datetime.date.today().strftime("%Y%m%d")

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generatefolder(f"./Adversarial_Attack_Test/", today)

#紀錄開始時間
getStartorEndtime("starttime",start_IDS,f"./Adversarial_Attack_Test/{today}")

# 加载CICIDS2017数据集
data = pd.read_csv('D:\\develop_Federated_Learning_Non_IID_Lab\\data\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20240502\\doFeatureSelect\\44\\ALLday_train_dataframes_AfterFeatureSelect.csv')

# 列出所有特征名称，排除最后一列
all_features = data.columns[:-1].tolist()
# print("所有特征名称（不包括最后一列）:", all_features)

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


model.to(device)

model.eval()

# 将PyTorch模型转换为ART分类器
classifier = PyTorchClassifier(
    model=model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=(44,),
    nb_classes=15,
    clip_values=(0.0, 1.0)
)


# 使用JSMA攻击方法生成对抗性样本并评估模型的鲁棒性
attack = SaliencyMapMethod(classifier=classifier, theta=0.1, gamma=0.5, verbose=True)

# 加载测试数据
x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)
y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)

# 并转换为float32
x_test = x_test.astype(np.float32)

# 将输入数据转移到GPU
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)



# 初始化存储对抗性样本的数组
source_samples = x_test.shape[0]
X_adv = np.zeros((source_samples, x_test.shape[1]))

print(y_test)

# Initialize lists to store successful attack information
successful_attacks = []
sample_indices = []
original_classes = []

# 迭代每個測試樣本生成對抗樣本
for sample_ind in range(source_samples):
    current_class = int(y_test[sample_ind])  # 直接將當前樣本的真實類別轉換為整數
    # 如果當前樣本的真實類別已經是0，則跳過對該樣本的對抗樣本生成步驟
    if current_class == 0:
        continue
    

    # 將輸入數據移動到 GPU 以生成對抗性樣本
    x_test_gpu = x_test[sample_ind: (sample_ind + 1)]

    # 轉換為 NumPy 陣列 从 GPU 转换回 CPU 并转换为 NumPy 数组的数据 
    x_test_gpu_np = x_test_gpu.cpu().numpy()

    # 調用JSMA生成對抗樣本
    x_test_adv_jsma = attack.generate(x=x_test_gpu_np)

    # # 將輸入數據移動到 CPU 以生成對抗性樣本
    # x_test_cpu = x_test[sample_ind: (sample_ind + 1)].cpu().numpy()

    # # 調用JSMA生成對抗樣本
    # x_test_adv_jsma = attack.generate(x=x_test_cpu)

    # 將生成的對抗樣本存儲在X_adv中
    X_adv[sample_ind] = x_test_adv_jsma.flatten()

    # 輸出生成的對抗樣本及其標籤
    print(f'Adversarial Sample {sample_ind + 1}:')
    # print(x_test_adv_jsma.flatten())
    print(f'Original Label: {current_class}')

    # 繪制對抗樣本的趨勢圖
    plt.figure(figsize=(10, 6))
    plt.plot(x_test_adv_jsma.flatten(), label='Adversarial Sample')
    plt.plot(x_test[sample_ind].cpu().numpy().flatten(), label='Original Sample')
    plt.title(f'Adversarial Sample {sample_ind + 1} vs Original Sample')
    plt.legend()
    plt.xticks(np.arange(len(all_features)), all_features, rotation=90, fontsize=8)
    plt.ylabel('Feature Values')  # 添加Y轴标签
    plt.tight_layout()
    # plt.savefig(f'./Adversarial_Attack_Test/{today}/adversarial_samples_{sample_ind + 1}.png')
    # plt.show()
    plt.close()

   # 對對抗性樣本進行預測
    predictions_adv = classifier.predict(np.expand_dims(x_test_adv_jsma.flatten(), axis=0))
    predicted_class = np.argmax(predictions_adv)

    # 計算並打印對抗性樣本的準確性
    accuracy_adv_jsma = np.sum(np.argmax(predictions_adv, axis=1) == current_class) / 1.0

    # 保存成功的攻擊信息到CSV文件
    if predicted_class != current_class:
        print(f'Attack successful! Predicted class: {predicted_class}, Original class: {current_class}, Accuracy: {accuracy_adv_jsma * 100}%')
        successful_attacks.append(sample_ind + 1)  # 添加成功攻擊的樣本編號
        sample_indices.append(sample_ind + 1)
        original_classes.append(current_class)

        # 打开 CSV 文件，如果不存在则创建新文件
        csv_file_path = f"./Adversarial_Attack_Test/{today}/successful_attacks.csv"
        with open(csv_file_path, "a+") as file:
            # 将成功攻击的样本信息保存到 CSV 文件中
            successful_df = pd.DataFrame({
                'Predicted Class': predicted_class,
                'Sample Index': sample_indices,
                'Original Class': original_classes,
                'Current Class': current_class,
                'Accuracy': accuracy_adv_jsma * 100
            })
            successful_df.to_csv(file, header=file.tell() == 0, index=False)  # 如果文件为空则写入头部，否则不写入

    else:
        print(f'Attack unsuccessful. Predicted class: {predicted_class}, Original class: {current_class}, Accuracy: {accuracy_adv_jsma * 100}%')
        # 打开 CSV 文件，如果不存在则创建新文件
        csv_file_path = f"./Adversarial_Attack_Test/{today}/unsuccessful_attacks.csv"
        with open(csv_file_path, "a+") as file:
            # 将未成功攻击的样本信息保存到 CSV 文件中
            unsuccessful_df = pd.DataFrame({
                'Predicted Class': predicted_class,
                'Sample Index': sample_indices,
                'Original Class': original_classes,
                'Current Class': current_class,
                'Accuracy': accuracy_adv_jsma * 100
            })
            unsuccessful_df.to_csv(file, header=file.tell() == 0, index=False)  # 如果文件为空则写入头部

# 保存所有生成的對抗樣本
np.save(f'./Adversarial_Attack_Test/{today}/adversarial_samples.npy', X_adv)

#紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime",end_IDS,f"./Adversarial_Attack_Test/{today}")