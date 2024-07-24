import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SaliencyMapMethod
from mytoolfunction import generatefolder, getStartorEndtime  # 假設這些是您自定義的函數

# 定義文件路徑和日期
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today().strftime("%Y%m%d")

# 如果GPU可用，將模型移動到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generatefolder(f"./Adversarial_Attack_Test/", today)  # 創建存儲結果的文件夾

# 記錄開始時間
start_time = time.time()
getStartorEndtime("starttime", start_time, f"./Adversarial_Attack_Test/{today}")

# 加載 CICIDS2017 數據集
data = pd.read_csv('D:\\develop_Federated_Learning_Non_IID_Lab\\data\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20240502\\doFeatureSelect\\44\\ALLday_train_dataframes_AfterFeatureSelect.csv')

# 列出所有特徵名稱，排除最後一列（類別）
all_features = data.columns[:-1].tolist()

# 定義 MLP 模型
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

# 假設有預先訓練的模型路徑
model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\20240522\\BaseLine\\CICIDS2017_chi45\\normal\\BaseLine_After_local_train_model.pth'

# 加載預訓練模型，重命名參數
model = MLP(input_size=44, output_size=15)
pretrained_dict = torch.load(model_path)
model_dict = model.state_dict()

rename_dict = {
    'layer1.weight': 'fc1.weight',
    'layer1.bias': 'fc1.bias',
    'layer5.weight': 'fc5.weight',
    'layer5.bias': 'fc5.bias'
}

pretrained_dict = {rename_dict[k]: v for k, v in pretrained_dict.items() if k in rename_dict}
model_dict.update(pretrained_dict)

model.load_state_dict(model_dict)
model.to(device)
model.eval()

# 將 PyTorch 模型轉換為 ART 分類器
classifier = PyTorchClassifier(
    model=model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=(44,),
    nb_classes=15,
    clip_values=(0.0, 1.0)
)

# 使用 JSMA 攻擊方法生成對抗性樣本並評估模型的穩健性
attack = SaliencyMapMethod(classifier=classifier, theta=0.1, gamma=0.5, verbose=True)

# 加載測試數據
x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\x_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)
y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\y_ALLday_test_cicids2017_AfterFeatureSelect44_20240502.npy", allow_pickle=True)

# 轉換為 float32
x_test = x_test.astype(np.float32)

# 將輸入數據移動到 GPU
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)

# 初始化數組以存儲對抗性樣本
source_samples = x_test.shape[0]
X_adv = np.zeros((source_samples, x_test.shape[1]))

# 初始化列表以存儲成功攻擊信息
successful_attacks = []
sample_indices = []
original_classes = []

# 計算每個類別的大約 10% 樣本數
class_counts = np.bincount(y_test.cpu().numpy())

# class_counts * 0.1: 这里乘以 0.1，意味着每个类别应该生成的对抗样本数是该类别样本数的10%。
# np.round(class_counts * 0.1): 将乘以 0.1 后的结果四舍五入为最接近的整数。这样可以得到每个类别理论上的对抗样本数量。

samples_per_class = np.maximum(np.round(class_counts * 0.1).astype(int), 1)  # 確保每個類別至少有 1 個樣本

# 逐個類別生成對抗性樣本
for class_label in range(0, 15):  # 假設類別從0開始（假設 y_test 從0開始）
    generatefolder(f"./Adversarial_Attack_Test/{today}/", f"Label_{class_label}")
    class_indices = np.where(y_test.cpu().numpy() == class_label)[0]
    selected_indices = np.random.choice(class_indices, size=samples_per_class[class_label], replace=False)

    for sample_ind in selected_indices:
        current_class = int(y_test[sample_ind])

        # 將輸入數據移動到 GPU 以生成對抗性樣本
        x_test_gpu = x_test[sample_ind: (sample_ind + 1)]
        x_test_gpu_np = x_test_gpu.cpu().numpy()

        # 使用 JSMA 攻擊生成對抗性樣本
        x_test_adv_jsma = attack.generate(x=x_test_gpu_np)

        # 將生成的對抗性樣本存儲在 X_adv 中
        X_adv[sample_ind] = x_test_adv_jsma.flatten()

        # 使用對抗性樣本進行預測
        predictions_adv = classifier.predict(np.expand_dims(x_test_adv_jsma.flatten(), axis=0))
        predicted_class = np.argmax(predictions_adv)

        # 計算對抗性樣本的準確率
        accuracy_adv_jsma = np.sum(np.argmax(predictions_adv, axis=1) == current_class) / 1.0

        # 將成功攻擊信息保存到 CSV 文件中
        if predicted_class != current_class:
            successful_attacks.append(sample_ind + 1)
            sample_indices.append(sample_ind + 1)
            original_classes.append(current_class)

            csv_file_path = f"./Adversarial_Attack_Test/{today}/successful_attacks.csv"
            with open(csv_file_path, "a+") as file:
                successful_df = pd.DataFrame({
                    'Predicted Class': predicted_class,
                    'Sample Index': sample_indices,
                    'Original Class': original_classes,
                    'Current Class': current_class,
                    'Accuracy': accuracy_adv_jsma * 100
                })
                successful_df.to_csv(file, header=file.tell() == 0, index=False)

        else:
            csv_file_path = f"./Adversarial_Attack_Test/{today}/unsuccessful_attacks.csv"
            with open(csv_file_path, "a+") as file:
                unsuccessful_df = pd.DataFrame({
                    'Predicted Class': predicted_class,
                    'Sample Index': sample_indices,
                    'Original Class': original_classes,
                    'Current Class': current_class,
                    'Accuracy': accuracy_adv_jsma * 100
                })
                unsuccessful_df.to_csv(file, header=file.tell() == 0, index=False)
        

        # 假設你有一個生成對抗樣本的過程
        original_sample = x_test[sample_ind].cpu().numpy().flatten()
        adversarial_sample = X_adv[sample_ind]

        # 創建一個新的圖形
        plt.figure(figsize=(10, 6))

        # 繪製原始樣本和對抗樣本的趨勢圖
        # plt.plot(original_sample, label='Original Sample', marker='o')
        # plt.plot(adversarial_sample, label='Adversarial Sample', marker='x')

        plt.plot(x_test_adv_jsma.flatten(), label='Adversarial Sample')
        plt.plot(x_test[sample_ind].cpu().numpy().flatten(), label='Original Sample')
    
        # 添加標題和圖例
        plt.title(f'Label {class_label} Sample {sample_ind + 1} - Original vs Adversarial')
        
        # plt.title(f'Sample {class_label} - Original vs Adversarial')
        plt.xticks(np.arange(len(all_features)), all_features, rotation=90, fontsize=8)
        plt.ylabel('Feature Values')  # 添加Y轴标签
        plt.legend()

        plt.tight_layout()
        # 保存圖像或顯示圖像
        plt.savefig(f'./Adversarial_Attack_Test/{today}/Label_{class_label}/adversarial_samples_{sample_ind + 1}.png')
        # plt.show()

        # 關閉當前圖形以釋放內存資源
        plt.close()

# 保存所有生成的對抗性樣本
np.save(f'./Adversarial_Attack_Test/{today}/adversarial_samples.npy', X_adv)

# 記錄結束時間
end_time = time.time()
getStartorEndtime("endtime", end_time, f"./Adversarial_Attack_Test/{today}")


# 計算每個類別的成功攻擊數量
# attack_counts = np.bincount(successful_attacks)

# 繪製趨勢圖
# Plot trend chart
# plt.figure(figsize=(10, 6))
# plt.bar(range(1, len(attack_counts)), attack_counts[1:], align='center', alpha=0.5)
# plt.xticks(range(1, len(attack_counts)), rotation=45)
# plt.xlabel('Class')
# plt.ylabel('Successful Attacks Count')
# plt.title('Trend Chart of Successful Attacks Count per Class')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()
# plt.savefig(f'./Adversarial_Attack_Test/{today}/adversarial_samples_Count.png')
