import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SaliencyMapMethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mytoolfunction import generatefolder, SaveDataToCsvfile, SaveDataframeTonpArray, ChooseUseModel,getStartorEndtime
from collections import Counter, defaultdict

# 定义文件路径和日期
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today().strftime("%Y%m%d")
start_IDS = time.time()

generatefolder(f"./Adversarial_Attack_Test/", today)
getStartorEndtime("starttime", start_IDS, f"./Adversarial_Attack_Test/{today}")

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载CICIDS2017 test after do labelencode and minmax chi_square45 75 25分
# afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20240502\\doFeatureSelect\\44\\ALLday_test_dataframes_AfterFeatureSelect.csv")
# 加载TONIOT test
afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\test_ToN-IoT_dataframes_20240523.csv")

# 加载TONIOT client3 train 均勻劃分
# afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_train_half3_20240523.csv")
# 加载TONIOT client3 train 隨機劃分
# afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\TONIOT\\20240523\\train_ToN-IoT_dataframes_train_half3_random_20240523.csv")

print("Dataset loaded.")

# 移除字符串类型特征
def RemoveStringTypeValueForJSMA(afterprocess_dataset):
    crop_dataset = afterprocess_dataset.iloc[:, :]
    #cicids2017 normal
    #columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Timestamp', 'Protocol']
    #cicids2017 normal chi-square45 後Protocol可能不見
    # columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Timestamp']
    #toniot
    columns_to_exclude = ['ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto']
    testdata_removestring = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
    undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col in columns_to_exclude]]

    SaveDataframeTonpArray(testdata_removestring, f"./Adversarial_Attack_Test/{today}", f"testdata_removestring", today)
    print(f"Removed string type columns: {columns_to_exclude}")
    return testdata_removestring, undoScalerdataset

# 创建并加载模型
# cicids2017 normal chi-square45 後Protocol可能不見
# 輸入39是特徵扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Timestamp', 'Label'
# model = ChooseUseModel("MLP", 39, 15).to(device)
#toniot
model = ChooseUseModel("MLP", 38, 10).to(device)

# 假设你有训练好的PyTorch模型的路径
# BaseLine每層神經元512下所訓練出來的model
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\20240719_TONIOT_神經元512\\BaseLine\\normal\\BaseLine_After_local_train_model.pth'

# BaseLine每層神經元64下所訓練出來的model
# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\20240729_TONIOT_神經元64\\BaseLine\\normal\\BaseLine_After_local_train_model.pth'

# model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\\Adversarial_Attack_Test\\20240722_FL_cleint3_.0.5_0.02\\After_JSMA_Attack_model.pth'


# def LoadingModelforeval(model):
#     # 加载预训练模型的状态字典
#     pretrained_dict = torch.load(model_path, map_location=device)
#     # 重命名预训练模型的键
#     rename_dict = {
#         # 'fc1.weight': 'layer1.weight',
#         # 'fc1.bias': 'layer1.bias',
#         # 'fc2.weight': 'layer2.weight',
#         # 'fc2.bias': 'layer2.bias',
#         # 'fc3.weight': 'layer3.weight',
#         # 'fc3.bias': 'layer3.bias',
#         # 'fc4.weight': 'layer4.weight',
#         # 'fc4.bias': 'layer4.bias',
#         # 'fc5.weight': 'layer5.weight',
#         # 'fc5.bias': 'layer5.bias'
#     }
#     # 更新模型字典
#     model_dict = model.state_dict()
#     updated_pretrained_dict = {rename_dict[k]: v for k, v in pretrained_dict.items() if k in rename_dict}
#     model_dict.update(updated_pretrained_dict)
#     # 加载更新后的模型参数
#     model.load_state_dict(model_dict)
#     # 将模型移动到GPU
#     model = model.to(device)
#     model.eval()
#     return model

# model = LoadingModelforeval(model)

# 将PyTorch模型转换为ART分类器
classifier = PyTorchClassifier(
    model=model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=(38,),  # 确保输入特征数量为39
    nb_classes=10,
    clip_values=(0.0, 1.0)
)
print("ART classifier created.")

# 使用JSMA攻击方法生成对抗性样本并评估模型的鲁棒性
# - theta：控制擾動的幅度。增加theta的值可以增加擾動的大小。
# - gamma：控制對每個特徵的影響程度。較大的gamma值可以使攻擊更加集中在少數幾個特徵上，增加其擾動的顯著性
attack = SaliencyMapMethod(classifier=classifier, theta=0.1, gamma=1.0, verbose=True)
# attack = SaliencyMapMethod(classifier=classifier, theta=0.05, gamma=0.02, verbose=True)
# attack = SaliencyMapMethod(classifier=classifier, theta=0.1, gamma=0.02, verbose=True)
# attack = SaliencyMapMethod(classifier=classifier, theta=0.15, gamma=0.02, verbose=True)
# attack = SaliencyMapMethod(classifier=classifier, theta=0.2, gamma=0.02, verbose=True)
# attack = SaliencyMapMethod(classifier=classifier, theta=0.25, gamma=0.02, verbose=True)




print("SaliencyMapMethod attack initialized.")

# 加载测试数据
testdata_removestring, undoScalerdataset = RemoveStringTypeValueForJSMA(afterprocess_dataset)
x_test_RemoveString_np = np.load(f"./Adversarial_Attack_Test/{today}/x_testdata_removestring_{today}.npy", allow_pickle=True)
y_test_RemoveString_np = np.load(f"./Adversarial_Attack_Test/{today}/y_testdata_removestring_{today}.npy", allow_pickle=True)
original_label_counter = Counter(y_test_RemoveString_np)
print(f"Loaded test data. Original sample count: {original_label_counter}, Feature count: {x_test_RemoveString_np.shape[1]}")

# 将输入数据转换为Float类型
x_test_RemoveString_np = x_test_RemoveString_np.astype(np.float32)

# 初始化存储对抗性样本的数组
source_samples = x_test_RemoveString_np.shape[0]
X_adv = np.zeros((source_samples, x_test_RemoveString_np.shape[1]), dtype=np.float32)

# 初始化存储成功攻击信息的列表
successful_attacks = []
sample_indices = []
original_classes = []
adv_labels = []

# 初始化计数器来跟踪每个标签生成的样本数量
successful_label_counts = defaultdict(int)
unsuccessful_label_counts = defaultdict(int)

# 迭代每个测试样本生成对抗样本
for sample_ind in range(source_samples):
    current_class = int(y_test_RemoveString_np[sample_ind])
    generatefolder(f"./Adversarial_Attack_Test/{today}/", f"Label_{current_class}")
    if current_class == 0:
        continue

    x_test_gpu = torch.tensor(x_test_RemoveString_np[sample_ind: (sample_ind + 1)]).to(device)
    x_test_gpu_np = x_test_gpu.cpu().numpy()

    # 调用JSMA生成对抗样本
    x_test_adv_jsma = attack.generate(x=x_test_gpu_np)

    # 将生成的对抗样本存储在X_adv中
    X_adv[sample_ind] = x_test_adv_jsma.flatten()

    # 计算对抗样本的预测结果
    predictions_adv = classifier.predict(x_test_adv_jsma)
    predicted_class = np.argmax(predictions_adv, axis=1)[0]
    accuracy_adv_jsma = np.sum(np.argmax(predictions_adv, axis=1) == current_class) / 1.0

    if predicted_class != current_class:
        print(f'Attack successful! Sample index: {sample_ind + 1}, Predicted class: {predicted_class}, Original class: {current_class}, Accuracy: {accuracy_adv_jsma * 100}%')
        successful_attacks.append(sample_ind + 1)
        sample_indices.append(sample_ind + 1)
        original_classes.append(current_class)

        csv_file_path = f"./Adversarial_Attack_Test/{today}/Label_{current_class}/successful_attacks.csv"
        with open(csv_file_path, "a+") as file:
            attack_info = pd.DataFrame({
                'Predicted Class': [predicted_class],
                'Sample Index': [sample_ind + 1],
                'Original Class': [current_class],
                'Accuracy': [accuracy_adv_jsma * 100]
            })
            attack_info.to_csv(file, header=file.tell() == 0, index=False)
        
        # 更新成功标签计数器
        successful_label_counts[current_class] += 1
    else:
        print(f'Attack unsuccessful. Sample index: {sample_ind + 1}, Predicted class: {predicted_class}, Original class: {current_class}, Accuracy: {accuracy_adv_jsma * 100}%')
        csv_file_path = f"./Adversarial_Attack_Test/{today}/Label_{current_class}/unsuccessful_attacks.csv"
        with open(csv_file_path, "a+") as file:
            attack_info = pd.DataFrame({
                'Predicted Class': [predicted_class],
                'Sample Index': [sample_ind + 1],
                'Original Class': [current_class],
                'Accuracy': [accuracy_adv_jsma * 100]
            })
            attack_info.to_csv(file, header=file.tell() == 0, index=False)
        
        # 更新未成功标签计数器
        unsuccessful_label_counts[current_class] += 1
    adv_labels.append(current_class)

print("Saving adversarial samples and labels.")
np.save(f'./Adversarial_Attack_Test/{today}/adversarial_samples.npy', X_adv)
np.save(f'./Adversarial_Attack_Test/{today}/adversarial_labels.npy', np.array(adv_labels))

# 打印每个标签生成的样本数量
print("Original, successful, and unsuccessful generated sample counts per label:")
for label, original_count in original_label_counter.items():
    successful_count = successful_label_counts[label]
    unsuccessful_count = unsuccessful_label_counts[label]
    print(f"Label {label}: Original samples: {original_count}, Successful generated samples: {successful_count}, Unsuccessful generated samples: {unsuccessful_count}")

def plot_detailed_feature_comparison(original_samples, adversarial_samples):
    feature_names = [
        'service', 'duration', 'src_bytes', 'dst_bytes', 
        'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 
        'dst_pkts', 'dst_ip_bytes', 'dns_query', 'dns_qclass', 'dns_qtype', 
        'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_version', 
        'ssl_cipher', 'ssl_resumed', 'ssl_established', 'ssl_subject', 
        'ssl_issuer', 'http_trans_depth', 'http_method', 'http_uri', 'http_version', 
        'http_request_body_len', 'http_response_body_len', 'http_status_code', 'http_user_agent', 'http_orig_mime_types', 
        'http_resp_mime_types', 'weird_name', 'weird_addl', 'weird_notice', 
        'label'
    ]

    original_min = np.min(original_samples, axis=0)
    original_median = np.median(original_samples, axis=0)
    original_max = np.max(original_samples, axis=0)
    adversarial_min = np.min(adversarial_samples, axis=0)
    adversarial_median = np.median(adversarial_samples, axis=0)
    adversarial_max = np.max(adversarial_samples, axis=0)
    plt.figure(figsize=(15, 6))
    # plt.plot(adversarial_min, 'g--', label='JSMA Min')
    plt.plot(adversarial_median, 'g-', label='JSMA Median')
    # plt.plot(adversarial_max, 'g:', label='JSMA Max')
    # plt.plot(original_min, 'b--', label='Ordinary Min')
    plt.plot(original_median, 'b-', label='Ordinary Median')
    # plt.plot(original_max, 'b:', label='Ordinary Max')
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Scaled Value')
    plt.title('Adversarial Samples vs Original Samples Feature Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./Adversarial_Attack_Test/{today}/detailed_comparison.png")
    plt.show()

print("Process completed.")
# 需要传入原始样本和对抗样本数据
plot_detailed_feature_comparison(x_test_RemoveString_np, X_adv)

np.save(f'./Adversarial_Attack_Test/{today}/adversarial_samples.npy', X_adv)
np.save(f'./Adversarial_Attack_Test/{today}/adversarial_labels.npy', np.array(adv_labels))

def compare_samples(save_dir):
    # 加载保存的对抗样本和原始样本
    x_test_RemoveString_np = np.load(os.path.join(save_dir, f"x_testdata_removestring_{today}.npy"), allow_pickle=True)
    y_test_RemoveString_np = np.load(os.path.join(save_dir, f"y_testdata_removestring_{today}.npy"), allow_pickle=True)
    
    adversarial_samples = np.load(os.path.join(save_dir, "adversarial_samples.npy"), allow_pickle=True)
    adversarial_labels = np.load(os.path.join(save_dir, "adversarial_labels.npy"), allow_pickle=True)

    adversarial_label_counter = Counter(adversarial_labels)
    print(f"Loaded adversarial_samples data. adversarial sample count:" + 
          f"{adversarial_label_counter}, Feature count: {adversarial_samples.shape[1]}")

    # 获取原始样本的特征数据
    original_samples = pd.DataFrame(x_test_RemoveString_np)
    original_labels = pd.DataFrame(y_test_RemoveString_np, columns=['Label'])

    # 移除 original_labels 和 adversarial_labels 中的空白数据
    original_labels = original_labels.dropna(subset=['Label'])
    adversarial_labels = pd.DataFrame(adversarial_labels, columns=['Label']).dropna(subset=['Label']).values.flatten()

    # 将对抗样本转换为 DataFrame 格式
    adversarial_samples_df = pd.DataFrame(adversarial_samples)
    adversarial_labels_df = pd.DataFrame(adversarial_labels, columns=['Label'])

    # 合并对抗样本和原始样本
    adversarial_data_with_labels = pd.concat([adversarial_samples_df, adversarial_labels_df], axis=1)

    # 找出未生成对抗样本的原始数据
    missing_targets = set(original_labels['Label'].unique()) - set(adversarial_labels)
    print("Missing targets:", missing_targets)

    # 将未生成对抗样本的原始数据补充到生成的对抗样本中
    for target in missing_targets:
        missing_data = original_samples[original_labels['Label'] == target]
        missing_labels = original_labels[original_labels['Label'] == target]
        missing_data_with_labels = pd.concat([missing_data.reset_index(drop=True), missing_labels.reset_index(drop=True)], axis=1)
        adversarial_data_with_labels = pd.concat([adversarial_data_with_labels, missing_data_with_labels])

    # 再次移除合并后 DataFrame 中的空白数据
    final_adversarial_examples_df = adversarial_data_with_labels.dropna(subset=['Label']).reset_index(drop=True)


    # 恢复未被删除的字符串类型特征数据
    finalDf = pd.concat([undoScalerdataset.reset_index(drop=True), final_adversarial_examples_df.reset_index(drop=True)], axis=1)
    
    # 保存补充后的对抗样本数据为 CSV 文件
    finalDf.to_csv(os.path.join(save_dir, "final_adver_examples_with_missing.csv"), index=False)

    SaveDataframeTonpArray(finalDf, save_dir, f"DoJSMA_test_theta_0.05", today)
    # SaveDataframeTonpArray(finalDf, save_dir, f"DoJSMA_train_half3", today)
    # SaveDataframeTonpArray(finalDf, save_dir, f"DoJSMA_train_half3_theta_0.15", today)
    # SaveDataframeTonpArray(finalDf, save_dir, f"DoJSMA_train_half3_theta_0.2", today)
    # SaveDataframeTonpArray(finalDf, save_dir, f"DoJSMA_train_half3_theta_0.25", today)


save_dir = f"./Adversarial_Attack_Test/{today}"
compare_samples(save_dir)


torch.save(model.state_dict(), f"./Adversarial_Attack_Test/{today}/After_JSMA_Attack_model.pth")

# 紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime", end_IDS, f"./Adversarial_Attack_Test/{today}")



