import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from mytoolfunction import generatefolder, getStartorEndtime,ChooseUseModel,SaveDataToCsvfile,SaveDataframeTonpArray
import torch.utils.data as Data
import pandas as pd
import os
from collections import Counter

# 定義文件路徑和日期
filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
today = datetime.date.today().strftime("%Y%m%d")
start_IDS = time.time()
# JSMA參數
theta = 0.05  # 扰动值
gamma = 0.02  # 最多扰动特征数占总特征数量的比例
# 创建保存路径
# save_dir = f"./Adversarial_Attack_Test/{today}"
# save_dir = f"./Adversarial_Attack_Test/20240717"
generatefolder(f"./Adversarial_Attack_Test/", today)
generatefolder(f"./Adversarial_Attack_Test/{today}/", f"theta_{theta}_gamma_{gamma}")
save_dir = f"./Adversarial_Attack_Test/{today}/theta_{theta}_gamma_{gamma}"
getStartorEndtime("starttime",start_IDS,f"./Adversarial_Attack_Test/{today}")

# 將模型移動到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 以CSV loading dataset
afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\CICIDS2017\\ALLday\\20240502\\doFeatureSelect\\44\\ALLday_test_dataframes_AfterFeatureSelect.csv")
crop_dataset=afterprocess_dataset.iloc[:,:]
columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
# 使用条件选择不等于这些列名的列
testdata_removestring = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
#除了Label外的特徵
# 选择DataFrame的最后一列
    # df.iloc[:, -1] : 
    #   选择数据框中的最后一列。
    # : 表示选择所有行。
    # -1 表示选择最后一列。
# 选择数据框中除了最后一列以外的所有列。
    # df.iloc[:, :-1] 选择数据框中的所有列，除了最后一列。
    # 选择数据框中除了最后一列以外的所有列。
    # : 表示选择所有行。
    # :-1 表示选择所有列，除了最后一列。
# 选择数据框中的所有列，
    # df.iloc[:, :]
    # 可以使用 : 表示所有行和所有列：

def RemoveStringtypeFeatureForJSMA():    
    SaveDataToCsvfile(testdata_removestring, f"./data/Adversarial_Attack_Test/CICIDS2017/ALLDay/{today}", f"testdata_removestring_{today}")
    SaveDataframeTonpArray(testdata_removestring, f"./data/Adversarial_Attack_Test/CICIDS2017/ALLDay/{today}", f"testdata_removestring",today)


x_test_RemoveString_np = np.load(f"./data/Adversarial_Attack_Test/CICIDS2017/ALLDay/20240717/x_testdata_removestring_20240717.npy", allow_pickle=True)
y_test_RemoveString_np = np.load(f"./data/Adversarial_Attack_Test/CICIDS2017/ALLDay/20240717/y_testdata_removestring_20240717.npy", allow_pickle=True)
counter = Counter(y_test_RemoveString_np)
print("test筆數",counter)
print("特徵數量",x_test_RemoveString_np.shape[1])
x_test_RemoveString_np = torch.tensor(x_test_RemoveString_np).float()
y_test_RemoveString_np = torch.tensor(y_test_RemoveString_np).long()

test_data = TensorDataset(x_test_RemoveString_np, y_test_RemoveString_np)
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
# 超參數設置
batch_size = len(test_data)
learning_rate = 0.001

# 加載預訓練的模型
model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\20240522\\BaseLine\\CICIDS2017_chi45\\normal\\BaseLine_After_local_train_model.pth'

# 加載預訓練模型，並調整與文本特徵相關的輸入層大小
# model = ChooseUseModel("MLP", 44, 15).to(device)
# chi-square 45-6個remove string type = 39
model = ChooseUseModel("MLP", 39, 15).to(device)

def LoadingModelforeval(model):
    # 加載預訓練模型的狀態字典
    pretrained_dict = torch.load(model_path, map_location=device)

    # 重命名預訓練模型的鍵
    rename_dict = {
        'fc1.weight': 'layer1.weight',
        'fc1.bias': 'layer1.bias',
        'fc5.weight': 'layer5.weight',
        'fc5.bias': 'layer5.bias'
    }

    # 更新模型字典
    model_dict = model.state_dict()
    pretrained_dict = {rename_dict[k]: v for k, v in pretrained_dict.items() if k in rename_dict}
    model_dict.update(pretrained_dict)

    # 加載更新後的模型參數
    model.load_state_dict(model_dict)

    # 將模型移動到GPU
    model = model.to(device)
    model.eval()

# 計算雅可比矩陣，即前向導數
def compute_jacobian(model, input):
    var_input = input.clone()
    var_input.detach_()
    var_input.requires_grad = True
    output = model(var_input)

    num_features = int(np.prod(var_input.shape[1:]))
    jacobian = torch.zeros([output.size()[1], num_features], device=device)
    for i in range(output.size()[1]):
        if var_input.grad is not None:
            var_input.grad.zero_()
        output[0][i].backward(retain_graph=True)
        jacobian[i] = var_input.grad.squeeze().view(-1, num_features).clone()

    return jacobian

# 計算顯著圖
def saliency_map(jacobian, target_index, increasing, search_space, nb_features):
    domain = torch.eq(search_space, 1).float()  # 搜索域
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_index]
    others_grad = all_sum - target_grad

    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float().to(device)
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(device)
    increase_coef = increase_coef.view(-1, nb_features)

    target_tmp = target_grad.clone()
    target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)
    others_tmp = others_grad.clone()
    others_tmp += increase_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte().to(device)

    if increasing:
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)
    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
    saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q

def perturbation_single(text_features, ys_target, theta, gamma, model):
    var_sample = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0).to(device)
    var_target = torch.tensor([ys_target], dtype=torch.long).to(device)

    if theta > 0:
        increasing = True
    else:
        increasing = False

    num_features = text_features.shape[0]
    shape = var_sample.size()

    max_iters = int(np.ceil(num_features * gamma / 2.0))

    if increasing:
        search_domain = torch.lt(var_sample, 0.99)
    else:
        search_domain = torch.gt(var_sample, 0.01)
    search_domain = search_domain.view(num_features)

    model.eval().to(device)
    output = model(var_sample)
    current = torch.max(output.data, 1)[1].cpu().numpy()

    iter = 0
    while (iter < max_iters) and (current[0] != ys_target) and (search_domain.sum() != 0):
        jacobian = compute_jacobian(model, var_sample)
        p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
        var_sample_flatten = var_sample.view(-1, num_features).clone().detach_()
        var_sample_flatten[0, p1] += theta
        var_sample_flatten[0, p2] += theta

        new_sample = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
        new_sample = new_sample.view(shape)
        search_domain[p1] = 0
        search_domain[p2] = 0
        var_sample = torch.tensor(new_sample, requires_grad=True, device=device)

        output = model(var_sample)
        current = torch.max(output.data, 1)[1].cpu().numpy()
        iter += 1

    adv_samples = var_sample.detach().cpu().numpy()
    return adv_samples.squeeze()

# 生成JSMA對抗樣本
def generateJSMASample(theta,gamma):
    # 初始化变量
    num_labels = 15  # 0到14共15个标签

    # 初始化总的存储变量，用于存储所有标签的对抗样本和原始样本
    total_adver_examples = np.zeros((sum(counter.values()), x_test_RemoveString_np.shape[1]))
    total_adver_targets = np.zeros(sum(counter.values()))
    total_clean_examples = np.zeros((sum(counter.values()), x_test_RemoveString_np.shape[1]))
    total_clean_targets = np.zeros(sum(counter.values()))

    # 初始化一个字典来存储每个样本的对抗样本和原始样本对比
    samples_comparison = {}
    
    start_idx = 0

    # 为每个标签生成对抗样本
    for ys_target in range(num_labels):
        adver_nums = counter[ys_target]  # 设置对抗样本数量为该标签的样本数量

        # 初始化用于存储对抗样本和原始样本的变量
        adver_examples_by_JSMA = torch.zeros((adver_nums, x_test_RemoveString_np.shape[1])).to(device)
        adver_targets = torch.zeros(adver_nums).to(device)
        clean_examples = torch.zeros((adver_nums, x_test_RemoveString_np.shape[1])).to(device)
        clean_targets = torch.zeros(adver_nums).to(device)

        count = 0  # 当前生成对抗样本的计数
        for i, (data, target) in enumerate(tqdm(testloader, desc=f"Processing label {ys_target}")):
            if count >= adver_nums:
                break  # 如果生成的对抗样本数量达到预设值，则停止生成

            # 筛选出当前批次中目标标签的样本
            mask = (target == ys_target)
            if mask.sum().item() == 0:
                continue  # 如果当前批次中没有目标标签的样本，跳过当前批次

            data = data[mask]
            target = target[mask]
            batch_size = data.size(0)
            cur_adver_examples_by_JSMA = torch.zeros_like(data).to(device)
            for j in range(batch_size):
                perturbed_text = perturbation_single(data[j].cpu().numpy(), ys_target, theta, gamma, model)  # 生成对抗样本
                cur_adver_examples_by_JSMA[j] = torch.from_numpy(perturbed_text).to(device)  # 存储对抗样本
                # 存储对抗样本和原始样本的对比信息
                samples_comparison[start_idx + count + j] = {
                    'original': data[j].cpu().numpy(),
                    'adversarial': perturbed_text
                }

            pred = model(cur_adver_examples_by_JSMA).max(1)[1]

            if count == 0:
                adver_examples_by_JSMA[:batch_size] = cur_adver_examples_by_JSMA
                clean_examples[:batch_size] = data
                clean_targets[:batch_size] = target
                adver_targets[:batch_size] = pred
            else:
                start = count
                end = count + batch_size
                adver_examples_by_JSMA[start:end] = cur_adver_examples_by_JSMA
                clean_examples[start:end] = data
                clean_targets[start:end] = target
                adver_targets[start:end] = pred

            count += batch_size

        end_idx = start_idx + adver_nums
        total_adver_examples[start_idx:end_idx] = adver_examples_by_JSMA.cpu().numpy()
        total_adver_targets[start_idx:end_idx] = adver_targets.cpu().numpy()
        total_clean_examples[start_idx:end_idx] = clean_examples.cpu().numpy()
        total_clean_targets[start_idx:end_idx] = clean_targets.cpu().numpy()

        start_idx = end_idx

        # 输出生成的原始样本
        print(f"Label {ys_target}: clean_examples")
        print(clean_examples)

    

    #确保指定目录存在，必要时创建所有缺失的父目录
    # os.makedirs(save_dir, exist_ok=True)
    # 组合路径，确保生成的路径在不同操作系统上都能正确使用
    # 保存为numpy数组
    np.save(os.path.join(save_dir, "all_adver_examples_by_JSMA.npy"), total_adver_examples)  # 保存对抗样本
    np.save(os.path.join(save_dir, "all_adver_targets.npy"), total_adver_targets)  # 保存对抗样本的目标标签
    np.save(os.path.join(save_dir, "all_clean_examples.npy"), total_clean_examples)  # 保存原始样本
    np.save(os.path.join(save_dir, "all_clean_targets.npy"), total_clean_targets)  # 保存原始样本的目标标签

    # 保存为CSV文件
    adver_df = pd.DataFrame(total_adver_examples)
    adver_df['target'] = total_adver_targets
    adver_df.to_csv(os.path.join(save_dir, "all_adver_examples_by_JSMA.csv"), index=False)

    clean_df = pd.DataFrame(total_clean_examples)
    clean_df['target'] = total_clean_targets
    clean_df.to_csv(os.path.join(save_dir, "all_clean_examples.csv"), index=False)

    # 保存样本对比信息
    comparison_df = pd.DataFrame.from_dict(samples_comparison, orient='index')
    comparison_df.to_csv(os.path.join(save_dir, "samples_comparison.csv"), index=False)


def plot_detailed_feature_comparison(original_samples, adversarial_samples):
    feature_names = [
        'Flow Duration', 'Total Fwd Packets', 'Total Length of Fwd Packets', 'Bwd Packet Length Max', 
        'Fwd Packet Length Max', 'Fwd Packet Length Std', 'Bwd Packet Length Min', 'Fwd Packet Length Mean', 
        'Bwd Packet Length Mean', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 
        'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 
        'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 
        'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 
        'Bwd Avg Bulk Rate'
    ]
    
    # 計算原始樣本和對抗樣本的最小值、中位數和最大值
    original_min = np.min(original_samples, axis=0)
    original_median = np.median(original_samples, axis=0)
    original_max = np.max(original_samples, axis=0)
    
    adversarial_min = np.min(adversarial_samples, axis=0)
    adversarial_median = np.median(adversarial_samples, axis=0)
    adversarial_max = np.max(adversarial_samples, axis=0)
    
    plt.figure(figsize=(15, 6))
    plt.plot(adversarial_min, 'g--', label='JSMA Min')
    plt.plot(adversarial_median, 'g-', label='JSMA Median')
    plt.plot(adversarial_max, 'g:', label='JSMA Max')
    
    plt.plot(original_min, 'b--', label='Ordinary Min')
    plt.plot(original_median, 'b-', label='Ordinary Median')
    plt.plot(original_max, 'b:', label='Ordinary Max')
    
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Scaled Value')
    plt.title('Adversarial Samples vs Original Samples Feature Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./Adversarial_Attack_Test/{today}/detailed_comparison.png")
    plt.show()


def compare_samples(save_dir):
    # 加载保存的对抗样本和原始样本
    all_adver_examples_by_JSMA_df = pd.read_csv(os.path.join(save_dir, "all_adver_examples_by_JSMA.csv"))
    all_clean_examples_df = pd.read_csv(os.path.join(save_dir, "all_clean_examples.csv"))

    # 調用這個函數繪製原始樣本與JSMA對抗樣本的趨勢
    adver_examples = np.load(f"{save_dir}/all_adver_examples_by_JSMA.npy")
    clean_examples  = np.load(f"{save_dir}/all_clean_examples.npy")
    plot_detailed_feature_comparison(clean_examples, adver_examples)

    # 获取对抗样本和原始样本的特征数据（包括最后一列 'target'）
    all_adver_examples = all_adver_examples_by_JSMA_df.values
    all_clean_examples = all_clean_examples_df.values

    # 找出哪些标签没有生成对抗样本
    adver_targets = set(all_adver_examples_by_JSMA_df['target'])
    all_targets = set(all_clean_examples_df['target'])
    missing_targets = all_targets - adver_targets
    print(missing_targets)

    # 将未生成对抗样本的原始数据补充到生成的对抗样本中
    for target in missing_targets:
        missing_data = all_clean_examples_df[all_clean_examples_df['target'] == target]
        all_adver_examples = np.vstack([all_adver_examples, missing_data.values])

    # 保存补充后的对抗样本数据为 CSV 文件
    final_adver_examples_df = pd.DataFrame(all_adver_examples, columns=all_clean_examples_df.columns)
    final_adver_examples_df.to_csv(os.path.join(save_dir, "final_adver_examples_with_missing.csv"), index=False)
    
    finalDf = pd.concat([undoScalerdataset,final_adver_examples_df], axis = 1)
    SaveDataframeTonpArray(finalDf, save_dir, f"DoJSMA_test",today)
    

# generateJSMASample(0.05,0.02)
compare_samples(save_dir)





# plot_detailed_feature_comparison(clean_examples, adver_examples)
# 紀錄結束時間
end_IDS = time.time()
getStartorEndtime("endtime",end_IDS,f"./Adversarial_Attack_Test/{today}")

# adver_examples = np.load(f"./Adversarial_Attack_Test/20240717/adver_examples_by_JSMA.npy")
# adver_targets = np.load(f"./Adversarial_Attack_Test/20240717/adver_targets.npy")

# clean_examples  = np.load(f"./Adversarial_Attack_Test/20240717/clean_examples.npy")
# clean_targets  = np.load(f"./Adversarial_Attack_Test/20240717/clean_targets.npy")
# # 转换为Tensor
# adver_examples_tensor = torch.tensor(adver_examples).float().to(device)
# adver_targets_tensor = torch.tensor(adver_targets).long().to(device)


# clean_examples_tensor = torch.tensor(clean_examples).float().to(device)
# clean_targets_tensor = torch.tensor(clean_targets).long().to(device)



# # 使用模型进行预测
# model.eval()
# with torch.no_grad():
#     # outputs = model(adver_examples_tensor)
#     outputs = model(clean_examples_tensor)
#     predicted = torch.max(outputs, 1)[1]
#     # correct = (predicted == adver_targets_tensor).sum().item()
#     correct = (predicted == clean_targets_tensor).sum().item()


# # accuracy = correct / len(adver_targets)
# accuracy = correct / len(clean_targets)
# print(f"Model accuracy on adversarial samples: {accuracy * 100:.2f}%")