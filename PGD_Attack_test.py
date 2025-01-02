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
from sklearn.metrics import classification_report
from mytoolfunction import ChooseUseModel, getStartorEndtime
from colorama import Fore, Back, Style, init
labelCount = 13

filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"
start_IDS = time.time()
client_str = "baseline_train"
Choose_method = "normal"
num_epochs = 1
# 設定設備
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using DEVICE: {DEVICE}")

def generatefolder(path, foldername):
    if not os.path.exists(os.path.join(path, foldername)):
        os.makedirs(os.path.join(path, foldername))

# 在Adversarial_Attack_Test產生天日期的資料夾
today = datetime.date.today()
today = today.strftime("%Y%m%d")
current_time = time.strftime("%Hh%Mm%Ss", time.localtime())
save_dir = f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}"

print(Fore.YELLOW+Style.BRIGHT+f"當前時間: {current_time}")
generatefolder(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/", today)
generatefolder(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/", client_str)
generatefolder(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/", Choose_method)
getStartorEndtime("starttime",start_IDS,f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}")


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

# 畫混淆矩陣
def draw_confusion_matrix(y_true, y_pred, plot_confusion_matrix = False,epsilon = None,step_epsilon= None):
    #混淆矩陣
    if plot_confusion_matrix:
        # df_cm的PD.DataFrame 接受三個參數：
        # arr：混淆矩陣的數據，這是一個二維陣列，其中包含了模型的預測和實際標籤之間的關係，以及它們在混淆矩陣中的計數。
        # class_names：類別標籤的清單，通常是一個包含每個類別名稱的字串清單。這將用作 Pandas 資料幀的行索引和列索引，以標識混淆矩陣中每個類別的位置。
        # class_names：同樣的類別標籤的清單，它作為列索引的標籤，這是可選的，如果不提供這個參數，將使用行索引的標籤作為列索引
        arr = confusion_matrix(y_true, y_pred)
        # # CICIDS2019
        class_names = {
                        #二元分類
                        # 0: '0_BENIGN', 
                        # 1: 'Attack', 
                        0: '0_BENIGN', 
                        1: '1_DrDoS_DNS', 
                        2: '2_DrDoS_LDAP', 
                        3: '3_DrDoS_MSSQL',
                        4: '4_DrDoS_NTP', 
                        5: '5_DrDoS_NetBIOS', 
                        6: '6_DrDoS_SNMP', 
                        7: '7_DrDoS_SSDP', 
                        8: '8_DrDoS_UDP', 
                        9: '9_Syn', 
                        10: '10_TFTP', 
                        11: '11_UDPlag', 
                        12: '12_WebDDoS'
                        # 13: '13_Web Attack Sql Injection', 
                        # 14: '14_Web Attack XSS'
                        # 15: '15_backdoor',
                        # 16: '16_dos',
                        # 17: '17_injection',
                        # 18: '18_mitm',
                        # 19: '19_password',
                        # 20: '20_ransomware',
                        # 21: '21_scanning',
                        # 22: '22_xss'
                        } 
        # df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names)
        df_cm = pd.DataFrame(arr, index=class_names.values(), columns=class_names.values())
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        
        # 固定子圖參數
        plt.subplots_adjust(
            left=0.19,    # 左邊界
            bottom=0.167,  # 下邊界
            right=1.0,     # 右邊界
            top=0.88,      # 上邊界
            wspace=0.207,  # 子圖間的寬度間隔
            hspace=0.195   # 子圖間的高度間隔
        )
        
        plt.title(client_str +"_"+ Choose_method)
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        # Rotate the x-axis labels (prediction categories)
        plt.xticks(rotation=30, ha='right',fontsize=9)
        # plt.savefig(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_confusion_matrix.png")
        if epsilon == None and step_epsilon == None:
            plt.savefig(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_epsilon_{epsilon}_confusion_matrix.png")
            plt.show()
        else:
            str_epsilon = f"epsilon_{epsilon}"
            str_step_epsilon = f"step_epsilon_{step_epsilon}"
            plt.savefig(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/{str_step_epsilon}/{client_str}_epochs_{num_epochs}_epsilon_{epsilon}_confusion_matrix.png")
            plt.close()
        

def save_to_csv(data, filepath):
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def train(args, model, DEVICE, train_loader, optimizer, epoch, losses):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
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

def test_simple(model, DEVICE, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
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

def test(net,testloader, start_time, client_str,plot_confusion_matrix):
    # print("測試中")
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    ave_loss = 0.0  # 初始化 ave_loss

    net.eval()  #PyTorch 中的一個方法，用於將神經網絡模型設置為測試模式
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 計算滑動平均損失
            ave_loss = ave_loss * 0.9 + loss * 0.1

            # 將標籤和預測結果轉換為 NumPy 陣列
            y_true = labels.data.cpu().numpy()
            y_pred = predicted.data.cpu().numpy()
        
            # 計算每個類別的召回率
            acc = classification_report(y_true, y_pred, digits=4, output_dict=True)
            accuracy = correct / total

            # 將每個類別的召回率寫入 "recall-baseline.csv" 檔案
            RecordRecall = ()
            RecordAccuracy = ()
           
            for i in range(labelCount):
                RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
                 
            RecordAccuracy = RecordAccuracy + (accuracy, time.time() - start_time,)
            RecordRecall = str(RecordRecall)[1:-1]

            # 標誌來跟踪是否已經添加了標題行
            header_written = False
            with open(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}/recall-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    # file.write("標籤," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                    header_written = True
                file.write(str(RecordRecall) + "\n")
        
            # 將總體準確度和其他信息寫入 "accuracy-baseline.csv" 檔案
            with open(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    # file.write("標籤," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                    header_written = True
                file.write(f"精確度,時間\n")
                file.write(f"{accuracy},{time.time() - start_time}\n")

                # 生成分類報告
                GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
                report_df = pd.DataFrame(GenrateReport).transpose()
                report_df.to_csv(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}/baseline_report_{client_str}.csv",header=True)

    draw_confusion_matrix(y_true, y_pred,plot_confusion_matrix)
    accuracy = correct / total
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+f"測試準確度:"+Fore.LIGHTWHITE_EX+ f"{accuracy:.4f}"+
          "\t"+Fore.LIGHTYELLOW_EX + Style.BRIGHT+f"loss:"+Fore.LIGHTWHITE_EX + f"{ave_loss:.4f}")
    # print(Fore.LIGHTYELLOW_EX + Style.BRIGHT+f"loss:"+Fore.LIGHTWHITE_EX + f"{ave_loss:.4f}")
    return accuracy
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

def pgd_attack_evaluation(model, DEVICE, test_loader, classifier, attack, save_dir, epsilon, step_epsilon):
    model.eval()  # 設置模型為評估模式
    successful_attacks = []  # 儲存成功的攻擊案例
    accuracies = []  # 儲存每個批次的準確率
    
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    ave_loss = 0.0  # 初始化 ave_loss

    y_true_all = []  # 儲存所有批次的真實標籤
    y_pred_all = []  # 儲存所有批次的預測標籤

    # 逐批次處理資料
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        data_np = data.cpu().numpy()

        # 1. 產生對抗樣本
        x_adv = attack.generate(x=data_np)
        x_adv = torch.from_numpy(x_adv).to(DEVICE)

        # 2. 在對抗樣本上進行預測並計算損失
        outputs_adv = model(x_adv)
        batch_loss = criterion(outputs_adv, target).item()
        loss += batch_loss
        ave_loss = ave_loss * 0.9 + batch_loss * 0.1  # 更新滑動平均損失
        _, predicted_adv = torch.max(outputs_adv, 1)

        # 3. 計算批次的準確率，並累計總正確數和樣本數
        total += target.size(0)
        correct += (predicted_adv == target).sum().item()

        # 4. 將批次的真實標籤和對抗樣本的預測結果保存下來
        y_true_all.extend(target.cpu().numpy())
        y_pred_all.extend(predicted_adv.cpu().numpy())

        # 記錄每個批次的準確率
        batch_accuracy = np.mean(predicted_adv.cpu().numpy() == target.cpu().numpy())
        accuracies.append(batch_accuracy)

        # 記錄成功攻擊的樣本
        for i in range(len(target)):
            if predicted_adv[i] != target[i].cpu().numpy():
                successful_attacks.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'original_class': target[i].item(),
                    'adversarial_class': predicted_adv[i].item()
                })

    # 計算平均準確率並顯示結果
    avg_accuracy = correct / total
    print(Fore.RED + Style.BRIGHT+f'Average accuracy under PGD attack (ε={epsilon})(eps_step{step_epsilon}): {avg_accuracy:.4f}')
    print(Fore.RED + Style.BRIGHT+f'loss: {ave_loss:.4f}')

    # 計算每個類別的召回率
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    acc = classification_report(y_true, y_pred, digits=4, output_dict=True)
    
    # 將每個類別的召回率寫入 "recall-baseline.csv" 檔案
    RecordRecall = ()
    RecordAccuracy = ()
    start_time = time.time()
    
    for i in range(len(acc) - 3):  # -3 因為 classification_report 會多出 'accuracy', 'macro avg', 'weighted avg'
        RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
    
    RecordAccuracy = RecordAccuracy + (avg_accuracy, time.time() - start_time,)
    RecordRecall = str(RecordRecall)[1:-1]

    str_epsilon = f"epsilon_{epsilon}"
    str_step_epsilon = f"step_epsilon_{step_epsilon}"
    generatefolder(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}/", str_epsilon)
    generatefolder(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/", str_step_epsilon)
    # 標誌來跟踪是否已經添加了標題行
    header_written = False
    with open(f"{save_dir}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/{str_step_epsilon}/recall-baseline_epsilon_{epsilon}_{str_step_epsilon}.csv", "a+") as file:
        if not header_written:
            header_written = True
        file.write(str(RecordRecall) + "\n")

    # 將總體準確度和其他信息寫入 "accuracy-baseline.csv" 檔案
    with open(f"{save_dir}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/{str_step_epsilon}/accuracy-baseline_epsilon_{epsilon}_{str_step_epsilon}.csv", "a+") as file:
        if not header_written:
            header_written = True
        file.write(f"精確度,時間,loss\n")
        file.write(f"{avg_accuracy},{time.time() - start_time},{ave_loss}\n")

    # 生成分類報告
    report_df = pd.DataFrame(acc).transpose()
    report_df.to_csv(f"{save_dir}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/{str_step_epsilon}/baseline_report_epsilon_{epsilon}_{str_step_epsilon}.csv", header=True)

    # 畫出混淆矩陣
    draw_confusion_matrix(y_true, y_pred, True, epsilon,step_epsilon)

    # 保存成功攻擊的結果到 CSV
    attack_results = pd.DataFrame(successful_attacks)
    attack_results.to_csv(os.path.join(f"{save_dir}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/{str_step_epsilon}/", f'successful_attacks_eps_{epsilon}_{str_step_epsilon}.csv'), index=False)

    # 將對抗樣本和標籤保存為 CSV 和 npy 檔案
    x_adv_np = np.concatenate([attack.generate(x=data.cpu().numpy()) for data, _ in test_loader], axis=0)
    y_adv_np = np.concatenate([target.cpu().numpy() for _, target in test_loader], axis=0)

    x_adv_df = pd.DataFrame(x_adv_np.reshape(x_adv_np.shape[0], -1))  # 展平每個樣本
    x_adv_df['label'] = y_adv_np  # 添加標籤欄位
    x_adv_df.to_csv(os.path.join(f"{save_dir}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/{str_step_epsilon}", f'CICIDS2019_adversarial_samples_eps{epsilon}_{str_step_epsilon}.csv'), index=False)

    np.save(os.path.join(f"{save_dir}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/{str_step_epsilon}", f'x_test_CICIDS2019_adversarial_samples_eps{epsilon}_{str_step_epsilon}.npy'), x_adv_np)
    np.save(os.path.join(f"{save_dir}/{current_time}/{client_str}/{Choose_method}/{str_epsilon}/{str_step_epsilon}", f'y_test_CICIDS2019_adversarial_labels_eps{epsilon}_{str_step_epsilon}.npy'), y_adv_np)

    return avg_accuracy, successful_attacks

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='PGD Attack on CICIDS2017')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=True)
    args = parser.parse_args()


    # Load CICIDS2019 dataset
    filepath = "D:\\develop_Federated_Learning_Non_IID_Lab\\data"

    # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
    x_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_train_20240502.npy", allow_pickle=True)
    y_train = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_train_20240502.npy", allow_pickle=True)
    # # 20240502 CIC-IDS2019 after do labelencode and minmax 75 25分
    x_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\x_01_12_test_20240502.npy", allow_pickle=True)
    y_test = np.load(filepath + "\\dataset_AfterProcessed\\CICIDS2019\\01_12\\y_01_12_test_20240502.npy", allow_pickle=True)
    # # 20241105 CIC-IDS2019 after do labelencode and minmax 75 25分 do PGD eps0.3
    # x_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/x_test_CICIDS2019_adversarial_samples_eps0.3.npy", allow_pickle=True)
    # y_test = np.load(f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/y_test_CICIDS2019_adversarial_labels_eps0.3.npy", allow_pickle=True)
    # 轉換為張量
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    
    # 創建用於訓練和測試的數據加載器
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=500, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    # 初始化模型
    input_size = x_train.shape[1]  # 特徵數量
    num_classes = len(np.unique(y_train))  # 類別數量
    model = MLP(input_size, num_classes).to(DEVICE)
    
    # 每層神經元512下所訓練出來的正常model
    model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2019\\BaseLine_After_local_train_model_bk.pth'
    # PGD攻擊後的模型
    # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\single_AnalyseReportFolder\\CICIDS2019\\model_0.1.pt'
    # model_path = 'D:\\develop_Federated_Learning_Non_IID_Lab\\\single_AnalyseReportFolder\\CICIDS2019\\BaseLine_After_local_train_model_e500CandW.pth'
    model.load_state_dict(torch.load(model_path))

    # 設定優化器和調度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 測試模型
    # accuracy, all_preds, all_labels = test_simple(model, DEVICE, test_loader)
    accuracy = test(model,test_loader, start_IDS, client_str,True)
    
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
    # epsilons = [0.1, 0.2, 0.3]
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,1.0]
    step_epsilons_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,1.0]
    # epsilons = [0.05, 0.1]
    # step_epsilons_list = [0.05, 0.1]
    for epsilon in epsilons:
        for step_epsilons in step_epsilons_list:
            attack = ProjectedGradientDescent(
                estimator=classifier,
                eps=epsilon,
                # eps_step=0.5,
                eps_step=step_epsilons,
                # max_iter=100,
                max_iter=100,
                targeted=False,
                num_random_init=0
            )
        
        # 執行攻擊並評估
        acc, successful_attacks = pgd_attack_evaluation(
            model, DEVICE, test_loader, classifier, attack, save_dir, epsilon, step_epsilons
        )
        
        # 模型權重需經過訓練後才會發生變化
        # 這邊經測試樣本只生成test的樣本未訓練
        # 所以不用存模型
        # torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epsilon}.pth"))

        #紀錄結束時間
        end_IDS = time.time()
        getStartorEndtime("endtime",end_IDS,f"./Adversarial_Attack_Test/CICIDS2019/PGD_Attack/{today}/{current_time}/{client_str}/{Choose_method}")


if __name__ == '__main__':
    main()