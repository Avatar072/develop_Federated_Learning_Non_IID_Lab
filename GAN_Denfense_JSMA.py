import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mytoolfunction import generatefolder

'''
以GNA 的生成器 noise替換成JSMA攻擊後的資料
主要作用是在鑑別器其功用是分別出正常流量跟JSMA攻擊後的流量
最大用意是要查出JSMA攻擊後的流量
所以gloss 和 dloss要差距越大不能收斂，dloss要越大
'''
# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, input_dim, data_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),         # 第一层全连接层
            nn.LeakyReLU(0.2),                 # Leaky ReLU 激活函数
            nn.BatchNorm1d(256, momentum=0.8), # 批归一化
            nn.Linear(256, 512),               # 第二层全连接层
            nn.LeakyReLU(0.2),                 # Leaky ReLU 激活函数
            nn.BatchNorm1d(512, momentum=0.8), # 批归一化
            nn.Linear(512, 256),               # 第三层全连接层
            nn.LeakyReLU(0.2),                 # Leaky ReLU 激活函数
            nn.BatchNorm1d(256, momentum=0.8), # 批归一化
            nn.Linear(256, data_shape),        # 输出层
            nn.Sigmoid()                       # Sigmoid 激活函数
        )

    def forward(self, z):
        return self.model(z)  # 前向传播

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_shape, 256),  # 第一层全连接层
            nn.ReLU(),                  # ReLU 激活函数
            nn.Linear(256, 512),        # 第二层全连接层
            nn.ReLU(),                  # ReLU 激活函数
            nn.Linear(512, 256),        # 第三层全连接层
            nn.LeakyReLU(0.2),          # Leaky ReLU 激活函数
            nn.Linear(256, 1),          # 输出层
            nn.Sigmoid()                # Sigmoid 激活函数
        )

    def forward(self, data):
        return self.model(data)  # 前向传播

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    num_name = [
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
    data = data[num_name]  # 只选择数值列
    # 创建数据集和数据加载器
    dataset = TensorDataset(torch.tensor(data.values, dtype=torch.float32), torch.zeros(len(data), dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=500, shuffle=True)  # 创建数据加载器
    return loader

# 训练判别器函数，包括对抗样本和正常样本
def train_discriminator(discriminator, optimizer, real_data, fake_data, loss_fn):
    N = real_data.size(0)
    optimizer.zero_grad()

    # 训练真实数据
    prediction_real = discriminator(real_data)
    error_real = loss_fn(prediction_real, torch.ones(N, 1).to(DEVICE))  # 真实数据标签为1

    # 训练生成数据
    prediction_fake = discriminator(fake_data)
    error_fake = loss_fn(prediction_fake, torch.zeros(N, 1).to(DEVICE))  # 生成数据标签为0

    # 计算总误差并反向传播
    error = (error_real + error_fake) / 2
    error.backward()
    optimizer.step()
    
    return error, prediction_real, prediction_fake

# 训练 GAN
def train_gan(epochs, jsma_data, batch_size=500, latent_dim=38):
    # 初始化生成器和判别器
    generator = Generator(latent_dim, 38).to(DEVICE)  # 注意这里的data_shape需要与数据的特征数一致
    discriminator = Discriminator(38).to(DEVICE)  # 同上
    
    # 定义损失函数和优化器
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 加载正常数据
    dataloader = load_data("./data/dataset_AfterProcessed/TONIOT/20240523/train_ToN-IoT_dataframes_train_half3_20240523.csv")
    
    d_losses, g_losses = [], []
    
    for epoch in range(epochs):
        for i, (real_data, _) in enumerate(dataloader):
            
            batch_size = real_data.size(0)
            
            # 训练判别器
            real_data = real_data.to(DEVICE)
            # noise使用jsma攻擊後的樣本
            z = jsma_data[np.random.choice(jsma_data.shape[0], batch_size, replace=False)]
            z = torch.tensor(z, dtype=torch.float32).to(DEVICE)
            gen_data = generator(z)
            d_loss, real_result, fake_result = train_discriminator(discriminator, optimizer_D, real_data, gen_data.detach(), adversarial_loss)
            
            # 训练生成器
            optimizer_G.zero_grad()
            # noise使用jsma攻擊後的樣本
            z = jsma_data[np.random.choice(jsma_data.shape[0], batch_size, replace=False)]
            z = torch.tensor(z, dtype=torch.float32).to(DEVICE)
            gen_data = generator(z)
            g_loss = adversarial_loss(discriminator(gen_data), torch.ones(batch_size, 1).to(DEVICE))
            g_loss.backward()
            optimizer_G.step()
            
        # 每 10000 个 epochs 保存一次模型和绘制损失曲线
        # if (epoch+1) % 10000 == 0:
        # 保存生成器和判别器模型
            # generatefolder(f"./Adversarial_Attack_Test/20240725/",{epoch})
            folder_path = f"./Adversarial_Attack_Test/20240725"
            generatefolder(folder_path,str({epoch+1}))

            torch.save(generator.state_dict(), f"./Adversarial_Attack_Test/20240725/{epochs}/generator_{epochs}.pth")
            torch.save(discriminator.state_dict(), f"./Adversarial_Attack_Test/20240725/{epochs}/discriminator_{epochs}.pth")
            # 绘制损失曲线
            plt.figure(figsize=(10, 5))
            plt.plot(d_losses, label="D Loss")
            plt.plot(g_losses, label="G Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Generator and Discriminator Loss During Training")
            plt.savefig(f"./Adversarial_Attack_Test/20240725/{epochs}/g_loss_d_loss.png")

        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")  # 打印损失
        d_losses.append(d_loss.item())  # 记录判别器损失
        g_losses.append(g_loss.item())  # 记录生成器损失
    
    return generator, discriminator, d_losses, g_losses  # 返回模型和损失列表

# 生成一些样本并可视化
def generate_samples(generator, latent_dim, n_samples=100):
    z = torch.randn(n_samples, latent_dim).to(DEVICE)
    samples = generator(z).cpu().detach().numpy()
    return samples

def visualize_samples(samples, n_samples=100):
    plt.figure(figsize=(10, 5))
    plt.plot(samples[:n_samples].T)
    plt.title("Generated Samples")
    plt.show()

# 评估判别器性能
def evaluate_discriminator(discriminator, generator, dataloader, jsma_data):
    real_correct = 0
    fake_correct = 0
    total = 0
    
    for real_data, _ in dataloader:
        batch_size = real_data.size(0)
        real_data = real_data.to(DEVICE)
        
        # 真实数据的预测
        real_pred = discriminator(real_data)
        real_correct += (real_pred >= 0.5).sum().item()
        
        # 生成数据的预测
        z = jsma_data[np.random.choice(jsma_data.shape[0], batch_size, replace=False)]
        z = torch.tensor(z, dtype=torch.float32).to(DEVICE)
        fake_data = generator(z)
        fake_pred = discriminator(fake_data)
        fake_correct += (fake_pred < 0.5).sum().item()
        
        total += batch_size
        
    real_accuracy = real_correct / total
    fake_accuracy = fake_correct / total
    
    print(f"Real Data Accuracy: {real_accuracy * 100:.2f}%")
    print(f"Fake Data Accuracy: {fake_accuracy * 100:.2f}%")

# 加载模型函数
def load_model(generator_path, discriminator_path, latent_dim, data_shape):
    generator = Generator(latent_dim, data_shape)
    discriminator = Discriminator(data_shape)
    
    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discriminator_path))
    
    generator.to(DEVICE)
    discriminator.to(DEVICE)
    
    return generator, discriminator

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备
    # 加载 JSMA 攻击后的数据
    jsma_data = np.load("./Adversarial_Attack_Test/20240725_FL_cleint3_use_train_.0.05_0.02/x_DoJSMA_adversarialsample_removestring_20240725.npy", allow_pickle=True)

    # 加载正常的数据
    # jsma_data = np.load("./Adversarial_Attack_Test/20240725_FL_cleint3_use_train_.0.05_0.02/x_testdata_removestring_20240725.npy", allow_pickle=True)

    generator, discriminator, d_losses, g_losses = train_gan(epochs=1, jsma_data=jsma_data)  # 训练 GAN

    # 載入模型驗證
    # generator, discriminator = load_model("./Adversarial_Attack_Test/20240725/generator.pth",
    #                                       "./Adversarial_Attack_Test/20240725/discriminator.pth",
    #                                       38,
    #                                       38,
    #                                       )
    # 绘制损失曲线
    # plt.figure(figsize=(10, 5))
    # plt.plot(d_losses, label="D Loss")
    # plt.plot(g_losses, label="G Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.show()


    # 加载 JSMA 攻击后的数据
    # dataloader = load_data("./Adversarial_Attack_Test/20240725_FL_cleint3_use_train_.0.05_0.02/final_adver_examples_with_missing.csv")

    # 加载正常的数据
    dataloader = load_data("./data/dataset_AfterProcessed/TONIOT/20240523/train_ToN-IoT_dataframes_train_half3_20240523.csv")
    
    evaluate_discriminator(discriminator, generator, dataloader, jsma_data)
    
    # 生成一些样本并可视化
    # samples = generate_samples(generator, latent_dim=38, n_samples=100)
    # visualize_samples(samples)
