import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
def load_data():
    data = pd.read_csv("./data/dataset_AfterProcessed/TONIOT/20240523/test_ToN-IoT_dataframes_20240523.csv")
    # 选择数值列
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
    loader = DataLoader(dataset, batch_size=64, shuffle=True)  # 创建数据加载器
    return loader

# 训练 GAN
def train_gan(epochs, batch_size=64, latent_dim=100):
    # 初始化生成器和判别器
    generator = Generator(latent_dim, 38).to(DEVICE)  # 注意这里的data_shape需要与数据的特征数一致
    discriminator = Discriminator(38).to(DEVICE)  # 同上
    
    # 定义损失函数和优化器
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 加载数据
    dataloader = load_data()
    
    d_losses, g_losses = [], []
    
    for epoch in range(epochs):
        for i, (data, _) in enumerate(dataloader):
            
            batch_size = data.size(0)
            valid = torch.ones(batch_size, 1).to(DEVICE)  # 标记真实样本
            fake = torch.zeros(batch_size, 1).to(DEVICE)  # 标记生成样本
            
            # 训练判别器
            real_data = data.to(DEVICE)
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(DEVICE)  # 生成噪声
            gen_data = generator(z)  # 使用生成器生成数据
            real_loss = adversarial_loss(discriminator(real_data), valid)  # 计算真实样本的损失
            fake_loss = adversarial_loss(discriminator(gen_data.detach()), fake)  # 计算生成样本的损失
            d_loss = 0.5 * (real_loss + fake_loss)  # 判别器的总损失
            d_loss.backward()  # 反向传播
            optimizer_D.step()  # 更新判别器参数
            
            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(DEVICE)  # 生成噪声
            gen_data = generator(z)  # 使用生成器生成数据
            g_loss = adversarial_loss(discriminator(gen_data), valid)  # 生成器的损失
            g_loss.backward()  # 反向传播
            optimizer_G.step()  # 更新生成器参数
            
        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")  # 打印损失
        d_losses.append(d_loss.item())  # 记录判别器损失
        g_losses.append(g_loss.item())  # 记录生成器损失
    
    return d_losses, g_losses  # 返回损失列表

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备
    d_losses, g_losses = train_gan(epochs=5000)  # 训练 GAN

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="D Loss")
    plt.plot(g_losses, label="G Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator and Discriminator Loss During Training")
    plt.show()
