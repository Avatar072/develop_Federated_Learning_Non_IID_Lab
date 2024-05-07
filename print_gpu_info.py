import torch

def main():
    # 返回gpu数量
    print("GPU数量:", torch.cuda.device_count())

    # 返回gpu名字，设备索引默认从0开始
    for i in range(torch.cuda.device_count()):
        print("GPU", i, "名称:", torch.cuda.get_device_name(i))

    # 返回当前设备索引
    print("当前设备索引:", torch.cuda.current_device())

if __name__ == "__main__":
    main()
