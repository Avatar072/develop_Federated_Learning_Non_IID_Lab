import torch
import math

# 定義兩個向量
# vector1 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],[5.0,5.0,5.0]])
# vector2 = torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0],[3.0,4.0,6.0]])

# 定義兩個向量
vector1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 8.0]])
vector2 = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

print("Vector1:", vector1)
print("Vector2:", vector2)

# 1. 計算單個向量的 L2 範數
print("\n1. 計算 Vector1 的 L2 範數:")

# 計算每個元素平方的總和，然後取平方根
squared_sum1 = torch.sum(vector1*vector1)  # 將每個元素平方，然後計算總和
norm1 = math.sqrt(squared_sum1.item())  # 將總和轉為標量並取平方根
print(f"手動計算: √({squared_sum1.item():.4f}) = {norm1:.4f}")

# 計算每個元素平方的總和，然後取平方根
squared_sum2 = torch.sum(vector2*vector2)  # 將每個元素平方，然後計算總和
norm2 = math.sqrt(squared_sum2.item())  # 將總和轉為標量並取平方根
print(f"手動計算: √({squared_sum2.item():.4f}) = {norm2:.4f}")


# 使用 torch.norm() 來計算 L2 範數
print(f"使用 torch.norm(): {torch.norm(vector1).item():.4f}")
print(f"使用 torch.norm(): {torch.norm(vector2).item():.4f}")


# 2. 計算兩個向量之間的歐幾里得距離
print("\n2. 計算 Vector1 和 Vector2 之間的歐幾里得距離:")

# 計算兩個向量之間的差異
diff = vector2 - vector1
# 計算差異的平方
squared_diff = diff*diff
# 計算差異平方和
squared_sum_diff = torch.sum(squared_diff)
# 計算平方和的平方根，得到歐幾里得距離
distance = math.sqrt(squared_sum_diff.item())

print("差異向量:", diff)
print("差異平方:", squared_diff)
print(f"平方和: {squared_sum_diff.item():.4f}")
print(f"歐幾里得距離: √{squared_sum_diff.item():.4f} = {distance:.4f}")
print(f"使用 torch.norm()求歐幾里得距離: {torch.norm(vector1 - vector2,p=2).item():.4f}")


# # 3. 計算向量的 L1 範數（曼哈頓距離）
# print("\n3. 計算 Vector1 的 L1 範數:")
# l1_norm1 = sum(abs(x) for x in vector1)
# print(f"手動計算: {' + '.join(f'|{x}|' for x in vector1)} = {l1_norm1:.4f}")
# print(f"使用 torch.norm(p=1): {torch.norm(vector1, p=1).item():.4f}")

# # 4. 計算兩個向量之間的 L1 距離
# print("\n4. 計算 Vector1 和 Vector2 之間的 L1 距離:")
# l1_distance = sum(abs(x) for x in diff)
# print(f"手動計算: {' + '.join(f'|{x:.2f}|' for x in diff)} = {l1_distance:.4f}")
# print(f"使用 torch.norm(p=1): {torch.norm(vector1 - vector2, p=1).item():.4f}")