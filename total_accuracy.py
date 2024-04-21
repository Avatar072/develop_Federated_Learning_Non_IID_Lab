# 數據
# Client1 Chi-Square BaseLine
precision = [100.00, 100.00, 100.00, 98.65, 98.89, 99.91, 99.31, 99.14, 100.00, 100.00, 100.00, 98.84, 99.12, 70.00, 98.58]
recall = [100.00, 100.00, 100.00, 98.50, 98.64, 100.00, 99.91, 99.14, 100.00, 100.00, 100.00, 98.84, 99.70, 100.00, 95.21]
support = [1965, 385, 1965, 2002, 1988, 1093, 1147, 1628, 4, 12, 1974, 1212, 338, 7, 146]
# Client1 PCA BaseLine
# precision = [99.90, 100.00, 99.59, 99.65, 99.45, 99.82, 99.91, 99.69, 100.00, 90.91, 100.00, 97.66, 99.12, 0.00, 97.04]
# recall = [99.85, 100.00, 100.00, 99.85, 99.95, 100.00, 100.00, 98.22, 100.00, 83.33, 99.59, 99.67, 99.70, 0.00, 89.73]
# support = [1965, 385, 1965, 2002, 1988, 1093, 1147, 1628, 4, 12, 1974, 1212, 338, 7, 146]

#Client1 normal
# precision = [90.26, 100.00, 100.00, 99.85, 98.22, 100.00, 100.00, 99.75, 100.00, 100.00, 44.79, 98.85, 100.00, 70.00, 99.28]
# recall = [100.00, 100.00, 89.33, 98.35, 100.00, 99.91, 100.00, 99.14, 100.00, 58.33, 100.00, 99.67, 100.00, 100.00, 94.52]
# support = [3965, 385, 3965, 2002, 1988, 1093, 1147, 1628, 4, 12, 1974, 1212, 338, 7, 146]

#Client1 SMOTE
# precision = [87.05, 100.00, 100.00, 98.30, 99.25, 97.76, 99.73, 99.94, 100.00, 100.00, 65.82, 97.19, 97.97, 66.67, 98.55]
# recall = [100.00, 100.00, 85.15, 97.95, 99.75, 99.73, 97.47, 97.85, 100.00, 91.67, 100.00, 99.92, 100.00, 85.71, 93.15]
# support = [3965, 385, 3965, 2002, 1988, 1093, 1147, 1628, 4, 12, 1974, 1212, 338, 7, 146]

#Client1 BL-SMOTE1
# precision = [100.00, 100.00, 100.00, 100.00, 99.10, 100.00, 100.00, 99.88, 100.00, 100.00, 33.51, 100.00, 100.00, 70.00, 100.00]
# recall = [100.00, 100.00, 100.00, 99.25, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.83, 100.00, 100.00, 95.89]
# support = [3965, 385, 3965, 2002, 1988, 1093, 1147, 1628, 4, 12, 1974, 1212, 338, 7, 146]


#Client1 BL-SMOTE2
# precision = [100.00, 100.00, 100.00, 99.11, 99.75, 100.00, 100.00, 97.60, 100.00, 100.00, 12.49, 100.00, 98.54, 70.00, 77.91]
# recall = [100.00, 100.00, 100.00, 99.90, 99.14, 99.91, 96.77, 100.00, 100.00, 100.00, 100.00, 96.70, 99.70, 100.00, 91.78]
# support = [3965, 385, 3965, 2002, 1988, 1093, 1147, 1628, 4, 12, 1974, 1212, 338, 7, 146]

#Client3 normal
# precision = [41.79, 99.50, 98.71, 99.60, 99.90, 100.00, 100.00, 98.95, 28.17, 99.90]
# recall = [100.00, 100.00, 99.45, 99.85, 99.70, 94.91, 99.90, 98.75, 99.85, 100.00]
# support = [3965, 3965, 2000, 2000, 2000, 216, 2000, 2000, 2000, 2000]

#用precision算
# # 計算每個類別的預測正確數量
# correct_predictions = [prec * sup / 100 for prec, sup in zip(precision, support)]

# # 總預測正確數量
# total_correct_predictions = sum(correct_predictions)

# # 總樣本數量
# total_samples = sum(support)

# # 計算準確率
# accuracy = total_correct_predictions / total_samples

# print("整體的準確率（Accuracy）: {:.2f}%".format(accuracy * 100))
#########################

#用recall算
# 計算每個類別的預測正確數量
correct_predictions = [rec * sup / 100 for rec, sup in zip(recall, support)]

# 總預測正確數量
total_correct_predictions = sum(correct_predictions)

# 總樣本數量
total_samples = sum(support)

# 計算準確率
accuracy = total_correct_predictions / total_samples

print("整體的準確率（Accuracy）: {:.2f}%".format(accuracy * 100))
