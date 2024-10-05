#把会用到的库全加载了
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv("/Users/dudu/Desktop/capstone/SCMS_Delivery_History_Dataset.csv")

#看一看数据
data

#看一下数据的信息
print(data.info())

#数据描述
print(data.describe())

#检查缺失值
missing_values = data1.isnull().sum()
print(missing_values)

# 检查数值列中的非数值数据（非数字的值会返回True）
non_numeric_data = data[['Weight (Kilograms)', 'Freight Cost (USD)', 'Line Item Value', 'Pack Price', 'Unit Price', 'Line Item Insurance (USD)']].apply(pd.to_numeric, errors='coerce')
print(non_numeric_data.isnull().sum())  # 检查哪些列有非数值数据

# 将应该是数值的列转换为数值类型（如无法转换则标记为 NaN）
columns_to_convert = ['Weight (Kilograms)', 'Freight Cost (USD)', 'Line Item Value', 
                      'Pack Price', 'Unit Price', 'Line Item Insurance (USD)']

# 使用 pd.to_numeric 函数将列转换为数值类型，无法转换的将被设置为 NaN
for column in columns_to_convert:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# 处理缺失值 - 使用均值、中位数等方式填充缺失值，或删除包含缺失值的行
# 示例：用均值填充 Weight 列
data['Weight (Kilograms)'].fillna(data['Weight (Kilograms)'].mean(), inplace=True)

# 处理 Shipment Mode 和 Dosage 列中的缺失值
# Shipment Mode 列可以用 "Unknown" 填充，因为这通常是分类列
data['Shipment Mode'].fillna('Unknown', inplace=True)

# Dosage 列可以用最常见的值填充
data['Dosage'].fillna(data1['Dosage'].mode()[0], inplace=True)

# 检查转换和填充后的结果
print(data.isnull().sum())

# 确认数据类型是否正确
print(data.dtypes)

# 如果需要删除含有 NaN 的行，可以执行以下操作：
# data.dropna(inplace=True)

numeric_columns = ['Weight (Kilograms)', 'Line Item Quantity', 'Line Item Value']

# 统计缺失值数量
print("各列缺失值数量：")
print(data[numeric_columns].isnull().sum())

# 去除包含缺失值的行
data = data.dropna(subset=numeric_columns)

# 检查是否存在零或负数的值
for col in numeric_columns:
    invalid_count = (data1[col] <= 0).sum()
    print(f"列 {col} 中小于等于零的值的数量：{invalid_count}")

# 去除小于等于零的值
for col in numeric_columns:
    data1 = data1[data[col] > 0]

 # 描述性统计
print("数值列的描述性统计：")
print(data[numeric_columns].describe())

# 绘制箱线图查看异常值
import matplotlib.pyplot as plt
import seaborn as sns

for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=data[col])
    plt.title(f"{col} 的箱线图")
    plt.show()

# 比较清洗前后的数据量
original_count = data.shape[0]
cleaned_count = data_cleaned.shape[0]
removed_count = original_count - cleaned_count

print(f"原始数据量：{original_count}")
print(f"清洗后数据量：{cleaned_count}")
print(f"共去除 {removed_count} 条异常值样本")

# 重新绘制箱线图查看清洗效果
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=data_cleaned[col])
    plt.title(f"清洗后 {col} 的箱线图")
    plt.show()



