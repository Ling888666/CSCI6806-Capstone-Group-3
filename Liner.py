import pandas as pd
import numpy as np

def linear_regression(x, y):
    n = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumxy = sum([x[i] * y[i] for i in range(n)])
    sumx2 = sum([x[i] * x[i] for i in range(n)])

    a1 = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx * sumx)
    a0 = (sumy / n) - a1 * (sumx / n)

    st = sum([(y[i] - np.mean(y)) ** 2 for i in range(n)])
    sr = sum([(y[i] - a1 * x[i] - a0) ** 2 for i in range(n)])

    syx = (sr / (n - 2)) ** 0.5
    r2 = (st - sr) / st

    return a0, a1, syx, r2

file_path = r'/Users/alan/Desktop/FDU/Capstone/SCMS_Delivery_History_Dataset.csv'
data = pd.read_csv(file_path)


print(data.head())

x = data['Line Item Quantity'].values
y = data['Line Item Value'].values

x = np.nan_to_num(x)
y = np.nan_to_num(y)

a0, a1, syx, r2 = linear_regression(x, y)

print(f"a0: {a0}")
print(f"a1: {a1}")
print(f"syx: {syx}")
print(f"r2: {r2}")
