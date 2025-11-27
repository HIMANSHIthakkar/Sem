# Sem
#import numpy as np

X = np.array([1,2,3,4])
y = np.array([3,5,7,9])   # y = 2x + 1

m = c = 0
lr = 0.01

for _ in range(1000):
    y_pred = m*X + c
    dm = -2 * sum(X * (y - y_pred)) / len(X)
    dc = -2 * sum(y - y_pred) / len(X)
    m -= lr * dm
    c -= lr * dc

print("m:", m, "c:", c)
