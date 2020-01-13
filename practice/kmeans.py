import tkinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = -2 * np.random.rand(100, 2)
x1 = 1 + 2* np.random.rand(50, 2)

x[50: 100, :] = x1

plt.scatter(x[ : , 0], x[ : , 1], s = 50, c = 'b')

plt.show()

Kmean = KMeans(n_clusters=2)

print(Kmean.fit(x))

print(Kmean.cluster_centers_)

plt.scatter(x[ : , 0], x[ : , 1], s =50, c='b')
plt.scatter(-0.94665068, -0.97138368, s=200, c='g', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()

print(Kmean.labels_)

sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
print(Kmean.predict(second_test))
