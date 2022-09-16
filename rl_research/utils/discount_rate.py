import matplotlib.pyplot as plt
import numpy as np
import sys

gamma = float(sys.argv[1])
n = 1000

inf_hor = []
running_inf_hor = 0
discount = []
for j in range(n):
    running_inf_hor += gamma**j
    inf_hor.append(running_inf_hor)
    discount.append(gamma**j)

true_hor = [1 / (1 - gamma)] * n


fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(np.arange(n), np.array(inf_hor), c='b')
ax1.plot(np.arange(n), np.array(true_hor), c='r')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.arange(n), np.array(discount), c='g')

plt.show()
