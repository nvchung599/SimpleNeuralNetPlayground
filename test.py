import numpy as np
from dataset_creator import *
from general import *
import math



data = gen_data()
m = data.shape[0]

Theta_1 = np.random.rand(4, 3)
Theta_2 = np.random.rand(1, 5)




Delta_1 = np.full_like(Theta_1, 0)
Delta_2 = np.full_like(Theta_2, 0)

for i in range(1):
    x_vec = data[i, [0, 1]].reshape(-1, 1) #2x1
    y = data[i, 2]
    h, A_2 = feed_fwd(x_vec, Theta_1, Theta_2)
    d_3 = h - y
    d_2 = np.matmul(np.transpose(Theta_2), np.array([[d_3]]))
    d_2 = np.multiply(d_2, A_2)
    d_2 = np.multiply(d_2, (1-A_2))
    Delta_1 = Delta_1 + np.matmul(d_2[1:], np.transpose(add_bias(x_vec))) #4x3
    Delta_2 = Delta_2 + np.matmul(np.array([[d_3]]), np.transpose(A_2))

    J = compute_cost(y, h)

Gradient_1 = (1/m) * Delta_1
Gradient_2 = (1/m) * Delta_2



# gradient check

for i in range(1):
    x_vec = data[i, [0, 1]].reshape(-1, 1) #2x1
    y = data[i, 2]
    h, A_2 = feed_fwd(x_vec, Theta_1, Theta_2)
    d_3 = h - y
    d_2 = np.matmul(np.transpose(Theta_2), np.array([[d_3]]))
    d_2 = np.multiply(d_2, A_2)
    d_2 = np.multiply(d_2, (1-A_2))
    Delta_1 = Delta_1 + np.matmul(d_2[1:], np.transpose(add_bias(x_vec))) #4x3
    Delta_2 = Delta_2 + np.matmul(np.array([[d_3]]), np.transpose(A_2))
