import numpy as np
from dataset_creator import *
from general import *
import math



data = gen_data()
m = data.shape[0]

Theta_1 = np.random.rand(4, 3)
Theta_2 = np.random.rand(1, 5)

alpha = 0.1
convergence_target = 0.005
cur_convergence = 99999999999
J_history = []


while cur_convergence > convergence_target:
    accumulated_J = 0
    Delta_1 = np.full_like(Theta_1, 0)
    Delta_2 = np.full_like(Theta_2, 0)

    for i in range(m):
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
        accumulated_J = J + accumulated_J

    Gradient_1 = (1/m) * Delta_1
    Gradient_2 = (1/m) * Delta_2

    if len(J_history) > 1:
        cur_convergence = J_history[-1] - accumulated_J
    J_history.append(accumulated_J)

    Theta_1 = Theta_1 - alpha * Gradient_1
    Theta_2 = Theta_2 - alpha * Gradient_2

    print("current convergence rate = %f" % cur_convergence)
    print("cost = %f" % accumulated_J)

plot_prediction_domain(Theta_1, Theta_2)
plt.show()

#for i in range(m):
#    x_vec = data[i, [0, 1]]

# 2 input, 4 hidden, 1 output

# add bias to X
# Z_2 = X*Theta_1
# A_2 = g(Z_2)
# add bias to A2
# Z_3 = A_2*Theta_2
# H   = g(Z_3)

# Theta_1 is 4x3
# Theta_2 is 1x5

# FOR ALL m {
# init thetas
# fwd prop
# get errors
# accumulate term
# }

# get gradients
# apply gradients and update
