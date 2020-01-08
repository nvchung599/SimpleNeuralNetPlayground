import numpy as np
import dataset_creator as gen


def cost_function(y_vec, h_vec):
    return 0

def gradient_check():
    return 0


data = gen.gen_data()
m = data.shape[0]

Theta_1 = np.random.rand(4, 3)
Theta_2 = np.random.rand(1, 5)

i = 0
x_vec = data[i, [0, 1]].reshape(-1, 1)
x_vec = np.vstack((1, x_vec))
print(x_vec.shape)
print(x_vec)
Z_2 = np.matmul(Theta_1, x_vec)
print(Z_2.shape)
print(Z_2)

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
