import dataset_creator as gen


def cost_function(y_vec, h_vec):
    return 0

def gradient_check():
    return 0


data = gen.gen_data()

# 2 input, 3 hidden, 2 output

# Z_2 = X*Theta_1
# A_2 = g(Z_2)
# Z_3 = A_2*Theta_2
# H   = g(Z_3)

# Theta_1 is 2x3
# Theta_2 is 4x2

# FOR ALL m {
# init thetas
# fwd prop
# get errors
# accumulate term
# }

# get gradients
# apply gradients and update
