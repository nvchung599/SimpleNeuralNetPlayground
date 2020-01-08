import numpy as np
from matplotlib import pyplot as plt
import pandas

def f_zero(x_vec):
    random_vec = np.random.rand(x_vec.shape[0]) * 1.5
    random_vec = random_vec.reshape(-1, 1)
    y_vec = np.sin(x_vec) + x_vec + random_vec
    return y_vec

def f_one(x_vec):
    random_vec = np.random.rand(x_vec.shape[0]) * 1.5
    random_vec = random_vec.reshape(-1, 1)
    y_vec = np.sin(x_vec) + x_vec + random_vec + 1
    return y_vec

def label_zero(x_vec):
    label_vec = np.full_like(x_vec, 0)
    return label_vec

def label_one(x_vec):
    label_vec = np.full_like(x_vec, 1)
    return label_vec

def gen_data():
    x_zero = np.linspace(0,10,100)
    x_zero = x_zero.reshape(-1, 1)
    y_zero = f_zero(x_zero)
    labels = label_zero(x_zero)
    data_zero = np.hstack((x_zero, y_zero, labels))

    x_one = np.linspace(0,10,100)
    x_one = x_one.reshape(-1, 1)
    y_one = f_one(x_one)
    labels = label_one(x_one)
    data_one = np.hstack((x_one, y_one, labels))

    data_matrix = np.vstack((data_zero, data_one))
    np.random.shuffle(data_matrix)

#    plt.scatter(x_zero, y_zero)
#    plt.scatter(x_one, y_one)
#    plt.legend(labels=['zero', 'one'])
#    plt.show()

    return data_matrix

