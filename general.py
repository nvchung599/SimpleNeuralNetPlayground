import numpy as np
from dataset_creator import *
from matplotlib import pyplot as plt
import random
import math


def gradient_check():
    return 0

def add_bias(vector):
    """input i*1 vector, stacks a 1 at the top"""
    return np.vstack((1, vector))

def feed_fwd(x_vec, Theta_1, Theta_2):
    x_vec = add_bias(x_vec) #3x1
    Z_2 = np.matmul(Theta_1, x_vec) # 4x3 * 3x1 = 4x1
    A_2 = sigmoid(Z_2) #4x1
    A_2 = add_bias(A_2) #5x1
    Z_3 = np.matmul(Theta_2, A_2) # 1x5 * 5x1 = 1x1
    A_3 = sigmoid(Z_3)
    return (A_3[0, 0], A_2)

def predict_mat(x0, x1, Theta_1, Theta_2):

    rows = x0.shape[0]
    cols = x1.shape[0]
    pred_mat = np.full_like(x0, 2)

    for i in range(rows):
        for j in range(cols):
            x_vec = np.array([[x0[i,j]], [x1[i,j]]])
            x_vec = add_bias(x_vec) #3x1
            Z_2 = np.matmul(Theta_1, x_vec) # 4x3 * 3x1 = 4x1
            A_2 = sigmoid(Z_2) #4x1
            A_2 = add_bias(A_2) #5x1
            Z_3 = np.matmul(Theta_2, A_2) # 1x5 * 5x1 = 1x1
            A_3 = sigmoid(Z_3)

            #pred = None
            #if A_3[0, 0] > 0.5:
            #    pred = 1
            #else:
            #    pred = 0
            pred = A_3[0, 0]
            pred_mat[i, j] = pred

    return pred_mat

def plot_prediction_domain(Theta_1, Theta_2):
    x_zero_space = np.linspace(0, 10, 100)
    x_one_space = np.linspace(0, 14, 100)
    x0, x1 = np.meshgrid(x_zero_space, x_one_space)
    y = predict_mat(x0, x1, Theta_1, Theta_2)
    plt.contour(x0, x1, y)
#    fig, ax = plt.subplots()
#    CS = ax.contour(x0, x1, y)
#    ax.clabel(CS, inline=1, fontsize=10)
#    ax.set_title('Simplest default with labels')

def compute_cost(y, h):
    J = -y*math.log(h) - (1-y)*math.log(1-h)
    return J

def sigmoid(x):
    """returns sigmoid of single value, or piece-wise sigmoid of matrix"""
    return (1/(1+np.exp(-x)))
