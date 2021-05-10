import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chisquare
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


def normalize(count_t):
    count_t_max = np.max(count_t)
    count_t_t = np.array([])
    for i in count_t:
        if i > 0:
            count_t_t = np.append(count_t_t, [(i + 1.) / count_t_max])
        else:
            count_t_t = np.append(count_t_t, [1. / count_t_max])
    return count_t_t


def count_12_t(img1, img2):
    values1 = pd.Series(img1.ravel()).value_counts()
    values2 = pd.Series(img2.ravel()).value_counts()

    max_1 = max(values1.index)
    max_2 = max(values1.index)
    max_t = max_1 if max_1 > max_2 else max_2

    count_1, bin_1 = np.histogram(pd.Series(img1.ravel()), bins=10, range=(0, max_t))
    count_2, bin_2 = np.histogram(pd.Series(img2.ravel()), bins=10, range=(0, max_t))

    count_1_t = normalize(count_1)
    count_2_t = normalize(count_2)

    return count_1_t, count_2_t

# chi_square

def chi_square(img1, img2):
    count_1_t, count_2_t = count_12_t(img1, img2)
    return chisquare(count_1_t, count_2_t)[0]

# mean_square_error

def metric_mse(img1, img2):
    return mean_squared_error(img1, img2)

# ssim

def metric_ssim(img1, img2):
    return ssim(img1, img2)