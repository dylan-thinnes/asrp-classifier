from scipy import ndimage
import numpy as np
import imageio

def get_center(image):
    return

# Obtaining individual RGB channels
def get_r(original):
    return np.dot(original, [1,0,0])
def get_g(original):
    return np.dot(original, [0,1,0])
def get_b(original):
    return np.dot(original, [0,0,1])

# Getting the minimum or maximum RGB for each pixel
def min_channel(original):
    r = get_r(original)
    g = get_g(original)
    b = get_b(original)
    min = np.minimum(np.minimum(r,g),b)

    return min

def max_channel(original):
    r = get_r(original)
    g = get_g(original)
    b = get_b(original)
    max = np.maximum(np.maximum(r,g),b)

    return max

# Get the threshold between black and brightly colored pixels
def histogram_trough(image):
    histogram = np.histogram(image,20)
    best_mean_ii = np.argmin(histogram[0])
    best_mean = histogram[1][best_mean_ii]
    print("Best mean found: " + str(best_mean))

    return best_mean;

