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

