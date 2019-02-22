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
