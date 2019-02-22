from scipy import ndimage
import numpy as np
import imageio

# EXTRACTING THE FEATURES OF AN IMAGE AT A PATH
def extract_features(path):
    original = imageio.imread(path);
    (x,y) = get_center(original)
    (r,g,b) = check_color(original, x, y)
    return {
        "r":r,
        "g":g,
        "b":b,
        "x":x,
        "y":y,
        "path":path
    }

# Curried function - allows us to apply tags to all features extracted w/ the
# resulting function
def extract_features_tagged(die, side):
    if (side not in die):
        raise Exception("Specified side '" + str(side) + "' is not in die " + str(die));
    def f(filename):
        result = extract_features(filename)
        result["die"] = die
        result["side"] = side
        return result
    return f

# FIND THE CENTER (THE POINT OF INTEREST) OF THE DIE
def get_center(image):
    maximized = max_channel(original)
    masked = filter_by_trough(maximized)
    (y,x) = ndimage.measurements.center_of_mass(masked)
    return (int(x),int(y))

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

# Filter all pixels not meeting the threshold to black, otherwise white
def filter_by_trough(image):
    mean = histogram_trough(image)
    return np.where(image > mean, True, False)

# Get the threshold between black and brightly colored pixels
def histogram_trough(image):
    histogram = np.histogram(image,20)
    best_mean_ii = np.argmin(histogram[0])
    best_mean = histogram[1][best_mean_ii]
    print("Best mean found: " + str(best_mean))

    return best_mean;

# CHECKING THE COLOR OF AN POINT OF INTEREST ON THE DIE
# Summing up all color values at a center to create a feature vector
def check_color(original, x, y):
    (height, width, _) = original.shape
    disc = disc_mask(x,y,50,width,height)
    r = disc * get_r(original)
    g = disc * get_g(original)
    b = disc * get_b(original)
    ru = np.nanmean(map_to_nan(r))
    gu = np.nanmean(map_to_nan(g))
    bu = np.nanmean(map_to_nan(b))
    return (ru,gu,bu)

# Producing a disc-shaped mask at coords (b,a) with radius r on image with
# dimensions w,h
# Lets us filter to only get results at point of interest
def disc_mask(b,a,r,w,h):
    y,x = np.ogrid[-a:h-a, -b:w-b]
    mask = x*x + y*y <= r*r

    return np.where(mask == True, 1, 0).astype(np.uint8)

# Turn zero-values to NaN so that they don't contribute to value averages
def map_to_nan(array):
    return np.where(array == 0, np.NaN, array)

