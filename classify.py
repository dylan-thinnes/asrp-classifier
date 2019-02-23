# Image manipulation & classification
from scipy import ndimage
import numpy as np
import imageio

# Filename manipulation
import json
from os.path import split
from glob import glob

DEBUG=False

# MAIN

# Run feature extraction on preclassified 4's & 3's, 
# write to tagged.json
def main():
    training_data = train_die([4,4,4,3])
    h = open("./tagged.json", "w")
    json.dump(training_data, h)
    h.close()
    return

# Finds all images for a die spec, extracts their features.
# e.g. train_die([4,4,4,3])
def train_die(die):
    print("Crunching training data for die " + str(die))
    distinct_sides = list(set(die))
    training_results = []
    for side in distinct_sides:
        paths = image_paths(die, side)
        for path in paths:
            training_results.append(extract_features_tagged(die,side)(path))
    return training_results

# Finds paths to all images for a specific side of a specific die
def image_paths(die, side):
    die_name = "".join(str(x) for x in die)
    side_name = str(side)
    all_images = glob("./pictures/" + die_name + "/" + side_name + "/*")
    return all_images

# EXTRACTING THE FEATURES OF AN IMAGE AT A PATH
# Return a dictionary expressing the extracted values of the path & some
# debugging info
def extract_features(path):
    print("Handling '" + path + "'...")
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
    maximized = max_channel(image)
    masked = filter_by_trough(maximized)
    (y,x) = ndimage.measurements.center_of_mass(masked)
    global DEBUG;
    if (DEBUG):
        print("X: " + str(int(x)))
        print("Y: " + str(int(y)))
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
    disc = to_pixel_triples(disc_mask(x,y,50,width,height))
    disc_nan = map_to_nan(disc)
    filtered = disc_nan * original
    r = get_r(filtered)
    g = get_g(filtered)
    b = get_b(filtered)
    ru = np.nanmean(r)
    gu = np.nanmean(g)
    bu = np.nanmean(b)
    global DEBUG;
    if (DEBUG):
        print("Red:   " + str(ru))
        print("Green: " + str(gu))
        print("Blue:  " + str(bu))
    return (ru,gu,bu)

# Producing a disc-shaped mask at coords (b,a) with radius r on image with
# dimensions w,h
# Lets us filter to only get results at point of interest
def disc_mask(b,a,r,w,h):
    y,x = np.ogrid[-a:h-a, -b:w-b]
    mask = x*x + y*y <= r*r

    return np.where(mask, 1, 0).astype(np.uint8)

# Turn zero-values to NaN so that they don't contribute to value averages
def map_to_nan(array):
    return np.where(array == 0, np.NaN, array)

# Turning a an n-dim array into an n-dim array of triples
# Turns bw masks into images w/ rgb values, so dot prod can be used
def to_pixel_triples(array):
    (a,b) = array.shape
    return np.transpose([array,array,array], (1,2,0))

# RUNNING MAIN
if (__name__ == "__main__"): main()
