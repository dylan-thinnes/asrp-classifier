import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from scipy.cluster.hierarchy import dendrogram, linkage

import json
from glob import glob

def main():
    vectors = get_training_vectors([4,4,4,3])
    # plot_dendrogram(vectors)
    # plot_3d(vectors)

# Cluster rgb vectors using cosine distance, show in a dendrogram
def plot_dendrogram(vectors):
    Z = linkage(vectors, metric="cosine")
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)
    plt.show()

# Plotting rgb vectors in 3d to see cosine distance yourself
def plot_3d(vectors):
    plot_vectors(vectors)

def plot_vector(ax, x, y, z):
    ax.plot([0,x],[0,y],[0,z], color=(x / 255, y / 255, z / 255))

def plot_vectors(vectors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for vector in vectors:
        plot_vector(ax, *vector)

    ax.set_xlim([0,256])
    ax.set_ylim([0,256])
    ax.set_zlim([0,256])
    plt.show()

# TAGGED ITEMS MANIPULATION
def features_to_vector(item):
    return (item["r"], item["g"], item["b"])

def get_vector(path):
    h = open(path, "r")
    features = json.load(h)
    h.close()
    vector = features_to_vector(features)
    return vector

def get_training_features_paths(die):
    die_name = "".join(str(x) for x in die)
    return glob("./pictures/" + die_name + "/*/*.json")

def get_training_vectors(die):
    return [get_vector(path) for path in get_training_features_paths(die)]

# RUN MAIN
if (__name__ == "__main__"): main()
