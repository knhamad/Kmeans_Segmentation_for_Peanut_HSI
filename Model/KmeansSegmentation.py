import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
def segment(hsi, n_clusters=2, show_figure=False):
    '''
    Segment peanut from RGB image using Kmeans
    '''
    row, col, band = hsi.shape
    img_reshape = np.reshape(hsi, (row*col,band))
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(img_reshape)
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = pairwise_distances_argmin(img_reshape, k_means_cluster_centers)
    labels = np.reshape(k_means_labels, (row, col))
    unique_lab, count = np.unique(labels, return_counts=True)
    sort_labels = np.zeros(labels.shape)
    # import pdb; pdb.set_trace()
    # re-assign lables such that peanut pixels always have same label
    for i, idx in enumerate(np.flip(np.argsort(count))):
        sort_labels[np.where(labels==unique_lab[idx])] = i
        
    if show_figure == True:                
        plt.figure()
        plt.imshow(sort_labels)
    
    return sort_labels