import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def return_knn(dataset, model, n_clusters, n_neighbors):

    intermediate_reps, true_labels, = [], []
    for i, value in enumerate(dataset):
        if i == 500:
            break
        data_point = value[0].cuda().reshape(1, *value[0].shape)
        activations, label = model(data_point)

        if label!=value[1]:
            continue
        intermediate_reps.append(activations.reshape(-1,).cpu().numpy())
        true_labels.append(value[1])
        del data_point

    intermediate_reps = np.array(intermediate_reps)
    true_labels = np.array(true_labels)

    kmeans_models = []
    for i in range(10):
        kmeans = KMeans(n_clusters = n_clusters, random_state = 0)
        kmeans.fit(intermediate_reps[true_labels == i])
        kmeans_models.append(kmeans)


    knn_X = []
    knn_y = []

    for i, kmeans in enumerate(kmeans_models):
        knn_X.append(kmeans_models[i].cluster_centers_)
        knn_y.append(np.array([i]*n_clusters))

    knn_X = np.concatenate(knn_X, axis=0)
    knn_y = np.concatenate(knn_y, axis=0)

    knn_classifier = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn_classifier.fit(knn_X, knn_y)
    
    
    return knn_classifier, knn_y



def knn_defense(data, model, knn_classifier, distance = 20):
    
    activations, label = model(data)
    
    distances, labels = knn_classifier.kneighbors(activations.reshape(1,-1).cpu().numpy())
    
    return distances, labels, label
    
    
