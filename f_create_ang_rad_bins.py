import numpy as np
from sklearn.neighbors import NearestNeighbors
from f_useful_functions import normalize_01

def create_ang_rad_bins(
    n_angular_bins,
    n_radial_bins, 
    features_patch,
    polar_coords_patch,
    n_nearest_neigh):

    max_rho = np.max(polar_coords_patch[:,0])


    angular, radial = np.mgrid[-np.pi:np.pi:72j, 0:max_rho:10j]
    z = np.zeros((angular.shape[0], angular.shape[1], features_patch.shape[1]))

    features_A_norm = np.zeros_like(features_patch)
    for column in range(features_patch.shape[1]):
        features_A_norm[:,column] = normalize_01(features_patch[:,column])

    knn = NearestNeighbors(n_neighbors=n_nearest_neigh)
    knn.fit(polar_coords_patch)

    for angular_bin in range(n_angular_bins):
        for radial_bin in range(n_radial_bins):
        
            dist, neighbors = knn.kneighbors([[radial[0, radial_bin], angular[angular_bin,0]]], return_distance=True)
        
            feature_array_neigbors = features_A_norm[neighbors][0]

            bin_desc = np.sum(feature_array_neigbors * (1/dist.T), axis = 0) / np.sum(1/dist.T)

            z[angular_bin, radial_bin]=bin_desc

    return z