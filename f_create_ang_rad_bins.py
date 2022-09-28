import numpy as np
from sklearn.neighbors import NearestNeighbors
from f_useful_functions import normalize_01
from f_useful_functions import normalize_columns_01
from f_useful_functions import polar_to_cart

def create_ang_rad_bins(
    n_angular_bins,
    n_radial_bins, 
    features_patch,
    polar_coords_patch,
    cart_coords_patch,
    n_nearest_neigh
    ):

    max_rho = np.max(polar_coords_patch[:,0])

    # Create meshgrid n_angular_bins x n_radial_bins
    angular, radial = np.mgrid[-np.pi:np.pi:complex(n_angular_bins), 0:max_rho:complex(n_radial_bins)]
    
    # Create empty feature array with one n_angular_bins x n_radial_bins x n_features
    z = np.zeros((angular.shape[0], angular.shape[1], features_patch.shape[1]))

    # Normalize the features (IF POSSIBLE DO THIS SOMEWHERE ELSE)
    #features_patch_n = normalize_columns_01(features_patch)

    # normalize the inputs for the kNN for it to work properly
    #angular_n = normalize_01(angular)
    #radial_n = normalize_01(radial)
    #polar_coords_patch_n = normalize_columns_01(polar_coords_patch)

    knn = NearestNeighbors(n_neighbors=n_nearest_neigh)
    knn.fit(cart_coords_patch)

    # Loop through each of the bins
    for angular_bin in range(n_angular_bins):
        for radial_bin in range(n_radial_bins):
            
            # The polar coordinates of the bin rho=radial, theta=angular
            theta = angular[angular_bin,0]
            rho = radial[0, radial_bin]

            # Convert to cartesian coordinates and look for nearest neighbors in cartesian coords
            cart = polar_to_cart(np.array([rho, theta]))

            # compute the bins nearest_neighbours in the cartesian representation of the patch
            dist, neighbors = knn.kneighbors( [cart], return_distance=True)
        
            feature_array_neigbors = features_patch[neighbors][0]

            # Compute a distance-weighted average of the feature vectors of all found nearest neighbors
            # and assign the resulting mean vector to the bin
            bin_desc = np.sum(feature_array_neigbors * (1/dist.T), axis = 0) / np.sum(1/dist.T)

            z[angular_bin, radial_bin]=bin_desc

    return z