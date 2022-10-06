import numpy as np
from sklearn.neighbors import NearestNeighbors
from f_helper_functions import normalize_01
from f_helper_functions import normalize_columns_01
from f_helper_functions import polar_to_cart

def create_ang_rad_bins(
    n_angular_bins,
    n_radial_bins,
    max_rho,
    features_patch,
    cart_coords_patch,
    n_nearest_neigh
    ):


    # Create meshgrid n_angular_bins x n_radial_bins (+1 to have the right numb or spaces in between)
    angular, radial = np.mgrid[-np.pi:np.pi:complex(n_angular_bins+1), 0:max_rho:complex(n_radial_bins+1)]
    
    # Create empty feature array with one n_angular_bins x n_radial_bins x n_features
    z = np.zeros((angular.shape[0]-1, angular.shape[1]-1, features_patch.shape[1]))

    knn = NearestNeighbors(n_neighbors=n_nearest_neigh)
    knn.fit(cart_coords_patch)

    # Loop through each of the bins
    for angular_bin in range(n_angular_bins):
        
        theta = np.mean([angular[angular_bin,0], angular[angular_bin+1, 0]])

        for radial_bin in range(n_radial_bins):
            
            rho = np.mean([radial[0, radial_bin], radial[0, radial_bin+1]])

            # Convert to cartesian coordinates and look for nearest neighbors in cartesian coords
            cart = polar_to_cart(np.array([rho, theta]))

            # compute the bins nearest_neighbours in the cartesian representation of the patch
            dist, neighbors = knn.kneighbors( [cart], return_distance=True)

            feature_array_neigbors = features_patch[neighbors][0]

            weights = ((1/dist.T)-1)
            weights[weights<0] = 0

            # Compute a distance-weighted average of the feature vectors of all found nearest neighbors
            # and assign the resulting mean vector to the bin
            if np.sum(weights > 0):
                bin_desc = np.sum(feature_array_neigbors * weights, axis = 0) / np.sum(weights)

            else: 
                bin_desc = np.sum(feature_array_neigbors * weights, axis = 0)

            z[angular_bin, radial_bin]=bin_desc
            
    return z