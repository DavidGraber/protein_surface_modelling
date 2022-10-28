import numpy as np
from sklearn.neighbors import NearestNeighbors
from f_helper_functions import polar_to_cart

def create_ang_rad_bins(
    angular,
    radial,
    features_patch,
    cart_coords_patch,
    n_nearest_neigh
    ):

    '''Translates a patch of points into a meshgrid of fixed size with features assigned to each bin based on 
    a weighted average of the features of the points in vicinity of that bin
    . 
    Function that takes 
    - a meshgrid (angular, radial)
    - a matrix with features (features_patch)
    - the cartesian coordinates of the patch points
    - a number of neirest neighbors that should be taken into account for calculating a bins feature vector
    
    Returns a tensor of size (angular_bins, radial_bins, n_features)'''

    if angular.shape != radial.shape:
        raise Exception('Meshgrid matrices must be of same size')

    n_angular_bins = angular.shape[0]-1
    n_radial_bins = radial.shape[1]-1
   
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

            feature_array_neighbors = features_patch[neighbors][0]
            weights = ((1/dist**2).T) # contribution of the neighbor to the bin features decreases exponentially with distance

            # Compute a distance-weighted average of the feature vectors of all found nearest neighbors
            # and assign the resulting mean vector to the bin
            bin_desc = np.sum(feature_array_neighbors * weights, axis = 0) / np.sum(weights)

            z[angular_bin, radial_bin]=bin_desc
            
    return z