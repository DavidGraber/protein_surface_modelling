import numpy as np

def normalize_01(array):
    array_norm = (array - np.min(array))/(np.max(array)-np.min(array))
    return array_norm

def normalize_columns_01(array):
    array_norm = np.zeros_like(array)
    for column in range(array.shape[1]):
        to_be_normed = array[:,column]
        column_norm = (to_be_normed - np.min(to_be_normed))/(np.max(to_be_normed)-np.min(to_be_normed))
        array_norm[:,column] = column_norm
    return array_norm

def convert_to_polar(cartesian_coords):
    x= cartesian_coords[:, 0]
    y = cartesian_coords[:,1]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    polar_coords = np.column_stack((rho, theta))    
    return polar_coords