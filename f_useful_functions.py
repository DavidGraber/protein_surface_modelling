import numpy as np

def normalize_01(array):
    array_norm = (array - np.min(array))/(np.max(array)-np.min(array))
    return array_norm


def convert_to_polar(cartesian_coords):
    x= cartesian_coords[:, 0]
    y = cartesian_coords[:,1]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    polar_coords = np.column_stack((rho, theta))    
    return polar_coords