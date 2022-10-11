import numpy as np

def rotate_circular_patch(patch_array):
    patch_rotated = np.zeros_like(patch_array)
    patch_rotated[0, :, :] = patch_array[-1, :, :]
    patch_rotated[1:, :, :] = patch_array[:-1, :, :]
    return patch_rotated


def calc_dot_prod_matrix(tensor1, tensor2):

    n_angular = tensor1.shape[0]
    n_radial = tensor1.shape[1]

    dot_matrix = np.zeros([n_angular,n_radial])

    for ang in range(n_angular):
        for rad in range(n_radial):
            dot_matrix[ang, rad] = np.dot(tensor1[ang,rad,:], tensor2[ang,rad,:])
    
    return dot_matrix



def compare_patches(tensor1, tensor2, max_rho):

    if not tensor1.ndim == tensor2.ndim:
        print("Tensors must have same dimension")

    else:        
        
        # Compute a weight vector representing the bin sizes from radius=0 to radius = max_rho
        radii = [i * max_rho/tensor1.shape[1] for i in range(tensor1.shape[1]+1)]
        areas = [ np.pi*np.square(radii[i+1]) - np.pi*np.square(radii[i]) for i in range(len(radii)-1)]
        weight_vector = np.array(areas) / tensor1.shape[0]

        # Calculate the complementarity in the original position
        best_dot_matrix = calc_dot_prod_matrix(tensor1, tensor2)*weight_vector
        comp_score = np.sum(best_dot_matrix)
        #print("Initial score: " + str(int(comp_score)))
        
        best_score = comp_score
        best_rotation = 0
        best_tensor2 = tensor2

        n_rotations = tensor1.shape[0]-1 # How many angular rotations?
        
        rot_tensor2 = tensor2
        for rotation in range(n_rotations):
            rot_tensor2 = rotate_circular_patch(rot_tensor2)
            dot_matrix = calc_dot_prod_matrix(tensor1, rot_tensor2)*weight_vector
            comp_score = np.sum(dot_matrix)
            # print("Rotation: " +str(rotation+1)+ ", Score: " + str(comp_score))

            if comp_score > best_score:
                best_score = comp_score
                best_rotation = rotation+1
                best_tensor2 = rot_tensor2
                best_dot_matrix = dot_matrix

        #print("Best Score: " + str(int(best_score)) + " after " + str(best_rotation) + " rotations")
        return best_score, best_rotation, best_tensor2, best_dot_matrix