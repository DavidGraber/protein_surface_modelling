from Bio.PDB.PDBParser import PDBParser
import numpy as np
from f_parse_pdb import parse_pdb
import os
from f_extract_surface_patch import *
from sklearn.manifold import MDS

from f_create_ang_rad_bins import create_ang_rad_bins
from f_helper_functions import *
from f_center_embedding import center_embedding
import time

# Read the content of features_dir, predcoords_dir and pdbs_dir
predfeatures_dir = [file for file in os.listdir('c:\\Users\\david\\MT_code\\data\\masif_site_outputs\\predfeatures')]
predcoords_dir = [file for file in os.listdir('c:\\Users\\david\\MT_code\\data\\masif_site_outputs\\predcoords')]
pdbs_dir = [file for file in os.listdir('c:\\Users\\david\\MT_code\\data\\alphafold_outputs\\pdbs')]

# Save the corresponding paths without filenames
source_dir_feat = 'C:/Users/david/MT_code/data/masif_site_outputs/predfeatures'
source_dir_cord = 'C:/Users/david/MT_code/data/masif_site_outputs/predcoords'
source_dir_pdbs = 'C:/Users/david/MT_code/data/alphafold_outputs/pdbs'

# Import the fitness data of the mutants
os.chdir('c:\\Users\\david\\MT_code\\data')
fitness = np.load('fitness_dict_short.npy', allow_pickle="TRUE").item()



### Select a mutant to extract a patch from
min_fitness = min(fitness.values())
max_fitness = max(fitness.values())
scores = list(fitness.values())
mutants = list(fitness.keys())

for i in range(1500):
    tic = time.time()
    
    # Choose to draw from positive or negative mutations
    beneficial = np.random.choice([True, False])
    
    if beneficial: 
        # draw from scores >1
        draw = np.random.uniform(1, max_fitness)
        score = min(scores, key=lambda x:abs(x-draw))
        mutant_name = mutants[scores.index(score)]
        mutants.remove(mutant_name)
        scores.remove(score)
        print(mutant_name, fitness[mutant_name])
    else:  
        # draw from scores < 1
        draw = np.random.uniform(min_fitness, 1)
        score = min(scores, key=lambda x:abs(x-draw))
        mutant_name = mutants[scores.index(score)]
        mutants.remove(mutant_name)
        scores.remove(score)
        print(mutant_name, fitness[mutant_name])

    

    ### Import the data corresponding to that mutant
    import fnmatch
    features_filename = fnmatch.filter(predfeatures_dir, mutant_name+'*')[0]
    predcoords_filename = fnmatch.filter(predcoords_dir, mutant_name+'*')[0]
    pdb_filename = fnmatch.filter(pdbs_dir, mutant_name+'*')[0]

    # Load the features of the mutant
    features = np.load(os.path.join(source_dir_feat, features_filename))
    features = features[:, 16:32]
    features = normalize_m11(features)
    
    # Load the predcoords of the mutant
    predcoords = np.load(os.path.join(source_dir_cord, predcoords_filename))
    
    # Parse the pdb of the mutant
    parser = PDBParser(PERMISSIVE=1)
    with open(os.path.join(source_dir_pdbs, pdb_filename)) as pdbfile: 
        gb1, atomcoords = parse_pdb(parser, mutant_name, pdbfile)

    
    
    ### Determine the center for patch extraction 
    
    atms27 = np.asarray(gb1[27]["atoms"])
    atms28 = np.asarray(gb1[28]["atoms"])
    atms31 = np.asarray(gb1[31]["atoms"])
    
    C_GLU27 = np.asarray(gb1[27]["coords"])[np.where(atms27 == 'C')]
    CA_LYS28 = np.asarray(gb1[28]["coords"])[np.where(atms28 == 'CA')]
    CA_LYS31 = np.asarray(gb1[31]["coords"])[np.where(atms31 == 'CA')]
    
    tolerance = 0.0
    center_coords = []
    while len(center_coords) < 1:
        for i, point in enumerate(predcoords):
            if 4.864-tolerance < np.linalg.norm(point-CA_LYS31) < 4.864+tolerance: # CA of LYS31 dist 4.864 (225)
                if 4.973-tolerance < np.linalg.norm(point-C_GLU27) < 4.973+tolerance: # C of GLU27 dist 4.973 (190)
                    if 5.072-tolerance < np.linalg.norm(point-CA_LYS28) < 5.072+tolerance: # CA of LYS28 dist 5.072 (198)
                        center_coords.append(list(predcoords[i])) 
        tolerance += 0.1
    
    center_index = np.where(predcoords == center_coords[0])[0][0]
    
    

    ### Extract the patch
    patch_GB1 = {}
    patch_GB1["indeces"], patch_GB1["coords"], patch_GB1["distance_matrix"] = extract_surface_patch(predcoords, center_index, 12)
    patch_GB1["features"]=features[patch_GB1["indeces"]]
    
    ### Generate MDS embedding
    mds = MDS(dissimilarity='precomputed', random_state=0)
    embedding = mds.fit_transform(patch_GB1["distance_matrix"])
    embedding = center_embedding(embedding)
    patch_GB1["cart_embedding"] = embedding
    
    ### convert to polar coordinates
    polar_coords = cart_to_polar(embedding)
    patch_GB1["polar_embedding"] = polar_coords
    

    ### Create angular and radial bins
    features_patch = patch_GB1["features"]
    
    # Set the number of radial and angular bins and how many of its nearest neighbors should be taken into 
    # account for the calculation of each bins feature vector
    max_rho = 12 # diameter of the patches
    n_angular_bins = 72
    n_radial_bins = 10
    n_neighbors = 8

    # Create meshgrid n_angular_bins x n_radial_bins (+1 to have the right numb or spaces in between)
    angular, radial = np.mgrid[-np.pi:np.pi:complex(n_angular_bins+1), 0:max_rho:complex(n_radial_bins+1)]

    # To create a feature vector (length 16) for each of the bins and save them in a np.array of shape (angular bins x radial bins x features)
    # Feed the meshgrid matrices, the features matrices, the cartesian coords and a number of nearest neighbors (for the bin feature computation)
    tensor = create_ang_rad_bins(angular, radial, features_patch, embedding, n_neighbors) 
    patch_GB1["tensor"] = tensor


    # Save the resulting dictionary
    os.chdir('c:\\Users\\david\\MT_code\\data\\extracted_patches\\mutants')
    np.save('{m}_patch'.format(m = mutant_name), patch_GB1, allow_pickle=True)
    
    toc = time.time()
    print("{m} extracted in {t:.2f} seconds".format(m=mutant_name, t = (toc-tic)))