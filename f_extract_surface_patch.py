import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
#import time   

def generate_graph(indeces, coords_sel, normals):

    '''Function that takes a set of points, with their label, coordinates and surface normals. Calculates for each point the 
    geodesic distance to its n nearest neighbors and saves that information in a dictionary representing a graph. '''
    
    #from sklearn.neighbors import NearestNeighbors
    #import numpy as np
    
    graph = {p:{} for p in indeces}

    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(coords_sel)

    #loop through each point that is within the radius and find its nearest neighbors and their euclidean distance
    for idx, point in enumerate(coords_sel):
        dist, neighbors = knn.kneighbors([point], return_distance=True)
                
        # loop through the nearest neighbors, calculate their geodesic distance to the point chosen above
        # Add the geodesic distance to a graph-dictionary
        
        for index, neighbor in enumerate(neighbors[0]):
            
            geo_dist = dist[0][index]*(2-np.dot(normals[indeces[idx]], normals[indeces[neighbor]]))        

            if geo_dist !=0:
                graph[indeces[idx]][indeces[neighbor]]=geo_dist
                graph[indeces[neighbor]][indeces[idx]]=geo_dist

    return graph




def distances_from_center(graph, center):
    
    '''Function that takes a graph and the starting node and returns a list of distances 
    from the starting node to every other node'''
    
    dist_from_center = {key:100 for key in graph}
    dist_from_center[center] = 0
    unseen_nodes = list(dist_from_center.keys())
    
    for _ in graph:

        # IDENTIFICATION OF THE NEXT POINT TO LOOK AT (SHORTES DISTANCE FROM START)
        dist = 101
        for node in unseen_nodes:
            if dist_from_center[node]<dist:
                dist = dist_from_center[node]
                loc = node

        # LOOP THROUGH ALL THE NEIGHBORS OF THE NODE AND ADJUST THE VALUES OF THOSE, IF NEEDED
        for neighbor, weight in graph[loc].items():               
            if dist + weight < dist_from_center[neighbor]:
                dist_from_center[neighbor] = dist + weight 
        unseen_nodes.remove(loc)
        
    return dist_from_center



def compute_pairwise_distances(graph, patch_indeces):

    '''Djikstra Implementation: Function that takes a graph and returns a matrix 
    of size n_nodes x n_nodes containing the pairwise distances'''

    distance_matrix = np.zeros([len(patch_indeces),len(patch_indeces)])
    keys = list(graph.keys())

    #next_point_time = 0
    #next_point = 0
    #looping_neighbors_time = 0
    #looping_neighbors = 0
    
    for idx, start in enumerate(patch_indeces):
        dist_from_s = {key:100 for key in keys}
        dist_from_s[start] = 0
        unseen_nodes = list(dist_from_s.keys())
    
        for _ in keys:

            # IDENTIFICATION OF THE NEXT POINT TO LOOK AT (SHORTES DISTANCE FROM START)
            #tic = time.time()
            dist = 101
            for node in unseen_nodes:
                if dist_from_s[node]<dist:
                    dist = dist_from_s[node]
                    loc = node
            #toc = time.time()
            #next_point_time += toc-tic
            #next_point+=1


            # LOOP THROUGH ALL THE NEIGHBORS OF THE NODE AND ADJUST THE VALUES OF THOSE, IF NEEDED
            #tic = time.time()
            for neighbor, weight in graph[loc].items():               
                if dist + weight < dist_from_s[neighbor]:
                    dist_from_s[neighbor] = dist + weight 
            unseen_nodes.remove(loc)
            #toc = time.time()
            #looping_neighbors_time += toc-tic
            #looping_neighbors += 1

        
        distance_matrix[idx, :] = distance_matrix[:, idx] = [dist_from_s[point] for point in patch_indeces]

    #print("Next Point Searching:", next_point_time, "(", next_point, " times) = {t:.10f}".format(t =  next_point_time/next_point))
    #print("Looping Neighbors:", looping_neighbors_time, "(", looping_neighbors, " times) = {t:.10f}".format(t =  looping_neighbors_time/looping_neighbors))

    return distance_matrix




def extract_surface_patch(coords, center_index, radius):
    
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(coords)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn = 5))
    pointcloud.orient_normals_consistent_tangent_plane(k=5)
    normals = np.asarray(pointcloud.normals)

    first_sel = [] # to save all the points that are within the non-geodesic radius

    #loop through all the points and calculate their euclidean distance to the selected center
    for index, point in enumerate(coords):
        dist = np.linalg.norm(coords[center_index]-point)

        # first selection with only those points that are close to the center point
        if dist < radius:
            first_sel.append(index)
            
    coords_sel = coords[first_sel]


    # generate a graph with the selected points
    #start = time.time()
    graph = generate_graph(first_sel, coords_sel, normals)
    #end = time.time()
    #print("Graph Generation: "+ str(end - start) + 's')


    # check for each point the GEODESIC distance to the center with djikstra
    #start = time.time()
    dist_from_center = distances_from_center(graph, center_index)
    #end = time.time()
    #print("Distances from center: "+ str(end - start)+ 's')


    # Collect the indeces of the points that within the geodesic radius from the center point
    #start = time.time()

    patch_indeces = []
    for key in dist_from_center:
        if dist_from_center[key]<=radius:
            patch_indeces.append(key)
    patch_coords = coords[patch_indeces]
    
    #end = time.time()
    #print("Extraction of patch members: "+ str(end - start)+ 's')


    # Generate a new graph including only the patch members
    #start = time.time()
    patch_graph = generate_graph(patch_indeces, patch_coords, normals)
    #end = time.time()
    #print("Generation of Patch Graph: " + str(end - start)+ 's')


    # Compute the pairwise distances between all points in the patch_graph: 
    #start = time.time()
    pairwise_distances = compute_pairwise_distances(patch_graph, patch_indeces)
    #end = time.time()
    #print("Computation of distance matrix: " + str(end - start)+ 's')

   
    return patch_indeces, patch_coords, pairwise_distances

