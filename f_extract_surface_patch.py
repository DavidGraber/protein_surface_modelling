import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import time   

def generate_graph(indeces, coords_sel, normals):

    '''Function that takes a set of points, with their label, coordinates and surface normals. Calculates for each point the 
    geodesic distance to its n nearest neighbors and saves that information in a dictionary representing a graph. '''
    
    #from sklearn.neighbors import NearestNeighbors
    #import numpy as np
    
    graph = {p:{} for p in indeces}

    knn = NearestNeighbors(n_neighbors=8)
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

    n = len(graph) # How many nodes are in the graph?
    # initialize a dictionary to save the distances of each node from the start node
    dist_from_center = {key:100 for key in graph}  
    # initialize a dictionary to save which node has been visited already
    visited = {key:False for key in graph}      
    # set the distance for the start to be 0
    dist_from_center[center] = 0
    
    for p in range(n):  
        # loop through all the nodes to check which one is not yet visited and has the lowest distance to the current node
        u = -1
        for key in graph:
            # if the node 'key' hasn't been visited and
            # we haven't processed it or the distance we have for it is less
            # than the distance we have to the "start" node
            
            # our start node will be selected first and assigned to u
            if not visited[key] and (u == -1 or dist_from_center[key] < dist_from_center[u]):
                u = key 
        
        # all the nodes have been visited or we can't reach this node
        if dist_from_center[u] == 1000:
            break
        
        # set the node as visited
        visited[u] = True
        
        # from the current selected node u, check what the distances to the next nodes are and update their dist from center
        # loop through all the points (and their weights) that can be reached from our current node
        for key in graph[u]:
            if dist_from_center[u] + graph[u][key] < dist_from_center[key]:
                dist_from_center[key]= dist_from_center[u] + graph[u][key]
    
    return dist_from_center
    


def compute_pairwise_distances(graph, patch_indeces):

    distance_matrix = np.zeros([len(patch_indeces),len(patch_indeces)])

    keys = list(graph.keys())
    lookup = {key:{} for key in keys}

    for idx, start in enumerate(patch_indeces):

        dist_from_s = {key:100 for key in keys}
        dist_from_s[start] = 0
        visited = {key:False for key in keys}

        while False in visited.values():
            dist = 100
            for key in dist_from_s:
                if dist_from_s[key]<dist and not visited[key]:
                    dist = dist_from_s[key]
                    loc = key

            if not lookup[loc] == {}:
                for key in dist_from_s:
                    if dist_from_s[loc] + lookup[loc][key] < dist_from_s[key]:
                        dist_from_s[key] = dist_from_s[loc] + lookup[loc][key]
                visited[loc] = True

            else: # loop through the neighbors of this node loc and update its neighbors
                for key in graph[loc]:  
                    if dist_from_s[loc] + graph[loc][key] < dist_from_s[key]:
                        dist_from_s[key] = dist_from_s[loc] + graph[loc][key] 

                visited[loc] = True

        lookup[start] = dist_from_s
        distance_matrix[idx, :] = distance_matrix[:, idx] = [dist_from_s[point] for point in patch_indeces]

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

