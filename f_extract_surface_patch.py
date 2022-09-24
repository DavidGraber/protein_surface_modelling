def generate_graph(indeces, coords_sel, normals):

    '''Function that takes a set of points, with their label, coordinates and surface normals. Calculates for each point the 
    geodesic distance to its n nearest neighbors and saves that information in a dictionary representing a graph. '''
    
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
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



def extract_surface_patch(coords, center_index, radius):
    
    import open3d as o3d
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import time               #Remove

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(coords)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn = 5))
    pointcloud.orient_normals_consistent_tangent_plane(k=5)
    normals = np.asarray(pointcloud.normals)

    first_sel = [center_index] # to save all the points that are within the non-geodesic radius


    #loop through all the points and calculate their euclidean distance to the selected center
    for index, point in enumerate(coords):
        dist = np.linalg.norm(coords[center_index]-point)

        # first selection with only those points that are close to the center point
        if dist < radius and dist != 0:
            first_sel.append(index)
            
    coords_sel = coords[first_sel]


    # generate a graph with the selected points
    start = time.time()
    graph = generate_graph(first_sel, coords_sel, normals)
    end = time.time()
    print("Graph Generation: "+ str(end - start) + 's')

    # check for each point the GEODESIC distance to the center with djikstra
    start = time.time()
    dist_from_center = distances_from_center(graph, center_index)
    end = time.time()
    print("Distances from center: "+ str(end - start)+ 's')



    # Collect the indeces of the points that within the geodesic radius from the center point
    start = time.time()

    patch_indeces = []
    for key in dist_from_center:
        if dist_from_center[key]<=radius:
            patch_indeces.append(key)
    patch_coords = coords[patch_indeces]
    
    end = time.time()
    print("Extraction of patch members: "+ str(end - start)+ 's')



    #Make a new graph containing only the points of the patch + their nearest neighbors outside of the patch
    start = time.time()
    knn = NearestNeighbors(n_neighbors=30)
    knn.fit(coords_sel)

    second_sel = []

    #Compute the nearest neighbors of the points of the patch and add them to second_sel
    for point in patch_coords: 
        neighbors = knn.kneighbors([point], return_distance=False)
        for neighbor in neighbors[0]:
            if first_sel[neighbor] not in second_sel:
                second_sel.append(first_sel[neighbor])
    
    coords_second_sel = coords[second_sel]
    
    end = time.time()
    print("Add nearest neighbors to second_sel: " + str(end - start)+ 's')


    start = time.time()
    patch_graph = generate_graph(second_sel, coords_second_sel, normals)
    end = time.time()
    print("Generation of Patch Graph: " + str(end - start)+ 's')



    # Generate a dict with the pairwise distances
    start = time.time()
    pairwise_dist_dict = {}
    for key in patch_graph:
        distances = distances_from_center(patch_graph, key)
        pairwise_dist_dict[key]=distances
    end = time.time()
    print("Pairwise Dist Dict " + str(end - start)+ 's')


    start = time.time()
    # Generate a quadratic dataframe for the pairwise distances between all points, label the columns and rows accordingly
    pairwise_distances = pd.DataFrame(np.zeros((len(patch_indeces),len(patch_indeces))))
    pairwise_distances.columns = patch_indeces
    pairwise_distances.index = patch_indeces

    for p in patch_indeces:
        for q in patch_indeces:
            pairwise_distances.at[p,q] = pairwise_dist_dict[p][q]    
    
    end = time.time()
    print("Pairwise Distance Matrix: " + str(end - start)+ 's')
   
    return patch_indeces, patch_coords, pairwise_distances, first_sel, second_sel

