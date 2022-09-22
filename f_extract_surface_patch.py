def generate_graph(indeces, coords_sel, normals):

    '''Function that takes a set of points, with their label, coordinates and surface normals. Calculates for each point the 
    geodesic distance to its n nearest neighbors and saves that information in a dictionary representing a graph. '''
    
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    graph = {p:{} for p in indeces}

    knn = NearestNeighbors(n_neighbors=4)
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


def dijkstra(graph, center):
    
    '''Function that takes a graph and the starting node and returns a list of distances 
    from the starting node to every other node'''

    n = len(graph) # How many nodes are in the graph?
    
    # initialize a dictionary to save the distances of each node from the start node
    dist_from_center = {}
    for key in graph:
        dist_from_center[key]=1000
        
    # initialize a dictionary to save which node has been visited already
    visited={}
    for key in graph:
        visited[key]=False
            
    # set the distance for the start to be 0
    dist_from_center[center] = 0
    
    
    
    for p in range(n):
        
        # loop through all the nodes to check which one is not yet visited and has the lowest distance to the current node
        u = -1
        for key in graph:
            # if the node 'key' hasn't been visited and
            # we haven't processed it or the distance we have for it is less
            # than the distance we have to the "start" node
            
            # our start node (4557) will be selected first and assigned to u
            if not visited[key] and (u == -1 or dist_from_center[key] < dist_from_center[u]):
                u = key 
        
        
        # all the nodes have been visited or we can't reach this node
        if dist_from_center[u] == 1000:
            #print("break")
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

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(coords)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn = 5))
    pointcloud.orient_normals_consistent_tangent_plane(k=5)
    normals = np.asarray(pointcloud.normals)

    #Select a random point of the cloud, around which to draw a geodesic circle, set a geodesic radius
    radius = radius
    center_index = center_index

    first_sel = [center_index] # to save all the points that are within the non-geodesic radius

    #loop through all the points and calculate their euclidean distance to the selected center
    for index, point in enumerate(coords):
        dist = np.linalg.norm(coords[center_index]-point)

        # first selection with only those points that are close to the center point
        if dist < radius and dist != 0:
            first_sel.append(index)
            
    coords_sel = coords[first_sel]

    # generate a graph with the selected points
    graph = generate_graph(first_sel, coords_sel, normals)

    # check for each point the GEODESIC distance to the center with djikstra
    dist_from_center = dijkstra(graph, center_index)


    # Collect the indeces of the points that are < radius away from the center point
    patch_indeces = []

    for key in dist_from_center:
        if dist_from_center[key]<=radius:
            patch_indeces.append(key)
            
    patch_coords = coords[patch_indeces]
    patch_normals = normals[patch_indeces]

    # Make a graph out of the extracted patch
    #patch_graph = generate_graph(patch_indeces, patch_coords, normals)
    #patch_graph = make_graph_bidirectional(patch_graph)
    
    # Generate a double dictionary where the distance between two points can be accessed with dict[point1][point2]
    pairwise_dist_dict = {}
    for idx in patch_indeces:
        distances = dijkstra(graph, idx)
        pairwise_dist_dict[idx]=distances
    
    
    
    # Generate a quadratic dataframe for the pairwise distances between all points, label the columns and rows accordingly
    pairwise_distances = pd.DataFrame(np.zeros((len(patch_indeces),len(patch_indeces))))
    pairwise_distances.columns = patch_indeces
    pairwise_distances.index = patch_indeces

    # Add the distance information stored in the pairwise_dist_dict to the dataframe
    for index in patch_indeces:
        for idx in patch_indeces:
            pairwise_distances.at[index, idx] = pairwise_dist_dict[index][idx]

    return patch_indeces, patch_coords, pairwise_distances, first_sel