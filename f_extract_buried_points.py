def extract_buried_points(subunit_coords, complex_coords, threshold):

    '''Function that takes as input the coordinates of two similar pointclouds (one subunit and
    one complex involving that subunit) and a threshold. Returns a list of indeces indicating 
    which points of the first pointcloud (subunit) are more than an certain distance (threshold)
    away from any point in the second pointcloud. These points are "buried" in the complex. The 
    point with the largest distance above the threshold is returned as the "center" of the buried patch'''

    from sklearn.neighbors import NearestNeighbors
    
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(complex_coords) 

    buried= []
    largest_dist = 0
    for i in range(len(subunit_coords)):
        if (dist := neigh.kneighbors([subunit_coords[i]])[0][0][0]) > threshold:
        #if neigh.kneighbors([subunit_coords[i]])[0][0][0] > threshold:
            buried.append(i)
            if dist > largest_dist:
                largest_dist = dist
                center = i
    return buried, center 