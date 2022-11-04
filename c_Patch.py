class Patch:

    '''Class storing the data of an extracted surface patch graph, including the '''

    def __init__(self, coords, distances, A, edge_index, feature_matrix):
        self.coords = coords
        self.distances = distances
        self.A = A
        self.edge_index = edge_index
        self.features = feature_matrix

    def num_nodes(self):
        return len(self.coords)

    def num_edges(self):
        return len(self.edge_index)

    def num_features(self):
        return self.features.shape[1]