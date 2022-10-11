import numpy as np

def center_embedding(embedding):
    diff_y = np.max(embedding[:,1]) - np.abs(np.min(embedding[:,1])) # calculate each y_value minus diff_y/2
    diff_x = np.max(embedding[:,0]) - np.abs(np.min(embedding[:,0])) # calculate each x_value minus diff_y/2

    correction = np.zeros_like(embedding)
    correction[:,0] = -diff_x/2
    correction[:,1] = -diff_y/2

    corrected_embedding = embedding + correction

    return corrected_embedding