import cv2 as cv
import numpy as np

def detect_edge(img, th1=10, th2=20, expand=5):
    edges = cv.Canny(img, th1, th2)
    mask = edges > 0
    mask[:, 0] = True
    mask[:, -1] = True
    mask[0, :] = True
    mask[-1, :] = True

    # expand the edge mask 
    for _ in range(expand):
        new_mask = mask.copy()
        new_mask[1:,:] = np.logical_or(new_mask[1:,:], mask[:-1,:])
        new_mask[:-1,:] = np.logical_or(new_mask[:-1,:], mask[1:,:])
        new_mask[:,1:] = np.logical_or(new_mask[:,1:], mask[:,:-1])
        new_mask[:,:-1] = np.logical_or(new_mask[:,:-1], mask[:,1:])
        mask = new_mask.copy()
    edges[mask] = 255
    return edges
