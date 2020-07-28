# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv

##===================================================================================================================##
##===================================================================================================================##
def set_projected_points(num_pts_w, num_pts_h, step_width, step_height):
    """
    Set the projected points.
    
    Input:
        num_pts_w, 
        num_pts_h, 
        step_width, 
        step_height
    
    Output:
        np.array(pts)
    """
    
    ## For Output
    pts = []
    
    ## Main loop
    for j in range(num_pts_h):
        for i in range(num_pts_w):
            
            pt = [0, 0, 1]

            pt[0] = i * step_width
            pt[1] = j * step_height
            
            pts.append(np.array(pt))
    
    ## Convert a list into np.array and return
    return np.array(pts)


##===================================================================================================================##
##===================================================================================================================##
def error_homography(size_slave, homography_gt, homography_est, num_pts_w=21, num_pts_h=21):
    """
    Calculate projection error.
    
    Input:
        size_slave,       ## Put as XXX.shape
        homography_gt,    ## Slave to Master
        homography_est,   ## Slave to Master
        num_pts_w=21,
        num_pts_h=21
    
    Output:
        error_proj, 
        pts, 
        pts_proj
    """
    
    ## Slave Image Size
    width  = size_slave[1]
    height = size_slave[0]
    
    ## Step of projected points
    step_width  =  width / (num_pts_w - 1)
    step_height = height / (num_pts_h - 1)    
    
    ## Set Projected Points
    pts = set_projected_points(num_pts_w, num_pts_h, step_width, step_height)
        
    ## Projection
    pts_proj_gt  = np.dot(pts, np.transpose(homography_gt))
    pts_proj_est = np.dot(pts, np.transpose(homography_est))
        
    ## Norm
    norms = np.linalg.norm(pts_proj_gt - pts_proj_est, axis=1)
        
    ## Re-Projection Error
    error_homography = np.average(norms)
    
    ## Return
    return [error_homography, pts_proj_est, pts_proj_gt]


## EOF