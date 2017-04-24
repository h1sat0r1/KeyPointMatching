# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
    KpMatch.py
        Created on Tue Jul 12 22:36:03 2016
        @author: h1sat0r1
----------------------------------------------------------------------------"""


""" Import """
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt


""" Const Numbers """
NN_DIST_RATIO      = 0.75
MIN_MATCH_COUNT    = 8
THRESH_RANSAC      = 0.50
PARAMS_DRAW        = dict(matchColor=(0,255,255),singlePointColor=(255,0,0),flags=0)
NUM_HIST_ANGLE     = 360 #DO NOT CHANGE THIS
NUM_HIST_OCTAVE    = 8
THRESH_HIST_ANGLE  = 15
THRESH_HIST_OCTAVE = 2



"""============================================================================
    createHist()
============================================================================"""
def createHist(_kp0, _kp1, _matches):

    """
    Creating histograms of dif of angle and octave
    """
    

    """ Create lists """
    hist_angle  = [0] * NUM_HIST_ANGLE
    hist_octave = [0] * (NUM_HIST_OCTAVE * 2 + 1)
    
    
    """ Roop for all matches """
    for m in _matches:

        """ Angle """
        gap_angle = int(_kp0[m.queryIdx].angle - _kp1[m.trainIdx].angle + 0.5)
        while (gap_angle < 0):
            gap_angle += NUM_HIST_ANGLE

        hist_angle[gap_angle] += 1


        """ Octave """        
        gap_octave = (_kp0[m.queryIdx].octave&(NUM_HIST_OCTAVE-1)) - (_kp1[m.trainIdx].octave&(NUM_HIST_OCTAVE-1))
        if ((gap_octave < -NUM_HIST_OCTAVE) or (NUM_HIST_OCTAVE < gap_octave)):
            continue

        hist_octave[gap_octave + NUM_HIST_OCTAVE] += 1


    return [hist_angle, hist_octave]



"""============================================================================
    calcDiffHistAngle()
============================================================================"""
def calcDiffHistAngle(_id0, _id1):

    """
    Calcurating the gap of two bins in angle histogram
    """
    

    """ Simple gap """
    dif = _id0 - _id1


    """ Clip in range[0-359] """
    while(not(0 <= dif < NUM_HIST_ANGLE)):
        
        if (dif < 0):
            dif += NUM_HIST_ANGLE
            
        elif(NUM_HIST_ANGLE <= dif):
            dif -= NUM_HIST_ANGLE

       
    return dif



"""============================================================================
    pickGoodMatches()
============================================================================"""
def pickGoodMatches(_kp0, _kp1, _matches):
    
    """
    Picking better matches
    """
    

    """ thresholded by distance """
    g  = []

    """ thresholded by distance, angle and octave """
    g_ = []


    """ Thresholding based on distance """    
    for m1,n1 in _matches:
        f_dist   = (m1.distance < NN_DIST_RATIO * n1.distance)
        if (f_dist):
            g.append(m1)


    """ Creating histograms """
    hist_angle, hist_octave = createHist(_kp0, _kp1, g)


    """ Preview of histograms """
    plt.figure(100)
    plt.title("Angle dif histogram")
    plt.plot(hist_angle)
    plt.figure(101)
    plt.title("Octave dif histogram")
    plt.plot(hist_octave)
    plt.pause(2.0)
    
    
    """ Get max and its index """
    num_max_hist_angle  = max(hist_angle)
    num_max_hist_octave = max(hist_octave)
    id_max_hist_angle   = hist_angle.index(num_max_hist_angle) 
    id_max_hist_octave  = hist_octave.index(num_max_hist_octave) 
    

    """ Thresholding based on angle and octave """    
    for m2 in g:

        """ Angle bin number """
        dif_angle  = int(_kp0[m2.queryIdx].angle - _kp1[m2.trainIdx].angle + 0.5)

        """ Octave bin number """
        dif_octave = (_kp0[m2.queryIdx].octave&(NUM_HIST_OCTAVE-1)) - (_kp1[m2.trainIdx].octave&(NUM_HIST_OCTAVE-1))
        dif_octave += NUM_HIST_OCTAVE + 1

        """ Calcurate the gap from max bin """
        dif_hist_angle  = calcDiffHistAngle(id_max_hist_angle, dif_angle)
        dif_hist_octave = abs(id_max_hist_octave - dif_octave)
        
        """ Flags """
        f_angle  = (dif_hist_angle  < THRESH_HIST_ANGLE)
        f_octave = (dif_hist_octave < THRESH_HIST_OCTAVE)

        """ Add for g_ """
        if (f_angle and f_octave):
            g_.append(m2)

    return g_



"""============================================================================
    Keypoint Match ()
============================================================================"""
def kpMatch(_img0, _img1):

    """ Glayscale """
    gry0 = cv2.cvtColor(_img0, cv2.COLOR_BGR2GRAY)
    gry1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2GRAY)
    
    
    """-------------------------------------------------------
       SIFT, SURF, ORB, etc.
       *Not all of detectors & descriptors
    -------------------------------------------------------"""

    """ Detector """
    detect   = cv2.xfeatures2d.SIFT_create()
    #detect   = cv2.xfeatures2d.SURF_create()
    #detect   = cv2.ORB_create()
    #detect   = cv2.AgastFeatureDetector_create()
    #detect   = cv2.AKAZE_create()
    
 
    """ Descriptor """
    descript = cv2.xfeatures2d.SIFT_create()
    #descript = cv2.xfeatures2d.SURF_create()
    #descript = cv2.xfeatures2d.DAISY_create()
    #descript = cv2.ORB_create()
    #descript = cv2.BRISK_create()
    #descript = cv2.xfeatures2d.FREAK_create()
    #descript = cv2.AKAZE_create()
    
    
    """ Detection """
    kp0 = detect.detect(gry0)
    kp1 = detect.detect(gry1)
    
    
    """ Description """
    kp0, dsc0 = descript.compute(gry0, kp0)
    kp1, dsc1 = descript.compute(gry1, kp1)
    
    
    """ Matching """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(dsc0, dsc1, k=2)
    good = pickGoodMatches(kp0, kp1, matches)
   
    
    """" Compute Homography"""
    if(len(good) < MIN_MATCH_COUNT):
        """ In case of few matches """
        print("[ERROR] Not enough matches are found...\n")
        sys.exit(-1)

    else:
        """ Enough number of matches """
        srcPts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        """ Calculating homography """
        proj2, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, THRESH_RANSAC)
    
    
    """ Draw Matching Result """
    img2 = cv2.drawMatchesKnn(_img0, kp0, _img1, kp1, [good], None, **PARAMS_DRAW)
    
    
    """ Draw Detection&Description Results """
    kpimg0 = cv2.drawKeypoints(_img0, kp0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpimg1 = cv2.drawKeypoints(_img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    
    """ imwrite """
    cv2.imwrite("Kp0.jpg", kpimg0)
    cv2.imwrite("Kp1.jpg", kpimg1)
    cv2.imwrite("Kps.jpg", img2)
    
    return [img2, proj2]
 
 
"""============================================================================
============================================================================"""
if __name__ == "__main__":
    img0 = cv2.imread("input/img1.ppm", cv2.IMREAD_COLOR)
    #img1 = cv2.imread("input/img1_rot90.ppm", cv2.IMREAD_COLOR)
    #img1 = cv2.imread("input/img1_rot180.ppm", cv2.IMREAD_COLOR)
    #img1 = cv2.imread("input/img1_scale0.5.ppm", cv2.IMREAD_COLOR)
    img1 = cv2.imread("input/img1_scale0.25.ppm", cv2.IMREAD_COLOR)

    img, proj = kpMatch(img0,img1)
    
    cv2.imshow("image",img)
    cv2.waitKey()
    cv2.destroyAllWindows()
