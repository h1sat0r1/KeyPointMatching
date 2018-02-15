# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
    KpMatch.py
        Created on Tue Jul 12 22:36:03 2016
        @author: h1sat0r1
----------------------------------------------------------------------------"""


""" Import """
import os
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt


""" Const Numbers """
PARAMS_DRAW        = dict(matchColor=(0,255,255), singlePointColor=(255,0,0), flags=0)
NUM_HIST_ANGLE     = 360 #DO NOT CHANGE THIS
NUM_HIST_OCTAVE    = 8

""" Thresh """
THRESH_NN_DIST_RATIO   = 0.70
THRESH_RANSAC          = 0.50
THRESH_MIN_MATCH_COUNT = 8
THRESH_HIST_ANGLE      = 15
THRESH_HIST_OCTAVE     = 2

""" Keypoint Detector """
DETECTOR   = 'SIFT' #'SIFT' 'SURF' 'ORB' 'AGAST' 'AKAZE'

""" Keypoint Descriptor """
DESCRIPTOR = 'SIFT' #'SIFT' 'SURF' 'DAISY' 'ORB' 'BRISK' 'FREAK' 'AKAZE'

""" Input Files """
#FILENAME_IMAGE_0   = "input/graffiti/img1.ppm"
#FILENAME_IMAGE_1   = "input/graffiti/img3.ppm"
FILENAME_IMAGE_0   = "input/boat/img1.pgm"
FILENAME_IMAGE_1   = "input/boat/img4.pgm"

""" Output directory """
DIRNAME_OUTPUT     = "./output"
DIR_OUTPUT         = DIRNAME_OUTPUT + "/"


"""============================================================================
    dCreate()
============================================================================"""
def dCreate(_detType, _desType):
    
    print('Detector  : ' + _detType)
    print('Descriptor: ' + _desType)
    
    """ Keypoint Detector """
    if   _detType == 'SIFT':
        detect_ = cv2.xfeatures2d.SIFT_create()

    elif _detType == 'SURF':
        detect_ = cv2.xfeatures2d.SURF_create()

    elif _detType == 'ORB':
        detect_ = cv2.ORB_create()

    elif _detType == 'AGAST':
        detect_ = cv2.AgastFeatureDetector_create()

    elif _detType == 'AKAZE':
        detect_ = cv2.AKAZE_create()
        
    else:
        detect_ = cv2.ORB_create()
        print('No keypoint detection method is specified.')
        print('Use ORB detector.')

 
    """ Keypoint Descriptor """
    if   _desType == 'SIFT':
        descript_ = cv2.xfeatures2d.SIFT_create()

    elif _desType == 'SURF':
        descript_ = cv2.xfeatures2d.SURF_create()
        
    elif _desType == 'DAISY':
        descript_ = cv2.xfeatures2d.DAISY_create()

    elif _desType == 'ORB':
        descript_ = cv2.ORB_create()

    elif _desType == 'BRISK':
        descript_ = cv2.BRISK_create()

    elif _desType == 'FREAK':
        descript_ = cv2.xfeatures2d.FREAK_create()

    elif _desType == 'AKAZE':
        descript_ = cv2.AKAZE_create()    

    else:
        descript_ = cv2.ORB_create()
        print('No keypoint description method is specified.')
        print('Use ORB descriptor.')
    
    return [detect_, descript_]



"""============================================================================
    createHist()
============================================================================"""
def createHist(_kp0, _kp1, _matches,
               _num_hist_angle=NUM_HIST_ANGLE,
               _num_hist_octave=NUM_HIST_OCTAVE
               ):

    """
    Creating histograms of dif of angle and octave
    """
    

    """ Create lists """
    hist_angle  = [0] * _num_hist_angle
    hist_octave = [0] * (_num_hist_octave * 2 + 1)
    
    
    """ Roop for all matches """
    for m in _matches:

        """ Angle """
        gap_angle = int(_kp1[m.trainIdx].angle - _kp0[m.queryIdx].angle + 0.5)
        while (gap_angle < 0):
            gap_angle += NUM_HIST_ANGLE

        hist_angle[gap_angle] += 1


        """ Octave """        
        gap_octave = ((_kp1[m.trainIdx].octave&0xFF) - (_kp0[m.queryIdx].octave&0xFF))
        if ((gap_octave < -_num_hist_octave) or (_num_hist_octave < gap_octave)):
            continue

        hist_octave[gap_octave + _num_hist_octave] += 1


    return [hist_angle, hist_octave]



"""============================================================================
    calcDiffHistAngle()
============================================================================"""
def calcDiffHistAngle(_id0, _id1,
                      _num_hist_angle=NUM_HIST_ANGLE
                      ):

    """
    Calcurating the gap of two bins in angle histogram
    """
    

    """ Simple gap """
    dif = _id1 - _id0


    """ Clip in range[0-359] """
    while(not(0 <= dif < _num_hist_angle)):
        
        if (dif < 0):
            dif += _num_hist_angle
            
        elif(NUM_HIST_ANGLE <= dif):
            dif -= _num_hist_angle

       
    return dif



"""============================================================================
    pickGoodMatches()
============================================================================"""
def pickGoodMatches(_kp0, _kp1, _matches,
                    _detType=DETECTOR,
                    _desType=DESCRIPTOR,
                    _thresh_NN_dist_ratio=THRESH_NN_DIST_RATIO,
                    _thresh_hist_angle=THRESH_HIST_ANGLE,
                    _thresh_hist_octave=THRESH_HIST_OCTAVE,
                    _thresh_min_match_count=THRESH_MIN_MATCH_COUNT,
                    _thresh_RANSAC=THRESH_RANSAC,
                    _num_hist_angle=NUM_HIST_ANGLE,
                    _num_hist_octave=NUM_HIST_OCTAVE
                    ):
    
    """
    Picking better matches
    """
    
    """ Thresholded by distance """
    g  = []

    """ Thresholded by distance, angle and octave """
    g_ = []


    """ Thresholding based on distance """    
    for m1,n1 in _matches:
        f_dist   = (m1.distance < _thresh_NN_dist_ratio * n1.distance)
        if (f_dist):
            g.append(m1)


    """ Creating histograms """
    hist_angle, hist_octave = createHist(_kp0, _kp1, g, _num_hist_angle=_num_hist_angle, _num_hist_octave=_num_hist_octave)


    """ Angle Hist Preview """
    plt.figure(100, figsize=(12, 6))
    plt.title("Angle dif histogram")
    plt.xticks( np.arange(0, 360, 15) )
    #plt.hist(hist_angle, bins=_num_hist_angle, range=(0,360))
    plt.plot(hist_angle)
    #plt.show()

    """ Octve Hist Preview """
    plt.figure(101, figsize=(12, 6))
    plt.title("Octave dif histogram")
    plt.xticks( np.arange(0, 2*_num_hist_octave+1, 1) )
    #plt.hist(hist_octave, bins=_num_hist_octave*2+1, range=(-_num_hist_octave,_num_hist_octave))
    plt.plot(hist_octave)
    #plt.show()    
    plt.pause(.0001)

    
    """ Get max and its index """
    num_max_hist_angle  = max(hist_angle)
    num_max_hist_octave = max(hist_octave)
    id_max_hist_angle   = hist_angle.index(num_max_hist_angle) 
    id_max_hist_octave  = hist_octave.index(num_max_hist_octave) 
    

    """ Thresholding based on angle and octave """    
    for m2 in g:

        """ Angle bin number """
        dif_angle  = int(_kp1[m2.trainIdx].angle - _kp0[m2.queryIdx].angle + 0.5)

        """ Octave bin number """
        dif_octave = (_kp1[m2.trainIdx].octave&0xFF) - (_kp0[m2.queryIdx].octave&0xFF)
        dif_octave += (_num_hist_octave + 1)

        """ Calcurate the gap from max bin """
        dif_hist_angle  = calcDiffHistAngle(id_max_hist_angle, dif_angle)
        dif_hist_octave = abs(id_max_hist_octave - dif_octave)
        
        """ Flags """
        f_angle  = (dif_hist_angle  <= _thresh_hist_angle)
        f_octave = (dif_hist_octave <= _thresh_hist_octave)

        """ Add for g_ """
        if (f_angle and f_octave):
            g_.append(m2)

    return g_


"""============================================================================
    Keypoint Match ()
============================================================================"""
def kpMatch(_img0, _img1,
            _detType=DETECTOR,
            _desType=DESCRIPTOR,
            _thresh_NN_dist_ratio=THRESH_NN_DIST_RATIO,
            _thresh_hist_angle=THRESH_HIST_ANGLE,
            _thresh_hist_octave=THRESH_HIST_OCTAVE,
            _thresh_min_match_count=THRESH_MIN_MATCH_COUNT,
            _thresh_RANSAC=THRESH_RANSAC,
            _num_hist_angle=NUM_HIST_ANGLE,
            _num_hist_octave=NUM_HIST_OCTAVE,
            _dir_output=DIR_OUTPUT
            ):

    """ Glayscale """
    gry0 = cv2.cvtColor(_img0, cv2.COLOR_BGR2GRAY)
    gry1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2GRAY)
    

    """ Detector and Descriptor"""
    detect, descript = dCreate(_detType, _desType)
    
    
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
    if(len(good) < _thresh_min_match_count):
        """ In case of few matches """
        print("[ERROR] Not enough matches are found...\n")
        sys.exit(-1)

    else:
        """ Enough number of matches """
        srcPts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        """ Calculating homography """
        proj2, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, _thresh_RANSAC)
    
    
    """ Draw Matching Result """
    img2 = cv2.drawMatchesKnn(_img0, kp0, _img1, kp1, [good], None, **PARAMS_DRAW)
    
    
    """ Draw Detection&Description Results """
    kpimg0 = cv2.drawKeypoints(_img0, kp0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpimg1 = cv2.drawKeypoints(_img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    
    """ imwrite """
    cv2.imwrite(_dir_output + "_Kp0.jpg", kpimg0)
    cv2.imwrite(_dir_output + "_Kp1.jpg", kpimg1)
    cv2.imwrite(_dir_output + "_Kps.jpg", img2)
    
    return [img2, proj2, mask]
 
 
"""============================================================================
============================================================================"""
if __name__ == "__main__":
    
    """ Input Files """
    img0 = cv2.imread(FILENAME_IMAGE_0, cv2.IMREAD_COLOR)
    img1 = cv2.imread(FILENAME_IMAGE_1, cv2.IMREAD_COLOR)
    

    """ Create Output Dir """
    try:
        os.mkdir(DIRNAME_OUTPUT)
        print('Created Output dir \"' + DIRNAME_OUTPUT + '\".')
        
    except FileExistsError as e:
        #print(e.strerror)
        #print(e.errno)
        #print(e.filename)
        print('Output dir \"'+ e.filename + '\" already exists.')


    """ Matching """    
    img, proj, mask = kpMatch(img0, img1, _detType='AKAZE', _desType='AKAZE')
    wrp = cv2.warpPerspective(img0, proj, (img1.shape[1],img1.shape[0]))


    """ Display Homography Matrix """
    print('Homography mat (img0->img1)')
    print(proj)    
    

    """ Save to File """
    sys.stdout = open(DIR_OUTPUT + 'homography.txt', 'w')
    print(proj, file=sys.stdout) 
    

    """ Merge """
    alpha = 0.5
    beta  = 0.5
    mrg  = cv2.addWeighted(img1, alpha, wrp, beta, 0)
    

    """ Show """
    cv2.imshow("warp", wrp)
    cv2.imshow("merge", mrg)
    cv2.imshow("image", img)
    cv2.imwrite(DIR_OUTPUT + "_Warp.jpg", wrp)
    cv2.imwrite(DIR_OUTPUT + "_Merge.jpg", mrg)
    cv2.waitKey()
    cv2.destroyAllWindows()

#EOF