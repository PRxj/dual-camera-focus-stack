# calculate sift value and align images

import numpy as np
import cv2

def align_images(images):

    images = []
    detector = cv2.xfeatures2d.SIFT_create()
    outimages.append(images[0])
    image1gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    image_1_kp, image_1_desc = detector.detectAndCompute(image1gray, None)

    for i in range(1,len(images)):
        image_i_kp, image_i_desc = detector.detectAndCompute(images[i], None)
        bf = cv2.BFMatcher()
        pairMatches = bf.knnMatch(image_i_desc,image_1_desc, k=2)
        rawMatches = []
        for m,n in pairMatches:
          if m.distance < 0.7*n.distance:
             rawMatches.append(m)
          else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            rawMatches = bf.match(image_i_desc, image_1_desc)

        sortMatches = sorted(rawMatches, key=lambda x: x.distance)
        matches = sortMatches[0:128]

        hom = findHomography(image_i_kp, image_1_kp, matches)
        newimage = cv2.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR)
        outimages.append(newimage)
        
    return outimages
