import cv2
import os
im = cv2.imread("image_scenery.jpg", 0)
blurred = cv2.GaussianBlur(im, (7, 7), 1.166) # apply gaussian blur to the image
blurred_sq = blurred * blurred
sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166)
sigma = (sigma - blurred_sq) ** 0.5
sigma = sigma + 1.0/255 # to make sure the denominator doesn't give DivideByZero Exception
mscn = (im - blurred)/sigma # final MSCN(i, j) image

with open("PIPAL.txt", 'r') as listFile:
    for line in listFile:
        scn_idx, ref, dis, score = line.split()
