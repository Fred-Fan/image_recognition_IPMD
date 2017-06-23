import cv2
import numpy as np
import copy
import math
import imutils
#from appscript import app

# parameters
cap_region_x_begin=0  # start point/total width
cap_region_y_end=0  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0
    
blurValue = 17
minthres = 210
maxthres = 260  
stepthres = 20
  
frame = cv2.imread("dataset\\hand2.jpg")
frame = imutils.resize(frame, width=500)
frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
#frame = cv2.flip(frame, 1)  # flip the frame horizontally
cv2.imshow('original', frame)


 
img = frame
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray to threshold
'''
ret, thresh_fin = cv2.threshold(gray, minthres, 255, cv2.THRESH_BINARY)
for i in range(minthres+stepthres, maxthres, stepthres):
    ret, thresh = cv2.threshold(gray, i, 255, cv2.THRESH_BINARY)
    thresh_fin += thresh
mb = cv2.medianBlur(thresh_fin,5)
cv2.imshow('thres', mb)'''

ret, thresh_gray = cv2.threshold(gray, minthres, 255, cv2.THRESH_BINARY)

# guassianblur to threshold 
blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
cv2.imshow('blur', blur)

ret, thresh_fin = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
thresh_blur =thresh_fin

for i in range(minthres, maxthres, stepthres):
    ret, thresh = cv2.threshold(blur, i, 255, cv2.THRESH_BINARY)
    thresh_fin += thresh
mb = cv2.medianBlur(thresh_fin,5)
cv2.imshow('median_blur', mb)



# get the coutours
thresh1 = copy.deepcopy(mb)
im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
length = len(contours)
maxArea = -1
if length > 0:
    for i in range(length):  # find the biggest contour (according to area)
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > maxArea:
            maxArea = area
            ci = i

    res = contours[ci]
    hull = cv2.convexHull(res)
    drawing = np.zeros(img.shape, np.uint8)
    cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

    isFinishCal,cnt = calculateFingers(res,drawing)
    if triggerSwitch is True:
        if isFinishCal is True and cnt <= 2:
            print(cnt)
            #app('System Events').keystroke(' ')  # simulate pressing blank space

cv2.imshow('contour', drawing)

# get the coutours
thresh1 = copy.deepcopy(thresh_gray)
im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
length = len(contours)
maxArea = -1
if length > 0:
    for i in range(length):  # find the biggest contour (according to area)
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > maxArea:
            maxArea = area
            ci = i

    res = contours[ci]
    hull = cv2.convexHull(res)
    drawing = np.zeros(img.shape, np.uint8)
    cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

    isFinishCal,cnt = calculateFingers(res,drawing)
    if triggerSwitch is True:
        if isFinishCal is True and cnt <= 2:
            print(cnt)
            #app('System Events').keystroke(' ')  # simulate pressing blank space

cv2.imshow('contour_blur', drawing)

k = cv2.waitKey(0)

