# import the necessary packages
from imutils import face_utils
import numpy as np
#import argparse
import imutils
import dlib
import cv2
import multiprocessing
from multiprocessing import Pool
import glob
import os


def conversion(f):
    head, tail = os.path.split(f)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(f)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left = []
        right = []
        up = []
        bottom = []
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                left.append(x)
                right.append(x+w)
                up.append(y)
                bottom.append(y+h)
            left_choose = min(left)
            right_choose = max(right)
            up_choose = min(up)
            bottom_choose = max(bottom)

            if (right_choose-left_choose)>=(bottom_choose-up_choose): 
                diff = round(((right_choose-left_choose)-(bottom_choose-up_choose))/2)
                roi = image[int(up_choose-10-diff/2):int(bottom_choose+10+diff/2), int(left_choose-10):int( right_choose+10)]
            else:
                diff = round(((bottom_choose-up_choose) - (right_choose-left_choose))/2)

                roi = image[int(up_choose-10):int(bottom_choose+10), int(left_choose-10-diff/2):int(right_choose+10+        diff/2)]
            #print(x,x+w,y,y+h )
            #roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=48)
            output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            outfile = 'output/'+tail
            print(outfile)
            cv2.imwrite(outfile,output)



def main():
    process_number = 1

    #pool = Pool(processes=process_number)

    for f in glob.glob('images/*.*'):
        print(f)
        #result = pool.apply_async(conversion, (f,))
        #result.get()
        conversion(f)
    #pool.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # must run for windows
    main()
