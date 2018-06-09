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
import pandas as pd
import re
import pickle
from shutil import rmtree
from landmarks import landmarks_convert
from combine import deep_convert
from googlevision import google_convert

def main():
    # set mulit-process number, equal to the number of cores
    # process_number = 1
    filepath = input("Filepath: ")
    start_number = int(input("start_id:"))
    summary_file = ''
    if os.path.isdir('output'):
        rmtree('output')
    os.mkdir('output')
    #pool = Pool(processes=process_number)
    output_array = []
    for f in glob.glob(filepath + '/**/*.*', recursive=True):
    #for f in glob.glob(filepath + '/*.*'):
        #print(f)
        print(str(start_number), f)
        head, tail = os.path.split(f)
        if re.search(r'^\d+ \w+.\w+$', tail) is not None:
        # for standard
            photo_id = re.findall(r'\d+', tail)[0]
        else:
        # for non standard
            photo_id = str(start_number)
        start_number += 1
        try:
            emotion = re.findall(r' ([A-Za-z]+)\.', tail)[0]
        except IndexError:
            emotion = 'Need to check'
        #result = pool.apply_async(conversion, (f,))
        #result.get()
        try:
            # whether needs to use the old 68 landmarks model
            #temp_output = landmarks_convert(f, save_img=True)
            temp_output = None
            if temp_output is None:
                print('deep convert')
                temp_output = deep_convert(f, save_img=True)
                if temp_output is None:
                    print('google convert')
                    temp_output = google_convert(f, save_img=True)
                    if temp_output is None:
                        summary_file += f + ': fail to find the front face\n'
                    else:
                        summary_file += f + ': use google convert\n'
                else:
                    summary_file += f + ': use deep convert\n'
            else:
                summary_file += f + ': use 68 landmarks\n'
            output_array.append([photo_id, temp_output, emotion, tail])
            # if cannot convert, temp_output == None
        except:
            summary_file += f + ': fail to convert\n'
            print(f, ': fail to extract front face\n')
    #pool.close()
    output_pd = pd.DataFrame(output_array, columns =['id', 'pixels', 'emotion', 'original_file'])

    with open('pixel.pd', 'wb') as fout:
        pickle.dump(output_pd, fout)
    with open('summary.txt', 'w', errors ='ignore') as fout:
        fout.write(summary_file)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # must run for windows
    main()
