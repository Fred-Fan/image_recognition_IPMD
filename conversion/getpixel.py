import cv2
import glob
import os
import re
import pickle
import pandas as pd


def main():
    # set mulit-process number, equal to the number of cores
    # process_number = 1
    filepath = input("Filepath: ")
    output_array = []
    i = 0
    for f in glob.glob(filepath + '/*.*'):
        #print(f)
        image = cv2.imread(f)
        output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = output.flatten()
        if len(output) != 96*96:
            print(f+': is not 96*96, will not be added to pd')
        else:
            head, tail = os.path.split(f)
            try:
                photo_id = re.findall(r'\d+', tail)[0]
            except IndexError:
                photo_id = i
            try:
                #for IPMD
                #emotion = re.findall(r'[ \-\_]([a-zA-Z]+\s?)[\.\(]', tail)[0]
                #for Kaggle
                emotion = re.findall(r'(\w+)((-T.*)|(\.[a-zA-Z]+))', tail)[0][0]
            except IndexError:
                print('Need to check'+f)
                emotion = 'Need to check'
            i += 1
            output_array.append([photo_id, output, emotion.strip().title(), tail])
    print(str(i)+' images are processed')
    output_pd = pd.DataFrame(output_array, columns =['id', 'pixels', 'emotion', 'original_file'])

    with open('pixel.pd', 'wb') as fout:
        pickle.dump(output_pd, fout)

if __name__ == "__main__":
    main()